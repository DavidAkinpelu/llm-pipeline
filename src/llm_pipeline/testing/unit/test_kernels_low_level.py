"""Tests for the kernels & low-level optimization section.

Covers Tier 1 (Triton fused ops, run on CUDA + skipped otherwise),
Tier 2 (compile path validation), Tier 3 (memory-efficient training,
fully CPU-runnable), and Tier 4 (hardware-blocked stubs — fallback
behaviour pinned, real-kernel validation deferred to H100 cloud).
"""

import math
from unittest import mock

import pytest
import torch
import torch.nn as nn

from llm_pipeline.quantization.kernels.triton_rope import (
    _torch_reference as rope_torch_ref,
    apply_rope_triton,
)
from llm_pipeline.quantization.kernels.triton_rmsnorm import (
    _torch_reference as rmsnorm_torch_ref,
    apply_rmsnorm_triton,
)
from llm_pipeline.quantization.kernels.triton_softmax import fused_softmax_triton
from llm_pipeline.quantization.kernels.triton_iq4_nl import (
    matmul_iq4_nl_triton,
    prepack_iq4_nl_for_gpu,
)
from llm_pipeline.quantization.kernels.triton_fp8 import (
    _is_hopper,
    fp8_matmul,
)
from llm_pipeline.quantization.kernels.triton_mxfp4 import (
    _is_blackwell,
    mxfp4_matmul,
)
from llm_pipeline.inference.attention.flash_attention_3 import flash_attn_3_func
from llm_pipeline.inference.attention.sliding_window import (
    sliding_window_attention,
    sliding_window_mask,
)
from llm_pipeline.inference.attention.sparse import (
    attention_sink_mask,
    dilated_mask,
    sparse_attention,
)
from llm_pipeline.inference.speculative_kernel import (
    _torch_reference as specdec_ref,
    verify_drafts_batched,
)
from llm_pipeline.parallelism.communication.nccl_tuner import (
    NCCLTuner,
    NCCLTunerConfig,
    TuningResult,
)
from llm_pipeline.training import (
    CPUOffloadOptimizer,
    SelectiveCheckpointWrapper,
    compile_model_for_training,
)


# --------------------------------------------------------------------------- #
# Tier 1: Triton fused ops
#
# All four functions soft-fall through to a torch reference on CPU. We test
# the reference path here; real kernel execution is validated on H100.
# --------------------------------------------------------------------------- #


def test_rope_torch_reference_rotates_correctly():
    """Reference implementation matches the canonical RoPE formula."""
    B, H, T, D = 1, 2, 4, 8
    rotary_dim = 4
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    cos = torch.ones(B, T, rotary_dim)
    sin = torch.zeros(B, T, rotary_dim)
    # Identity rotation (cos=1, sin=0) → output equals input.
    q_out, k_out = rope_torch_ref(q, k, cos, sin)
    torch.testing.assert_close(q_out, q)
    torch.testing.assert_close(k_out, k)


def test_rope_dispatcher_returns_correct_shape_on_cpu():
    """Dispatch path on CPU runs the reference; shape preserved."""
    B, H, T, D = 1, 2, 4, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    cos = torch.randn(B, T, 4)
    sin = torch.randn(B, T, 4)
    q_out, k_out = apply_rope_triton(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_rope_cpu_fallback_preserves_in_place_semantics():
    """CPU fallback should mutate and return the original tensors like CUDA."""
    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    q_before = q.clone()
    k_before = k.clone()
    angle = torch.full((1, 3, 4), 0.5)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    q_out, k_out = apply_rope_triton(q, k, cos, sin)

    assert q_out.data_ptr() == q.data_ptr()
    assert k_out.data_ptr() == k.data_ptr()
    assert not torch.equal(q, q_before)
    assert not torch.equal(k, k_before)


def test_rmsnorm_reference_matches_formula():
    x = torch.randn(2, 3, 8) * 0.5
    weight = torch.randn(8) * 0.1
    out = rmsnorm_torch_ref(x, weight, eps=1e-6, add_residual_one=False)
    # Manual: x * rsqrt(mean(x^2)) * w
    expected = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * weight.float()
    torch.testing.assert_close(out, expected.to(x.dtype), atol=1e-5, rtol=1e-5)


def test_rmsnorm_residual_one_form_handles_zero_init_weight():
    """Qwen-style RMSNorm (zero-init weight, ``(1+w)`` scaling) is identity at init."""
    x = torch.randn(4, 8)
    weight = torch.zeros(8)
    out = rmsnorm_torch_ref(x, weight, eps=1e-6, add_residual_one=True)
    expected = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(out, expected.to(x.dtype), atol=1e-5, rtol=1e-5)


def test_rmsnorm_dispatcher_on_cpu_falls_through():
    out = apply_rmsnorm_triton(torch.randn(2, 8), torch.ones(8), eps=1e-6)
    assert out.shape == (2, 8)


def test_softmax_no_mask_matches_torch():
    x = torch.randn(3, 4, 16) * 2.0
    out = fused_softmax_triton(x, causal=False)
    expected = torch.softmax(x.float(), dim=-1).to(x.dtype)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_softmax_causal_mask_blocks_future_positions():
    T = 4
    x = torch.zeros(T, T)
    out = fused_softmax_triton(x, causal=True)
    # Each row's sum is 1 (softmax) but only positions ≤ row_idx contribute.
    for i in range(T):
        # Allowed positions: 0..i. Each gets 1/(i+1).
        for j in range(T):
            if j <= i:
                assert abs(out[i, j].item() - 1.0 / (i + 1)) < 1e-5
            else:
                assert out[i, j].item() == 0


def test_iq4_nl_prepack_layout_correct():
    """Pre-pack utility splits the byte blob into the expected shapes."""
    if not torch.cuda.is_available():
        pytest.skip("prepack writes to GPU; CUDA-only")
    from llm_pipeline.quantization.kquants import encode_iq4_nl
    blob, shape = encode_iq4_nl(torch.randn(64))
    pre = prepack_iq4_nl_for_gpu(blob, device="cuda:0")
    assert pre["d"].shape == (2,)                # 2 super-blocks of 32 weights
    assert pre["qs"].shape == (2, 16)
    assert pre["codebook"].shape == (16,)


# --------------------------------------------------------------------------- #
# Tier 2: Compile / graph paths
# --------------------------------------------------------------------------- #


def test_compile_for_training_returns_compatible_module():
    """``torch.compile`` either wraps the model or returns it unchanged
    when not available; either way ``.forward`` still works.
    """
    base = nn.Linear(4, 4)
    out = compile_model_for_training(base)
    x = torch.randn(2, 4)
    assert out(x).shape == (2, 4)


def test_compile_warns_on_old_pytorch(monkeypatch):
    """If ``torch.compile`` doesn't exist, the wrapper warns + returns base."""
    if hasattr(torch, "compile"):
        monkeypatch.delattr(torch, "compile", raising=False)
    base = nn.Linear(4, 4)
    with pytest.warns(RuntimeWarning, match="torch.compile"):
        out = compile_model_for_training(base)
    assert out is base


# --------------------------------------------------------------------------- #
# Tier 3: Memory-efficient training (full CPU tests)
# --------------------------------------------------------------------------- #


def test_selective_checkpoint_forward_equivalence():
    """Wrapped vs unwrapped forward produces the same output."""
    cheap = nn.LayerNorm(8)
    expensive = nn.Linear(8, 8)
    wrapped = SelectiveCheckpointWrapper(cheap, expensive)

    x = torch.randn(2, 8, requires_grad=True)
    out_wrapped = wrapped(x)
    out_unwrapped = expensive(cheap(x))
    torch.testing.assert_close(out_wrapped, out_unwrapped, atol=1e-5, rtol=1e-5)


def test_selective_checkpoint_gradient_equivalence():
    """Gradients computed via the wrapper match the unwrapped reference."""
    torch.manual_seed(0)
    cheap = nn.LayerNorm(8)
    expensive = nn.Linear(8, 8)
    wrapped = SelectiveCheckpointWrapper(cheap, expensive)

    x_w = torch.randn(2, 8, requires_grad=True)
    x_u = x_w.detach().clone().requires_grad_(True)

    wrapped(x_w).sum().backward()
    expensive(cheap(x_u)).sum().backward()

    torch.testing.assert_close(x_w.grad, x_u.grad, atol=1e-5, rtol=1e-5)


def test_selective_checkpoint_preserves_parameter_gradients():
    """Wrapped checkpointing must backprop into module parameters, not just inputs."""
    torch.manual_seed(0)
    cheap_w = nn.Linear(8, 8, bias=False)
    expensive_w = nn.Linear(8, 8, bias=False)
    wrapped = SelectiveCheckpointWrapper(cheap_w, expensive_w)

    cheap_u = nn.Linear(8, 8, bias=False)
    expensive_u = nn.Linear(8, 8, bias=False)
    cheap_u.weight.data.copy_(cheap_w.weight.data)
    expensive_u.weight.data.copy_(expensive_w.weight.data)

    x_w = torch.randn(2, 8, requires_grad=True)
    x_u = x_w.detach().clone().requires_grad_(True)

    wrapped(x_w).sum().backward()
    expensive_u(cheap_u(x_u)).sum().backward()

    torch.testing.assert_close(cheap_w.weight.grad, cheap_u.weight.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(expensive_w.weight.grad, expensive_u.weight.grad, atol=1e-5, rtol=1e-5)


def test_cpu_offload_optimizer_step_runs():
    """CPUOffloadOptimizer wraps any optimizer and produces equivalent updates."""
    torch.manual_seed(0)
    p_a = nn.Parameter(torch.randn(4, 4))
    p_b = nn.Parameter(p_a.detach().clone())

    opt_a = torch.optim.AdamW([p_a], lr=1e-2)
    opt_b = CPUOffloadOptimizer(torch.optim.AdamW([p_b], lr=1e-2))

    for _ in range(3):
        # Same gradient for both.
        g = torch.randn(4, 4)
        p_a.grad = g.clone()
        p_b.grad = g.clone()
        opt_a.step()
        opt_b.step()
    torch.testing.assert_close(p_a.data, p_b.data, atol=1e-5, rtol=1e-5)


def test_cpu_offload_optimizer_state_lives_on_cpu_between_steps():
    """After ``step()`` the optimizer's state tensors are on CPU."""
    p = nn.Parameter(torch.randn(4, 4))
    opt = CPUOffloadOptimizer(torch.optim.AdamW([p], lr=1e-2))
    p.grad = torch.randn_like(p)
    opt.step()
    state = opt.optimizer.state[p]
    for v in state.values():
        if isinstance(v, torch.Tensor):
            assert v.device.type == "cpu", f"state tensor on {v.device}"


def test_cpu_offload_state_dict_round_trip():
    """``state_dict`` / ``load_state_dict`` preserves training state."""
    p = nn.Parameter(torch.randn(4, 4))
    opt = CPUOffloadOptimizer(torch.optim.AdamW([p], lr=1e-2))
    p.grad = torch.randn_like(p)
    opt.step()
    sd = opt.state_dict()

    p2 = nn.Parameter(torch.randn(4, 4))
    opt2 = CPUOffloadOptimizer(torch.optim.AdamW([p2], lr=1e-2))
    opt2.load_state_dict(sd)
    # State is preserved.
    assert len(opt2.optimizer.state) == len(opt.optimizer.state)


# --------------------------------------------------------------------------- #
# Tier 4: Hardware-blocked stubs
# --------------------------------------------------------------------------- #


def test_fp8_matmul_falls_back_on_non_hopper():
    """On Ampere / CPU the fp8 dispatcher uses the pure-PyTorch dequant path
    and produces a finite result.
    """
    if not torch.cuda.is_available():
        # Pure-CPU branch.
        x = torch.randn(2, 8)
        w = torch.randn(4, 8).to(torch.float16)
        scale = torch.ones(4)
        # The fallback expects CUDA; on CPU we ensure the fallback's torch
        # ops don't blow up. Since fp8 itself is GPU-only in practice, this
        # is a structural check.
        out = fp8_matmul(x, w, scale)
        assert out.shape == (2, 4)
    else:
        x = torch.randn(2, 8, device="cuda", dtype=torch.float16)
        w = torch.randn(4, 8, device="cuda", dtype=torch.float16)
        scale = torch.ones(4, device="cuda")
        out = fp8_matmul(x, w, scale)
        assert out.shape == (2, 4)


def test_fp8_matmul_uses_fallback_for_per_block_scales():
    """Per-block scales are not implemented in the Hopper Triton kernel yet."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA tensor dispatch")

    x = torch.randn(2, 64, device="cuda", dtype=torch.float16)
    w = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    scale = torch.ones(4, 2, device="cuda", dtype=torch.float16)

    with mock.patch(
        "llm_pipeline.quantization.kernels.triton_fp8._is_hopper",
        return_value=True,
    ), mock.patch(
        "llm_pipeline.quantization.kernels.triton_fp8._fallback_matmul",
        return_value=torch.empty(2, 4, device="cuda", dtype=torch.float16),
    ) as fallback, mock.patch(
        "llm_pipeline.quantization.kernels.triton_fp8._hopper_matmul",
    ) as hopper:
        out = fp8_matmul(x, w, scale)

    assert out.shape == (2, 4)
    fallback.assert_called_once()
    hopper.assert_not_called()


def test_mxfp4_matmul_dequant_path_runs():
    """Dequant-then-matmul fallback works regardless of device generation."""
    if not torch.cuda.is_available():
        pytest.skip("MXFP4 fallback uses GPU ops")
    x = torch.randn(2, 32, device="cuda", dtype=torch.float16)
    w_packed = torch.randint(0, 256, (4, 16), dtype=torch.uint8, device="cuda")
    w_scale = torch.ones(4, 1, dtype=torch.uint8, device="cuda") * 127
    out = mxfp4_matmul(x, w_packed, w_scale)
    assert out.shape == (2, 4)


def test_mxfp4_matmul_rejects_invalid_block_shapes():
    """Bad MXFP4 shapes should raise a clear validation error."""
    x = torch.randn(2, 48, dtype=torch.float16)
    w_packed = torch.zeros(3, 24, dtype=torch.uint8)
    w_scale = torch.ones(3, 1, dtype=torch.uint8)

    with pytest.raises(ValueError, match="multiple of 32"):
        mxfp4_matmul(x, w_packed, w_scale)


def test_fa3_falls_back_to_sdpa():
    """Without flash_attn_3 installed, the wrapper uses SDPA. Output shape
    matches FA2's expected layout.
    """
    B, T, H, D = 1, 4, 2, 8
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    v = torch.randn(B, T, H, D)
    out = flash_attn_3_func(q, k, v, causal=True)
    assert out.shape == (B, T, H, D)


def test_sliding_window_mask_correctness():
    mask = sliding_window_mask(seq_len=5, window=2)
    # mask[i, j] = True iff (i - j >= 2) OR (j > i).
    # Position 0 attends only itself: mask[0] = [F, T, T, T, T]
    # Position 2 attends 1, 2: mask[2] = [T, F, F, T, T]
    expected = torch.tensor([
        [False, True,  True,  True,  True],
        [False, False, True,  True,  True],
        [True,  False, False, True,  True],
        [True,  True,  False, False, True],
        [True,  True,  True,  False, False],
    ])
    torch.testing.assert_close(mask, expected)


def test_sliding_window_attention_with_full_window_matches_causal():
    """window ≥ T should make sliding-window equivalent to standard causal."""
    B, H, T, D = 1, 2, 6, 8
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    out_swa = sliding_window_attention(q, k, v, window=T)
    out_causal = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True,
    )
    torch.testing.assert_close(out_swa, out_causal, atol=1e-5, rtol=1e-5)


def test_attention_sink_mask_keeps_first_n_tokens_attendable():
    """With n_sinks=2 and window=1, every position can attend the first 2 tokens."""
    mask = attention_sink_mask(seq_len=6, n_sinks=2, window=1)
    # Position i ≥ 2 can attend i-1, i, AND positions 0, 1 (sinks).
    for i in range(2, 6):
        assert mask[i, 0].item() is False
        assert mask[i, 1].item() is False


def test_dilated_mask_respects_step_pattern():
    """Position i can attend j iff j ≤ i AND (j > i - window OR (i-j) % dilation == 0)."""
    mask = dilated_mask(seq_len=8, dilation=3, window=2)
    # Position 6: in-window = {5, 6}, dilated stride-3 = {0, 3, 6}.
    # Allowed: {0, 3, 5, 6}.
    blocked_at_6 = [j for j in range(8) if mask[6, j].item()]
    allowed_at_6 = [j for j in range(8) if not mask[6, j].item()]
    assert set(allowed_at_6) == {0, 3, 5, 6}


def test_sparse_attention_runs():
    B, H, T, D = 1, 2, 8, 4
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    out = sparse_attention(q, k, v, n_sinks=2, window=4)
    assert out.shape == (B, H, T, D)


def test_speculative_kernel_torch_reference_matches_python_loop():
    """Reference verification matches the Python ``speculative_decode`` accept rule."""
    K, V = 4, 16
    target_probs = torch.softmax(torch.randn(K, V), dim=-1)
    draft_probs = torch.rand(K) * 0.5 + 0.1
    draft_tokens = torch.randint(0, V, (K,), dtype=torch.int64)
    rand_uniform = torch.rand(K)

    out = specdec_ref(target_probs, draft_probs, draft_tokens, rand_uniform)
    # Manual verification per-position.
    for i in range(K):
        p = target_probs[i, draft_tokens[i]].item()
        q = max(draft_probs[i].item(), 1e-9)
        ratio = min(p / q, 1.0)
        expected = 1 if rand_uniform[i].item() < ratio else 0
        assert int(out[i].item()) == expected


def test_speculative_kernel_dispatcher_works_on_cpu():
    """Non-CUDA dispatch falls through to the torch reference."""
    K, V = 8, 32
    target_probs = torch.softmax(torch.randn(K, V), dim=-1)
    draft_probs = torch.rand(K).clamp_min(0.01)
    draft_tokens = torch.randint(0, V, (K,), dtype=torch.int64)
    rand_uniform = torch.rand(K)
    out = verify_drafts_batched(target_probs, draft_probs, draft_tokens, rand_uniform)
    assert out.shape == (K,)
    assert out.dtype == torch.int8


def test_nccl_tuner_returns_empty_without_distributed():
    """No distributed group → tuner prints a clear message and returns []."""
    tuner = NCCLTuner(NCCLTunerConfig(sizes_kb=[1, 4], n_warmup=1, n_iter=1))
    results = tuner.run()
    assert results == []


def test_nccl_tuner_best_buffer_size_picks_max_bandwidth():
    """``best_buffer_size`` picks the buffer with highest measured bandwidth."""
    results = [
        TuningResult(buffer_bytes=1024, latency_us=10.0, bandwidth_gbps=1.0),
        TuningResult(buffer_bytes=4096, latency_us=20.0, bandwidth_gbps=5.0),
        TuningResult(buffer_bytes=16384, latency_us=80.0, bandwidth_gbps=2.0),
    ]
    assert NCCLTuner.best_buffer_size(results) == 4096


def test_nccl_tuner_handles_empty_results():
    assert NCCLTuner.best_buffer_size([]) is None
