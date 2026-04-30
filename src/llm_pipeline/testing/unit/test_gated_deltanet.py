"""Tests for the Gated DeltaNet linear-attention module (Qwen3.5/3.6)."""

import pytest
import torch

from llm_pipeline.models.qwen3_5 import (
    GatedDeltaNetConfig,
    Qwen3_5GatedDeltaNet,
    RMSNormGated,
    l2norm,
    recurrent_gated_delta_rule,
)


def _cfg(**overrides) -> GatedDeltaNetConfig:
    base = dict(
        hidden_size=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,        # GQA: 2 → 4 (each k head shared by 2 v heads)
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    base.update(overrides)
    return GatedDeltaNetConfig(**base)


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def test_l2norm_returns_unit_vectors():
    x = torch.randn(3, 5, 7)
    y = l2norm(x, dim=-1)
    norms = y.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-3, rtol=1e-3)


def test_l2norm_handles_zero_input():
    """The eps inside rsqrt prevents NaN on zero vectors — the result is just
    a vector with very small magnitude, not unit. This matches the FLA convention.
    """
    x = torch.zeros(2, 4)
    y = l2norm(x, dim=-1)
    assert torch.isfinite(y).all()
    assert (y == 0).all()


def test_rmsnorm_gated_without_z_is_plain_rmsnorm():
    norm = RMSNormGated(8)
    x = torch.randn(4, 8)
    y = norm(x)  # no z
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + norm.eps)
    torch.testing.assert_close(y, expected, atol=1e-5, rtol=1e-5)


def test_rmsnorm_gated_with_z_applies_silu_gate():
    norm = RMSNormGated(8)
    x = torch.randn(4, 8)
    z = torch.randn(4, 8)
    y = norm(x, z)
    # Reproducing the formula: rmsnorm(x * silu(z))
    gated = x * torch.nn.functional.silu(z)
    expected = gated * torch.rsqrt(gated.pow(2).mean(dim=-1, keepdim=True) + norm.eps)
    torch.testing.assert_close(y, expected, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# Pure-recurrence numerical properties
# --------------------------------------------------------------------------- #


def _rand_recurrence_inputs(B=2, T=6, H=3, D_k=4, D_v=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, D_k, generator=g)
    k = torch.randn(B, T, H, D_k, generator=g)
    v = torch.randn(B, T, H, D_v, generator=g)
    # log-decay: small negative numbers → α just under 1.
    g_log = -torch.rand(B, T, H, generator=g) * 0.1
    beta = torch.rand(B, T, H, generator=g)
    return q, k, v, g_log, beta


def test_recurrence_output_shape():
    q, k, v, g, beta = _rand_recurrence_inputs(D_v=5)
    out, state = recurrent_gated_delta_rule(q, k, v, g=g, beta=beta)
    assert out.shape == (2, 6, 3, 5)
    assert state.shape == (2, 3, 4, 5)


def test_recurrence_with_beta_zero_keeps_initial_state_decaying():
    """β=0 disables the delta update → state is purely α-decayed across steps,
    so the readout y_t = q_t @ (α^t · S_0) should equal the closed-form decay.
    """
    B, T, H, D_k, D_v = 1, 5, 1, 3, 3
    q = torch.randn(B, T, H, D_k)
    k = torch.randn(B, T, H, D_k)        # ignored (β=0)
    v = torch.randn(B, T, H, D_v)        # ignored (β=0)
    g = torch.full((B, T, H), -0.1)      # α = exp(-0.1)
    beta = torch.zeros(B, T, H)

    initial = torch.randn(B, H, D_k, D_v)
    out, _ = recurrent_gated_delta_rule(q, k, v, g=g, beta=beta,
                                        initial_state=initial, use_qk_l2norm=False)
    # Closed-form: at step t the state is alpha^(t+1) * initial; output is q_t @ that.
    alpha = torch.tensor(-0.1).exp()
    scale = D_k ** -0.5  # query is scaled inside the recurrence
    for t in range(T):
        S_t = (alpha ** (t + 1)) * initial.float()
        expected = (q[:, t].float() * scale).unsqueeze(-1) * S_t
        expected = expected.sum(dim=-2)
        torch.testing.assert_close(out[:, t].float(), expected, atol=1e-4, rtol=1e-4)


def test_recurrence_is_strictly_causal():
    """Changing input at step t must NOT affect outputs at steps < t."""
    q, k, v, g, beta = _rand_recurrence_inputs()
    out_baseline, _ = recurrent_gated_delta_rule(q, k, v, g=g, beta=beta)

    # Perturb step T-1 and confirm earlier outputs are unchanged.
    v2 = v.clone()
    v2[:, -1] = torch.randn_like(v2[:, -1]) * 100.0
    out_perturbed, _ = recurrent_gated_delta_rule(q, k, v2, g=g, beta=beta)

    torch.testing.assert_close(out_baseline[:, :-1], out_perturbed[:, :-1])
    assert not torch.allclose(out_baseline[:, -1], out_perturbed[:, -1])


def test_recurrence_state_chaining_matches_one_shot():
    """Splitting a sequence at position s and feeding the state from the
    first half as initial_state for the second half should reproduce the
    one-shot output exactly.
    """
    B, T, H, D_k, D_v = 1, 8, 2, 4, 4
    q, k, v, g, beta = _rand_recurrence_inputs(B=B, T=T, H=H, D_k=D_k, D_v=D_v, seed=42)

    # One-shot.
    full_out, final_state = recurrent_gated_delta_rule(q, k, v, g=g, beta=beta)

    # Split into two halves; chain the state.
    s = 3
    a_out, a_state = recurrent_gated_delta_rule(
        q[:, :s], k[:, :s], v[:, :s], g=g[:, :s], beta=beta[:, :s],
        output_final_state=True,
    )
    b_out, b_state = recurrent_gated_delta_rule(
        q[:, s:], k[:, s:], v[:, s:], g=g[:, s:], beta=beta[:, s:],
        initial_state=a_state, output_final_state=True,
    )

    chained = torch.cat([a_out, b_out], dim=1)
    torch.testing.assert_close(chained, full_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(b_state, final_state, atol=1e-4, rtol=1e-4)


def test_recurrence_initial_state_zero_equals_no_initial():
    q, k, v, g, beta = _rand_recurrence_inputs()
    B, _, H, D_k = q.shape
    D_v = v.shape[-1]
    zero_state = torch.zeros(B, H, D_k, D_v)
    out_a, _ = recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None)
    out_b, _ = recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=zero_state)
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# Full module
# --------------------------------------------------------------------------- #


def test_module_forward_shape_preserved():
    cfg = _cfg()
    mod = Qwen3_5GatedDeltaNet(cfg)
    x = torch.randn(2, 7, cfg.hidden_size)
    y, state = mod(x)
    assert y.shape == x.shape
    assert state is None  # default return_final_state=False


def test_module_returns_cache_on_request():
    """``return_final_state=True`` returns ``(recurrent_state, conv_state)``.

    Recurrent shape: ``[B, num_v_heads, D_k, D_v]``.
    Conv state shape: ``[B, conv_dim, conv_kernel - 1]`` where
    conv_dim = key_dim*2 + value_dim.
    """
    cfg = _cfg()
    mod = Qwen3_5GatedDeltaNet(cfg)
    x = torch.randn(2, 7, cfg.hidden_size)
    _, cache = mod(x, return_final_state=True)
    rec, conv = cache
    assert rec.shape == (2, cfg.linear_num_value_heads,
                         cfg.linear_key_head_dim, cfg.linear_value_head_dim)
    expected_conv_dim = (
        cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
        + cfg.linear_num_value_heads * cfg.linear_value_head_dim
    )
    assert conv.shape == (2, expected_conv_dim, cfg.linear_conv_kernel_dim - 1)


def test_module_rejects_invalid_kv_ratio():
    """num_v_heads must be a multiple of num_k_heads for GQA repeat."""
    with pytest.raises(ValueError, match="multiple"):
        _cfg(linear_num_key_heads=3, linear_num_value_heads=4)
        Qwen3_5GatedDeltaNet(GatedDeltaNetConfig(
            hidden_size=32,
            linear_num_key_heads=3, linear_num_value_heads=4,
            linear_key_head_dim=8, linear_value_head_dim=8,
        ))


def test_module_is_strictly_causal_end_to_end():
    cfg = _cfg()
    mod = Qwen3_5GatedDeltaNet(cfg).eval()
    x = torch.randn(1, 6, cfg.hidden_size)
    y_baseline, _ = mod(x)

    x2 = x.clone()
    x2[:, 4:] = torch.randn_like(x2[:, 4:]) * 5.0
    y_perturbed, _ = mod(x2)

    # Steps 0..3 must be unchanged. The conv1d kernel is 4 with causal
    # padding, so step 4's output already sees its own input — no leak earlier.
    torch.testing.assert_close(y_baseline[:, :4], y_perturbed[:, :4], atol=1e-5, rtol=1e-5)


def test_module_gradient_flows_through_all_projections():
    cfg = _cfg()
    mod = Qwen3_5GatedDeltaNet(cfg)
    x = torch.randn(2, 5, cfg.hidden_size, requires_grad=True)
    y, _ = mod(x)
    y.sum().backward()
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        proj = getattr(mod, name)
        assert proj.weight.grad is not None and proj.weight.grad.abs().sum() > 0, (
            f"no gradient on {name}.weight"
        )
    assert mod.A_log.grad is not None and mod.A_log.grad.abs().sum() > 0
    assert mod.dt_bias.grad is not None and mod.dt_bias.grad.abs().sum() > 0
    assert mod.conv1d.weight.grad is not None and mod.conv1d.weight.grad.abs().sum() > 0


def test_module_full_cache_chaining_matches_one_shot():
    """End-to-end streaming test: feed a sequence in two chunks via the full
    ``(recurrent_state, conv_state)`` cache and confirm the output matches a
    single-shot run on the same sequence.

    This is the property that makes hybrid KV decode correct.
    """
    cfg = _cfg()
    mod = Qwen3_5GatedDeltaNet(cfg).eval()
    x = torch.randn(1, 10, cfg.hidden_size)

    full, _ = mod(x)

    s = 4
    a_out, a_cache = mod(x[:, :s], return_final_state=True)
    b_out, _ = mod(
        x[:, s:],
        initial_state=a_cache[0],
        conv_state=a_cache[1],
        return_final_state=False,
    )
    chained = torch.cat([a_out, b_out], dim=1)
    torch.testing.assert_close(chained, full, atol=1e-4, rtol=1e-4)


def test_module_no_conv_state_matches_zero_init_padding():
    """Calling without a conv_state should be equivalent to calling with
    zero conv_state — the explicit pad path is the same as the implicit one.
    """
    cfg = _cfg()
    mod = Qwen3_5GatedDeltaNet(cfg).eval()
    x = torch.randn(1, 5, cfg.hidden_size)

    out_implicit, _ = mod(x)
    conv_dim = (
        cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
        + cfg.linear_num_value_heads * cfg.linear_value_head_dim
    )
    zero_conv = torch.zeros(1, conv_dim, cfg.linear_conv_kernel_dim - 1)
    out_explicit, _ = mod(x, conv_state=zero_conv)
    torch.testing.assert_close(out_implicit, out_explicit, atol=1e-5, rtol=1e-5)
