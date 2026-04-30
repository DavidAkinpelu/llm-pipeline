"""Triton Q4_K kernel correctness tests.

Skipped when CUDA isn't available or Triton isn't installed — both common
in CI runners.
"""

import pytest
import torch

from llm_pipeline.quantization.kquants import decode_q4_k, encode_q4_k


cuda_available = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


pytestmark = pytest.mark.skipif(
    not (cuda_available and _HAS_TRITON),
    reason="Triton kernel tests require CUDA + triton",
)


@pytest.fixture
def gpu_weight():
    """Build a small Q4_K-encoded weight on GPU + the FP16 reference."""
    from llm_pipeline.quantization.kernels import prepack_q4_k_for_gpu
    torch.manual_seed(0)
    n = 1024
    w_orig = torch.randn(n) * 0.05
    blob, _ = encode_q4_k(w_orig)
    gpu = prepack_q4_k_for_gpu(blob, device="cuda:0")
    ref = decode_q4_k(blob, (n,)).to("cuda:0").to(torch.float16)
    return gpu, ref, w_orig


def test_dequant_kernel_bit_exact_against_numpy(gpu_weight):
    """Triton dequant must match the numpy decode bit-for-bit (in FP16)."""
    from llm_pipeline.quantization.kernels import dequant_q4_k_triton
    gpu, ref, _ = gpu_weight
    out = dequant_q4_k_triton(gpu)[: ref.numel()]
    diff = (out.float() - ref.float()).abs().max().item()
    # FP16 reference + FP16 output → identical bytes mean diff == 0.
    assert diff == 0.0, f"max abs diff = {diff}"


def test_dequant_kernel_handles_arbitrary_super_block_count():
    """The kernel's BLOCK_M=8 must handle non-multiple-of-8 block counts."""
    from llm_pipeline.quantization.kernels import (
        dequant_q4_k_triton,
        prepack_q4_k_for_gpu,
    )
    torch.manual_seed(0)
    # 5 super-blocks = 1280 weights — partial last block tile.
    n = 5 * 256
    w = torch.randn(n) * 0.05
    blob, _ = encode_q4_k(w)
    gpu = prepack_q4_k_for_gpu(blob, device="cuda:0")
    ref = decode_q4_k(blob, (n,)).to("cuda:0").to(torch.float16)
    out = dequant_q4_k_triton(gpu)
    assert out.shape == ref.shape
    assert (out.float() - ref.float()).abs().max().item() == 0.0


def test_matmul_kernel_matches_dequant_then_fp16_matmul():
    """Fused Q4_K matmul must agree with (dequantize → FP16 matmul) within
    FP16 noise."""
    from llm_pipeline.quantization.kernels import (
        matmul_q4_k_triton,
        prepack_q4_k_for_gpu,
    )
    torch.manual_seed(0)
    M, N, K = 8, 256, 512        # K = 2 super-blocks per row
    W = torch.randn(N, K) * 0.02
    blobs = b"".join(encode_q4_k(W[n])[0] for n in range(N))
    gpu_w = prepack_q4_k_for_gpu(blobs, device="cuda:0")
    W_dq = decode_q4_k(blobs, (N, K)).to("cuda:0").to(torch.float16)
    X = torch.randn(M, K, device="cuda:0", dtype=torch.float16)

    y_ref = (X.float() @ W_dq.float().T).to(torch.float16)
    y_triton = matmul_q4_k_triton(X, gpu_w, n_out=N, n_blocks_per_row=K // 256)

    rel = (
        (y_triton.float() - y_ref.float()).pow(2).sum().sqrt()
        / y_ref.float().pow(2).sum().sqrt()
    ).item()
    assert rel < 1e-2, f"rel L2 = {rel}"


def test_matmul_kernel_rejects_invalid_K():
    """K must be a multiple of 256 (the super-block size)."""
    from llm_pipeline.quantization.kernels import (
        matmul_q4_k_triton,
        prepack_q4_k_for_gpu,
    )
    blob, _ = encode_q4_k(torch.zeros(256))
    gpu_w = prepack_q4_k_for_gpu(blob, device="cuda:0")
    X = torch.randn(2, 200, device="cuda:0", dtype=torch.float16)  # K=200 ≠ multiple of 256
    with pytest.raises(ValueError, match="must equal n_blocks_per_row"):
        matmul_q4_k_triton(X, gpu_w, n_out=1, n_blocks_per_row=1)


def test_prepack_recovers_original_weights():
    """Round-trip: encode → prepack → dequant on GPU → matches numpy decode
    within FP16 storage rounding."""
    from llm_pipeline.quantization.kernels import (
        dequant_q4_k_triton,
        prepack_q4_k_for_gpu,
    )
    torch.manual_seed(7)
    w = torch.randn(2048) * 0.05
    blob, _ = encode_q4_k(w)
    gpu = prepack_q4_k_for_gpu(blob, device="cuda:0")
    # The kernel stores FP16, so compare against an FP16-cast reference.
    out = dequant_q4_k_triton(gpu).cpu()
    ref_fp16 = decode_q4_k(blob, (2048,)).to(torch.float16)
    assert torch.equal(out, ref_fp16)
