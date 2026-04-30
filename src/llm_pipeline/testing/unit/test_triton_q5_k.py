"""Triton Q5_K kernel correctness tests.

Mirrors test_triton_q4_k.py with the qh high-bit lane that Q5_K adds.
Skipped when CUDA isn't available or Triton isn't installed.
"""

import pytest
import torch

from llm_pipeline.quantization.kquants import decode_q5_k, encode_q5_k


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


def test_q5_k_dequant_kernel_bit_exact_against_numpy():
    from llm_pipeline.quantization.kernels import (
        dequant_q5_k_triton,
        prepack_q5_k_for_gpu,
    )
    torch.manual_seed(0)
    n = 1024
    w = torch.randn(n) * 0.05
    blob, _ = encode_q5_k(w)
    gpu = prepack_q5_k_for_gpu(blob, device="cuda:0")
    ref = decode_q5_k(blob, (n,)).to("cuda:0").to(torch.float16)
    out = dequant_q5_k_triton(gpu)[:n]
    assert (out.float() - ref.float()).abs().max().item() == 0.0


def test_q5_k_dequant_handles_partial_super_block():
    from llm_pipeline.quantization.kernels import (
        dequant_q5_k_triton,
        prepack_q5_k_for_gpu,
    )
    torch.manual_seed(0)
    n = 5 * 256          # 5 super-blocks
    w = torch.randn(n) * 0.05
    blob, _ = encode_q5_k(w)
    gpu = prepack_q5_k_for_gpu(blob, device="cuda:0")
    ref = decode_q5_k(blob, (n,)).to("cuda:0").to(torch.float16)
    out = dequant_q5_k_triton(gpu)
    assert (out.float() - ref.float()).abs().max().item() == 0.0


def test_q5_k_matmul_kernel_matches_dequant_then_fp16_matmul():
    from llm_pipeline.quantization.kernels import (
        matmul_q5_k_triton,
        prepack_q5_k_for_gpu,
    )
    torch.manual_seed(0)
    M, N, K = 8, 256, 512
    W = torch.randn(N, K) * 0.02
    blobs = b"".join(encode_q5_k(W[n])[0] for n in range(N))
    gpu_w = prepack_q5_k_for_gpu(blobs, device="cuda:0")
    W_dq = decode_q5_k(blobs, (N, K)).to("cuda:0").to(torch.float16)
    X = torch.randn(M, K, device="cuda:0", dtype=torch.float16)

    y_ref = (X.float() @ W_dq.float().T).to(torch.float16)
    y_triton = matmul_q5_k_triton(X, gpu_w, n_out=N, n_blocks_per_row=K // 256)
    rel = (
        (y_triton.float() - y_ref.float()).pow(2).sum().sqrt()
        / y_ref.float().pow(2).sum().sqrt()
    ).item()
    assert rel < 1e-2, f"rel L2 = {rel}"


def test_q5_k_matmul_kernel_rejects_invalid_K():
    from llm_pipeline.quantization.kernels import (
        matmul_q5_k_triton,
        prepack_q5_k_for_gpu,
    )
    blob, _ = encode_q5_k(torch.zeros(256))
    gpu_w = prepack_q5_k_for_gpu(blob, device="cuda:0")
    X = torch.randn(2, 200, device="cuda:0", dtype=torch.float16)
    with pytest.raises(ValueError, match="must equal n_blocks_per_row"):
        matmul_q5_k_triton(X, gpu_w, n_out=1, n_blocks_per_row=1)


def test_q5_k_matmul_better_than_q4_k_at_same_shape():
    """5-bit dequant fed into the same dot should give a closer answer than
    4-bit. Verifies the qh lane is actually contributing precision."""
    from llm_pipeline.quantization.kernels import (
        matmul_q4_k_triton, matmul_q5_k_triton,
        prepack_q4_k_for_gpu, prepack_q5_k_for_gpu,
    )
    from llm_pipeline.quantization.kquants import encode_q4_k

    torch.manual_seed(0)
    M, N, K = 32, 256, 512
    W = torch.randn(N, K) * 0.02
    X = torch.randn(M, K, device="cuda:0", dtype=torch.float16)

    blobs4 = b"".join(encode_q4_k(W[n])[0] for n in range(N))
    blobs5 = b"".join(encode_q5_k(W[n])[0] for n in range(N))
    gpu_q4 = prepack_q4_k_for_gpu(blobs4, device="cuda:0")
    gpu_q5 = prepack_q5_k_for_gpu(blobs5, device="cuda:0")
    W_fp16 = W.to("cuda:0").to(torch.float16)
    y_ref = (X.float() @ W_fp16.float().T).to(torch.float16)

    y4 = matmul_q4_k_triton(X, gpu_q4, n_out=N, n_blocks_per_row=K // 256)
    y5 = matmul_q5_k_triton(X, gpu_q5, n_out=N, n_blocks_per_row=K // 256)

    err4 = (y4.float() - y_ref.float()).pow(2).sum().sqrt().item()
    err5 = (y5.float() - y_ref.float()).pow(2).sum().sqrt().item()
    # Q5_K should produce a more accurate matmul than Q4_K with the same X.
    assert err5 < err4, f"err5={err5} ≥ err4={err4} (Q5_K should be more accurate)"
