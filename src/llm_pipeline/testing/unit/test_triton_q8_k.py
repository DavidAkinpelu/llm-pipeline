"""Triton Q8_K kernel correctness tests.

Skipped when CUDA isn't available or Triton isn't installed.
"""

import pytest
import torch

from llm_pipeline.quantization.kquants import decode_q8_k, encode_q8_k


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


def test_q8_k_dequant_kernel_bit_exact_against_numpy():
    from llm_pipeline.quantization.kernels import (
        dequant_q8_k_triton,
        prepack_q8_k_for_gpu,
    )
    torch.manual_seed(0)
    n = 1024
    w = torch.randn(n) * 0.05
    blob, _ = encode_q8_k(w)
    gpu = prepack_q8_k_for_gpu(blob, device="cuda:0")
    ref = decode_q8_k(blob, (n,)).to("cuda:0").to(torch.float16)
    out = dequant_q8_k_triton(gpu)[:n]
    assert (out.float() - ref.float()).abs().max().item() == 0.0


def test_q8_k_dequant_handles_partial_super_block():
    from llm_pipeline.quantization.kernels import (
        dequant_q8_k_triton,
        prepack_q8_k_for_gpu,
    )
    torch.manual_seed(0)
    n = 5 * 256
    w = torch.randn(n) * 0.05
    blob, _ = encode_q8_k(w)
    gpu = prepack_q8_k_for_gpu(blob, device="cuda:0")
    ref = decode_q8_k(blob, (n,)).to("cuda:0").to(torch.float16)
    out = dequant_q8_k_triton(gpu)
    assert (out.float() - ref.float()).abs().max().item() == 0.0


def test_q8_k_matmul_kernel_matches_dequant_then_fp16_matmul():
    from llm_pipeline.quantization.kernels import (
        matmul_q8_k_triton,
        prepack_q8_k_for_gpu,
    )
    torch.manual_seed(0)
    M, N, K = 8, 256, 512
    W = torch.randn(N, K) * 0.02
    blobs = b"".join(encode_q8_k(W[n])[0] for n in range(N))
    gpu_w = prepack_q8_k_for_gpu(blobs, device="cuda:0")
    W_dq = decode_q8_k(blobs, (N, K)).to("cuda:0").to(torch.float16)
    X = torch.randn(M, K, device="cuda:0", dtype=torch.float16)

    y_ref = (X.float() @ W_dq.float().T).to(torch.float16)
    y_triton = matmul_q8_k_triton(X, gpu_w, n_out=N, n_blocks_per_row=K // 256)
    rel = (
        (y_triton.float() - y_ref.float()).pow(2).sum().sqrt()
        / y_ref.float().pow(2).sum().sqrt()
    ).item()
    assert rel < 1e-2, f"rel L2 = {rel}"


def test_q8_k_matmul_sparse_correctness():
    from llm_pipeline.quantization.kernels import (
        matmul_q8_k_triton,
        prepack_q8_k_for_gpu,
    )
    N, K = 1, 256
    W = torch.zeros(N, K)
    W[0, 0] = 0.5; W[0, 1] = -0.25; W[0, 5] = 0.1
    blob = encode_q8_k(W[0])[0]
    gpu_w = prepack_q8_k_for_gpu(blob, device="cuda:0")
    W_dq = decode_q8_k(blob, (N, K)).to("cuda:0").to(torch.float16)
    X = torch.zeros(1, K, device="cuda:0", dtype=torch.float16)
    X[0, 0] = 1.0; X[0, 1] = 1.0; X[0, 5] = 1.0
    y_ref = (X.float() @ W_dq.float().T).to(torch.float16)
    y_triton = matmul_q8_k_triton(X, gpu_w, n_out=N, n_blocks_per_row=K // 256)
    assert (y_triton.float() - y_ref.float()).abs().max().item() < 1e-3


def test_q8_k_matmul_kernel_rejects_invalid_K():
    from llm_pipeline.quantization.kernels import (
        matmul_q8_k_triton,
        prepack_q8_k_for_gpu,
    )
    blob, _ = encode_q8_k(torch.zeros(256))
    gpu_w = prepack_q8_k_for_gpu(blob, device="cuda:0")
    X = torch.randn(2, 200, device="cuda:0", dtype=torch.float16)
    with pytest.raises(ValueError, match="must equal n_blocks_per_row"):
        matmul_q8_k_triton(X, gpu_w, n_out=1, n_blocks_per_row=1)


def test_q8_k_matmul_most_accurate_in_family():
    """8-bit should produce a more accurate matmul than 4-bit AND 6-bit."""
    from llm_pipeline.quantization.kernels import (
        matmul_q4_k_triton, matmul_q6_k_triton, matmul_q8_k_triton,
        prepack_q4_k_for_gpu, prepack_q6_k_for_gpu, prepack_q8_k_for_gpu,
    )
    from llm_pipeline.quantization.kquants import encode_q4_k, encode_q6_k

    torch.manual_seed(0)
    M, N, K = 32, 256, 512
    W = torch.randn(N, K) * 0.02
    X = torch.randn(M, K, device="cuda:0", dtype=torch.float16)

    blobs4 = b"".join(encode_q4_k(W[n])[0] for n in range(N))
    blobs6 = b"".join(encode_q6_k(W[n])[0] for n in range(N))
    blobs8 = b"".join(encode_q8_k(W[n])[0] for n in range(N))
    gpu_q4 = prepack_q4_k_for_gpu(blobs4, device="cuda:0")
    gpu_q6 = prepack_q6_k_for_gpu(blobs6, device="cuda:0")
    gpu_q8 = prepack_q8_k_for_gpu(blobs8, device="cuda:0")
    W_fp16 = W.to("cuda:0").to(torch.float16)
    y_ref = (X.float() @ W_fp16.float().T).to(torch.float16)

    err4 = (matmul_q4_k_triton(X, gpu_q4, n_out=N, n_blocks_per_row=K // 256).float() - y_ref.float()).pow(2).sum().sqrt().item()
    err6 = (matmul_q6_k_triton(X, gpu_q6, n_out=N, n_blocks_per_row=K // 256).float() - y_ref.float()).pow(2).sum().sqrt().item()
    err8 = (matmul_q8_k_triton(X, gpu_q8, n_out=N, n_blocks_per_row=K // 256).float() - y_ref.float()).pow(2).sum().sqrt().item()
    assert err8 < err6 < err4, f"Expected err8 < err6 < err4; got {err8}, {err6}, {err4}"
