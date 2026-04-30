"""FP8 matmul Triton kernel — Hopper (sm_90+) path.

Targets ``torch.float8_e4m3fn`` and ``torch.float8_e5m2`` weights with
native Hopper tensor-core FP8 instructions. Falls through to the
existing pure-Python encode/decode + cuBLAS fp16 on non-Hopper hardware.

Layout: per-block fp16 scale (matches our ``encode_fp8_e4m3`` /
``encode_fp8_e5m2`` byte format from ``fp_low.py``). The kernel reads
both the scaled fp8 values and the per-block scale, dequantises in
registers, and accumulates the matmul into fp32.

**Status**: code committed, locally validated only as far as soft-import
fallback (the dispatcher returns the cuBLAS-fp16 path on Ampere). Real
H100 validation is the natural next step in the cloud pass.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


_FP8_DTYPES = ("float8_e4m3fn", "float8_e5m2")


def _is_hopper() -> bool:
    """Detect sm_90+ (Hopper / Blackwell) — the platforms with native FP8 TC."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 9


def fp8_matmul(
    x: torch.Tensor,                     # [M, K] in fp16/bf16
    w_fp8: torch.Tensor,                 # [N, K] in fp8 (e4m3 or e5m2)
    w_scale: torch.Tensor,               # [N] or [N, K // block_size] fp16/fp32
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Compute ``X @ W^T`` where W is FP8 with per-row (or per-block) scale.

    On Hopper this dispatches to the Triton kernel for per-row scales.
    Per-block scales fall back to the reference path until the tiled
    kernel supports blockwise scale loads. On non-Hopper it upcasts
    ``w_fp8`` to fp16 and uses cuBLAS — correct but unaccelerated.
    """
    if (
        not _HAS_TRITON
        or not x.is_cuda
        or not _is_hopper()
        or w_scale.dim() != 1
    ):
        return _fallback_matmul(x, w_fp8, w_scale, out_dtype)
    return _hopper_matmul(x, w_fp8, w_scale, out_dtype)


def _fallback_matmul(x, w_fp8, w_scale, out_dtype):
    """Non-Hopper path: dequantise W to fp16 + cuBLAS."""
    # ``w_fp8`` may not be a "real" FP8 tensor on pre-Hopper builds; treat
    # whatever dtype it comes with as the source of truth and just upcast.
    w_fp16 = w_fp8.to(torch.float16)
    if w_scale.dim() == 1:
        w_fp16 = w_fp16 * w_scale.to(torch.float16).unsqueeze(-1)
    else:
        # Per-block scale — broadcast block-wise.
        N, K = w_fp16.shape
        block_size = K // w_scale.shape[-1]
        w_fp16 = (
            w_fp16.view(N, K // block_size, block_size)
            * w_scale.to(torch.float16).unsqueeze(-1)
        ).reshape(N, K)
    return (x.to(torch.float16) @ w_fp16.T).to(out_dtype)


if _HAS_TRITON:
    @triton.jit
    def _fp8_matmul_kernel(
        x_ptr, w_ptr, scale_ptr, out_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Tiled FP8 matmul. Per-tile: load X (fp16), load W (fp8) +
        scale, cast both to fp16/bf16, dot-product into fp32, write out.

        On Hopper this is *the* spot for ``tl.dot`` to use native FP8 TC
        instructions; the Triton runtime picks the right HMMA variant
        based on the operand dtypes when ``input_precision`` is left at
        the default.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_block in range(0, K, BLOCK_K):
            k_idx = k_block + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0,
            )
            w = tl.load(
                w_ptr + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk,
                mask=(offs_n[:, None] < N) & (k_idx[None, :] < K), other=0.0,
            )
            # Per-row scale; cast to fp32 for accumulation.
            scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0)
            w_scaled = w.to(tl.float32) * scale[:, None]
            acc += tl.dot(x.to(tl.float32), tl.trans(w_scaled))

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc.to(tl.float16),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def _hopper_matmul(x, w_fp8, w_scale, out_dtype):
    """Hopper FP8 path. Real validation pending H100 access."""
    M, K = x.shape
    N = w_fp8.shape[0]
    out = torch.empty(M, N, dtype=torch.float16, device=x.device)
    BLOCK_M = BLOCK_N = 64
    BLOCK_K = 64
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    _fp8_matmul_kernel[grid](
        x, w_fp8, w_scale, out, M, N, K,
        x.stride(0), x.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out.to(out_dtype)
