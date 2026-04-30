"""Triton kernels for Q8_K — the simplest K-quant.

Block layout (see ``kquants.q8_k``)::

    d:        fp32   (4 bytes)         — single per-super-block scale
    qs[256]:  int8   (256 bytes)       — signed 8-bit values
    bsums[16]: int16 (32 bytes)        — per-sub-block sums (precomputed
                                         for fast dot products in inference;
                                         our kernel doesn't read them)

Total: 292 bytes per 256 weights ≈ 9.1 bits/weight.

Decode: ``w_i = d · qs_i``. No bit packing, no per-sub-block scale, no
min lane. Just `int8 * fp32 scalar`. The simplest dequant+matmul in the
family.

Because there's no sub-block hierarchy, the K-loop can do the *full
super-block* (K=256) in one ``tl.dot``. That gives the largest possible
inner-dim per memory load (cuBLAS-style), which should land near or
below cuBLAS FP16 latency on RTX 3090.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

import triton
import triton.language as tl


SUPER_BLOCK = 256


# --------------------------------------------------------------------------- #
# Host-side prepack
# --------------------------------------------------------------------------- #


@dataclass
class Q8KGPUWeights:
    """GPU-resident Q8_K weights split into kernel-friendly fields.

    ``bsums`` is the per-sub-block precomputed sum kept in the file format
    for a particular dot-product fast-path that's only relevant when the
    other operand is also Q8_K-encoded. For our X-is-FP16 use case we
    ignore it on the GPU side; left here as ``None`` for clarity.
    """
    d: torch.Tensor       # (N,) fp32
    qs: torch.Tensor      # (N, 256) int8 — SIGNED

    @property
    def n_super_blocks(self) -> int:
        return int(self.d.shape[0])

    @property
    def device(self) -> torch.device:
        return self.d.device


def prepack_q8_k_for_gpu(blob: bytes, device: torch.device | str = "cuda:0") -> Q8KGPUWeights:
    """Split a Q8_K blob into GPU-resident tensors. 292 bytes per super-block."""
    if len(blob) % 292 != 0:
        raise ValueError(f"Q8_K blob length {len(blob)} not a multiple of 292 bytes/super-block.")
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(-1, 292)
    n_blocks = arr.shape[0]

    d = np.frombuffer(arr[:, 0:4].tobytes(), dtype=np.float32)
    # Reinterpret 256 uint8 as 256 int8 (signed) — same bytes, different sign interpretation.
    qs = np.frombuffer(arr[:, 4:4 + 256].tobytes(), dtype=np.int8).reshape(n_blocks, 256)
    # bsums (arr[:, 260:292]) skipped — we don't use it for the matmul path.

    dev = torch.device(device)
    return Q8KGPUWeights(
        d=torch.from_numpy(d.copy()).to(dev),
        qs=torch.from_numpy(qs.copy()).to(dev),
    )


# --------------------------------------------------------------------------- #
# Pure dequant kernel
# --------------------------------------------------------------------------- #


@triton.jit
def _dequant_q8_k_kernel(
    d_ptr, qs_ptr,
    out_ptr,
    n_blocks,
    BLOCK_M: tl.constexpr,
    SUPER_BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    blk = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    in_range = blk < n_blocks

    d = tl.load(d_ptr + blk, mask=in_range, other=0.0)                 # fp32 (BM,)

    w_off = tl.arange(0, SUPER_BLOCK_C)
    qs = tl.load(
        qs_ptr + blk[:, None] * SUPER_BLOCK_C + w_off[None, :],
        mask=in_range[:, None], other=0,
    ).to(tl.float32)                                                    # int8 → float32 (sign-extending)

    out = d[:, None] * qs                                               # (BM, 256)
    out_addr = out_ptr + blk[:, None] * SUPER_BLOCK_C + w_off[None, :]
    tl.store(out_addr, out.to(tl.float16), mask=in_range[:, None])


def dequant_q8_k_triton(weights: Q8KGPUWeights) -> torch.Tensor:
    n = weights.n_super_blocks
    out = torch.empty(n * SUPER_BLOCK, device=weights.device, dtype=torch.float16)
    BLOCK_M = 8
    grid = ((n + BLOCK_M - 1) // BLOCK_M,)
    _dequant_q8_k_kernel[grid](
        weights.d, weights.qs,
        out, n,
        BLOCK_M=BLOCK_M, SUPER_BLOCK_C=SUPER_BLOCK,
    )
    return out


# --------------------------------------------------------------------------- #
# Fused dequant + matmul
# --------------------------------------------------------------------------- #


@triton.jit
def _matmul_q8_k_kernel(
    x_ptr, y_ptr,
    d_ptr, qs_ptr,
    M, N, K,
    n_blocks_per_row,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SUPER_BLOCK_C: tl.constexpr,
):
    """``Y = X @ W^T`` for Q8_K.

    K-loop steps in chunks of one full super-block (256 weights). Each
    iteration loads ``(BN, 256)`` int8 weights + 1 fp32 scale per output
    row, dequantizes to FP16, and feeds a K=256 ``tl.dot`` (16 HMMA
    tiles). With no bit packing and no sub-block gather, this is the
    cleanest of the K-quant matmul kernels.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_off < M
    n_mask = n_off < N
    sb_valid = n_mask[:, None]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    w_off = tl.arange(0, SUPER_BLOCK_C)                                  # (256,)

    for sb_k in range(0, n_blocks_per_row):
        global_sb = n_off[:, None] * n_blocks_per_row + sb_k             # (BN, 1)

        d = tl.load(d_ptr + global_sb, mask=sb_valid, other=0.0)         # (BN, 1) fp32

        qs = tl.load(
            qs_ptr + global_sb * SUPER_BLOCK_C + w_off[None, :],
            mask=sb_valid, other=0,
        ).to(tl.float32)                                                  # (BN, 256) int8 → fp32 (sign-extend)

        w_block_fp16 = (d * qs).to(tl.float16)                            # (BN, 256)

        k_off = sb_k * SUPER_BLOCK_C + w_off
        k_valid = k_off < K
        x_block_fp16 = tl.load(
            x_ptr + m_off[:, None] * K + k_off[None, :],
            mask=m_mask[:, None] & k_valid[None, :], other=0.0,
        )                                                                 # (BM, 256) fp16

        accumulator += tl.dot(x_block_fp16, tl.trans(w_block_fp16))

    y_addr = y_ptr + m_off[:, None] * N + n_off[None, :]
    tl.store(y_addr, accumulator.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


def matmul_q8_k_triton(
    x: torch.Tensor,
    weights: Q8KGPUWeights,
    n_out: int,
    n_blocks_per_row: int,
) -> torch.Tensor:
    M, K = x.shape
    if K != n_blocks_per_row * SUPER_BLOCK:
        raise ValueError(f"K={K} must equal n_blocks_per_row * {SUPER_BLOCK}")
    if weights.n_super_blocks != n_out * n_blocks_per_row:
        raise ValueError(
            f"weights have {weights.n_super_blocks} super-blocks; expected "
            f"{n_out * n_blocks_per_row} = n_out({n_out}) * n_blocks_per_row({n_blocks_per_row})"
        )

    y = torch.empty(M, n_out, device=x.device, dtype=torch.float16)
    BLOCK_M = 16 if M <= 16 else 32
    BLOCK_N = 64
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (n_out + BLOCK_N - 1) // BLOCK_N,
    )
    _matmul_q8_k_kernel[grid](
        x, y,
        weights.d, weights.qs,
        M, n_out, K,
        n_blocks_per_row,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        SUPER_BLOCK_C=SUPER_BLOCK,
    )
    return y
