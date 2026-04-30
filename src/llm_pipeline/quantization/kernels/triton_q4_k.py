"""Triton kernels for Q4_K: pure dequant and fused dequant+matmul.

Two-step plan:

1. **Pre-pack on host.** A Q4_K blob is awkward to handle directly in
   Triton because the per-sub-block scales/mins are 6-bit-packed and
   would require nontrivial bit shuffling on the GPU. We instead split a
   blob into 5 GPU-friendly tensors during a one-time prepack:

     - ``d``        ``(N,)``    FP16  — master scale per super-block.
     - ``dmin``     ``(N,)``    FP16  — master min   per super-block.
     - ``scales``   ``(N, 8)``  uint8 — per-sub-block 6-bit scale, debyted.
     - ``mins``     ``(N, 8)``  uint8 — per-sub-block 6-bit min,   debyted.
     - ``qs``       ``(N, 128)`` uint8 — 4-bit values, two-per-byte packed.

   ``N`` = number of super-blocks. The total memory is identical to the
   original blob (~144 bytes/super-block) — we just don't bit-pack the
   scales/mins on the GPU side because the savings aren't worth the
   in-kernel unpacking cost.

2. **Kernel.** Each Triton program handles ``BLOCK_M`` super-blocks. For
   each weight position ``w ∈ [0, 256)`` we compute

       q4    = (qs[i, w // 2] >> (4 * (w & 1))) & 0xF
       scale = d[i]    * scales[i, w // 32]
       off   = -dmin[i] * mins[i,  w // 32]
       out[i, w] = scale * q4 + off

The fused matmul kernel reuses the same dequant logic but folds it into
the K-loop of a standard Triton matmul tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

import triton
import triton.language as tl


SUPER_BLOCK = 256
SUB_BLOCK = 32
N_SUB = 8


# --------------------------------------------------------------------------- #
# Host-side prepack: blob bytes → 5 GPU tensors
# --------------------------------------------------------------------------- #


@dataclass
class Q4KGPUWeights:
    """GPU-resident Q4_K weights split into kernel-friendly fields.

    ``d`` and ``dmin`` are FP16; ``scales``, ``mins``, ``qs`` are uint8.
    All five live on the same CUDA device with the same number of
    super-blocks ``N`` along the leading axis.
    """

    d: torch.Tensor       # (N,) fp16
    dmin: torch.Tensor    # (N,) fp16
    scales: torch.Tensor  # (N, 8) uint8
    mins: torch.Tensor    # (N, 8) uint8
    qs: torch.Tensor      # (N, 128) uint8

    @property
    def n_super_blocks(self) -> int:
        return int(self.d.shape[0])

    @property
    def device(self) -> torch.device:
        return self.d.device


def _unpack_6bit_blockwise(packed: np.ndarray) -> np.ndarray:
    """Inverse of the encoder's ``_pack_6bit_batched``. Input ``(N, 6)`` →
    output ``(N, 8)`` uint8 in [0, 63]."""
    bits = np.zeros(packed.shape[0], dtype=np.uint64)
    for i in range(6):
        bits |= packed[:, i].astype(np.uint64) << np.uint64(i * 8)
    out = np.zeros((packed.shape[0], 8), dtype=np.uint8)
    for i in range(8):
        out[:, i] = ((bits >> np.uint64(i * 6)) & np.uint64(0x3F)).astype(np.uint8)
    return out


def prepack_q4_k_for_gpu(blob: bytes, device: torch.device | str = "cuda:0") -> Q4KGPUWeights:
    """Split a Q4_K blob into GPU-resident kernel-friendly tensors."""
    if len(blob) % 144 != 0:
        raise ValueError(f"Q4_K blob length {len(blob)} is not a multiple of 144 bytes/super-block.")
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(-1, 144)
    n_blocks = arr.shape[0]

    d = np.frombuffer(arr[:, 0:2].tobytes(), dtype=np.float16)
    dmin = np.frombuffer(arr[:, 2:4].tobytes(), dtype=np.float16)
    scales = _unpack_6bit_blockwise(arr[:, 4:10])     # (N, 8) uint8
    mins = _unpack_6bit_blockwise(arr[:, 10:16])      # (N, 8) uint8
    qs = arr[:, 16:144].copy()                          # (N, 128) uint8

    dev = torch.device(device)
    return Q4KGPUWeights(
        d=torch.from_numpy(d.copy()).to(dev),
        dmin=torch.from_numpy(dmin.copy()).to(dev),
        scales=torch.from_numpy(scales).to(dev),
        mins=torch.from_numpy(mins).to(dev),
        qs=torch.from_numpy(qs).to(dev),
    )


# --------------------------------------------------------------------------- #
# Pure dequant kernel
# --------------------------------------------------------------------------- #


@triton.jit
def _dequant_q4_k_kernel(
    d_ptr, dmin_ptr, scales_ptr, mins_ptr, qs_ptr,
    out_ptr,
    n_blocks,
    BLOCK_M: tl.constexpr,
    N_SUB_C: tl.constexpr,          # 8
    SUB_BLOCK_C: tl.constexpr,       # 32
    SUPER_BLOCK_C: tl.constexpr,     # 256
):
    pid = tl.program_id(0)
    blk = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    in_range = blk < n_blocks

    d = tl.load(d_ptr + blk, mask=in_range, other=0.0).to(tl.float32)
    dmin = tl.load(dmin_ptr + blk, mask=in_range, other=0.0).to(tl.float32)

    sub_off = tl.arange(0, N_SUB_C)
    scales_addr = scales_ptr + blk[:, None] * N_SUB_C + sub_off[None, :]
    mins_addr = mins_ptr + blk[:, None] * N_SUB_C + sub_off[None, :]
    sb_mask = in_range[:, None]
    sb_scales = tl.load(scales_addr, mask=sb_mask, other=0).to(tl.float32)
    sb_mins = tl.load(mins_addr, mask=sb_mask, other=0).to(tl.float32)

    eff_scale = d[:, None] * sb_scales                                       # (BM, 8)
    eff_min = -dmin[:, None] * sb_mins                                        # (BM, 8)

    w_off = tl.arange(0, SUPER_BLOCK_C)
    sub_idx = w_off // SUB_BLOCK_C
    byte_idx = w_off // 2
    is_upper = (w_off & 1).to(tl.int32)

    qs_addr = qs_ptr + blk[:, None] * 128 + byte_idx[None, :]
    q_byte = tl.load(qs_addr, mask=in_range[:, None], other=0).to(tl.int32)
    q4 = (q_byte >> (4 * is_upper[None, :])) & 0xF                            # (BM, 256)

    # Gather per-sub-block scale/min by sub_idx (cheap "1-hot" multiply).
    gather = (sub_off[None, :] == sub_idx[:, None]).to(tl.float32)            # (256, 8)
    eff_scale_w = tl.sum(eff_scale[:, None, :] * gather[None, :, :], axis=2)
    eff_min_w = tl.sum(eff_min[:, None, :] * gather[None, :, :], axis=2)

    out = eff_scale_w * q4.to(tl.float32) + eff_min_w
    out_addr = out_ptr + blk[:, None] * SUPER_BLOCK_C + w_off[None, :]
    tl.store(out_addr, out.to(tl.float16), mask=in_range[:, None])


def dequant_q4_k_triton(weights: Q4KGPUWeights) -> torch.Tensor:
    """Dequantize Q4_K → FP16 on GPU. Returns shape ``(N_super_blocks * 256,)``."""
    n = weights.n_super_blocks
    out = torch.empty(n * SUPER_BLOCK, device=weights.device, dtype=torch.float16)
    BLOCK_M = 8
    grid = ((n + BLOCK_M - 1) // BLOCK_M,)
    _dequant_q4_k_kernel[grid](
        weights.d, weights.dmin, weights.scales, weights.mins, weights.qs,
        out,
        n,
        BLOCK_M=BLOCK_M,
        N_SUB_C=N_SUB,
        SUB_BLOCK_C=SUB_BLOCK,
        SUPER_BLOCK_C=SUPER_BLOCK,
    )
    return out


# --------------------------------------------------------------------------- #
# Fused dequant + matmul:  Y = X @ W^T  where W is Q4_K-encoded.
# --------------------------------------------------------------------------- #


@triton.jit
def _matmul_q4_k_kernel(
    x_ptr, y_ptr,
    d_ptr, dmin_ptr, scales_ptr, mins_ptr, qs_ptr,
    M, N, K,
    n_blocks_per_row,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SUB_BLOCK_C: tl.constexpr, N_SUB_C: tl.constexpr, SUPER_BLOCK_C: tl.constexpr,
):
    """``Y = X @ W^T`` where ``W`` is Q4_K-encoded.

    K-loop iterates **per sub-block** (32 weights). Each step loads exactly
    one (per-row) scale/min pair and 16 bytes of quantized values per
    output row. No gather needed — the metadata is already aligned. The
    inner dot is FP16×FP16 → FP32 (HMMA) with K=32 (tensor-core legal).

    Total K-steps = ``n_blocks_per_row * 8`` (8 sub-blocks per super-block).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_off < M
    n_mask = n_off < N
    sb_valid = n_mask[:, None]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    sub_w = tl.arange(0, SUB_BLOCK_C)                             # (32,)
    half_w = tl.arange(0, SUB_BLOCK_C // 2)                       # (16,) — unique byte indices

    for sb_k in range(0, n_blocks_per_row):
        global_sb = n_off[:, None] * n_blocks_per_row + sb_k       # (BN, 1)

        d = tl.load(d_ptr + global_sb, mask=sb_valid, other=0.0).to(tl.float32)
        dmin = tl.load(dmin_ptr + global_sb, mask=sb_valid, other=0.0).to(tl.float32)

        for s_idx in tl.static_range(N_SUB_C):
            scale_q = tl.load(
                scales_ptr + global_sb * N_SUB_C + s_idx,
                mask=sb_valid, other=0,
            ).to(tl.float32)
            min_q = tl.load(
                mins_ptr + global_sb * N_SUB_C + s_idx,
                mask=sb_valid, other=0,
            ).to(tl.float32)
            eff_scale = d * scale_q                                 # (BN, 1)
            eff_min = -dmin * min_q                                  # (BN, 1)

            # Load 16 *unique* bytes per row (instead of 32 with duplicate
            # accesses); produce 32 4-bit values via interleave.
            q_packed = tl.load(
                qs_ptr + global_sb * 128 + (s_idx * 16) + half_w[None, :],
                mask=sb_valid, other=0,
            ).to(tl.int32)                                          # (BN, 16)
            q_low = (q_packed & 0xF).to(tl.int32)                   # even positions
            q_high = ((q_packed >> 4) & 0xF).to(tl.int32)            # odd positions
            q4 = tl.interleave(q_low, q_high)                        # (BN, 32)
            w_block_fp16 = (eff_scale * q4.to(tl.float32) + eff_min).to(tl.float16)

            k_start = sb_k * SUPER_BLOCK_C + s_idx * SUB_BLOCK_C
            k_off = k_start + sub_w
            k_valid = k_off < K
            x_block_fp16 = tl.load(
                x_ptr + m_off[:, None] * K + k_off[None, :],
                mask=m_mask[:, None] & k_valid[None, :],
                other=0.0,
            )                                                      # (BM, 32) fp16

            accumulator += tl.dot(x_block_fp16, tl.trans(w_block_fp16))

    y_addr = y_ptr + m_off[:, None] * N + n_off[None, :]
    tl.store(y_addr, accumulator.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


def matmul_q4_k_triton(
    x: torch.Tensor,                # (M, K) FP16
    weights: Q4KGPUWeights,         # represents (N, K) with K = n_blocks_per_row * 256
    n_out: int,                     # N (= rows of W)
    n_blocks_per_row: int,          # K // 256
) -> torch.Tensor:
    """``Y = X @ W^T`` where ``W`` is supplied as Q4_K-encoded GPU tensors.

    Args:
        x: (M, K) FP16 input.
        weights: ``Q4KGPUWeights`` whose flat super-blocks are organised so
            that row ``n`` of ``W`` is super-blocks ``[n * n_blocks_per_row,
            (n + 1) * n_blocks_per_row)`` of the prepacked tensors.
        n_out: number of output features ``N``.
        n_blocks_per_row: ``K // 256``.

    Returns:
        ``(M, N)`` FP16 output.
    """
    M, K = x.shape
    if K != n_blocks_per_row * SUPER_BLOCK:
        raise ValueError(f"K={K} must equal n_blocks_per_row * {SUPER_BLOCK}")
    if weights.n_super_blocks != n_out * n_blocks_per_row:
        raise ValueError(
            f"weights have {weights.n_super_blocks} super-blocks; expected "
            f"{n_out * n_blocks_per_row} = n_out({n_out}) * n_blocks_per_row({n_blocks_per_row})"
        )

    y = torch.empty(M, n_out, device=x.device, dtype=torch.float16)
    # Larger tiles work now because the per-sub-block K-loop scales O(BN), not
    # O(BN * 256 * 8) per K-step.
    BLOCK_M = 16 if M <= 16 else 32
    BLOCK_N = 64
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (n_out + BLOCK_N - 1) // BLOCK_N,
    )
    _matmul_q4_k_kernel[grid](
        x, y,
        weights.d, weights.dmin, weights.scales, weights.mins, weights.qs,
        M, n_out, K,
        n_blocks_per_row,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        SUB_BLOCK_C=SUB_BLOCK, N_SUB_C=N_SUB, SUPER_BLOCK_C=SUPER_BLOCK,
    )
    return y
