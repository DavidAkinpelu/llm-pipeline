"""Triton kernels for Q5_K — same shape as Q4_K plus a qh high-bit lane.

The block layout (see ``kquants.q5_k``) is::

    d:         fp16  (2 bytes)
    dmin:      fp16  (2 bytes)
    scales[8]: 6-bit unsigned, packed (6 bytes)
    mins[8]:   6-bit unsigned, packed (6 bytes)
    qs[128]:   4-bit unsigned, two-per-byte (128 bytes)
    qh[32]:    1-bit unsigned, eight-per-byte (32 bytes)

Total: 176 bytes per 256 weights = 5.5 bits/weight.

Decode formula::

    u5_i = (qh_i << 4) | qs_i              # u5 ∈ [0, 31]
    w_i  = d * scale_sb * u5_i  -  dmin * min_sb

The host prepacker splits the blob into six GPU-friendly tensors. The
kernel iterates per sub-block (32 weights), loads:

  * 16 unique bytes of qs (as Q4_K does, then ``tl.interleave`` low/high
    nibbles → 32 4-bit values);
  * 4 unique bytes of qh, expanded to 32 1-bit values via a (BN, 4, 8)
    broadcast then reshape to (BN, 32);
  * the FP16 scale/min for that sub-block (one each per output row).

Tensor-core dot follows the same FP16-input pattern as the Q4_K kernel.
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
# Host-side prepack
# --------------------------------------------------------------------------- #


@dataclass
class Q5KGPUWeights:
    """GPU-resident Q5_K weights split into kernel-friendly fields."""
    d: torch.Tensor       # (N,) fp16
    dmin: torch.Tensor    # (N,) fp16
    scales: torch.Tensor  # (N, 8) uint8
    mins: torch.Tensor    # (N, 8) uint8
    qs: torch.Tensor      # (N, 128) uint8 — low 4 bits, two-per-byte
    qh: torch.Tensor      # (N, 32) uint8 — high bit, eight-per-byte

    @property
    def n_super_blocks(self) -> int:
        return int(self.d.shape[0])

    @property
    def device(self) -> torch.device:
        return self.d.device


def _unpack_6bit_blockwise(packed: np.ndarray) -> np.ndarray:
    bits = np.zeros(packed.shape[0], dtype=np.uint64)
    for i in range(6):
        bits |= packed[:, i].astype(np.uint64) << np.uint64(i * 8)
    out = np.zeros((packed.shape[0], 8), dtype=np.uint8)
    for i in range(8):
        out[:, i] = ((bits >> np.uint64(i * 6)) & np.uint64(0x3F)).astype(np.uint8)
    return out


def prepack_q5_k_for_gpu(blob: bytes, device: torch.device | str = "cuda:0") -> Q5KGPUWeights:
    """Split a Q5_K blob into GPU-resident tensors. 176 bytes per super-block."""
    if len(blob) % 176 != 0:
        raise ValueError(f"Q5_K blob length {len(blob)} not a multiple of 176 bytes/super-block.")
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(-1, 176)
    n_blocks = arr.shape[0]

    d = np.frombuffer(arr[:, 0:2].tobytes(), dtype=np.float16)
    dmin = np.frombuffer(arr[:, 2:4].tobytes(), dtype=np.float16)
    scales = _unpack_6bit_blockwise(arr[:, 4:10])
    mins = _unpack_6bit_blockwise(arr[:, 10:16])
    qs = arr[:, 16:144].copy()                          # (N, 128)
    qh = arr[:, 144:176].copy()                         # (N, 32)

    dev = torch.device(device)
    return Q5KGPUWeights(
        d=torch.from_numpy(d.copy()).to(dev),
        dmin=torch.from_numpy(dmin.copy()).to(dev),
        scales=torch.from_numpy(scales).to(dev),
        mins=torch.from_numpy(mins).to(dev),
        qs=torch.from_numpy(qs).to(dev),
        qh=torch.from_numpy(qh).to(dev),
    )


# --------------------------------------------------------------------------- #
# Pure dequant kernel
# --------------------------------------------------------------------------- #


@triton.jit
def _dequant_q5_k_kernel(
    d_ptr, dmin_ptr, scales_ptr, mins_ptr, qs_ptr, qh_ptr,
    out_ptr,
    n_blocks,
    BLOCK_M: tl.constexpr,
    N_SUB_C: tl.constexpr, SUB_BLOCK_C: tl.constexpr, SUPER_BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    blk = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    in_range = blk < n_blocks

    d = tl.load(d_ptr + blk, mask=in_range, other=0.0).to(tl.float32)
    dmin = tl.load(dmin_ptr + blk, mask=in_range, other=0.0).to(tl.float32)

    sub_off = tl.arange(0, N_SUB_C)
    sb_scales = tl.load(
        scales_ptr + blk[:, None] * N_SUB_C + sub_off[None, :],
        mask=in_range[:, None], other=0,
    ).to(tl.float32)
    sb_mins = tl.load(
        mins_ptr + blk[:, None] * N_SUB_C + sub_off[None, :],
        mask=in_range[:, None], other=0,
    ).to(tl.float32)
    eff_scale = d[:, None] * sb_scales
    eff_min = -dmin[:, None] * sb_mins

    # Per-weight indices.
    w_off = tl.arange(0, SUPER_BLOCK_C)
    sub_idx = w_off // SUB_BLOCK_C
    qs_byte_idx = w_off // 2
    qs_is_upper = (w_off & 1).to(tl.int32)
    qh_byte_idx = w_off // 8
    qh_bit_idx = (w_off & 7).to(tl.int32)

    qs_byte = tl.load(
        qs_ptr + blk[:, None] * 128 + qs_byte_idx[None, :],
        mask=in_range[:, None], other=0,
    ).to(tl.int32)
    q4 = (qs_byte >> (4 * qs_is_upper[None, :])) & 0xF             # (BM, 256)

    qh_byte = tl.load(
        qh_ptr + blk[:, None] * 32 + qh_byte_idx[None, :],
        mask=in_range[:, None], other=0,
    ).to(tl.int32)
    q_high = (qh_byte >> qh_bit_idx[None, :]) & 0x1
    u5 = (q_high << 4) | q4                                          # (BM, 256), 0..31

    # Gather per-sub-block scale/min by sub_idx.
    gather = (sub_off[None, :] == sub_idx[:, None]).to(tl.float32)
    eff_scale_w = tl.sum(eff_scale[:, None, :] * gather[None, :, :], axis=2)
    eff_min_w = tl.sum(eff_min[:, None, :] * gather[None, :, :], axis=2)

    out = eff_scale_w * u5.to(tl.float32) + eff_min_w
    out_addr = out_ptr + blk[:, None] * SUPER_BLOCK_C + w_off[None, :]
    tl.store(out_addr, out.to(tl.float16), mask=in_range[:, None])


def dequant_q5_k_triton(weights: Q5KGPUWeights) -> torch.Tensor:
    n = weights.n_super_blocks
    out = torch.empty(n * SUPER_BLOCK, device=weights.device, dtype=torch.float16)
    BLOCK_M = 8
    grid = ((n + BLOCK_M - 1) // BLOCK_M,)
    _dequant_q5_k_kernel[grid](
        weights.d, weights.dmin, weights.scales, weights.mins, weights.qs, weights.qh,
        out, n,
        BLOCK_M=BLOCK_M,
        N_SUB_C=N_SUB, SUB_BLOCK_C=SUB_BLOCK, SUPER_BLOCK_C=SUPER_BLOCK,
    )
    return out


# --------------------------------------------------------------------------- #
# Fused dequant + matmul
# --------------------------------------------------------------------------- #


@triton.jit
def _matmul_q5_k_kernel(
    x_ptr, y_ptr,
    d_ptr, dmin_ptr, scales_ptr, mins_ptr, qs_ptr, qh_ptr,
    M, N, K,
    n_blocks_per_row,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SUB_BLOCK_C: tl.constexpr, N_SUB_C: tl.constexpr, SUPER_BLOCK_C: tl.constexpr,
):
    """``Y = X @ W^T`` for Q5_K-encoded ``W`` (per-sub-block K-loop, HMMA dot)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_off < M
    n_mask = n_off < N
    sb_valid = n_mask[:, None]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    sub_w = tl.arange(0, SUB_BLOCK_C)                               # (32,)
    half_w = tl.arange(0, SUB_BLOCK_C // 2)                          # (16,) — unique qs bytes
    qh_byte_off = tl.arange(0, SUB_BLOCK_C // 8)                     # (4,) — unique qh bytes
    qh_bit_pos = tl.arange(0, 8)                                      # (8,) — bit positions

    for sb_k in range(0, n_blocks_per_row):
        global_sb = n_off[:, None] * n_blocks_per_row + sb_k         # (BN, 1)

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
            eff_scale = d * scale_q
            eff_min = -dmin * min_q

            # ---- qs: 16 unique bytes per row → 32 4-bit values ----
            q_packed = tl.load(
                qs_ptr + global_sb * 128 + (s_idx * 16) + half_w[None, :],
                mask=sb_valid, other=0,
            ).to(tl.int32)
            q_low = (q_packed & 0xF).to(tl.int32)
            q_high4 = ((q_packed >> 4) & 0xF).to(tl.int32)
            q4 = tl.interleave(q_low, q_high4)                        # (BN, 32)

            # ---- qh: 4 unique bytes per row → 32 1-bit values ----
            qh_bytes = tl.load(
                qh_ptr + global_sb * 32 + (s_idx * 4) + qh_byte_off[None, :],
                mask=sb_valid, other=0,
            ).to(tl.int32)                                            # (BN, 4)
            # Expand: each byte's 8 bits become 8 columns.
            qh_3d = (qh_bytes[:, :, None] >> qh_bit_pos[None, None, :]) & 1  # (BN, 4, 8)
            q_high = tl.reshape(qh_3d, (BLOCK_N, SUB_BLOCK_C))          # (BN, 32)

            u5 = (q_high << 4) | q4                                    # (BN, 32) ∈ [0, 31]
            w_block_fp16 = (eff_scale * u5.to(tl.float32) + eff_min).to(tl.float16)

            k_start = sb_k * SUPER_BLOCK_C + s_idx * SUB_BLOCK_C
            k_off = k_start + sub_w
            k_valid = k_off < K
            x_block_fp16 = tl.load(
                x_ptr + m_off[:, None] * K + k_off[None, :],
                mask=m_mask[:, None] & k_valid[None, :], other=0.0,
            )

            accumulator += tl.dot(x_block_fp16, tl.trans(w_block_fp16))

    y_addr = y_ptr + m_off[:, None] * N + n_off[None, :]
    tl.store(y_addr, accumulator.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


def matmul_q5_k_triton(
    x: torch.Tensor,
    weights: Q5KGPUWeights,
    n_out: int,
    n_blocks_per_row: int,
) -> torch.Tensor:
    """``Y = X @ W^T`` where ``W`` is Q5_K-encoded GPU tensors."""
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
    _matmul_q5_k_kernel[grid](
        x, y,
        weights.d, weights.dmin, weights.scales, weights.mins, weights.qs, weights.qh,
        M, n_out, K,
        n_blocks_per_row,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        SUB_BLOCK_C=SUB_BLOCK, N_SUB_C=N_SUB, SUPER_BLOCK_C=SUPER_BLOCK,
    )
    return y
