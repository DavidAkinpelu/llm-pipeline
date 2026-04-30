"""Triton kernels for Q6_K — symmetric 6-bit, the fattest K-quant.

Block layout (see ``kquants.q6_k``)::

    d:           fp16   (2 bytes)
    scales[16]:  int8   (16 bytes, signed)
    ql[128]:     4-bit, two-per-byte  (128 bytes — low 4 bits of each weight)
    qh[64]:      2-bit, four-per-byte (64 bytes — high 2 bits of each weight)

Total: 210 bytes per 256 weights = 6.5625 bits/weight. 16 sub-blocks of
16 weights each (note: smaller sub-blocks than Q4_K/Q5_K's 8×32).

Decode formula::

    u6_i = (qh_i << 4) | ql_i              # u6 ∈ [0, 63]
    q6_i = u6_i - 32                       # q6 ∈ [-32, 31]   (signed)
    w_i  = d * scale_sb * q6_i

The kernel mirrors the Q4_K / Q5_K template but with three differences:

  * No min lane (symmetric quant; single ``scale * q6`` term).
  * Sub-block scales are signed int8 (not unsigned 6-bit), so we load
    them as int8 and cast — Triton sign-extends on the cast.
  * Inner dot is K=16 (one HMMA tile) instead of K=32. The K-loop iterates
    16 sub-blocks per super-block (vs 8 for Q4_K/Q5_K) but each iteration
    is half as wide, keeping total work per super-block constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

import triton
import triton.language as tl


SUPER_BLOCK = 256
SUB_BLOCK = 16
N_SUB = 16


# --------------------------------------------------------------------------- #
# Host-side prepack
# --------------------------------------------------------------------------- #


@dataclass
class Q6KGPUWeights:
    """GPU-resident Q6_K weights split into kernel-friendly fields.

    Note ``scales`` is ``int8`` (signed) — the kernel sign-extends on cast.
    """
    d: torch.Tensor       # (N,) fp16
    scales: torch.Tensor  # (N, 16) int8 — SIGNED
    ql: torch.Tensor      # (N, 128) uint8 — low 4 bits, two-per-byte
    qh: torch.Tensor      # (N, 64) uint8 — high 2 bits, four-per-byte

    @property
    def n_super_blocks(self) -> int:
        return int(self.d.shape[0])

    @property
    def device(self) -> torch.device:
        return self.d.device


def prepack_q6_k_for_gpu(blob: bytes, device: torch.device | str = "cuda:0") -> Q6KGPUWeights:
    """Split a Q6_K blob into GPU-resident tensors. 210 bytes per super-block."""
    if len(blob) % 210 != 0:
        raise ValueError(f"Q6_K blob length {len(blob)} not a multiple of 210 bytes/super-block.")
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(-1, 210)
    n_blocks = arr.shape[0]

    d = np.frombuffer(arr[:, 0:2].tobytes(), dtype=np.float16)
    # Reinterpret 16 uint8 as 16 int8 (signed) — same bytes, different sign interpretation.
    scales = np.frombuffer(arr[:, 2:18].tobytes(), dtype=np.int8).reshape(n_blocks, 16)
    ql = arr[:, 18:18 + 128].copy()             # (N, 128)
    qh = arr[:, 18 + 128:18 + 128 + 64].copy()  # (N, 64)

    dev = torch.device(device)
    return Q6KGPUWeights(
        d=torch.from_numpy(d.copy()).to(dev),
        scales=torch.from_numpy(scales.copy()).to(dev),       # int8
        ql=torch.from_numpy(ql).to(dev),
        qh=torch.from_numpy(qh).to(dev),
    )


# --------------------------------------------------------------------------- #
# Pure dequant kernel
# --------------------------------------------------------------------------- #


@triton.jit
def _dequant_q6_k_kernel(
    d_ptr, scales_ptr, ql_ptr, qh_ptr,
    out_ptr,
    n_blocks,
    BLOCK_M: tl.constexpr,
    N_SUB_C: tl.constexpr, SUB_BLOCK_C: tl.constexpr, SUPER_BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    blk = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    in_range = blk < n_blocks

    d = tl.load(d_ptr + blk, mask=in_range, other=0.0).to(tl.float32)

    sub_off = tl.arange(0, N_SUB_C)
    sb_scales = tl.load(
        scales_ptr + blk[:, None] * N_SUB_C + sub_off[None, :],
        mask=in_range[:, None], other=0,
    ).to(tl.float32)                                  # int8 → float32 (sign-extending)
    eff_scale = d[:, None] * sb_scales                 # (BM, 16)

    # Per-weight indices.
    w_off = tl.arange(0, SUPER_BLOCK_C)
    sub_idx = w_off // SUB_BLOCK_C                     # which sub-block (0..15)
    ql_byte_idx = w_off // 2
    ql_is_upper = (w_off & 1).to(tl.int32)
    qh_byte_idx = w_off // 4
    qh_shift = ((w_off & 3) * 2).to(tl.int32)

    ql_byte = tl.load(
        ql_ptr + blk[:, None] * 128 + ql_byte_idx[None, :],
        mask=in_range[:, None], other=0,
    ).to(tl.int32)
    q_low = (ql_byte >> (4 * ql_is_upper[None, :])) & 0xF        # (BM, 256), low 4 bits

    qh_byte = tl.load(
        qh_ptr + blk[:, None] * 64 + qh_byte_idx[None, :],
        mask=in_range[:, None], other=0,
    ).to(tl.int32)
    q_high = (qh_byte >> qh_shift[None, :]) & 0x3                 # (BM, 256), high 2 bits

    u6 = (q_high << 4) | q_low                                     # (BM, 256), 0..63
    q6 = u6 - 32                                                    # signed [-32, 31]

    # Gather per-sub-block scale by sub_idx.
    gather = (sub_off[None, :] == sub_idx[:, None]).to(tl.float32)  # (256, 16)
    eff_scale_w = tl.sum(eff_scale[:, None, :] * gather[None, :, :], axis=2)

    out = eff_scale_w * q6.to(tl.float32)
    out_addr = out_ptr + blk[:, None] * SUPER_BLOCK_C + w_off[None, :]
    tl.store(out_addr, out.to(tl.float16), mask=in_range[:, None])


def dequant_q6_k_triton(weights: Q6KGPUWeights) -> torch.Tensor:
    n = weights.n_super_blocks
    out = torch.empty(n * SUPER_BLOCK, device=weights.device, dtype=torch.float16)
    BLOCK_M = 8
    grid = ((n + BLOCK_M - 1) // BLOCK_M,)
    _dequant_q6_k_kernel[grid](
        weights.d, weights.scales, weights.ql, weights.qh,
        out, n,
        BLOCK_M=BLOCK_M,
        N_SUB_C=N_SUB, SUB_BLOCK_C=SUB_BLOCK, SUPER_BLOCK_C=SUPER_BLOCK,
    )
    return out


# --------------------------------------------------------------------------- #
# Fused dequant + matmul
# --------------------------------------------------------------------------- #


@triton.jit
def _matmul_q6_k_kernel(
    x_ptr, y_ptr,
    d_ptr, scales_ptr, ql_ptr, qh_ptr,
    M, N, K,
    n_blocks_per_row,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SUB_BLOCK_C: tl.constexpr, N_SUB_C: tl.constexpr, SUPER_BLOCK_C: tl.constexpr,
    PAIR_K: tl.constexpr,           # 2 * SUB_BLOCK_C; the inner-dot K width
    N_SUB_PAIRS: tl.constexpr,      # N_SUB_C // 2
):
    """``Y = X @ W^T`` for Q6_K.

    Inner K-loop processes **two sub-blocks (32 weights) per iteration**.
    A first attempt with K=16 dots (one sub-block per iteration) gave
    wrong results inside this kernel — the dot accumulator dropped
    contributions from successive K=16 ops. Stacking pairs of sub-blocks
    into a K=32 tile mirrors what the Q4_K and Q5_K kernels do and
    produces bit-correct matmul output here.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_off < M
    n_mask = n_off < N
    sb_valid = n_mask[:, None]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    pair_w = tl.arange(0, PAIR_K)                                       # (32,) — full pair width
    pair_half = tl.arange(0, PAIR_K // 2)                               # (16,) — unique ql bytes per pair
    pair_qh_byte = tl.arange(0, PAIR_K // 4)                            # (8,)  — unique qh bytes per pair
    qh_pos = tl.arange(0, 4) * 2                                         # [0,2,4,6]

    for sb_k in range(0, n_blocks_per_row):
        global_sb = n_off[:, None] * n_blocks_per_row + sb_k             # (BN, 1)

        d = tl.load(d_ptr + global_sb, mask=sb_valid, other=0.0).to(tl.float32)

        for pair_idx in tl.static_range(N_SUB_PAIRS):
            s0 = pair_idx * 2                                              # first sub-block
            s1 = pair_idx * 2 + 1                                          # second sub-block
            scale0 = tl.load(
                scales_ptr + global_sb * N_SUB_C + s0,
                mask=sb_valid, other=0,
            ).to(tl.float32)                                                # (BN, 1)
            scale1 = tl.load(
                scales_ptr + global_sb * N_SUB_C + s1,
                mask=sb_valid, other=0,
            ).to(tl.float32)                                                # (BN, 1)
            eff_scale0 = d * scale0
            eff_scale1 = d * scale1

            # Build a (BN, 32) effective-scale tile: first 16 weights use
            # ``eff_scale0``, second 16 use ``eff_scale1``.
            half_idx = tl.arange(0, PAIR_K) // SUB_BLOCK_C                  # (32,) ∈ {0, 1}
            scale_for_w = tl.where(
                half_idx[None, :] == 0, eff_scale0, eff_scale1,
            )                                                              # (BN, 32)

            # ---- ql: 16 unique bytes per pair → 32 4-bit values ----
            ql_packed = tl.load(
                ql_ptr + global_sb * 128 + (pair_idx * 16) + pair_half[None, :],
                mask=sb_valid, other=0,
            ).to(tl.int32)                                                  # (BN, 16)
            q_low_a = (ql_packed & 0xF).to(tl.int32)
            q_low_b = ((ql_packed >> 4) & 0xF).to(tl.int32)
            q_low = tl.interleave(q_low_a, q_low_b)                          # (BN, 32)

            # ---- qh: 8 unique bytes per pair → 32 2-bit values ----
            qh_bytes = tl.load(
                qh_ptr + global_sb * 64 + (pair_idx * 8) + pair_qh_byte[None, :],
                mask=sb_valid, other=0,
            ).to(tl.int32)                                                  # (BN, 8)
            qh_3d = (qh_bytes[:, :, None] >> qh_pos[None, None, :]) & 3      # (BN, 8, 4)
            q_high = tl.reshape(qh_3d, (BLOCK_N, PAIR_K))                    # (BN, 32)

            u6 = (q_high << 4) | q_low                                       # 0..63
            q6 = u6 - 32                                                      # signed [-32, 31]
            w_block_fp16 = (scale_for_w * q6.to(tl.float32)).to(tl.float16)

            k_start = sb_k * SUPER_BLOCK_C + pair_idx * PAIR_K
            k_off = k_start + pair_w
            k_valid = k_off < K
            x_block_fp16 = tl.load(
                x_ptr + m_off[:, None] * K + k_off[None, :],
                mask=m_mask[:, None] & k_valid[None, :], other=0.0,
            )                                                                # (BM, 32) fp16

            accumulator += tl.dot(x_block_fp16, tl.trans(w_block_fp16))

    y_addr = y_ptr + m_off[:, None] * N + n_off[None, :]
    tl.store(y_addr, accumulator.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


def matmul_q6_k_triton(
    x: torch.Tensor,
    weights: Q6KGPUWeights,
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
    _matmul_q6_k_kernel[grid](
        x, y,
        weights.d, weights.scales, weights.ql, weights.qh,
        M, n_out, K,
        n_blocks_per_row,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        SUB_BLOCK_C=SUB_BLOCK, N_SUB_C=N_SUB, SUPER_BLOCK_C=SUPER_BLOCK,
        PAIR_K=2 * SUB_BLOCK,
        N_SUB_PAIRS=N_SUB // 2,
    )
    return y
