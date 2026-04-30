"""Q5_K: 5-bit K-quant. Q4_K's bigger sibling.

Block structure (256 weights per super-block):

  super_block = [
      d:    fp16,                    # master scale (for the 8 sub-block scales)
      dmin: fp16,                    # master min   (for the 8 sub-block mins)
      scales[8]: 6-bit unsigned,     # per-sub-block scale (relative to d)
      mins[8]:   6-bit unsigned,     # per-sub-block min   (relative to dmin)
      qs[256]:   4-bit unsigned,     # low 4 bits of each 5-bit value
      qh[256]:   1-bit unsigned,     # high bit of each 5-bit value
  ]

Bits per weight:
    d, dmin: 2 × 16              = 32
    scales:  8 × 6               = 48
    mins:    8 × 6               = 48
    qs:      256 × 4             = 1024
    qh:      256 × 1             = 256
    -----------------------------------
    total                        = 1408 / 256 = 5.5 bits per weight

Decoding: ``w_i ≈ d · scale_sb · q5_i  -  dmin · min_sb`` where
``q5_i = (qh_i << 4) | qs_i  ∈ [0, 31]``. Same closed-form fit as Q4_K
just with ``nmax=31``.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from .q4_k import (
    _fit_sub_blocks_qkx2_batched,
    _fit_sub_blocks_qkx3_batched,
    _pack_6bit_batched,
)


SUPER_BLOCK = 256
SUB_BLOCK = 32
N_SUB = SUPER_BLOCK // SUB_BLOCK  # 8


def _encode_super_blocks_batched(
    blocks: np.ndarray,
    importance: np.ndarray,
    use_qkx3: bool = False,
) -> np.ndarray:
    """Encode a batch of Q5_K super-blocks. Returns ``(N, 176)`` uint8.

    ``use_qkx3=True`` swaps the per-sub-block scale-fit for the iterative
    ``make_qkx3_quants`` refinement; output is wire-compatible.
    """
    N = blocks.shape[0]
    sb = blocks.reshape(N, N_SUB, SUB_BLOCK)
    imp_view = importance.reshape(N, N_SUB, SUB_BLOCK)

    # Step 1: per-sub-block fit, but with nmax=31.
    fitter = _fit_sub_blocks_qkx3_batched if use_qkx3 else _fit_sub_blocks_qkx2_batched
    fit_scales, fit_mins, _ = fitter(
        sb.reshape(N * N_SUB, SUB_BLOCK),
        imp_view.reshape(N * N_SUB, SUB_BLOCK),
        nmax=31,
    )
    fit_scales = fit_scales.reshape(N, N_SUB)
    fit_mins = fit_mins.reshape(N, N_SUB)

    # Step 2: master FP16 scales.
    max_sc = fit_scales.max(axis=1)
    max_min = fit_mins.max(axis=1)
    d = np.where(max_sc < 1e-12, 1.0, max_sc / 63.0).astype(np.float32)
    dmin = np.where(max_min < 1e-12, 1.0, max_min / 63.0).astype(np.float32)
    sb_scales_q = np.clip(np.round(fit_scales / d[:, None]), 0, 63).astype(np.uint8)
    sb_mins_q = np.clip(np.round(fit_mins / dmin[:, None]), 0, 63).astype(np.uint8)

    # Step 3: re-quantize each weight against the post-quant deq line.
    deq_scale = sb_scales_q.astype(np.float32) * d[:, None]
    deq_min = -sb_mins_q.astype(np.float32) * dmin[:, None]
    deq_scale_safe = np.where(deq_scale == 0, 1.0, deq_scale)
    deq_b = deq_scale[:, :, None]
    deq_m_b = deq_min[:, :, None]
    deq_safe_b = deq_scale_safe[:, :, None]
    q_real = (sb - deq_m_b) / deq_safe_b
    q_floor = np.clip(np.floor(q_real).astype(np.int32), 0, 31)
    q_ceil = np.clip(q_floor + 1, 0, 31)
    err_floor = imp_view * (sb - (deq_b * q_floor + deq_m_b)) ** 2
    err_ceil = imp_view * (sb - (deq_b * q_ceil + deq_m_b)) ** 2
    q5 = np.where(err_ceil < err_floor, q_ceil, q_floor).astype(np.uint8)
    q5_flat = q5.reshape(N, SUPER_BLOCK)                # [N, 256], values 0..31

    # Split into low-4-bits (qs) and high-1-bit (qh).
    qs = q5_flat & 0xF                                  # low 4 bits, [N, 256]
    qh = (q5_flat >> 4) & 0x1                           # high 1 bit, [N, 256]

    out = np.zeros((N, 176), dtype=np.uint8)
    out[:, 0:2] = np.frombuffer(d.astype(np.float16).tobytes(), dtype=np.uint8).reshape(N, 2)
    out[:, 2:4] = np.frombuffer(dmin.astype(np.float16).tobytes(), dtype=np.uint8).reshape(N, 2)
    out[:, 4:10] = _pack_6bit_batched(sb_scales_q)
    out[:, 10:16] = _pack_6bit_batched(sb_mins_q)
    # qs: 256 × 4 bits = 128 bytes per super-block.
    out[:, 16:144] = qs[:, 0::2] | (qs[:, 1::2] << 4)
    # qh: 256 × 1 bit = 32 bytes per super-block (8 weights per byte).
    out[:, 144:176] = (
        qh[:, 0::8] | (qh[:, 1::8] << 1) | (qh[:, 2::8] << 2) | (qh[:, 3::8] << 3) |
        (qh[:, 4::8] << 4) | (qh[:, 5::8] << 5) | (qh[:, 6::8] << 6) | (qh[:, 7::8] << 7)
    )
    return out


def _decode_super_blocks_batched(arr: np.ndarray) -> np.ndarray:
    """``arr`` shape ``(N, 176)`` uint8 → ``(N, 256)`` fp32."""
    N = arr.shape[0]
    d = np.frombuffer(arr[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32)
    dmin = np.frombuffer(arr[:, 2:4].tobytes(), dtype=np.float16).astype(np.float32)

    # Unpack 6-bit scales + mins.
    sb_scales_q = np.zeros((N, 8), dtype=np.uint8)
    sb_mins_q = np.zeros((N, 8), dtype=np.uint8)
    for blk_off, dest in ((4, sb_scales_q), (10, sb_mins_q)):
        # Build a 64-bit value per super-block from the 6 packed bytes.
        bits = np.zeros(N, dtype=np.uint64)
        for i in range(6):
            bits |= arr[:, blk_off + i].astype(np.uint64) << np.uint64(i * 8)
        for i in range(8):
            dest[:, i] = ((bits >> np.uint64(i * 6)) & np.uint64(0x3F)).astype(np.uint8)

    deq_scale = sb_scales_q.astype(np.float32) * d[:, None]
    deq_min = -sb_mins_q.astype(np.float32) * dmin[:, None]

    qs_packed = arr[:, 16:144]
    qs = np.zeros((N, SUPER_BLOCK), dtype=np.uint8)
    qs[:, 0::2] = qs_packed & 0xF
    qs[:, 1::2] = qs_packed >> 4

    qh_packed = arr[:, 144:176]
    qh = np.zeros((N, SUPER_BLOCK), dtype=np.uint8)
    for bit in range(8):
        qh[:, bit::8] = (qh_packed >> bit) & 0x1
    q5 = (qh.astype(np.uint8) << 4) | qs                # [N, 256]

    out = np.zeros((N, SUPER_BLOCK), dtype=np.float32)
    for s in range(N_SUB):
        sub = q5[:, s * SUB_BLOCK : (s + 1) * SUB_BLOCK].astype(np.float32)
        out[:, s * SUB_BLOCK : (s + 1) * SUB_BLOCK] = (
            deq_scale[:, s : s + 1] * sub + deq_min[:, s : s + 1]
        )
    return out


def encode_q5_k(
    tensor: torch.Tensor,
    importance: Optional[torch.Tensor] = None,
    use_qkx3: bool = False,
) -> Tuple[bytes, Tuple[int, ...]]:
    flat = tensor.detach().to(torch.float32).cpu().numpy().reshape(-1)
    if importance is not None:
        imp_flat = importance.detach().to(torch.float32).cpu().numpy().reshape(-1)
    else:
        imp_flat = np.ones_like(flat)
    n = flat.shape[0]
    n_blocks = (n + SUPER_BLOCK - 1) // SUPER_BLOCK
    pad = n_blocks * SUPER_BLOCK - n
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
        imp_flat = np.concatenate([imp_flat, np.zeros(pad, dtype=imp_flat.dtype)])
    sb = flat.reshape(n_blocks, SUPER_BLOCK)
    imp = imp_flat.reshape(n_blocks, SUPER_BLOCK)
    encoded = _encode_super_blocks_batched(sb, imp, use_qkx3=use_qkx3)
    return encoded.tobytes(), tuple(tensor.shape)


def decode_q5_k(blob: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    n_elems = 1
    for s in shape:
        n_elems *= s
    n_blocks = (n_elems + SUPER_BLOCK - 1) // SUPER_BLOCK
    expected = n_blocks * 176
    if len(blob) != expected:
        raise ValueError(f"blob length {len(blob)} != expected {expected} for shape {shape}")
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(n_blocks, 176)
    decoded = _decode_super_blocks_batched(arr)         # [n_blocks, 256]
    return torch.from_numpy(decoded.reshape(-1)[:n_elems].copy()).reshape(*shape)
