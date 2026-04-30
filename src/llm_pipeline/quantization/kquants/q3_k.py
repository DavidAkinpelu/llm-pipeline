"""Q3_K: 3-bit signed K-quant. The lowest-bit K-quant in the family.

Block structure (256 weights per super-block):

  super_block = [
      d:           fp16,             # master scale (for the 16 sub-block scales)
      scales[16]:  6-bit signed,     # per-sub-block scale (12 bytes packed)
      hmask[256]:  1-bit per weight, # high bit of each 3-bit value (32 bytes)
      qs[256]:     2-bit per weight, # low 2 bits of each 3-bit value (64 bytes)
  ]

Bits per weight:
    d:      16
    scales: 16 × 6        = 96
    hmask:  256 × 1       = 256
    qs:     256 × 2       = 512
    -------------------------------
    total                = 880 / 256 = 3.4375 bits per weight

Decoding: ``w_i = d · scale_sb · q3_i`` where ``q3_i ∈ [-4, 3]`` is the
signed 3-bit value reconstructed from the (hmask, qs) split. We use the
same symmetric closed-form fit as Q6_K but with ``nmax=3``.

The byte layout is *not* bit-identical to llama.cpp's ``block_q3_K`` —
in particular, llama.cpp's signed 6-bit scale packing uses a different
6-bit encoding scheme. This module's packing is what's described in the
docstring above and round-trips losslessly via the matching decoder
below.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from .q6_k import _fit_sub_blocks_qx_batched


SUPER_BLOCK = 256
SUB_BLOCK = 16
N_SUB = SUPER_BLOCK // SUB_BLOCK  # 16


def _pack_signed_6bit_batched(values: np.ndarray) -> np.ndarray:
    """Pack ``(N, 16)`` int8 values (each in [-32, 31]) into ``(N, 12)`` bytes.

    We bias to unsigned in [0, 63] (add 32), then bit-pack 16 × 6 bits =
    96 bits = 12 bytes per row, low-bit-first.
    """
    biased = (values.astype(np.int32) + 32).astype(np.uint64)        # [N, 16]
    out = np.zeros((biased.shape[0], 12), dtype=np.uint8)
    # Build a uint128-ish layout via two uint64's per row.
    lo = np.zeros(biased.shape[0], dtype=np.uint64)
    hi = np.zeros(biased.shape[0], dtype=np.uint64)
    for i in range(16):
        bit = i * 6
        if bit + 6 <= 64:
            lo |= (biased[:, i] & np.uint64(0x3F)) << np.uint64(bit)
        elif bit >= 64:
            hi |= (biased[:, i] & np.uint64(0x3F)) << np.uint64(bit - 64)
        else:
            # Crosses the 64-bit boundary: split.
            n_lo = 64 - bit
            mask_lo = (np.uint64(1) << np.uint64(n_lo)) - np.uint64(1)
            lo |= (biased[:, i] & mask_lo) << np.uint64(bit)
            hi |= (biased[:, i] >> np.uint64(n_lo)) & np.uint64(0x3F)
    for i in range(8):
        out[:, i] = ((lo >> np.uint64(i * 8)) & np.uint64(0xFF)).astype(np.uint8)
    for i in range(4):
        out[:, 8 + i] = ((hi >> np.uint64(i * 8)) & np.uint64(0xFF)).astype(np.uint8)
    return out


def _unpack_signed_6bit_batched(packed: np.ndarray) -> np.ndarray:
    """Inverse of ``_pack_signed_6bit_batched``. ``packed`` shape ``(N, 12)``."""
    N = packed.shape[0]
    lo = np.zeros(N, dtype=np.uint64)
    hi = np.zeros(N, dtype=np.uint64)
    for i in range(8):
        lo |= packed[:, i].astype(np.uint64) << np.uint64(i * 8)
    for i in range(4):
        hi |= packed[:, 8 + i].astype(np.uint64) << np.uint64(i * 8)

    out = np.zeros((N, 16), dtype=np.int32)
    for i in range(16):
        bit = i * 6
        if bit + 6 <= 64:
            v = (lo >> np.uint64(bit)) & np.uint64(0x3F)
        elif bit >= 64:
            v = (hi >> np.uint64(bit - 64)) & np.uint64(0x3F)
        else:
            n_lo = 64 - bit
            mask_lo = (np.uint64(1) << np.uint64(n_lo)) - np.uint64(1)
            v_low = (lo >> np.uint64(bit)) & mask_lo
            v_high = (hi & np.uint64(0x3F)) >> np.uint64(0)  # placeholder
            # actually: high_bits_count = 6 - n_lo
            high_bits = (hi & ((np.uint64(1) << np.uint64(6 - n_lo)) - np.uint64(1))) << np.uint64(n_lo)
            v = v_low | high_bits
        out[:, i] = v.astype(np.int32) - 32  # debias
    return out


def _encode_super_blocks_batched(blocks: np.ndarray, importance: np.ndarray) -> np.ndarray:
    """Encode a batch of Q3_K super-blocks. Returns ``(N, 110)`` uint8."""
    N = blocks.shape[0]
    sb = blocks.reshape(N, N_SUB, SUB_BLOCK)
    imp_view = importance.reshape(N, N_SUB, SUB_BLOCK)

    # Step 1: per-sub-block symmetric fit with nmax=3 (3-bit signed).
    fit_scales, _q3_per_sub = _fit_sub_blocks_qx_batched(
        sb.reshape(N * N_SUB, SUB_BLOCK),
        imp_view.reshape(N * N_SUB, SUB_BLOCK),
        nmax=3,
    )
    fit_scales = fit_scales.reshape(N, N_SUB)

    # Step 2: master FP16 scale.
    max_sc = np.abs(fit_scales).max(axis=1)
    d = np.where(max_sc < 1e-12, 1.0, max_sc / 31.0).astype(np.float32)  # signed 6-bit max = 31
    sb_scales_q = np.clip(np.round(fit_scales / d[:, None]), -32, 31).astype(np.int8)

    # Step 3: re-quantize each weight against the post-quant scale.
    deq_scale = sb_scales_q.astype(np.float32) * d[:, None]
    deq_scale_safe = np.where(deq_scale == 0, 1.0, deq_scale)
    deq_b = deq_scale[:, :, None]
    deq_safe_b = deq_scale_safe[:, :, None]
    q_real = sb / deq_safe_b
    q_floor = np.clip(np.floor(q_real).astype(np.int32), -4, 3)
    q_ceil = np.clip(q_floor + 1, -4, 3)
    err_floor = imp_view * (sb - deq_b * q_floor) ** 2
    err_ceil = imp_view * (sb - deq_b * q_ceil) ** 2
    q3 = np.where(err_ceil < err_floor, q_ceil, q_floor)               # signed [-4, 3]

    # Convert signed → unsigned u3 ∈ [0, 7] for packing.
    u3 = (q3 + 4).astype(np.uint8).reshape(N, SUPER_BLOCK)
    qs = u3 & 0x3                                                       # low 2 bits
    hmask = (u3 >> 2) & 0x1                                             # high 1 bit

    out = np.zeros((N, 110), dtype=np.uint8)
    out[:, 0:2] = np.frombuffer(d.astype(np.float16).tobytes(), dtype=np.uint8).reshape(N, 2)
    out[:, 2:14] = _pack_signed_6bit_batched(sb_scales_q)
    # hmask: 256 × 1 bit = 32 bytes per super-block (8 weights per byte).
    out[:, 14:46] = (
        hmask[:, 0::8] | (hmask[:, 1::8] << 1) | (hmask[:, 2::8] << 2) | (hmask[:, 3::8] << 3) |
        (hmask[:, 4::8] << 4) | (hmask[:, 5::8] << 5) | (hmask[:, 6::8] << 6) | (hmask[:, 7::8] << 7)
    )
    # qs: 256 × 2 bits = 64 bytes per super-block (4 weights per byte).
    out[:, 46:110] = (
        qs[:, 0::4] | (qs[:, 1::4] << 2) | (qs[:, 2::4] << 4) | (qs[:, 3::4] << 6)
    )
    return out


def _decode_super_blocks_batched(arr: np.ndarray) -> np.ndarray:
    """``arr`` shape ``(N, 110)`` uint8 → ``(N, 256)`` fp32."""
    N = arr.shape[0]
    d = np.frombuffer(arr[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32)
    sb_scales_q = _unpack_signed_6bit_batched(arr[:, 2:14])             # [N, 16] signed
    deq_scale = sb_scales_q.astype(np.float32) * d[:, None]             # [N, 16]

    hmask_packed = arr[:, 14:46]
    hmask = np.zeros((N, SUPER_BLOCK), dtype=np.uint8)
    for bit in range(8):
        hmask[:, bit::8] = (hmask_packed >> bit) & 0x1

    qs_packed = arr[:, 46:110]
    qs = np.zeros((N, SUPER_BLOCK), dtype=np.uint8)
    for k in range(4):
        qs[:, k::4] = (qs_packed >> (2 * k)) & 0x3

    u3 = (hmask << 2) | qs                                              # [N, 256], unsigned [0,7]
    q3 = u3.astype(np.int32) - 4                                        # signed [-4, 3]

    out = np.zeros((N, SUPER_BLOCK), dtype=np.float32)
    for s in range(N_SUB):
        sub = q3[:, s * SUB_BLOCK : (s + 1) * SUB_BLOCK].astype(np.float32)
        out[:, s * SUB_BLOCK : (s + 1) * SUB_BLOCK] = deq_scale[:, s : s + 1] * sub
    return out


def encode_q3_k(
    tensor: torch.Tensor,
    importance: Optional[torch.Tensor] = None,
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
    encoded = _encode_super_blocks_batched(sb, imp)
    return encoded.tobytes(), tuple(tensor.shape)


def decode_q3_k(blob: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    n_elems = 1
    for s in shape:
        n_elems *= s
    n_blocks = (n_elems + SUPER_BLOCK - 1) // SUPER_BLOCK
    expected = n_blocks * 110
    if len(blob) != expected:
        raise ValueError(f"blob length {len(blob)} != expected {expected} for shape {shape}")
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(n_blocks, 110)
    decoded = _decode_super_blocks_batched(arr)
    return torch.from_numpy(decoded.reshape(-1)[:n_elems].copy()).reshape(*shape)
