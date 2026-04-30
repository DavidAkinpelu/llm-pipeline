"""Q6_K: 6-bit K-quant.

Block structure (256 weights per super-block):

  super_block = [
      d:        fp16,                # master scale
      scales[16]: int8,              # per-sub-block scale (signed, 16 sub-blocks of 16)
      qh[64]:    2-bit per weight,   # high 2 bits of each 6-bit value
      ql[128]:   4-bit per weight,   # low 4 bits of each 6-bit value
  ]

Bits per weight:
    d:      16
    scales: 16 × 8        = 128
    qh:     256 × 2       = 512
    ql:     256 × 4       = 1024
    -------------------------------
    total                = 1680 / 256 = 6.5625 bits per weight

Decode: ``w_i = d · scale_sb · q6_i`` where ``q6_i = (qh<<4 | ql) - 32`` is
a *signed* 6-bit value in ``[-32, 31]``.

Like Q4_K, this is algorithmically correct but the byte layout is not
bit-exact with llama.cpp's ``block_q6_K``.
"""

from typing import Optional, Tuple

import numpy as np
import torch


SUPER_BLOCK = 256
SUB_BLOCK = 16
N_SUB = SUPER_BLOCK // SUB_BLOCK  # 16


def _fit_sub_blocks_qx_batched(
    x: np.ndarray,
    w: np.ndarray,
    nmax: int = 31,        # symmetric: signed 6-bit values in [-(nmax+1), nmax]
    nstep: int = 20,
    rmin: float = -1.0,
    rdelta: float = 0.1,
) -> tuple:
    """Vectorized symmetric ``make_qx_quants`` (no min, no offset).

    Same idea as the asymmetric Q4_K fit but for symmetric quantization
    around 0 — appropriate when the dequant rule is ``w ≈ scale · q`` with
    ``q`` signed. Closed-form for fixed ``q``: ``s* = Σ w·q·x / Σ w·q²``.

    Returns ``(scales[B], q[B, n])`` where q is signed int32.
    """
    B, n = x.shape
    abs_max = np.abs(x).max(axis=1)                          # [B]
    valid = abs_max > 0
    iscale0 = np.where(valid, nmax / np.where(valid, abs_max, 1.0), 0.0)
    q0 = np.clip(np.rint(iscale0[:, None] * x), -(nmax + 1), nmax).astype(np.int32)
    scale0 = np.where(valid, 1.0 / np.where(valid, iscale0, 1.0), 0.0)
    diff0 = scale0[:, None] * q0 - x
    best_mad = (w * diff0 * diff0).sum(axis=1)
    scales = scale0.copy().astype(np.float32)
    q_best = q0.copy()

    for is_step in range(nstep + 1):
        cand_iscale = np.where(
            valid,
            (rmin + rdelta * is_step + nmax) / np.where(valid, abs_max, 1.0),
            0.0,
        )
        positive = cand_iscale > 0
        q = np.clip(np.rint(cand_iscale[:, None] * x), -(nmax + 1), nmax).astype(np.int32)
        qf = q.astype(np.float32)
        SQ2 = (w * qf * qf).sum(axis=1)
        SXQ = (w * qf * x).sum(axis=1)
        sl2_ok = SQ2 > 0
        SQ2_safe = np.where(sl2_ok, SQ2, 1.0)
        cand_scale = SXQ / SQ2_safe
        usable = positive & sl2_ok
        diff = cand_scale[:, None] * qf - x
        mad = (w * diff * diff).sum(axis=1)
        improved = usable & (mad < best_mad)
        best_mad = np.where(improved, mad, best_mad)
        q_best = np.where(improved[:, None], q, q_best)
        scales = np.where(improved, cand_scale, scales)

    return scales.astype(np.float32), q_best.astype(np.int32)


def _quantize_super_block(
    block: np.ndarray,
    importance: Optional[np.ndarray] = None,
) -> bytes:
    """Encode 256 weights → 210 bytes per the layout above.

    Uses ``make_qx_quants`` (closed-form symmetric scale-fit) per sub-block —
    the same algorithm production llama.cpp uses for Q6_K.
    """
    assert block.shape == (SUPER_BLOCK,), block.shape
    block = block.astype(np.float32)
    imp = np.ones(SUPER_BLOCK, dtype=np.float32) if importance is None else importance.astype(np.float32)

    sb = block.reshape(N_SUB, SUB_BLOCK)
    imp_view = imp.reshape(N_SUB, SUB_BLOCK)

    # ---- Step 1: vectorized per-sub-block scale fit. ----
    fit_scales, q6_per_sub = _fit_sub_blocks_qx_batched(sb, imp_view)

    # ---- Step 2: master scale d (FP16) so per-sub-block scales fit in int8. ----
    max_sc = float(np.abs(fit_scales).max()) if fit_scales.size else 0.0
    if max_sc < 1e-12:
        d = np.float32(1.0)
    else:
        d = np.float32(max_sc / 127.0)

    sb_scales_q = np.clip(np.round(fit_scales / d), -127, 127).astype(np.int8)

    # ---- Step 3: re-quantize each weight against the post-quant scale. ----
    deq_scale = sb_scales_q.astype(np.float32) * d
    deq_scale_safe = np.where(deq_scale == 0, 1.0, deq_scale)

    q6 = np.zeros(SUPER_BLOCK, dtype=np.int32)
    for s in range(N_SUB):
        sub_w = sb[s]
        sub_imp = imp_view[s]
        q_real = sub_w / deq_scale_safe[s]
        q_floor = np.clip(np.floor(q_real).astype(np.int32), -32, 31)
        q_ceil = np.clip(q_floor + 1, -32, 31)
        err_floor = sub_imp * (sub_w - deq_scale[s] * q_floor) ** 2
        err_ceil = sub_imp * (sub_w - deq_scale[s] * q_ceil) ** 2
        chosen = np.where(err_ceil < err_floor, q_ceil, q_floor)
        q6[s * SUB_BLOCK : (s + 1) * SUB_BLOCK] = chosen

    # Convert signed q6 ∈ [-32, 31] → unsigned u6 ∈ [0, 63] for packing.
    u6 = (q6 + 32).astype(np.uint8)
    ql = (u6 & 0xF).astype(np.uint8)            # low 4 bits
    qh = ((u6 >> 4) & 0x3).astype(np.uint8)     # high 2 bits

    out = bytearray()
    out.extend(np.float16(d).tobytes())
    out.extend(sb_scales_q.tobytes())
    # ql packed two-per-byte: 256 × 4 bits = 128 bytes.
    ql_packed = (ql[0::2] | (ql[1::2] << 4)).astype(np.uint8)
    out.extend(ql_packed.tobytes())
    # qh packed four-per-byte: 256 × 2 bits = 64 bytes.
    qh_packed = (qh[0::4] | (qh[1::4] << 2) | (qh[2::4] << 4) | (qh[3::4] << 6)).astype(np.uint8)
    out.extend(qh_packed.tobytes())
    assert len(out) == 2 + 16 + 128 + 64, len(out)
    return bytes(out)


def _dequantize_super_block(blob: bytes) -> np.ndarray:
    if len(blob) != 210:
        raise ValueError(f"Q6_K block must be 210 bytes; got {len(blob)}")
    arr = np.frombuffer(blob, dtype=np.uint8)
    d = np.frombuffer(arr[0:2].tobytes(), dtype=np.float16).astype(np.float32)[0]
    sb_scales_q = np.frombuffer(arr[2:18].tobytes(), dtype=np.int8).astype(np.float32)
    ql_packed = arr[18 : 18 + 128]
    qh_packed = arr[18 + 128 : 18 + 128 + 64]

    ql = np.zeros(SUPER_BLOCK, dtype=np.uint8)
    ql[0::2] = ql_packed & 0xF
    ql[1::2] = ql_packed >> 4

    qh = np.zeros(SUPER_BLOCK, dtype=np.uint8)
    qh[0::4] = qh_packed & 0x3
    qh[1::4] = (qh_packed >> 2) & 0x3
    qh[2::4] = (qh_packed >> 4) & 0x3
    qh[3::4] = (qh_packed >> 6) & 0x3

    u6 = (qh.astype(np.int32) << 4) | ql.astype(np.int32)
    q6 = u6 - 32                                 # back to signed [-32, 31]
    deq_scale = sb_scales_q * d                  # [N_SUB]
    out = np.zeros(SUPER_BLOCK, dtype=np.float32)
    for s in range(N_SUB):
        out[s * SUB_BLOCK : (s + 1) * SUB_BLOCK] = q6[s * SUB_BLOCK : (s + 1) * SUB_BLOCK] * deq_scale[s]
    return out


def _encode_super_blocks_batched(
    blocks: np.ndarray,            # (N, 256)
    importance: np.ndarray,         # (N, 256)
) -> np.ndarray:                    # (N, 210) bytes
    """Encode a batch of Q6_K super-blocks at once."""
    N = blocks.shape[0]
    sb = blocks.reshape(N, N_SUB, SUB_BLOCK)
    imp_view = importance.reshape(N, N_SUB, SUB_BLOCK)

    # Step 1: per-sub-block symmetric scale fit.
    fit_scales, q6_per_sub = _fit_sub_blocks_qx_batched(
        sb.reshape(N * N_SUB, SUB_BLOCK),
        imp_view.reshape(N * N_SUB, SUB_BLOCK),
    )
    fit_scales = fit_scales.reshape(N, N_SUB)            # [N, N_SUB]

    # Step 2: master FP16 scale d.
    max_sc = np.abs(fit_scales).max(axis=1)              # [N]
    d = np.where(max_sc < 1e-12, 1.0, max_sc / 127.0).astype(np.float32)
    sb_scales_q = np.clip(np.round(fit_scales / d[:, None]), -127, 127).astype(np.int8)

    # Step 3: re-quantize each weight against the post-quant scale.
    deq_scale = sb_scales_q.astype(np.float32) * d[:, None]    # [N, N_SUB]
    deq_scale_safe = np.where(deq_scale == 0, 1.0, deq_scale)
    deq_b = deq_scale[:, :, None]
    deq_safe_b = deq_scale_safe[:, :, None]
    q_real = sb / deq_safe_b
    q_floor = np.clip(np.floor(q_real).astype(np.int32), -32, 31)
    q_ceil = np.clip(q_floor + 1, -32, 31)
    err_floor = imp_view * (sb - deq_b * q_floor) ** 2
    err_ceil = imp_view * (sb - deq_b * q_ceil) ** 2
    q6 = np.where(err_ceil < err_floor, q_ceil, q_floor)        # [N, N_SUB, SUB_BLOCK]

    # Convert signed q6 ∈ [-32, 31] → unsigned u6 ∈ [0, 63] for packing.
    u6 = (q6 + 32).astype(np.uint8).reshape(N, SUPER_BLOCK)
    ql = (u6 & 0xF)                                            # low 4 bits, [N, 256]
    qh = ((u6 >> 4) & 0x3)                                     # high 2 bits, [N, 256]

    out = np.zeros((N, 210), dtype=np.uint8)
    out[:, 0:2] = np.frombuffer(d.astype(np.float16).tobytes(), dtype=np.uint8).reshape(N, 2)
    out[:, 2:18] = sb_scales_q.view(np.uint8)                  # int8 reinterpret
    out[:, 18:18 + 128] = ql[:, 0::2] | (ql[:, 1::2] << 4)
    out[:, 18 + 128:18 + 128 + 64] = (
        qh[:, 0::4] | (qh[:, 1::4] << 2) | (qh[:, 2::4] << 4) | (qh[:, 3::4] << 6)
    )
    return out


def encode_q6_k(
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


def decode_q6_k(blob: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    n_elems = 1
    for s in shape:
        n_elems *= s
    n_blocks = (n_elems + SUPER_BLOCK - 1) // SUPER_BLOCK
    expected = n_blocks * 210
    if len(blob) != expected:
        raise ValueError(f"blob length {len(blob)} != expected {expected} for shape {shape}")
    out = np.zeros(n_blocks * SUPER_BLOCK, dtype=np.float32)
    for i in range(n_blocks):
        out[i * SUPER_BLOCK : (i + 1) * SUPER_BLOCK] = _dequantize_super_block(
            blob[i * 210 : (i + 1) * 210]
        )
    return torch.from_numpy(out[:n_elems].copy()).reshape(*shape)
