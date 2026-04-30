"""IQ4_XS: 4-bit super-block I-quant (4.25 bpw).

A scaled-up cousin of ``IQ4_NL``: 256-weight super-blocks divided into
eight 32-weight sub-blocks, each with its own 6-bit scale relative to a
master fp16 scale. The actual weight values are the same 16-entry
non-linear codebook used by IQ4_NL.

Block structure (256 weights):

  super_block = [
      d:        fp16,                # master scale for the per-sub-block scales
      scales[8]: 6-bit unsigned,     # one scale per 32-weight sub-block (6 bytes packed)
      qs[256]:   4-bit unsigned,     # 16-entry codebook indices (128 bytes packed)
  ]

Bits per weight:
    d:        16
    scales:    8 × 6           = 48
    qs:        256 × 4          = 1024
    -------------------------------
    total                       = 1088 / 256 = 4.25 bits per weight

Decode: ``w_i = d · scale_sb · CODEBOOK[qs_i]`` where ``scale_sb`` is the
6-bit-quantized per-sub-block scale.

Algorithmically faithful to llama.cpp's ``IQ4_XS``; the byte layout is
*not* guaranteed bit-exact to its C struct (uses the same packing trick
as our other K-quant references — see the wire-format note in
``__init__.py``).
"""

from typing import Optional, Tuple

import numpy as np
import torch

from .iq4 import IQ4_NL_CODEBOOK
from .q4_k import _pack_6bit, _unpack_6bit


SUPER_BLOCK = 256
SUB_BLOCK = 32
N_SUB = SUPER_BLOCK // SUB_BLOCK            # 8


def _fit_subblock_iq4(
    sub: np.ndarray, importance: np.ndarray,
) -> Tuple[float, np.ndarray, float]:
    """Pick a per-sub-block scale + codebook indices minimising weighted MSE.

    Returns ``(scale, indices, mad)`` where ``mad`` is the weighted MSE
    of the chosen fit. The scale is the unquantized real value; sub-block
    quantisation to 6 bits happens at super-block level.
    """
    abs_max = float(np.abs(sub).max())
    if abs_max < 1e-12:
        return 0.0, np.zeros(SUB_BLOCK, dtype=np.uint8), 0.0

    cb = IQ4_NL_CODEBOOK
    best_scale = abs_max / 113.0
    best_idx = np.zeros(SUB_BLOCK, dtype=np.uint8)
    best_mad = float("inf")

    for r in (90.0, 100.0, 113.0, 125.0, 140.0):
        scale = abs_max / r
        decoded = scale * cb                                # [16]
        sq = importance[:, None] * (sub[:, None] - decoded[None, :]) ** 2
        idx = np.argmin(sq, axis=1).astype(np.uint8)
        cb_vals = cb[idx]
        denom = float((importance * cb_vals * cb_vals).sum())
        if denom > 0:
            num = float((importance * cb_vals * sub).sum())
            scale_ref = num / denom
        else:
            scale_ref = scale
        decoded2 = scale_ref * cb
        sq2 = importance[:, None] * (sub[:, None] - decoded2[None, :]) ** 2
        idx2 = np.argmin(sq2, axis=1).astype(np.uint8)
        deq = scale_ref * cb[idx2]
        mad = float((importance * (sub - deq) ** 2).sum())
        if mad < best_mad:
            best_mad = mad
            best_scale = scale_ref
            best_idx = idx2

    return best_scale, best_idx, best_mad


def _quantize_super_block(block: np.ndarray, importance: Optional[np.ndarray] = None) -> bytes:
    assert block.shape == (SUPER_BLOCK,), block.shape
    block = block.astype(np.float32)
    imp = (
        np.ones(SUPER_BLOCK, dtype=np.float32)
        if importance is None
        else importance.astype(np.float32)
    )
    sb = block.reshape(N_SUB, SUB_BLOCK)
    imp_view = imp.reshape(N_SUB, SUB_BLOCK)

    # Step 1: per-sub-block scale-and-indices fit.
    scales = np.zeros(N_SUB, dtype=np.float32)
    all_idx = np.zeros((N_SUB, SUB_BLOCK), dtype=np.uint8)
    for s in range(N_SUB):
        scales[s], all_idx[s], _ = _fit_subblock_iq4(sb[s], imp_view[s])

    # Step 2: master scale that quantises the per-sub-block scales to 6 bits.
    abs_scales = np.abs(scales)
    max_sc = float(abs_scales.max())
    d = np.float32(1.0) if max_sc < 1e-12 else np.float32(max_sc / 31.0)
    # Scales are signed (sub-blocks can have positive or negative sign convention
    # baked in via the codebook), but for IQ4_XS we use the absolute value with
    # the codebook covering both ranges. So map abs scale to a 6-bit unsigned.
    sb_scales_q = np.clip(np.round(abs_scales / d), 0, 63).astype(np.uint8)

    # Step 3: re-pick indices using the *quantized* per-sub-block scales.
    cb = IQ4_NL_CODEBOOK
    qs = np.zeros(SUPER_BLOCK, dtype=np.uint8)
    for s in range(N_SUB):
        deq_scale = float(sb_scales_q[s]) * d * (1.0 if scales[s] >= 0 else -1.0)
        if abs(deq_scale) < 1e-12:
            qs[s * SUB_BLOCK:(s + 1) * SUB_BLOCK] = 0
            continue
        decoded = deq_scale * cb
        diffs = sb[s][:, None] - decoded[None, :]
        sq = imp_view[s][:, None] * (diffs ** 2)
        qs[s * SUB_BLOCK:(s + 1) * SUB_BLOCK] = np.argmin(sq, axis=1).astype(np.uint8)

    out = bytearray()
    out.extend(np.float16(d).tobytes())
    out.extend(_pack_6bit(sb_scales_q).tobytes())
    qs_packed = (qs[0::2] | (qs[1::2] << 4)).astype(np.uint8)
    out.extend(qs_packed.tobytes())
    # Sign byte: bit i tells whether sub-block i had a negative scale.
    sign_byte = 0
    for s in range(N_SUB):
        if scales[s] < 0:
            sign_byte |= (1 << s)
    out.extend(bytes([sign_byte]))
    assert len(out) == 2 + 6 + 128 + 1, len(out)             # 137 bytes / 256 weights
    return bytes(out)


def _dequantize_super_block(blob: bytes) -> np.ndarray:
    if len(blob) != 137:
        raise ValueError(f"IQ4_XS block must be 137 bytes; got {len(blob)}")
    arr = np.frombuffer(blob, dtype=np.uint8)
    d = np.frombuffer(arr[0:2].tobytes(), dtype=np.float16).astype(np.float32)[0]
    sb_scales_q = _unpack_6bit(arr[2:8])
    qs_packed = arr[8:136]
    sign_byte = int(arr[136])

    qs = np.zeros(SUPER_BLOCK, dtype=np.uint8)
    qs[0::2] = qs_packed & 0xF
    qs[1::2] = qs_packed >> 4

    out = np.zeros(SUPER_BLOCK, dtype=np.float32)
    for s in range(N_SUB):
        sign = -1.0 if (sign_byte >> s) & 1 else 1.0
        sub_scale = float(sb_scales_q[s]) * d * sign
        sub_idx = qs[s * SUB_BLOCK:(s + 1) * SUB_BLOCK]
        out[s * SUB_BLOCK:(s + 1) * SUB_BLOCK] = sub_scale * IQ4_NL_CODEBOOK[sub_idx]
    return out


def encode_iq4_xs(
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
    out = bytearray()
    for b in range(n_blocks):
        sl = slice(b * SUPER_BLOCK, (b + 1) * SUPER_BLOCK)
        out.extend(_quantize_super_block(flat[sl], imp_flat[sl]))
    return bytes(out), tuple(tensor.shape)


def decode_iq4_xs(blob: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    n_elems = 1
    for s in shape:
        n_elems *= s
    n_blocks = (n_elems + SUPER_BLOCK - 1) // SUPER_BLOCK
    out = np.zeros(n_blocks * SUPER_BLOCK, dtype=np.float32)
    for b in range(n_blocks):
        out[b * SUPER_BLOCK:(b + 1) * SUPER_BLOCK] = _dequantize_super_block(
            blob[b * 137:(b + 1) * 137]
        )
    return torch.from_numpy(out[:n_elems].copy()).reshape(*shape)
