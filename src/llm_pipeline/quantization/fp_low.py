"""FP8 (E4M3 / E5M2) and MXFP4 low-precision float quantization.

These formats trade integer precision for a wider dynamic range than
N-bit linear quantization. They're the workhorse formats for Hopper
H100 / Blackwell B200 inference paths and the OCP MX standard;
``torch.float8_e4m3fn`` and friends ship native CUDA support.

What's here
-----------

Pure-Python encode/decode reference implementations:

- **``encode_fp8_e4m3`` / ``decode_fp8_e4m3``** — FP8 with 4-bit
  exponent + 3-bit mantissa (sign+E4+M3 = 8 bits). Range ±448, smallest
  normal ±2⁻⁶. Block-scaled: one fp16 scale per 32-weight block.
- **``encode_fp8_e5m2`` / ``decode_fp8_e5m2``** — FP8 with 5-bit
  exponent + 2-bit mantissa. Wider range (±57344) at the cost of mantissa
  precision. Same block-scaled wrapper.
- **``encode_mxfp4`` / ``decode_mxfp4``** — OCP MX-style 4-bit float
  (sign+E2+M1 = 4 bits) with a shared 8-bit power-of-two (E8M0) exponent
  per 32-weight block. ~4.25 bpw. The smallest format that's natively
  supported by Blackwell.

Caveat
------

For Hopper/Blackwell deployment, ``torch`` provides native dtypes
(``torch.float8_e4m3fn``, ``torch.float8_e5m2``) and the matmul kernels
that operate on them. The encoders here exist to (a) study the bit
layouts directly and (b) round-trip between fp32 and the packed
representation without needing a CUDA device. For production inference,
load weights into the native dtype and let cuBLAS / cuDNN handle the
rest.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


BLOCK = 32                              # weights per block-scale group


# --------------------------------------------------------------------------- #
# E4M3 / E5M2 helpers (block-scaled FP8)
# --------------------------------------------------------------------------- #


def _fp8_max(exp_bits: int, mant_bits: int) -> float:
    """Max positive finite value our reference FP8 encoder will emit.

    Standard E4M3 reserves the all-1-exponent + max-mantissa code as NaN,
    capping the max finite at 448. Standard E5M2 follows IEEE inf/nan
    rules. This reference encoder allows biased=max with full mantissa,
    which gives one extra grid step beyond standard. That keeps the
    encode → decode round-trip clean (the encoded value decodes to itself
    exactly) at the cost of one wasted bit pattern that production
    formats reserve for NaN.

    Returned values:
      - E4M3 (4,3): 480 = (1 + 7/8) · 2^8     (vs standard 448)
      - E5M2 (5,2): 114688 = (1 + 3/4) · 2^16 (vs standard 57344)
    """
    bias = (1 << (exp_bits - 1)) - 1
    max_biased = (1 << exp_bits) - 1
    max_ev = max_biased - bias
    return float((1 << max_ev) * (2.0 - 2 ** -mant_bits))


def _quantize_to_fp8_bits(values: np.ndarray, exp_bits: int, mant_bits: int) -> np.ndarray:
    """Convert fp32 values into packed FP8 byte values (one byte per weight).

    The encoder is straightforward: scale and round to nearest representable
    FP8 grid point, then pack into the canonical ``<sign|exp|mantissa>``
    bit layout.
    """
    n = values.size
    out = np.zeros(n, dtype=np.uint8)
    bias = (1 << (exp_bits - 1)) - 1
    mant_mask = (1 << mant_bits) - 1

    for i in range(n):
        v = float(values[i])
        if v == 0.0 or not np.isfinite(v):
            out[i] = 0
            continue
        sign_bit = 0 if v >= 0 else 1
        v = abs(v)
        # Decompose: v = (1 + frac) · 2^exp where 0 ≤ frac < 1.
        ev = int(np.floor(np.log2(v))) if v > 0 else 0
        biased = ev + bias
        if biased <= 0:
            # Subnormal — flush to zero for this reference.
            out[i] = sign_bit << 7
            continue
        max_biased = (1 << exp_bits) - 1
        if biased > max_biased:
            # The unbiased exponent is past the format's range — saturate.
            # ``biased == max_biased`` is *valid* (the format permits the
            # full mantissa range there) and falls through to the
            # mantissa-computation branch below.
            biased = max_biased
            mant_q = mant_mask
        else:
            scale = 2.0 ** ev
            frac = v / scale - 1.0
            mant_q = int(round(frac * (1 << mant_bits)))
            if mant_q > mant_mask:
                # Mantissa rounded up past the field width — carry into
                # the exponent. Mantissa resets to 0 (NOT mant_mask: that
                # would jump from 1.0·2^(ev+1) all the way to 1.875·2^(ev+1)).
                mant_q = 0
                biased += 1
                if biased > max_biased:
                    # The carry pushed us past max — true saturation.
                    biased = max_biased
                    mant_q = mant_mask
        out[i] = (sign_bit << 7) | (biased << mant_bits) | mant_q
    return out


def _dequantize_fp8_bits(packed: np.ndarray, exp_bits: int, mant_bits: int) -> np.ndarray:
    bias = (1 << (exp_bits - 1)) - 1
    mant_mask = (1 << mant_bits) - 1
    n = packed.size
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        b = int(packed[i])
        sign = -1.0 if (b >> 7) & 1 else 1.0
        biased = (b >> mant_bits) & ((1 << exp_bits) - 1)
        mant = b & mant_mask
        if biased == 0:
            out[i] = 0.0           # subnormal flushed
            continue
        ev = biased - bias
        out[i] = sign * (1.0 + mant / (1 << mant_bits)) * (2.0 ** ev)
    return out


# --------------------------------------------------------------------------- #
# Public FP8 API (block-scaled)
# --------------------------------------------------------------------------- #


def _encode_fp8(
    tensor: torch.Tensor, exp_bits: int, mant_bits: int,
) -> Tuple[bytes, Tuple[int, ...]]:
    flat = tensor.detach().to(torch.float32).cpu().numpy().reshape(-1)
    n = flat.size
    n_blocks = (n + BLOCK - 1) // BLOCK
    pad = n_blocks * BLOCK - n
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    fp8_max = _fp8_max(exp_bits, mant_bits)

    out = bytearray()
    for b in range(n_blocks):
        sl = slice(b * BLOCK, (b + 1) * BLOCK)
        block = flat[sl]
        absmax = float(np.abs(block).max())
        scale = np.float32(1.0) if absmax < 1e-12 else np.float32(absmax / fp8_max)
        # Round each value to FP8 after scaling.
        normalised = block / float(scale) if absmax > 0 else block
        packed = _quantize_to_fp8_bits(normalised, exp_bits, mant_bits)
        out.extend(np.float16(scale).tobytes())              # 2 bytes
        out.extend(packed.tobytes())                          # 32 bytes
    return bytes(out), tuple(tensor.shape)


def _decode_fp8(
    blob: bytes, shape: Tuple[int, ...], exp_bits: int, mant_bits: int,
) -> torch.Tensor:
    n_elems = 1
    for s in shape:
        n_elems *= s
    n_blocks = (n_elems + BLOCK - 1) // BLOCK
    block_size = 2 + BLOCK
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(n_blocks, block_size)
    out = np.zeros(n_blocks * BLOCK, dtype=np.float32)
    for b in range(n_blocks):
        scale = np.frombuffer(arr[b, :2].tobytes(), dtype=np.float16).astype(np.float32)[0]
        packed = arr[b, 2:]
        decoded = _dequantize_fp8_bits(packed, exp_bits, mant_bits)
        out[b * BLOCK:(b + 1) * BLOCK] = decoded * scale
    return torch.from_numpy(out[:n_elems].copy()).reshape(*shape)


def encode_fp8_e4m3(tensor: torch.Tensor) -> Tuple[bytes, Tuple[int, ...]]:
    return _encode_fp8(tensor, exp_bits=4, mant_bits=3)


def decode_fp8_e4m3(blob: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    return _decode_fp8(blob, shape, exp_bits=4, mant_bits=3)


def encode_fp8_e5m2(tensor: torch.Tensor) -> Tuple[bytes, Tuple[int, ...]]:
    return _encode_fp8(tensor, exp_bits=5, mant_bits=2)


def decode_fp8_e5m2(blob: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    return _decode_fp8(blob, shape, exp_bits=5, mant_bits=2)


# --------------------------------------------------------------------------- #
# MXFP4 — OCP MX 4-bit float (E2M1) with E8M0 shared exponent per 32-weight block
# --------------------------------------------------------------------------- #


# Fixed E2M1 lookup: 4-bit codes → fp32 values.
# Layout: bit 3 = sign, bits 2-1 = exponent (bias 1), bit 0 = mantissa.
# Special: code 0b0000 = +0, 0b1000 = -0.
_MXFP4_LUT = np.array([
    +0.0,        # 0000
    +0.5,        # 0001
    +1.0,        # 0010
    +1.5,        # 0011
    +2.0,        # 0100
    +3.0,        # 0101
    +4.0,        # 0110
    +6.0,        # 0111
    -0.0,        # 1000
    -0.5,        # 1001
    -1.0,        # 1010
    -1.5,        # 1011
    -2.0,        # 1100
    -3.0,        # 1101
    -4.0,        # 1110
    -6.0,        # 1111
], dtype=np.float32)


def _quantize_to_mxfp4_codes(values: np.ndarray) -> np.ndarray:
    """Round each fp32 input to the closest MXFP4 grid value, return 4-bit codes."""
    diffs = (values[:, None] - _MXFP4_LUT[None, :]) ** 2
    return np.argmin(diffs, axis=1).astype(np.uint8)


def encode_mxfp4(tensor: torch.Tensor) -> Tuple[bytes, Tuple[int, ...]]:
    """OCP MX FP4: per-32-weight block scale stored as E8M0 (one byte =
    a power of 2). Each weight is a 4-bit code into ``_MXFP4_LUT``.
    Two weights packed per byte.
    """
    flat = tensor.detach().to(torch.float32).cpu().numpy().reshape(-1)
    n = flat.size
    n_blocks = (n + BLOCK - 1) // BLOCK
    pad = n_blocks * BLOCK - n
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])

    out = bytearray()
    for b in range(n_blocks):
        sl = slice(b * BLOCK, (b + 1) * BLOCK)
        block = flat[sl]
        absmax = float(np.abs(block).max())
        if absmax < 1e-12:
            scale_exp = 0
            normalised = block
        else:
            # Pick scale = 2^k such that absmax / 2^k ≤ 6 (LUT max).
            scale_exp = int(np.ceil(np.log2(absmax / 6.0)))
            scale_exp = max(min(scale_exp, 127), -127)
            normalised = block / (2.0 ** scale_exp)
        codes = _quantize_to_mxfp4_codes(normalised)
        # E8M0 byte: store ``scale_exp + 127`` so the byte is always in [0, 254].
        out.extend(bytes([scale_exp + 127]))
        # Pack 32 codes into 16 bytes (low nibble first).
        packed = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8)
        out.extend(packed.tobytes())                                # 16 bytes
    return bytes(out), tuple(tensor.shape)


def decode_mxfp4(blob: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    n_elems = 1
    for s in shape:
        n_elems *= s
    n_blocks = (n_elems + BLOCK - 1) // BLOCK
    block_size = 1 + BLOCK // 2                                  # 17 bytes
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(n_blocks, block_size)
    out = np.zeros(n_blocks * BLOCK, dtype=np.float32)
    for b in range(n_blocks):
        scale_exp = int(arr[b, 0]) - 127
        scale = 2.0 ** scale_exp
        packed = arr[b, 1:]
        codes = np.zeros(BLOCK, dtype=np.uint8)
        codes[0::2] = packed & 0xF
        codes[1::2] = packed >> 4
        out[b * BLOCK:(b + 1) * BLOCK] = _MXFP4_LUT[codes] * scale
    return torch.from_numpy(out[:n_elems].copy()).reshape(*shape)
