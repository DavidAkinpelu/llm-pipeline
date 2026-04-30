"""IQ4_NL: 4-bit non-linear codebook quant.

Unlike linear quantization, an I-quant doesn't store ``q × scale``. Each
4-bit value is an *index* into a fixed 16-entry codebook of carefully
chosen float values shaped to match the empirical distribution of LLM
weights (heavy tail near zero, sparse outliers).

Block structure (32 weights per block):

  block = [
      d:    fp16,                # one scale per 32-weight block
      qs[32]: 4-bit,             # 16-entry codebook indices
  ]

Decode: ``w_i = d · codebook[qs_i]``.

Bits per weight:
    d:      16
    qs:     32 × 4       = 128
    -----------------------------
    total              = 144 / 32 = 4.5 bits per weight

The codebook below is the canonical IQ4_NL one used by llama.cpp.
"""

from typing import Optional, Tuple

import numpy as np
import torch


# Canonical 16-entry non-linear codebook (matches llama.cpp's kvalues_iq4nl).
IQ4_NL_CODEBOOK = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)


BLOCK = 32


def _quantize_block(block: np.ndarray, importance: Optional[np.ndarray] = None) -> bytes:
    """IQ4_NL block encoder with codebook-aware scale search.

    The naive scale ``d = max(|w|) / 113`` is suboptimal — for any chosen
    set of codebook indices, the closed-form best scale is
    ``s* = Σ w · cb[idx] · x / Σ w · cb[idx]²``. We search a few candidate
    initial scales (around the trivial), assign indices greedily, then
    refine with the closed-form, and keep the candidate with the lowest
    weighted MSE.
    """
    assert block.shape == (BLOCK,), block.shape
    block = block.astype(np.float32)
    imp = np.ones(BLOCK, dtype=np.float32) if importance is None else importance.astype(np.float32)

    abs_max = float(np.abs(block).max())
    if abs_max < 1e-12:
        out = bytearray()
        out.extend(np.float16(1.0).tobytes())
        out.extend(b"\x00" * 16)
        return bytes(out)

    cb = IQ4_NL_CODEBOOK                                       # [16]
    best_mad = float("inf")
    best_d = np.float32(abs_max / 113.0)
    best_indices = np.zeros(BLOCK, dtype=np.uint8)

    # Candidate initial scales: several linear samples around the trivial.
    candidates = [abs_max / r for r in (90.0, 100.0, 113.0, 125.0, 140.0)]
    for d_init in candidates:
        d_cand = np.float32(d_init)
        decoded = d_cand * cb                                  # [16]
        diffs = block[:, None] - decoded[None, :]              # [32, 16]
        sq_err = imp[:, None] * (diffs ** 2)
        idx = np.argmin(sq_err, axis=1).astype(np.uint8)       # [32]
        # Closed-form refinement: given fixed indices, the optimal scale.
        cb_vals = cb[idx]
        denom = float((imp * cb_vals * cb_vals).sum())
        if denom > 0:
            num = float((imp * cb_vals * block).sum())
            d_refined = np.float32(num / denom)
        else:
            d_refined = d_cand
        # Re-pick indices at the refined scale.
        decoded2 = d_refined * cb
        diffs2 = block[:, None] - decoded2[None, :]
        sq_err2 = imp[:, None] * (diffs2 ** 2)
        idx2 = np.argmin(sq_err2, axis=1).astype(np.uint8)
        deq2 = d_refined * cb[idx2]
        mad = float((imp * (block - deq2) ** 2).sum())
        if mad < best_mad:
            best_mad = mad
            best_d = d_refined
            best_indices = idx2

    out = bytearray()
    out.extend(np.float16(best_d).tobytes())
    packed = (best_indices[0::2] | (best_indices[1::2] << 4)).astype(np.uint8)
    out.extend(packed.tobytes())
    assert len(out) == 18, len(out)
    return bytes(out)


def _dequantize_block(blob: bytes) -> np.ndarray:
    if len(blob) != 18:
        raise ValueError(f"IQ4_NL block must be 18 bytes; got {len(blob)}")
    arr = np.frombuffer(blob, dtype=np.uint8)
    d = np.frombuffer(arr[0:2].tobytes(), dtype=np.float16).astype(np.float32)[0]
    packed = arr[2:18]
    indices = np.zeros(BLOCK, dtype=np.uint8)
    indices[0::2] = packed & 0xF
    indices[1::2] = packed >> 4
    return d * IQ4_NL_CODEBOOK[indices]


def encode_iq4_nl(
    tensor: torch.Tensor,
    importance: Optional[torch.Tensor] = None,
) -> Tuple[bytes, Tuple[int, ...]]:
    flat = tensor.detach().to(torch.float32).cpu().numpy().reshape(-1)
    imp_flat = (
        importance.detach().to(torch.float32).cpu().numpy().reshape(-1)
        if importance is not None
        else None
    )
    n = flat.shape[0]
    n_blocks = (n + BLOCK - 1) // BLOCK
    pad = n_blocks * BLOCK - n
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
        if imp_flat is not None:
            imp_flat = np.concatenate([imp_flat, np.zeros(pad, dtype=imp_flat.dtype)])
    blob = bytearray()
    for i in range(n_blocks):
        s = i * BLOCK
        e = s + BLOCK
        blob.extend(_quantize_block(
            flat[s:e],
            imp_flat[s:e] if imp_flat is not None else None,
        ))
    return bytes(blob), tuple(tensor.shape)


def decode_iq4_nl(blob: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    n_elems = 1
    for s in shape:
        n_elems *= s
    n_blocks = (n_elems + BLOCK - 1) // BLOCK
    expected = n_blocks * 18
    if len(blob) != expected:
        raise ValueError(f"blob length {len(blob)} != expected {expected} for shape {shape}")
    out = np.zeros(n_blocks * BLOCK, dtype=np.float32)
    for i in range(n_blocks):
        out[i * BLOCK : (i + 1) * BLOCK] = _dequantize_block(blob[i * 18 : (i + 1) * 18])
    return torch.from_numpy(out[:n_elems].copy()).reshape(*shape)
