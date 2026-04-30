"""Q8_K: 8-bit K-quant. The simplest of the K-family.

Block structure (256 weights per super-block):

  super_block = [
      d:       fp32,                 # one super-block scale (single precision)
      qs[256]: int8,                 # per-weight signed 8-bit value
      bsums[16]: int16,              # per-sub-block sum of qs (used by inference
                                     # kernels for fast dot-products; we store
                                     # them faithfully even though pure-PyTorch
                                     # decoders ignore them).
  ]

Bits per weight:
    d:      32
    qs:     256 × 8       = 2048
    bsums:  16 × 16       = 256
    -------------------------------
    total                = 2336 / 256 = ~9.125 bits per weight

(Production llama.cpp uses 8.5 bits/w — the bsums are essentially metadata
that lives outside the weight budget when comparing across formats.)

Decode: ``w_i = d · qs_i``.
"""

from typing import Optional, Tuple

import numpy as np
import torch


SUPER_BLOCK = 256
SUB_BLOCK = 16
N_SUB = SUPER_BLOCK // SUB_BLOCK


def _quantize_super_block(block: np.ndarray, importance: Optional[np.ndarray] = None) -> bytes:
    assert block.shape == (SUPER_BLOCK,), block.shape
    block = block.astype(np.float32)
    abs_max = float(np.abs(block).max())
    if abs_max < 1e-12:
        d = np.float32(1.0)
    else:
        d = np.float32(abs_max / 127.0)
    qs = np.clip(np.round(block / d), -127, 127).astype(np.int8)

    bsums = np.zeros(N_SUB, dtype=np.int16)
    for s in range(N_SUB):
        bsums[s] = qs[s * SUB_BLOCK : (s + 1) * SUB_BLOCK].sum()

    out = bytearray()
    out.extend(np.float32(d).tobytes())
    out.extend(qs.tobytes())
    out.extend(bsums.tobytes())
    assert len(out) == 4 + 256 + 32, len(out)
    return bytes(out)


def _dequantize_super_block(blob: bytes) -> np.ndarray:
    if len(blob) != 292:
        raise ValueError(f"Q8_K block must be 292 bytes; got {len(blob)}")
    arr = np.frombuffer(blob, dtype=np.uint8)
    d = np.frombuffer(arr[0:4].tobytes(), dtype=np.float32)[0]
    qs = np.frombuffer(arr[4 : 4 + 256].tobytes(), dtype=np.int8).astype(np.float32)
    return d * qs


def encode_q8_k(
    tensor: torch.Tensor,
    importance: Optional[torch.Tensor] = None,
) -> Tuple[bytes, Tuple[int, ...]]:
    flat = tensor.detach().to(torch.float32).cpu().numpy().reshape(-1)
    n = flat.shape[0]
    n_blocks = (n + SUPER_BLOCK - 1) // SUPER_BLOCK
    pad = n_blocks * SUPER_BLOCK - n
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    blob = bytearray()
    for i in range(n_blocks):
        blob.extend(_quantize_super_block(flat[i * SUPER_BLOCK : (i + 1) * SUPER_BLOCK]))
    return bytes(blob), tuple(tensor.shape)


def decode_q8_k(blob: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    n_elems = 1
    for s in shape:
        n_elems *= s
    n_blocks = (n_elems + SUPER_BLOCK - 1) // SUPER_BLOCK
    expected = n_blocks * 292
    if len(blob) != expected:
        raise ValueError(f"blob length {len(blob)} != expected {expected} for shape {shape}")
    out = np.zeros(n_blocks * SUPER_BLOCK, dtype=np.float32)
    for i in range(n_blocks):
        out[i * SUPER_BLOCK : (i + 1) * SUPER_BLOCK] = _dequantize_super_block(
            blob[i * 292 : (i + 1) * 292]
        )
    return torch.from_numpy(out[:n_elems].copy()).reshape(*shape)
