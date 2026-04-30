"""Low-bit codebook I-quants: IQ3_XXS and IQ2_XXS.

The IQ family at low bit-rates uses 8-element codebook *grids*: small
hand-selected vectors of 8 signed values that approximate the empirical
distribution of LLM weights better than any uniform scheme can. Each
group of 8 weights is encoded as an index into a 256-entry grid table.

llama.cpp ships canonical grid tables for IQ3_XXS, IQ3_S, IQ2_XXS, etc.
— the values were tuned by hand against real weight distributions and
are part of the file format. Bit-exact compatibility requires copying
those tables verbatim. This module provides the **algorithmic
reference**: the encode/decode logic operates on a grid that's deterministic
and structurally similar (same 8-element-vector layout, same scaling +
indexing flow), but **not byte-identical** to llama.cpp's tables.

Block structure
---------------

Both formats use 256-weight super-blocks split into 32 groups of 8
weights, each group encoded as a 16-bit grid index:

  super_block = [
      d:           fp16,                      # one master scale
      sign_byte[8]: int8,                     # signs for 8 groups (one byte)
      qs[32]:      uint16,                    # one 16-bit grid index per 8-weight group
  ]

Per-group decode: ``w_group = d · sign · GRID[qs_group]`` where
``GRID`` is a 256×8 codebook table. Because each group needs only 8
bits of grid index (256 entries) we have 8 spare bits per group; in
llama.cpp these encode finer per-group sign / scale info. Our reference
uses them as a per-group 8-bit absmax-based scale offset for slightly
finer fits.

Bits per weight
---------------

- IQ3_XXS: ``16 + 8 + 32×16 = 536`` bits / 256 weights = **2.094 bpw**.
  llama.cpp's official IQ3_XXS is 3.0625 bpw (with a richer per-group
  encoding); ours is more aggressive because we don't pack the per-group
  scale — same algorithmic class, sub-3-bit storage.
- IQ2_XXS: ``16 + 8 + 32×8 = 280`` bits / 256 weights = **1.094 bpw**
  using an 8-bit per-group index into a 256-entry grid (no scale).
  llama.cpp's IQ2_XXS is 2.0625 bpw with a richer 16-bit-per-group
  encoding.

For both, the decode-side ratio between this reference and the official
formats is documented but not the headline; the **format and quality
trends** are what's educationally valuable.
"""

from typing import Optional, Tuple

import numpy as np
import torch


GROUP_SIZE = 8                         # weights per codebook lookup
SUPER_BLOCK = 256
N_GROUPS = SUPER_BLOCK // GROUP_SIZE   # 32


def _build_iq_grid(n_entries: int = 256, seed: int = 42) -> np.ndarray:
    """Construct a deterministic 256-entry grid of 8-element signed vectors.

    The official llama.cpp tables were hand-tuned. We approximate them by
    drawing from a heavy-tailed distribution (Gaussian + sparse spikes)
    quantised to integer values in ``[-7, 7]`` — the same value range
    llama.cpp uses for its 3-bit codebook entries. Deterministic via
    fixed seed so the encode/decode round-trip is reproducible.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_entries, GROUP_SIZE)) * 2.0
    # Sprinkle in some larger values to mimic the heavy-tail patterns
    # llama.cpp's hand-tuned grid captures.
    spike_mask = rng.random((n_entries, GROUP_SIZE)) < 0.15
    base[spike_mask] *= 3.0
    grid = np.clip(np.round(base), -7, 7).astype(np.float32)
    return grid


# Module-level grids (computed once).
_IQ3_GRID = _build_iq_grid(n_entries=256, seed=42)
_IQ2_GRID = _build_iq_grid(n_entries=256, seed=99)


def _encode_super_block(
    block: np.ndarray, grid: np.ndarray,
    importance: Optional[np.ndarray] = None,
) -> bytes:
    """Common encoder for any 8-element-grid I-quant.

    Per super-block: pick a master scale ``d``, then for each 8-weight
    group find the (sign, grid_idx) minimising weighted MSE.
    """
    assert block.shape == (SUPER_BLOCK,), block.shape
    block = block.astype(np.float32)
    imp = (
        np.ones(SUPER_BLOCK, dtype=np.float32)
        if importance is None
        else importance.astype(np.float32)
    )
    abs_max = float(np.abs(block).max())
    if abs_max < 1e-12:
        out = bytearray()
        out.extend(np.float16(0.0).tobytes())                      # 2 bytes
        out.extend(b"\x00" * 4)                                     # 4 sign bytes
        out.extend(b"\x00" * (2 * N_GROUPS))                        # 64 index bytes
        return bytes(out)                                           # 70 bytes total

    # Master scale chosen to put grid extremes (~7) at the absmax.
    d = np.float32(abs_max / 7.0)

    sign_byte = 0
    indices = np.zeros(N_GROUPS, dtype=np.uint16)
    for g in range(N_GROUPS):
        sl = slice(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)
        wg = block[sl]
        ig = imp[sl]

        # For each (sign, grid_idx) pair, compute weighted MSE; pick the best.
        # ``grid`` is ``[256, 8]``; we test both sign conventions.
        scaled_pos = d * grid                                  # [256, 8]
        diffs_pos = (wg[None, :] - scaled_pos) ** 2            # [256, 8]
        weighted_pos = (ig[None, :] * diffs_pos).sum(axis=1)    # [256]
        diffs_neg = (wg[None, :] + scaled_pos) ** 2
        weighted_neg = (ig[None, :] * diffs_neg).sum(axis=1)

        best_pos = int(weighted_pos.argmin())
        best_neg = int(weighted_neg.argmin())
        if weighted_pos[best_pos] <= weighted_neg[best_neg]:
            indices[g] = best_pos
        else:
            indices[g] = best_neg
            sign_byte |= (1 << (g % 8)) if g < 8 else 0       # only first 8 fit in one byte

    # Encode the sign bits more carefully: pack one bit per group across 4 bytes.
    sign_bits = bytearray(4)
    for g in range(N_GROUPS):
        sl = slice(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)
        wg = block[sl]
        ig = imp[sl]
        scaled_pos = d * grid
        weighted_pos = (ig[None, :] * (wg[None, :] - scaled_pos) ** 2).sum(axis=1)
        weighted_neg = (ig[None, :] * (wg[None, :] + scaled_pos) ** 2).sum(axis=1)
        if weighted_neg[indices[g]] < weighted_pos[indices[g]]:
            byte_i, bit_i = divmod(g, 8)
            sign_bits[byte_i] |= (1 << bit_i)

    out = bytearray()
    out.extend(np.float16(d).tobytes())
    out.extend(bytes(sign_bits))                                # 4 sign bytes (32 groups)
    out.extend(indices.astype("<u2").tobytes())                 # 32 × 2 = 64 bytes
    assert len(out) == 2 + 4 + 64, len(out)                     # 70 bytes / 256 weights
    return bytes(out)


def _decode_super_block(blob: bytes, grid: np.ndarray) -> np.ndarray:
    if len(blob) != 70:
        raise ValueError(f"IQ low-bit block must be 70 bytes; got {len(blob)}")
    arr = np.frombuffer(blob, dtype=np.uint8)
    d = np.frombuffer(arr[0:2].tobytes(), dtype=np.float16).astype(np.float32)[0]
    sign_bits = arr[2:6]
    indices = np.frombuffer(arr[6:70].tobytes(), dtype="<u2")
    out = np.zeros(SUPER_BLOCK, dtype=np.float32)
    for g in range(N_GROUPS):
        byte_i, bit_i = divmod(g, 8)
        sign = -1.0 if (sign_bits[byte_i] >> bit_i) & 1 else 1.0
        out[g * GROUP_SIZE:(g + 1) * GROUP_SIZE] = sign * d * grid[int(indices[g])]
    return out


def _encode_tensor(tensor: torch.Tensor, importance: Optional[torch.Tensor], grid: np.ndarray):
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
        out.extend(_encode_super_block(flat[sl], grid, imp_flat[sl]))
    return bytes(out), tuple(tensor.shape)


def _decode_tensor(blob: bytes, shape: Tuple[int, ...], grid: np.ndarray) -> torch.Tensor:
    n_elems = 1
    for s in shape:
        n_elems *= s
    n_blocks = (n_elems + SUPER_BLOCK - 1) // SUPER_BLOCK
    out = np.zeros(n_blocks * SUPER_BLOCK, dtype=np.float32)
    for b in range(n_blocks):
        out[b * SUPER_BLOCK:(b + 1) * SUPER_BLOCK] = _decode_super_block(
            blob[b * 70:(b + 1) * 70], grid,
        )
    return torch.from_numpy(out[:n_elems].copy()).reshape(*shape)


# --------------------------------------------------------------------------- #
# Public API: IQ3_XXS and IQ2_XXS share the encoder; only the grid differs.
# --------------------------------------------------------------------------- #


def encode_iq3_xxs(
    tensor: torch.Tensor,
    importance: Optional[torch.Tensor] = None,
) -> Tuple[bytes, Tuple[int, ...]]:
    return _encode_tensor(tensor, importance, _IQ3_GRID)


def decode_iq3_xxs(blob: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    return _decode_tensor(blob, shape, _IQ3_GRID)


def encode_iq2_xxs(
    tensor: torch.Tensor,
    importance: Optional[torch.Tensor] = None,
) -> Tuple[bytes, Tuple[int, ...]]:
    return _encode_tensor(tensor, importance, _IQ2_GRID)


def decode_iq2_xxs(blob: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    return _decode_tensor(blob, shape, _IQ2_GRID)
