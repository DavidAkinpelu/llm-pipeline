"""Shared outlier-aware encode/decode helpers.

The outlier trick (GPTQ/AWQ/SpQR-style) is identical across K-quant
formats: detect the top-K extreme weights per super-block, store them
separately at FP16, and fit the chosen base format on the bulk only. On
decode, dequantize the bulk and overwrite the outlier positions.

This module factors that logic out so each of ``q4_k_out``, ``q5_k_out``,
``q6_k_out``, ``q3_k_out`` is a five-line wrapper that just plugs in the
base encoder/decoder and the base block-byte size.

The encoder/decoder signature is::

    base_encoder(blocks: np.ndarray[N, 256], importance: np.ndarray[N, 256])
        -> np.ndarray[N, base_block_bytes]

    base_decoder(blob: bytes, shape: Tuple[int, ...]) -> torch.Tensor

Block layout (per super-block of 256 weights with ``K`` outliers)::

    [base_block_bytes Q*_K data]  +  K bytes positions  +  K × 2 bytes fp16 values
"""

from typing import Callable, Optional, Tuple

import numpy as np
import torch


SUPER_BLOCK = 256


def block_size_with_outliers(base_block_bytes: int, outlier_k: int) -> int:
    """Bytes per super-block when ``outlier_k`` outliers are stored alongside."""
    return base_block_bytes + 3 * outlier_k


def encode_with_outliers(
    tensor: torch.Tensor,
    importance: Optional[torch.Tensor],
    outlier_k: int,
    base_encoder_batched: Callable[[np.ndarray, np.ndarray], np.ndarray],
    base_block_bytes: int,
) -> Tuple[bytes, Tuple[int, ...]]:
    """Generic outlier-aware encoder.

    ``base_encoder_batched`` must accept ``(N, 256)`` weights + importance and
    return ``(N, base_block_bytes)`` uint8.
    """
    if outlier_k < 0 or outlier_k >= SUPER_BLOCK:
        raise ValueError(f"outlier_k must be in [0, {SUPER_BLOCK}); got {outlier_k}")

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

    sb = flat.reshape(n_blocks, SUPER_BLOCK).copy()
    imp = imp_flat.reshape(n_blocks, SUPER_BLOCK).copy()

    if outlier_k > 0:
        # Top-K outliers per super-block by absolute magnitude.
        abs_sb = np.abs(sb)
        outlier_pos = np.argpartition(abs_sb, SUPER_BLOCK - outlier_k, axis=1)[:, -outlier_k:]
        outlier_pos = outlier_pos.astype(np.int32)
        rows = np.arange(n_blocks, dtype=np.int32)[:, None]
        outlier_vals = sb[rows, outlier_pos].astype(np.float16)

        # Zero outliers in the bulk; their importance is also zeroed since
        # they'll be overwritten by the FP16 sidecar on decode.
        sb[rows, outlier_pos] = 0.0
        imp[rows, outlier_pos] = 0.0
    else:
        outlier_pos = np.zeros((n_blocks, 0), dtype=np.int32)
        outlier_vals = np.zeros((n_blocks, 0), dtype=np.float16)

    base_bytes = base_encoder_batched(sb, imp)                    # (N, base_block_bytes)
    if base_bytes.shape[1] != base_block_bytes:
        raise RuntimeError(
            f"base encoder returned {base_bytes.shape[1]} bytes/block; expected {base_block_bytes}"
        )

    # Sort outlier positions ascending so the on-disk layout is deterministic.
    sorted_idx = np.argsort(outlier_pos, axis=1)
    rows = np.arange(n_blocks, dtype=np.int32)[:, None]
    outlier_pos = outlier_pos[rows, sorted_idx]
    outlier_vals = outlier_vals[rows, sorted_idx]

    pos_bytes = outlier_pos.astype(np.uint8)                       # (N, K)
    val_bytes = np.frombuffer(outlier_vals.tobytes(), dtype=np.uint8).reshape(n_blocks, 2 * outlier_k)
    block_bytes = np.concatenate([base_bytes, pos_bytes, val_bytes], axis=1)
    expected = block_size_with_outliers(base_block_bytes, outlier_k)
    assert block_bytes.shape[1] == expected, f"got {block_bytes.shape[1]}, expected {expected}"
    return block_bytes.tobytes(), tuple(tensor.shape)


def decode_with_outliers(
    blob: bytes,
    shape: Tuple[int, ...],
    outlier_k: int,
    base_decoder: Callable[[bytes, Tuple[int, ...]], torch.Tensor],
    base_block_bytes: int,
) -> torch.Tensor:
    """Generic outlier-aware decoder. ``base_decoder`` must crop to the given shape."""
    bs = block_size_with_outliers(base_block_bytes, outlier_k)
    if len(blob) % bs != 0:
        raise ValueError(f"blob length {len(blob)} not a multiple of {bs} (block size for K={outlier_k})")
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(-1, bs)
    n_blocks = arr.shape[0]

    base_bytes = arr[:, :base_block_bytes].tobytes()
    bulk = base_decoder(base_bytes, (n_blocks * SUPER_BLOCK,)).numpy().reshape(n_blocks, SUPER_BLOCK)

    if outlier_k > 0:
        pos = arr[:, base_block_bytes : base_block_bytes + outlier_k].astype(np.int32)
        val_bytes = arr[:, base_block_bytes + outlier_k : base_block_bytes + 3 * outlier_k].tobytes()
        vals = np.frombuffer(val_bytes, dtype=np.float16).reshape(n_blocks, outlier_k).astype(np.float32)
        rows = np.arange(n_blocks, dtype=np.int32)[:, None]
        bulk[rows, pos] = vals

    n_elems = 1
    for s in shape:
        n_elems *= s
    return torch.from_numpy(bulk.reshape(-1)[:n_elems].copy()).reshape(*shape)
