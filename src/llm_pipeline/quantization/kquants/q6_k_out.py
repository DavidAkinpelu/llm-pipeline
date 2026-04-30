"""Outlier-aware Q6_K (``Q6_K_OUT``).

Same outlier-aware trick as ``Q4_K_OUT``, on top of Q6_K's 6-bit
symmetric K-quant. Pretty close to lossless on the bulk; the outlier
sidecar is mainly insurance against pathological tails.

Block layout (256 weights, K=3 default outliers): 210 bytes Q6_K bulk +
3 × (1 byte position + 2 byte fp16 value) = 219 bytes.

   K=0 → 210 bytes / 256 = 6.5625 bits/weight (= vanilla Q6_K)
   K=3 → 219 bytes / 256 = 6.84 bits/weight   (default)
"""

from typing import Optional, Tuple

import torch

from ._outlier_common import (
    decode_with_outliers,
    encode_with_outliers,
)
from .q6_k import _encode_super_blocks_batched as _encode_q6_k_batched
from .q6_k import decode_q6_k


DEFAULT_OUTLIER_K = 3
_BASE_BLOCK_BYTES = 210


def encode_q6_k_out(
    tensor: torch.Tensor,
    importance: Optional[torch.Tensor] = None,
    outlier_k: int = DEFAULT_OUTLIER_K,
) -> Tuple[bytes, Tuple[int, ...]]:
    return encode_with_outliers(
        tensor, importance, outlier_k,
        base_encoder_batched=_encode_q6_k_batched,
        base_block_bytes=_BASE_BLOCK_BYTES,
    )


def decode_q6_k_out(
    blob: bytes,
    shape: Tuple[int, ...],
    outlier_k: int = DEFAULT_OUTLIER_K,
) -> torch.Tensor:
    return decode_with_outliers(
        blob, shape, outlier_k,
        base_decoder=decode_q6_k,
        base_block_bytes=_BASE_BLOCK_BYTES,
    )
