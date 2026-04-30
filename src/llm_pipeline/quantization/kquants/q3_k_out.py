"""Outlier-aware Q3_K (``Q3_K_OUT``).

Same outlier-aware trick as ``Q4_K_OUT``, applied on top of Q3_K's
3-bit signed K-quant. Q3_K has by far the highest base error (~17%
rel-err on Gaussian) so the outlier sidecar buys a *lot* on heavy-tailed
weights — outliers are exactly the values that would otherwise blow up
the per-sub-block scale.

Block layout (256 weights, K=3 default outliers): 110 bytes Q3_K bulk +
3 × (1 byte position + 2 byte fp16 value) = 119 bytes.

   K=0 → 110 bytes / 256 = 3.4375 bits/weight (= vanilla Q3_K)
   K=3 → 119 bytes / 256 = 3.72 bits/weight   (default)
"""

from typing import Optional, Tuple

import torch

from ._outlier_common import (
    decode_with_outliers,
    encode_with_outliers,
)
from .q3_k import _encode_super_blocks_batched as _encode_q3_k_batched
from .q3_k import decode_q3_k


DEFAULT_OUTLIER_K = 3
_BASE_BLOCK_BYTES = 110


def encode_q3_k_out(
    tensor: torch.Tensor,
    importance: Optional[torch.Tensor] = None,
    outlier_k: int = DEFAULT_OUTLIER_K,
) -> Tuple[bytes, Tuple[int, ...]]:
    return encode_with_outliers(
        tensor, importance, outlier_k,
        base_encoder_batched=_encode_q3_k_batched,
        base_block_bytes=_BASE_BLOCK_BYTES,
    )


def decode_q3_k_out(
    blob: bytes,
    shape: Tuple[int, ...],
    outlier_k: int = DEFAULT_OUTLIER_K,
) -> torch.Tensor:
    return decode_with_outliers(
        blob, shape, outlier_k,
        base_decoder=decode_q3_k,
        base_block_bytes=_BASE_BLOCK_BYTES,
    )
