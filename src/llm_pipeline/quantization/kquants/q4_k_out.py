"""Outlier-aware Q4_K (``Q4_K_OUT``).

See ``_outlier_common`` for the shared algorithm. This module is a thin
wrapper that plugs Q4_K's encoder/decoder into the outlier-aware path.

Block layout (256 weights, K=3 default outliers): 144 bytes Q4_K bulk +
3 × (1 byte position + 2 byte fp16 value) = 153 bytes.

   K=0 → 144 bytes / 256 = 4.5 bits/weight (= vanilla Q4_K)
   K=3 → 153 bytes / 256 = 4.78 bits/weight   (default)
   K=8 → 168 bytes / 256 = 5.25 bits/weight
"""

from typing import Optional, Tuple

import torch

from ._outlier_common import (
    decode_with_outliers,
    encode_with_outliers,
)
from .q4_k import _encode_super_blocks_batched as _encode_q4_k_batched
from .q4_k import decode_q4_k


DEFAULT_OUTLIER_K = 3
_BASE_BLOCK_BYTES = 144


def encode_q4_k_out(
    tensor: torch.Tensor,
    importance: Optional[torch.Tensor] = None,
    outlier_k: int = DEFAULT_OUTLIER_K,
) -> Tuple[bytes, Tuple[int, ...]]:
    return encode_with_outliers(
        tensor, importance, outlier_k,
        base_encoder_batched=_encode_q4_k_batched,
        base_block_bytes=_BASE_BLOCK_BYTES,
    )


def decode_q4_k_out(
    blob: bytes,
    shape: Tuple[int, ...],
    outlier_k: int = DEFAULT_OUTLIER_K,
) -> torch.Tensor:
    return decode_with_outliers(
        blob, shape, outlier_k,
        base_decoder=decode_q4_k,
        base_block_bytes=_BASE_BLOCK_BYTES,
    )
