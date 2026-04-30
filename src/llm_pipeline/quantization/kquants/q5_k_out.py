"""Outlier-aware Q5_K (``Q5_K_OUT``).

Same outlier-aware trick as ``Q4_K_OUT``, on top of Q5_K's 5-bit
asymmetric K-quant. Useful when you want a near-Q6 quality level at
slightly lower memory than Q5_K + an extra bulk-bit, or when the weight
distribution has clear outliers worth peeling off.

Block layout (256 weights, K=3 default outliers): 176 bytes Q5_K bulk +
3 × (1 byte position + 2 byte fp16 value) = 185 bytes.

   K=0 → 176 bytes / 256 = 5.5 bits/weight  (= vanilla Q5_K)
   K=3 → 185 bytes / 256 = 5.78 bits/weight  (default)
"""

from typing import Optional, Tuple

import torch

from ._outlier_common import (
    decode_with_outliers,
    encode_with_outliers,
)
from .q5_k import _encode_super_blocks_batched as _encode_q5_k_batched
from .q5_k import decode_q5_k


DEFAULT_OUTLIER_K = 3
_BASE_BLOCK_BYTES = 176


def encode_q5_k_out(
    tensor: torch.Tensor,
    importance: Optional[torch.Tensor] = None,
    outlier_k: int = DEFAULT_OUTLIER_K,
) -> Tuple[bytes, Tuple[int, ...]]:
    return encode_with_outliers(
        tensor, importance, outlier_k,
        base_encoder_batched=_encode_q5_k_batched,
        base_block_bytes=_BASE_BLOCK_BYTES,
    )


def decode_q5_k_out(
    blob: bytes,
    shape: Tuple[int, ...],
    outlier_k: int = DEFAULT_OUTLIER_K,
) -> torch.Tensor:
    return decode_with_outliers(
        blob, shape, outlier_k,
        base_decoder=decode_q5_k,
        base_block_bytes=_BASE_BLOCK_BYTES,
    )
