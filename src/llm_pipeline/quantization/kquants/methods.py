"""Quantization-method enum and shared block-shape metadata."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QuantMethod(str, Enum):
    """All quantization formats this module knows about.

    The string value matches the canonical GGUF type name so logs are
    grep-friendly. Variants like ``Q4_K_S`` / ``Q4_K_M`` differ only in the
    *per-tensor* policy (which tensors get bumped to Q6_K) — not in the
    block-level encoding. The ``Quantizer`` facade implements the policy.
    """

    # Legacy
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q8_0 = "Q8_0"

    # K-quants (per-block format)
    Q3_K = "Q3_K"
    Q3_K_OUT = "Q3_K_OUT"
    Q4_K = "Q4_K"
    Q4_K_OUT = "Q4_K_OUT"
    Q5_K = "Q5_K"
    Q5_K_OUT = "Q5_K_OUT"
    Q6_K = "Q6_K"
    Q6_K_OUT = "Q6_K_OUT"
    Q8_K = "Q8_K"

    # K-quant variants (per-tensor policy on top of Q4_K)
    Q4_K_S = "Q4_K_S"
    Q4_K_M = "Q4_K_M"

    # I-quants
    IQ4_NL = "IQ4_NL"


@dataclass(frozen=True)
class BlockShape:
    """How the format groups weights.

    - ``super_block_size``: how many weights one super-block covers (32 for
      legacy formats, 256 for K-quants).
    - ``sub_block_size``: for K-quants, the size of each sub-block scale
      group (32 for Q4_K, 16 for Q6_K). For non-K-quants this equals
      ``super_block_size``.
    - ``bits_per_weight``: the *effective* bits per weight including scales
      etc. — useful for size estimation.
    """

    super_block_size: int
    sub_block_size: int
    bits_per_weight: float


_SHAPES = {
    QuantMethod.Q4_0: BlockShape(32, 32, 4.5),
    QuantMethod.Q4_1: BlockShape(32, 32, 5.0),
    QuantMethod.Q5_0: BlockShape(32, 32, 5.5),
    QuantMethod.Q5_1: BlockShape(32, 32, 6.0),
    QuantMethod.Q8_0: BlockShape(32, 32, 8.5),
    QuantMethod.Q3_K: BlockShape(256, 16, 3.4375),
    QuantMethod.Q3_K_OUT: BlockShape(256, 16, 3.71875),    # 119 / 256 bits/w with K=3
    QuantMethod.Q4_K: BlockShape(256, 32, 4.5),
    QuantMethod.Q4_K_OUT: BlockShape(256, 32, 4.78125),    # 153 / 256
    QuantMethod.Q5_K: BlockShape(256, 32, 5.5),
    QuantMethod.Q5_K_OUT: BlockShape(256, 32, 5.78125),    # 185 / 256
    QuantMethod.Q6_K: BlockShape(256, 16, 6.5625),
    QuantMethod.Q6_K_OUT: BlockShape(256, 16, 6.84375),    # 219 / 256
    QuantMethod.Q8_K: BlockShape(256, 16, 8.5),
    QuantMethod.Q4_K_S: BlockShape(256, 32, 4.5),
    QuantMethod.Q4_K_M: BlockShape(256, 32, 4.85),  # mixed Q4_K + Q6_K
    QuantMethod.IQ4_NL: BlockShape(32, 32, 4.5),
}


def get_block_shape(method: QuantMethod) -> BlockShape:
    return _SHAPES[method]
