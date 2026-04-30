"""
Configuration for attention mechanisms.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    use_flash_attention: bool = True
    attention_dropout: float = 0.0
    scale_attention_softmax_in_fp32: bool = True
    attention_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2"
