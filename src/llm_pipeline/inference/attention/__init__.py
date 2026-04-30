"""
Attention mechanisms for inference optimization.
"""

from .flash_attention import FlashAttentionLayer, StandardAttentionLayer, get_attention_layer
from .config import AttentionConfig
from .flash_attention_3 import flash_attn_3_func
from .sliding_window import sliding_window_attention, sliding_window_mask
from .sparse import attention_sink_mask, dilated_mask, sparse_attention

__all__ = [
    "FlashAttentionLayer",
    "StandardAttentionLayer",
    "get_attention_layer",
    "AttentionConfig",
    "flash_attn_3_func",
    "sliding_window_attention",
    "sliding_window_mask",
    "attention_sink_mask",
    "dilated_mask",
    "sparse_attention",
]
