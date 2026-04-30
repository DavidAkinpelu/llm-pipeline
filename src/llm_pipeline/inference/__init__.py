"""Qwen3-focused LLM inference framework with all advanced features."""

from .qwen3_engine import Qwen3InferenceEngine, Qwen3InferenceConfig
from .sampling import Sampler, SamplingConfig
from .kv_cache import KVCache
from .streaming import StreamingGenerator
from .batching import BatchProcessor, BatchConfig
from .continuous_batching import ContinuousBatcher, ContinuousBatchConfig
from .paged_attention import PagedAttentionManager, PagedAttentionConfig
from .caching import PrefixCache, PrefixCacheConfig
from .attention import FlashAttentionLayer, StandardAttentionLayer, get_attention_layer, AttentionConfig
from .speculative import (
    SpecDecStats,
    mtp_speculative_decode,
    speculative_decode,
)
from .speculative_kernel import verify_drafts_batched
from .advanced_sampling import (
    BeamHypothesis,
    beam_search_decode,
    contrastive_search_decode,
)
from .constraints import (
    JSONSchemaConstraint,
    LogitConstraint,
    PrefixConstraint,
    RegexConstraint,
)
from ..parallelism import Qwen3TensorParallelism, TensorParallelConfig, ParallelismFactory

__all__ = [
    # Core Qwen3 inference (primary interface)
    "Qwen3InferenceEngine",
    "Qwen3InferenceConfig",
    
    # Centralized Sampling
    "Sampler",
    "SamplingConfig",
    
    # KV Cache
    "KVCache",
    
    # Streaming
    "StreamingGenerator",
    
    # Batching
    "BatchProcessor",
    "BatchConfig",
    
    # Continuous Batching
    "ContinuousBatcher",
    "ContinuousBatchConfig",
    
    # Paged Attention
    "PagedAttentionManager",
    "PagedAttentionConfig",
    
    # Core prefix caching (always enabled)
    "PrefixCache",
    "PrefixCacheConfig",
    
    # Flash Attention (auto-detected)
    "FlashAttentionLayer",
    "StandardAttentionLayer",
    "get_attention_layer",
    "AttentionConfig",
    
    # Tensor Parallelism (optional)
    "Qwen3TensorParallelism",
    "TensorParallelConfig",
    "ParallelismFactory",

    # Speculative decoding
    "SpecDecStats",
    "mtp_speculative_decode",
    "speculative_decode",

    # Advanced sampling
    "BeamHypothesis",
    "beam_search_decode",
    "contrastive_search_decode",

    # Constrained generation
    "JSONSchemaConstraint",
    "LogitConstraint",
    "PrefixConstraint",
    "RegexConstraint",
]