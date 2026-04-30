"""
Caching module for inference optimization.
"""

from .prefix_cache import PrefixCache, PrefixCacheConfig

__all__ = [
    "PrefixCache",
    "PrefixCacheConfig",
]
