"""Prefix caching system for inference optimization."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch

PastKeyValues = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class PrefixCacheConfig:
    """Configuration for prefix caching."""
    max_cache_size: int = 1000
    max_prefix_length: int = 128
    enable_cache_stats: bool = True


class PrefixCache:
    """
    Core prefix caching system that automatically caches KV states for common input prefixes.
    
    This is always enabled by default to provide transparent performance improvements
    without requiring any configuration from the user.
    """
    
    def __init__(self, config: Optional[PrefixCacheConfig] = None):
        """
        Initialize the prefix cache.
        
        Args:
            config: Optional configuration. If None, uses default settings.
        """
        self.config = config or PrefixCacheConfig()
        # Cache stores (past_key_values, prefix_length) tuples
        self.cache: Dict[str, Tuple[PastKeyValues, int]] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0

    def _truncate_cache_tensor(self, tensor: torch.Tensor, prefix_len: int) -> torch.Tensor:
        """Clone only the cached prefix portion of a KV-cache tensor."""
        if tensor.dim() >= 3 and tensor.shape[2] >= prefix_len:
            return tensor[:, :, :prefix_len, :].detach().clone()
        if tensor.dim() >= 2 and tensor.shape[1] >= prefix_len:
            return tensor[:, :prefix_len, ...].detach().clone()
        return tensor.detach().clone()
    
    def _compute_hash(self, token_ids: List[int]) -> str:
        """Compute a stable hash for a sequence of token IDs.
        Uses a simple rolling hash to avoid large allocations/str conversions.
        """
        h = 2166136261
        for t in token_ids:
            h ^= int(t) & 0xFFFFFFFF
            h *= 16777619
            h &= 0xFFFFFFFF
        return f"{h:08x}:{len(token_ids)}"
    
    def get_cached_prefix(self, input_ids: torch.Tensor) -> Tuple[Optional[PastKeyValues], int]:
        """
        Get cached KV states for input prefix.
        
        Args:
            input_ids: Input token IDs tensor
            
        Returns:
            Tuple of (cached KV states if found, prefix length)
        """
        self.total_requests += 1
        
        # Convert to list for hashing
        token_ids = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
        
        # Try different prefix lengths (longest first for best cache hit)
        for prefix_len in range(min(len(token_ids), self.config.max_prefix_length), 0, -1):
            prefix_hash = self._compute_hash(token_ids[:prefix_len])
            if prefix_hash in self.cache:
                self.cache_hits += 1
                self.access_count[prefix_hash] += 1
                cached_kv, cached_prefix_len = self.cache[prefix_hash]
                return cached_kv, cached_prefix_len
        
        self.cache_misses += 1
        return None, 0
    
    def cache_prefix(self, input_ids: torch.Tensor, past_key_values: PastKeyValues) -> None:
        """
        Cache KV states for input prefix.
        
        Args:
            input_ids: Input token IDs tensor
            past_key_values: Full layered past_key_values structure from model
        """
        token_ids = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
        prefix_len = min(len(token_ids), self.config.max_prefix_length)
        
        if prefix_len > 0 and past_key_values is not None:
            prefix_hash = self._compute_hash(token_ids[:prefix_len])
            
            # Cache only the prefix-aligned KV slice; reusing a longer cache for
            # a shorter prefix would make subsequent decode steps semantically
            # incorrect.
            cloned_kv = tuple(
                (
                    self._truncate_cache_tensor(layer_kv[0], prefix_len),
                    self._truncate_cache_tensor(layer_kv[1], prefix_len),
                )
                for layer_kv in past_key_values
            )
            
            # Store (past_key_values, prefix_length)
            self.cache[prefix_hash] = (cloned_kv, prefix_len)
            self.access_count[prefix_hash] = 1
            
            # Evict if cache is full
            if len(self.cache) > self.config.max_cache_size:
                self._evict_least_recently_used()
    
    def _evict_least_recently_used(self) -> None:
        """Evict the least recently used cache entry."""
        if not self.cache:
            return
        
        # Find the entry with the lowest access count
        lru_key = min(self.cache.keys(), key=lambda k: self.access_count[k])
        del self.cache[lru_key]
        del self.access_count[lru_key]
    
    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.access_count.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        hit_rate = self.cache_hits / max(self.total_requests, 1)
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": self.total_requests,
            "hit_rate": hit_rate,
            "max_cache_size": self.config.max_cache_size,
            "max_prefix_length": self.config.max_prefix_length,
        }
    
    def print_stats(self) -> None:
        """Print cache statistics to console."""
        stats = self.get_stats()
        print("Prefix Cache Stats:")
        print(f"  Cache Size: {stats['cache_size']}/{stats['max_cache_size']}")
        print(f"  Hit Rate: {stats['hit_rate']:.2%}")
        print(f"  Hits: {stats['cache_hits']}, Misses: {stats['cache_misses']}")
        print(f"  Total Requests: {stats['total_requests']}")
