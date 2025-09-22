"""KV cache implementation for inference optimization."""

import torch
import time
from typing import Dict, Optional, Any, Tuple
from collections import OrderedDict
import hashlib


class KVCache:
    """Key-Value cache for transformer models."""
    
    def __init__(
        self, 
        max_size: int = 1000,
        cache_type: str = "lru",
        device: Optional[torch.device] = None
    ):
        """Initialize KV cache.
        
        Args:
            max_size: Maximum number of cached sequences
            cache_type: Type of cache eviction ("lru", "fifo")
            device: Device to store cache on
        """
        self.max_size = max_size
        self.cache_type = cache_type
        self.device = device or torch.device("cpu")
        
        if cache_type == "lru":
            self.cache = OrderedDict()
        else:
            self.cache = {}
            
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        
    def get(self, key: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached KV states.
        
        Args:
            key: Cache key (usually sequence hash)
            
        Returns:
            Cached KV states or None
        """
        self.stats["total_requests"] += 1
        
        if key in self.cache:
            self.stats["hits"] += 1
            
            if self.cache_type == "lru":
                # Move to end (most recently used)
                kv_states = self.cache.pop(key)
                self.cache[key] = kv_states
                
            return self.cache[key]
        else:
            self.stats["misses"] += 1
            return None
            
    def store(self, key: str, kv_states: Dict[str, torch.Tensor]):
        """Store KV states in cache.
        
        Args:
            key: Cache key
            kv_states: Key-value states to cache
        """
        if len(self.cache) >= self.max_size:
            self._evict()
            
        self.cache[key] = kv_states
        
    def clear(self):
        """Clear all cached states."""
        self.cache.clear()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        hit_rate = (
            self.stats["hits"] / self.stats["total_requests"] 
            if self.stats["total_requests"] > 0 else 0
        )
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_size": self.max_size
        }
        
    def _evict(self):
        """Evict least recently used or oldest entry."""
        self.stats["evictions"] += 1
        
        if self.cache_type == "lru":
            # Remove least recently used (first item)
            self.cache.popitem(last=False)
        else:
            # Remove arbitrary item (first key)
            key = next(iter(self.cache))
            del self.cache[key]
            
    def estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes.
        
        Returns:
            Estimated memory usage in bytes
        """
        total_bytes = 0
        for kv_states in self.cache.values():
            for tensor in kv_states.values():
                if isinstance(tensor, torch.Tensor):
                    total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes
        
    def create_cache_key(self, input_ids: torch.Tensor) -> str:
        """Create cache key from input sequence.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Cache key string
        """
        # Create hash from input sequence
        sequence_str = input_ids.tolist()
        return hashlib.md5(str(sequence_str).encode()).hexdigest()
        
    def is_cacheable(self, sequence_length: int, max_cacheable_length: int = 2048) -> bool:
        """Check if sequence is worth caching.
        
        Args:
            sequence_length: Length of input sequence
            max_cacheable_length: Maximum length to cache
            
        Returns:
            True if sequence should be cached
        """
        return sequence_length <= max_cacheable_length and sequence_length > 10
