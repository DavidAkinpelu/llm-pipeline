"""Cache manager for merged models."""

from typing import Dict, Any, Optional
import time
import hashlib


class MergeCacheManager:
    """Cache manager for merged models."""
    
    def __init__(self, max_cache_size: int = 5):
        """Initialize merge cache manager.
        
        Args:
            max_cache_size: Maximum number of cached merged models
        """
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_times = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        
    def get_merged_model(self, base_model, adapter_name: str, merge_strategy: str) -> Optional[Any]:
        """Get cached merged model.
        
        Args:
            base_model: Base model
            adapter_name: Adapter name
            merge_strategy: Merge strategy used
            
        Returns:
            Cached merged model or None
        """
        cache_key = self._generate_cache_key(base_model, adapter_name, merge_strategy)
        
        if cache_key in self.cache:
            self.cache_stats["hits"] += 1
            self.access_times[cache_key] = time.time()
            return self.cache[cache_key]
        else:
            self.cache_stats["misses"] += 1
            return None
            
    def cache_merged_model(
        self, 
        base_model, 
        adapter_name: str, 
        merge_strategy: str, 
        merged_model: Any
    ):
        """Cache merged model.
        
        Args:
            base_model: Base model
            adapter_name: Adapter name
            merge_strategy: Merge strategy used
            merged_model: Merged model to cache
        """
        cache_key = self._generate_cache_key(base_model, adapter_name, merge_strategy)
        
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()
            
        self.cache[cache_key] = merged_model
        self.access_times[cache_key] = time.time()
        
    def clear_cache(self):
        """Clear all cached merged models."""
        self.cache.clear()
        self.access_times.clear()
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size
        }
        
    def _generate_cache_key(self, base_model, adapter_name: str, merge_strategy: str) -> str:
        """Generate cache key for merged model.
        
        Args:
            base_model: Base model
            adapter_name: Adapter name
            merge_strategy: Merge strategy
            
        Returns:
            Cache key string
        """
        # Create hash from model info and adapter
        model_info = f"{type(base_model).__name__}_{id(base_model)}"
        cache_data = f"{model_info}_{adapter_name}_{merge_strategy}"
        return hashlib.md5(cache_data.encode()).hexdigest()
        
    def _evict_oldest(self):
        """Evict least recently used cached model."""
        if not self.access_times:
            return
            
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        self.cache_stats["evictions"] += 1
