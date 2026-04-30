"""KV Cache for efficient inference."""

import torch
from typing import Dict, Any, Optional, Tuple


class KVCache:
    """Key-Value cache for transformer inference."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        
    def get(self, cache_key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if cache_key in self.cache:
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            return self.cache[cache_key]
        return None
        
    def store(self, cache_key: str, key_states: torch.Tensor, value_states: torch.Tensor):
        if len(self.cache) >= self.max_size and cache_key not in self.cache:
            self._evict_oldest()
            
        self.cache[cache_key] = (key_states.clone(), value_states.clone())
        
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)
        
    def _evict_oldest(self):
        if self.access_order:
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
                
    def clear(self):
        self.cache.clear()
        self.access_order.clear()
        
    def size(self) -> int:
        return len(self.cache)
        
    def get_stats(self) -> Dict[str, Any]:
        total_memory = 0
        for key_states, value_states in self.cache.values():
            total_memory += key_states.numel() * key_states.element_size()
            total_memory += value_states.numel() * value_states.element_size()
            
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "memory_bytes": total_memory,
            "memory_mb": total_memory / (1024 * 1024)
        }
