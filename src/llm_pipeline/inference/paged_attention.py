"""Paged Attention implementation for efficient memory management in LLM inference."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import threading
import time


@dataclass
class PagedAttentionConfig:
    """Configuration for paged attention system.
    
    Args:
        block_size: Tokens per block
        num_blocks: Total number of blocks
        max_sequence_length: Maximum sequence length
        memory_fraction: Fraction of GPU memory to use
        enable_kv_cache: Enable KV cache optimization
        cache_dtype: KV cache data type
    """
    block_size: int = 16
    num_blocks: int = 1024
    max_sequence_length: int = 2048
    memory_fraction: float = 0.8
    enable_kv_cache: bool = True
    cache_dtype: torch.dtype = torch.float16


class MemoryBlock:
    """Represents a memory block for paged attention."""
    
    def __init__(self, block_id: int, block_size: int, device: torch.device, 
                 hidden_size: int, num_heads: int, head_dim: int, cache_dtype: torch.dtype):
        self.block_id = block_id
        self.block_size = block_size
        self.device = device
        self.is_allocated = False
        self.sequence_id = None
        self.offset = 0
        self.last_accessed = time.time()
        
        self.k_cache = torch.zeros(
            block_size, num_heads, head_dim,
            dtype=cache_dtype, device=device
        )
        self.v_cache = torch.zeros(
            block_size, num_heads, head_dim,
            dtype=cache_dtype, device=device
        )
        
        self.num_tokens = 0
        self.is_dirty = False
    
    def clear(self):
        """Clear the block and reset metadata."""
        self.is_allocated = False
        self.sequence_id = None
        self.offset = 0
        self.num_tokens = 0
        self.is_dirty = False
        self.last_accessed = time.time()
        
        self.k_cache.zero_()
        self.v_cache.zero_()
    
    def is_empty(self) -> bool:
        """Check if the block is empty."""
        return not self.is_allocated and self.num_tokens == 0


class PagedAttentionManager:
    """Manages memory blocks and paged attention operations."""
    
    def __init__(self, config: PagedAttentionConfig, model_config: Any, device: torch.device):
        self.config = config
        self.device = device
        self.model_config = model_config
        
        self.hidden_size = getattr(model_config, 'hidden_size', 4096)
        self.num_heads = getattr(model_config, 'num_attention_heads', 32)
        self.head_dim = self.hidden_size // self.num_heads
        
        self.blocks = [
            MemoryBlock(i, config.block_size, device, 
                       self.hidden_size, self.num_heads, self.head_dim, config.cache_dtype)
            for i in range(config.num_blocks)
        ]
        self.free_blocks = list(range(config.num_blocks))
        self.allocated_blocks = {}
        
        self.sequences = {}
        
        self.lock = threading.Lock()
        
        self.stats = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'memory_usage': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def allocate_sequence(self, sequence_id: str, sequence_length: int) -> List[int]:
        """Allocate blocks for a sequence."""
        with self.lock:
            num_blocks_needed = (sequence_length + self.config.block_size - 1) // self.config.block_size
            
            if len(self.free_blocks) < num_blocks_needed:
                self._evict_blocks(num_blocks_needed)
                
                if len(self.free_blocks) < num_blocks_needed:
                    raise RuntimeError(
                        f"Not enough free blocks. Need {num_blocks_needed}, "
                        f"have {len(self.free_blocks)} after eviction"
                    )
            
            allocated_block_ids = self.free_blocks[:num_blocks_needed]
            self.free_blocks = self.free_blocks[num_blocks_needed:]
            
            for block_id in allocated_block_ids:
                self.blocks[block_id].is_allocated = True
                self.blocks[block_id].sequence_id = sequence_id
                self.blocks[block_id].last_accessed = time.time()
            
            self.sequences[sequence_id] = {
                'block_ids': allocated_block_ids,
                'length': sequence_length,
                'allocated_blocks': num_blocks_needed,
                'created_at': time.time(),
                'last_accessed': time.time()
            }
            
            self.stats['total_allocations'] += 1
            self.stats['memory_usage'] += num_blocks_needed
            
            return allocated_block_ids
    
    def deallocate_sequence(self, sequence_id: str):
        """Deallocate blocks for a sequence."""
        with self.lock:
            if sequence_id not in self.sequences:
                return
            
            block_ids = self.sequences[sequence_id]['block_ids']
            
            for block_id in block_ids:
                self.blocks[block_id].clear()
            
            self.free_blocks.extend(block_ids)
            
            self.stats['total_deallocations'] += 1
            self.stats['memory_usage'] -= len(block_ids)
            
            del self.sequences[sequence_id]
    
    def get_block_mapping(self, sequence_id: str) -> Dict[str, torch.Tensor]:
        """Get block mapping for a sequence."""
        with self.lock:
            if sequence_id not in self.sequences:
                raise ValueError(f"Sequence {sequence_id} not found")
            
            sequence_info = self.sequences[sequence_id]
            block_ids = sequence_info['block_ids']
            sequence_length = sequence_info['length']
            
            sequence_info['last_accessed'] = time.time()
            for block_id in block_ids:
                self.blocks[block_id].last_accessed = time.time()
            
            block_tables = []
            slot_mappings = []
            
            for i, block_id in enumerate(block_ids):
                block_tables.append(block_id)
                
                for j in range(self.config.block_size):
                    slot_mappings.append(i * self.config.block_size + j)
            
            slot_mappings = slot_mappings[:sequence_length]
            
            return {
                'block_tables': torch.tensor([block_tables], device=self.device),
                'slot_mappings': torch.tensor([slot_mappings], device=self.device),
                'context_lens': torch.tensor([sequence_length], device=self.device)
            }
    
    def update_kv_cache(self, sequence_id: str, layer_idx: int, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """Update KV cache for a sequence.
        
        Args:
            sequence_id: Sequence identifier
            layer_idx: Layer index
            k_cache: Key cache tensor [B, H, T, D]
            v_cache: Value cache tensor [B, H, T, D]
        """
        with self.lock:
            if sequence_id not in self.sequences:
                return
            
            if k_cache.dim() == 4:
                k_cache = k_cache[0]  # [H, T, D]
                v_cache = v_cache[0]  # [H, T, D]
            
            k_cache = k_cache.transpose(0, 1)  # [T, H, D]
            v_cache = v_cache.transpose(0, 1)  # [T, H, D]
            
            block_ids = self.sequences[sequence_id]['block_ids']
            sequence_length = min(k_cache.shape[0], self.sequences[sequence_id]['length'])
            
            start_idx = 0
            for block_id in block_ids:
                block = self.blocks[block_id]
                end_idx = min(start_idx + self.config.block_size, sequence_length)
                
                if end_idx > start_idx:
                    tokens_in_block = end_idx - start_idx
                    
                    block.k_cache[:tokens_in_block].copy_(k_cache[start_idx:end_idx])
                    block.v_cache[:tokens_in_block].copy_(v_cache[start_idx:end_idx])
                    
                    block.num_tokens = tokens_in_block
                    block.is_dirty = True
                
                start_idx = end_idx
    
    def get_kv_cache(self, sequence_id: str, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get KV cache for a sequence.
        
        Returns:
            Tuple of (k_cache, v_cache) in [B, H, T, D] format, or (None, None) if not found
        """
        with self.lock:
            if sequence_id not in self.sequences:
                raise ValueError(f"Sequence {sequence_id} not found")
            
            block_ids = self.sequences[sequence_id]['block_ids']
            sequence_length = self.sequences[sequence_id]['length']
            
            k_caches = []
            v_caches = []
            
            for block_id in block_ids:
                block = self.blocks[block_id]
                if block.num_tokens > 0:
                    k_caches.append(block.k_cache[:block.num_tokens])
                    v_caches.append(block.v_cache[:block.num_tokens])
            
            if k_caches:
                k_cache = torch.cat(k_caches, dim=0)
                v_cache = torch.cat(v_caches, dim=0)
                
                k_cache = k_cache.transpose(0, 1).unsqueeze(0)  # [1, H, T, D]
                v_cache = v_cache.transpose(0, 1).unsqueeze(0)  # [1, H, T, D]
                
                self.stats['cache_hits'] += 1
                
                return k_cache, v_cache
            else:
                self.stats['cache_misses'] += 1
                return None, None
    
    def _evict_blocks(self, num_blocks_needed: int):
        """Evict least recently used blocks to free memory."""
        if len(self.free_blocks) >= num_blocks_needed:
            return
        
        # Sort sequences by last accessed time (oldest first)
        sorted_sequences = sorted(
            self.sequences.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        blocks_to_evict = []
        for sequence_id, sequence_info in sorted_sequences:
            blocks_to_evict.extend(sequence_info['block_ids'])
            if len(blocks_to_evict) >= num_blocks_needed:
                break
        
        # Evict the selected sequences
        for sequence_id, sequence_info in sorted_sequences:
            if sequence_info['block_ids'][0] in blocks_to_evict:
                self.deallocate_sequence(sequence_id)
                if len(self.free_blocks) >= num_blocks_needed:
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get paged attention statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats.update({
                'free_blocks': len(self.free_blocks),
                'allocated_blocks': self.config.num_blocks - len(self.free_blocks),
                'active_sequences': len(self.sequences),
                'memory_efficiency': (len(self.free_blocks) / self.config.num_blocks) * 100,
                'cache_hit_rate': (stats['cache_hits'] / max(stats['cache_hits'] + stats['cache_misses'], 1)) * 100
            })
            return stats
    
    def cleanup(self):
        """Clean up all allocated sequences and free memory."""
        with self.lock:
            # Deallocate all sequences
            sequence_ids = list(self.sequences.keys())
            for sequence_id in sequence_ids:
                self.deallocate_sequence(sequence_id)
            
            # Clear all blocks
            for block in self.blocks:
                block.clear()
            
            # Reset free blocks list
            self.free_blocks = list(range(self.config.num_blocks))
            
            # Clear statistics
            self.stats = {
                'total_allocations': 0,
                'total_deallocations': 0,
                'memory_usage': 0,
                'cache_hits': 0,
                'cache_misses': 0
            }

