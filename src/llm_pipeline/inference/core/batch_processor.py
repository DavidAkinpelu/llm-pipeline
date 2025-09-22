"""Batch processing for efficient multi-request handling."""

import torch
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 8
    max_sequence_length: int = 2048
    padding_strategy: str = "longest"  # "longest", "max_length", "no_padding"
    truncation: bool = True
    dynamic_batching: bool = True
    prefetch_factor: int = 2


class BatchProcessor:
    """Batch processor for efficient multi-request handling."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self.batch_stats = {
            "total_batches": 0,
            "total_requests": 0,
            "average_batch_size": 0.0,
            "total_processing_time": 0.0
        }
        
    def process_batch(
        self,
        engine,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """Process multiple prompts in batches.
        
        Args:
            engine: Inference engine
            prompts: List of prompts to process
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        start_time = time.time()
        
        # Determine optimal batch size
        batch_size = self._calculate_optimal_batch_size(len(prompts))
        
        # Process in batches
        results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = self._process_single_batch(
                engine, batch_prompts, max_tokens, **kwargs
            )
            results.extend(batch_results)
            
        # Update statistics
        processing_time = time.time() - start_time
        self._update_stats(len(prompts), processing_time)
        
        return results
        
    def process_with_dynamic_batching(
        self,
        engine,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """Process prompts with dynamic batching.
        
        Args:
            engine: Inference engine
            prompts: List of prompts to process
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        if not self.config.dynamic_batching:
            return self.process_batch(engine, prompts, max_tokens, **kwargs)
            
        # Group prompts by length for better batching
        grouped_prompts = self._group_by_length(prompts)
        
        results = []
        for length_group, group_prompts in grouped_prompts.items():
            batch_results = self.process_batch(
                engine, group_prompts, max_tokens, **kwargs
            )
            results.extend(batch_results)
            
        return results
        
    def _process_single_batch(
        self,
        engine,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """Process a single batch of prompts.
        
        Args:
            engine: Inference engine
            prompts: List of prompts in batch
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        if len(prompts) == 1:
            # Single prompt - use regular generation
            return [engine.generate(prompts[0], max_tokens=max_tokens, **kwargs)]
            
        # Multiple prompts - use batch generation
        return self._batch_generate(engine, prompts, max_tokens, **kwargs)
        
    def _batch_generate(
        self,
        engine,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts simultaneously.
        
        Args:
            engine: Inference engine
            prompts: List of prompts
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        # For now, fall back to individual generation
        # In a real implementation, this would use vectorized batch generation
        results = []
        for prompt in prompts:
            result = engine.generate(prompt, max_tokens=max_tokens, **kwargs)
            results.append(result)
        return results
        
    def _calculate_optimal_batch_size(self, total_prompts: int) -> int:
        """Calculate optimal batch size based on available resources.
        
        Args:
            total_prompts: Total number of prompts to process
            
        Returns:
            Optimal batch size
        """
        # Simple heuristic - can be improved with actual memory monitoring
        if total_prompts <= 4:
            return total_prompts
        elif total_prompts <= 16:
            return min(4, total_prompts)
        else:
            return min(self.config.max_batch_size, total_prompts)
            
    def _group_by_length(self, prompts: List[str]) -> Dict[int, List[str]]:
        """Group prompts by length for better batching.
        
        Args:
            prompts: List of prompts
            
        Returns:
            Dictionary mapping length to prompts
        """
        groups = {}
        for prompt in prompts:
            length = len(prompt.split())
            if length not in groups:
                groups[length] = []
            groups[length].append(prompt)
        return groups
        
    def _update_stats(self, num_requests: int, processing_time: float):
        """Update batch processing statistics.
        
        Args:
            num_requests: Number of requests processed
            processing_time: Time taken for processing
        """
        self.batch_stats["total_requests"] += num_requests
        self.batch_stats["total_batches"] += 1
        self.batch_stats["total_processing_time"] += processing_time
        
        # Update average batch size
        total_batches = self.batch_stats["total_batches"]
        self.batch_stats["average_batch_size"] = (
            self.batch_stats["total_requests"] / total_batches
        )
        
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics.
        
        Returns:
            Dictionary with batch processing statistics
        """
        stats = self.batch_stats.copy()
        
        if stats["total_processing_time"] > 0:
            stats["requests_per_second"] = (
                stats["total_requests"] / stats["total_processing_time"]
            )
        else:
            stats["requests_per_second"] = 0
            
        return stats
        
    def clear_stats(self):
        """Clear batch processing statistics."""
        self.batch_stats = {
            "total_batches": 0,
            "total_requests": 0,
            "average_batch_size": 0.0,
            "total_processing_time": 0.0
        }
