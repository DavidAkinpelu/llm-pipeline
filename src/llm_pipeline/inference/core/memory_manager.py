"""Memory management for inference optimization."""

import torch
import psutil
import gc
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_gb: float
    available_memory_gb: float
    model_memory_gb: float
    cache_memory_gb: float
    activation_memory_gb: float
    peak_memory_gb: float


class MemoryManager:
    """Memory management for inference optimization."""
    
    def __init__(self, max_memory_gb: float = 8.0):
        """Initialize memory manager.
        
        Args:
            max_memory_gb: Maximum memory budget in GB
        """
        self.max_memory_gb = max_memory_gb
        self.peak_memory = 0.0
        self.memory_history = []
        
    def get_system_memory(self) -> MemoryStats:
        """Get current system memory statistics.
        
        Returns:
            Memory statistics
        """
        # Get system memory
        system_memory = psutil.virtual_memory()
        total_memory_gb = system_memory.total / (1024**3)
        available_memory_gb = system_memory.available / (1024**3)
        
        # Get GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            total_memory_gb += gpu_memory / (1024**3)
            
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            available_memory_gb -= gpu_allocated + gpu_reserved
            
        # Estimate current usage
        model_memory = self._estimate_model_memory()
        cache_memory = self._estimate_cache_memory()
        activation_memory = self._estimate_activation_memory()
        
        return MemoryStats(
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            model_memory_gb=model_memory,
            cache_memory_gb=cache_memory,
            activation_memory_gb=activation_memory,
            peak_memory_gb=self.peak_memory
        )
        
    def estimate_memory_usage(
        self,
        model,
        batch_size: int = 1,
        sequence_length: int = 512,
        include_cache: bool = True
    ) -> float:
        """Estimate memory usage for inference.
        
        Args:
            model: The model to estimate memory for
            batch_size: Batch size for inference
            sequence_length: Input sequence length
            include_cache: Whether to include KV cache memory
            
        Returns:
            Estimated memory usage in GB
        """
        # Model parameters memory
        model_memory = self._estimate_model_memory()
        
        # Activation memory (scales with batch size and sequence length)
        activation_memory = self._estimate_activation_memory(batch_size, sequence_length)
        
        # KV cache memory
        cache_memory = 0.0
        if include_cache:
            cache_memory = self._estimate_kv_cache_memory(batch_size, sequence_length)
            
        total_memory = model_memory + activation_memory + cache_memory
        
        return total_memory
        
    def optimize_memory(self, model, target_memory_gb: float) -> Dict[str, Any]:
        """Optimize model for target memory usage.
        
        Args:
            model: Model to optimize
            target_memory_gb: Target memory usage in GB
            
        Returns:
            Optimization results
        """
        optimizations = {}
        current_memory = self.estimate_memory_usage(model)
        
        if current_memory <= target_memory_gb:
            optimizations["status"] = "no_optimization_needed"
            optimizations["memory_saved"] = 0.0
            return optimizations
            
        # Apply optimizations
        memory_saved = 0.0
        
        # 1. Clear cache
        if hasattr(model, 'clear_cache'):
            model.clear_cache()
            memory_saved += 0.1  # Estimate
            
        # 2. Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            optimizations["gradient_checkpointing"] = True
            memory_saved += 0.5  # Estimate
            
        # 3. Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            memory_saved += 0.2  # Estimate
            
        optimizations["status"] = "optimized"
        optimizations["memory_saved"] = memory_saved
        optimizations["final_memory"] = current_memory - memory_saved
        
        return optimizations
        
    def monitor_memory_usage(self, func, *args, **kwargs):
        """Monitor memory usage during function execution.
        
        Args:
            func: Function to monitor
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (function result, memory stats)
        """
        # Record initial memory
        initial_stats = self.get_system_memory()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Record final memory
        final_stats = self.get_system_memory()
        
        # Update peak memory
        memory_used = final_stats.model_memory_gb - initial_stats.model_memory_gb
        self.peak_memory = max(self.peak_memory, memory_used)
        
        # Record in history
        self.memory_history.append({
            "initial": initial_stats,
            "final": final_stats,
            "delta": memory_used
        })
        
        return result, final_stats
        
    def get_memory_recommendations(self, model) -> List[str]:
        """Get memory optimization recommendations.
        
        Args:
            model: Model to analyze
            
        Returns:
            List of recommendations
        """
        recommendations = []
        current_memory = self.estimate_memory_usage(model)
        available_memory = self.get_system_memory().available_memory_gb
        
        if current_memory > available_memory:
            recommendations.append("Reduce batch size")
            recommendations.append("Enable gradient checkpointing")
            recommendations.append("Use mixed precision (FP16)")
            
        if current_memory > self.max_memory_gb:
            recommendations.append("Consider model quantization")
            recommendations.append("Reduce sequence length")
            recommendations.append("Clear KV cache more frequently")
            
        if not recommendations:
            recommendations.append("Memory usage is optimal")
            
        return recommendations
        
    def _estimate_model_memory(self) -> float:
        """Estimate model parameter memory usage."""
        # This would be implemented based on actual model analysis
        # For now, return a placeholder
        return 2.0  # GB
        
    def _estimate_cache_memory(self) -> float:
        """Estimate KV cache memory usage."""
        # This would be implemented based on cache analysis
        return 0.5  # GB
        
    def _estimate_activation_memory(self, batch_size: int = 1, sequence_length: int = 512) -> float:
        """Estimate activation memory usage."""
        # Rough estimation: batch_size * sequence_length * hidden_size * 4 bytes
        hidden_size = 4096  # Typical hidden size
        activation_bytes = batch_size * sequence_length * hidden_size * 4
        return activation_bytes / (1024**3)  # Convert to GB
        
    def _estimate_kv_cache_memory(self, batch_size: int, sequence_length: int) -> float:
        """Estimate KV cache memory usage."""
        # KV cache stores key and value tensors
        # Rough estimation: batch_size * sequence_length * hidden_size * 2 * 4 bytes
        hidden_size = 4096
        kv_cache_bytes = batch_size * sequence_length * hidden_size * 2 * 4
        return kv_cache_bytes / (1024**3)  # Convert to GB
        
    def get_current_usage(self) -> float:
        """Get current memory usage."""
        stats = self.get_system_memory()
        return stats.model_memory_gb + stats.cache_memory_gb + stats.activation_memory_gb
