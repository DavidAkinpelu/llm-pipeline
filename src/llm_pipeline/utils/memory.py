"""Memory optimization utilities."""

import torch
import psutil
import gc
from typing import Dict, Tuple, Optional, Any
import numpy as np
from .optimizer_memory import (
    OptimizerMemoryCalculator, 
    get_memory_footprint_with_optimizer,
    estimate_training_memory_with_optimizer
)


def get_memory_footprint(
    model: torch.nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None,
    precision: str = "fp32"
) -> Dict[str, Any]:
    """Get detailed memory footprint information with optional optimizer.
    
    Args:
        model: PyTorch model
        optimizer: Optional optimizer instance for accurate memory estimation
        precision: Model precision ("fp32", "fp16", "bf16")
        
    Returns:
        Dictionary with memory breakdown
    """
    return get_memory_footprint_with_optimizer(model, optimizer, precision)


def estimate_training_memory(
    model: torch.nn.Module,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: Optional[int] = None,
    optimizer_type: str = "adam",
    precision: str = "fp32"
) -> Dict[str, float]:
    """Estimate memory usage during training.
    
    Args:
        model: PyTorch model
        batch_size: Training batch size
        sequence_length: Input sequence length
        hidden_size: Model hidden size
        num_layers: Number of layers (optional)
        optimizer_type: Type of optimizer ("adam", "sgd", "rmsprop", etc.)
        precision: Model precision ("fp32", "fp16", "bf16")
        
    Returns:
        Dictionary with memory estimates
    """
    
    # Model memory with specific optimizer type
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Use the new optimizer-aware estimation
    training_memory = estimate_training_memory_with_optimizer(
        trainable_params, optimizer_type, batch_size, sequence_length, precision
    )
    
    # Add model-specific calculations
    model_footprint = get_memory_footprint(model, precision=precision)
    model_memory_mb = model_footprint["estimated_total_mb"]
    
    # Activation memory (rough estimate)
    # Each layer stores activations for backprop
    if num_layers is None:
        # Try to estimate from model
        num_layers = len([n for n, _ in model.named_modules() if 'layer' in n.lower()])
        if num_layers == 0:
            num_layers = 12  # Default estimate
    
    # Activation memory per token per layer
    activation_memory_per_token = hidden_size * 4  # bytes
    total_activations = batch_size * sequence_length * num_layers * activation_memory_per_token
    activation_memory_mb = total_activations / (1024 * 1024)
    
    # Gradient accumulation memory
    grad_accum_memory_mb = model_footprint["gradient_memory_mb"]
    
    # Peak memory (during backward pass)
    peak_memory_mb = model_memory_mb + activation_memory_mb + grad_accum_memory_mb
    
    return {
        "model_memory_mb": model_memory_mb,
        "activation_memory_mb": activation_memory_mb,
        "gradient_memory_mb": grad_accum_memory_mb,
        "optimizer_memory_mb": training_memory["optimizer_memory_mb"],
        "peak_memory_mb": peak_memory_mb,
        "peak_memory_gb": peak_memory_mb / 1024,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "memory_per_sample_mb": peak_memory_mb / batch_size,
        "optimizer_type": optimizer_type,
        "precision": precision,
    }


def optimize_batch_size(
    model: torch.nn.Module,
    max_memory_gb: float,
    sequence_length: int,
    hidden_size: int,
    safety_margin: float = 0.8
) -> Tuple[int, Dict[str, Any]]:
    """Find optimal batch size given memory constraints"""
    
    available_memory_mb = max_memory_gb * 1024 * safety_margin
    
    # Binary search for optimal batch size
    min_batch = 1
    max_batch = 512
    optimal_batch = 1
    
    while min_batch <= max_batch:
        test_batch = (min_batch + max_batch) // 2
        
        memory_estimate = estimate_training_memory(
            model, test_batch, sequence_length, hidden_size
        )
        
        if memory_estimate["peak_memory_mb"] <= available_memory_mb:
            optimal_batch = test_batch
            min_batch = test_batch + 1
        else:
            max_batch = test_batch - 1
    
    # Get final memory estimate
    final_estimate = estimate_training_memory(
        model, optimal_batch, sequence_length, hidden_size
    )
    
    return optimal_batch, final_estimate


def estimate_memory_usage(
    num_parameters: int,
    batch_size: int = 1,
    sequence_length: int = 512,
    precision: str = "fp32",
    optimizer_type: str = "adam"
) -> Dict[str, float]:
    """Estimate memory usage for given parameters with specific optimizer.
    
    Args:
        num_parameters: Number of parameters
        batch_size: Batch size
        sequence_length: Sequence length
        precision: Model precision
        optimizer_type: Type of optimizer
        
    Returns:
        Dictionary with memory estimates
    """
    
    return estimate_training_memory_with_optimizer(
        num_parameters, optimizer_type, batch_size, sequence_length, precision
    )


def get_system_memory_info() -> Dict[str, float]:
    """Get current system memory information"""
    
    # System memory
    system_memory = psutil.virtual_memory()
    
    # GPU memory if available
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory
            gpu_allocated = torch.cuda.memory_allocated(i)
            gpu_reserved = torch.cuda.memory_reserved(i)
            
            gpu_info[f"gpu_{i}"] = {
                "total_gb": gpu_memory / (1024**3),
                "allocated_gb": gpu_allocated / (1024**3),
                "reserved_gb": gpu_reserved / (1024**3),
                "free_gb": (gpu_memory - gpu_reserved) / (1024**3)
            }
    
    return {
        "system_memory_gb": system_memory.total / (1024**3),
        "available_memory_gb": system_memory.available / (1024**3),
        "used_memory_gb": system_memory.used / (1024**3),
        "memory_usage_percent": system_memory.percent,
        "gpu_info": gpu_info
    }


def cleanup_memory():
    """Clean up memory and run garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
