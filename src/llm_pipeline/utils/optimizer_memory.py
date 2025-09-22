"""Optimizer memory estimation utilities."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union


class OptimizerMemoryCalculator:
    """Calculate memory usage for different optimizer types."""
    
    @staticmethod
    def calculate_optimizer_memory(optimizer: torch.optim.Optimizer) -> int:
        """Calculate actual memory usage of the optimizer.
        
        Args:
            optimizer: PyTorch optimizer instance
            
        Returns:
            Memory usage in bytes
        """
        total_memory = 0
        
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    # Base parameter size
                    param_size = param.numel() * param.element_size()
                    
                    # Optimizer state memory
                    if param in optimizer.state:
                        state = optimizer.state[param]
                        for state_tensor in state.values():
                            if isinstance(state_tensor, torch.Tensor):
                                total_memory += state_tensor.numel() * state_tensor.element_size()
                    
        return total_memory
    
    @staticmethod
    def estimate_optimizer_memory_by_type(
        trainable_params: int, 
        optimizer_type: str, 
        precision: str = "fp32"
    ) -> int:
        """Estimate memory based on optimizer type.
        
        Args:
            trainable_params: Number of trainable parameters
            optimizer_type: Type of optimizer ("adam", "sgd", "rmsprop", etc.)
            precision: Precision type ("fp32", "fp16", "bf16")
            
        Returns:
            Estimated memory usage in bytes
        """
        bytes_per_param = 4 if precision == "fp32" else 2
        
        optimizer_memory_map = {
            "adam": trainable_params * bytes_per_param * 2,  # momentum + velocity
            "adamw": trainable_params * bytes_per_param * 2,  # momentum + velocity
            "sgd": trainable_params * bytes_per_param * 0,    # no additional states
            "rmsprop": trainable_params * bytes_per_param * 1,  # square_avg
            "adagrad": trainable_params * bytes_per_param * 1,  # sum
            "adamax": trainable_params * bytes_per_param * 2,   # exp_avg + exp_inf
            "asgd": trainable_params * bytes_per_param * 1,     # ax
            "rprop": trainable_params * bytes_per_param * 2,    # prev_grad + step_size
            "lbfgs": trainable_params * bytes_per_param * 5,    # complex state
            "adafactor": trainable_params * bytes_per_param * 1,  # approximated second moments
            "adahessian": trainable_params * bytes_per_param * 3,  # hessian diagonal + momentum
            "lion": trainable_params * bytes_per_param * 1,       # momentum
            "eve": trainable_params * bytes_per_param * 3,        # momentum + variance + c
        }
        
        return optimizer_memory_map.get(optimizer_type.lower(), 
                                      trainable_params * bytes_per_param * 2)  # Default to Adam
    
    @staticmethod
    def get_optimizer_type_from_instance(optimizer: torch.optim.Optimizer) -> str:
        """Get optimizer type string from optimizer instance.
        
        Args:
            optimizer: PyTorch optimizer instance
            
        Returns:
            Optimizer type string
        """
        optimizer_class = type(optimizer).__name__.lower()
        
        # Map common optimizer classes to standard names
        optimizer_mapping = {
            "adam": "adam",
            "adamw": "adamw", 
            "sgd": "sgd",
            "rmsprop": "rmsprop",
            "adagrad": "adagrad",
            "adamax": "adamax",
            "asgd": "asgd",
            "rprop": "rprop",
            "lbfgs": "lbfgs",
            "adafactor": "adafactor",
            "adahessian": "adahessian",
            "lion": "lion",
            "eve": "eve",
        }
        
        return optimizer_mapping.get(optimizer_class, "adam")  # Default to adam
    
    @staticmethod
    def get_optimizer_info(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Get detailed information about optimizer memory usage.
        
        Args:
            optimizer: PyTorch optimizer instance
            
        Returns:
            Dictionary with optimizer information
        """
        total_memory = OptimizerMemoryCalculator.calculate_optimizer_memory(optimizer)
        optimizer_type = OptimizerMemoryCalculator.get_optimizer_type_from_instance(optimizer)
        
        param_count = 0
        state_count = 0
        
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    param_count += param.numel()
                    if param in optimizer.state:
                        state_count += len(optimizer.state[param])
        
        return {
            "optimizer_type": optimizer_type,
            "total_memory_bytes": total_memory,
            "total_memory_mb": total_memory / (1024 * 1024),
            "total_memory_gb": total_memory / (1024 * 1024 * 1024),
            "parameter_count": param_count,
            "state_count": state_count,
            "memory_per_param_bytes": total_memory / param_count if param_count > 0 else 0
        }


def get_memory_footprint_with_optimizer(
    model: nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None,
    precision: str = "fp32"
) -> Dict[str, Any]:
    """Get detailed memory footprint with actual optimizer.
    
    Args:
        model: PyTorch model
        optimizer: Optional optimizer instance
        precision: Model precision ("fp32", "fp16", "bf16")
        
    Returns:
        Dictionary with memory breakdown
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    bytes_per_param = 4 if precision == "fp32" else 2
    
    param_memory = total_params * bytes_per_param
    grad_memory = trainable_params * bytes_per_param
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Use actual optimizer if provided
    if optimizer is not None:
        optimizer_memory = OptimizerMemoryCalculator.calculate_optimizer_memory(optimizer)
        optimizer_type = OptimizerMemoryCalculator.get_optimizer_type_from_instance(optimizer)
        optimizer_info = OptimizerMemoryCalculator.get_optimizer_info(optimizer)
    else:
        # Fallback to Adam estimate
        optimizer_memory = OptimizerMemoryCalculator.estimate_optimizer_memory_by_type(
            trainable_params, "adam", precision
        )
        optimizer_type = "estimated_adam"
        optimizer_info = None
    
    total_memory = param_memory + grad_memory + optimizer_memory + buffer_memory
    
    result = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_memory_mb": param_memory / (1024 * 1024),
        "gradient_memory_mb": grad_memory / (1024 * 1024),
        "optimizer_memory_mb": optimizer_memory / (1024 * 1024),
        "buffer_memory_mb": buffer_memory / (1024 * 1024),
        "estimated_total_mb": total_memory / (1024 * 1024),
        "memory_efficiency": trainable_params / total_params * 100 if total_params > 0 else 0,
        "optimizer_type": optimizer_type,
        "precision": precision
    }
    
    if optimizer_info:
        result["optimizer_info"] = optimizer_info
    
    return result


def estimate_training_memory_with_optimizer(
    trainable_params: int,
    optimizer_type: str = "adam",
    batch_size: int = 1,
    sequence_length: int = 512,
    precision: str = "fp32",
    include_activations: bool = True
) -> Dict[str, float]:
    """Estimate training memory with specific optimizer type.
    
    Args:
        trainable_params: Number of trainable parameters
        optimizer_type: Type of optimizer
        batch_size: Training batch size
        sequence_length: Input sequence length
        precision: Model precision
        include_activations: Whether to include activation memory
        
    Returns:
        Dictionary with memory estimates
    """
    bytes_per_param = 4 if precision == "fp32" else 2
    
    # Model parameters
    model_memory = trainable_params * bytes_per_param
    
    # Gradients
    gradient_memory = trainable_params * bytes_per_param
    
    # Optimizer states
    optimizer_memory = OptimizerMemoryCalculator.estimate_optimizer_memory_by_type(
        trainable_params, optimizer_type, precision
    )
    
    # Activations (rough estimate)
    activation_memory = 0
    if include_activations:
        # Very rough estimate: batch_size * seq_len * hidden_size * bytes_per_param
        # Using a typical hidden size of 1024
        hidden_size = 1024
        activation_memory = batch_size * sequence_length * hidden_size * bytes_per_param
    
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
    
    return {
        "model_memory_mb": model_memory / (1024 * 1024),
        "gradient_memory_mb": gradient_memory / (1024 * 1024),
        "optimizer_memory_mb": optimizer_memory / (1024 * 1024),
        "activation_memory_mb": activation_memory / (1024 * 1024),
        "total_memory_mb": total_memory / (1024 * 1024),
        "total_memory_gb": total_memory / (1024 * 1024 * 1024),
        "optimizer_type": optimizer_type,
        "precision": precision,
        "batch_size": batch_size,
        "sequence_length": sequence_length
    }


def compare_optimizer_memory_usage(
    trainable_params: int,
    optimizer_types: Optional[list] = None,
    precision: str = "fp32"
) -> Dict[str, Dict[str, float]]:
    """Compare memory usage across different optimizer types.
    
    Args:
        trainable_params: Number of trainable parameters
        optimizer_types: List of optimizer types to compare
        precision: Model precision
        
    Returns:
        Dictionary with memory usage for each optimizer
    """
    if optimizer_types is None:
        optimizer_types = ["adam", "adamw", "sgd", "rmsprop", "adagrad", "adamax"]
    
    results = {}
    
    for optimizer_type in optimizer_types:
        memory = OptimizerMemoryCalculator.estimate_optimizer_memory_by_type(
            trainable_params, optimizer_type, precision
        )
        
        results[optimizer_type] = {
            "memory_bytes": memory,
            "memory_mb": memory / (1024 * 1024),
            "memory_gb": memory / (1024 * 1024 * 1024),
            "memory_per_param_bytes": memory / trainable_params if trainable_params > 0 else 0
        }
    
    return results
