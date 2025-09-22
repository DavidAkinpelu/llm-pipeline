"""Quantization utilities and analysis tools."""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
import time

from .configs import QuantizationConfig, BnBConfig, AQLMConfig, LoftQConfig


def get_quantization_info(model: nn.Module) -> Dict[str, Any]:
    """Get comprehensive quantization information about a model"""
    
    info = {
        "total_modules": 0,
        "quantized_modules": 0,
        "quantization_types": {},
        "parameter_breakdown": {},
        "memory_savings": {}
    }
    
    total_params = 0
    quantized_params = 0
    
    for name, module in model.named_modules():
        info["total_modules"] += 1
        
        if hasattr(module, 'get_memory_footprint'):
            # BnB quantized module
            footprint = module.get_memory_footprint()
            quant_type = footprint.get('quantization_type', 'unknown')
            
            info["quantized_modules"] += 1
            info["quantization_types"][quant_type] = info["quantization_types"].get(quant_type, 0) + 1
            
        elif hasattr(module, 'get_compression_info'):
            # AQLM quantized module  
            comp_info = module.get_compression_info()
            info["quantized_modules"] += 1
            info["quantization_types"]["2-bit"] = info["quantization_types"].get("2-bit", 0) + 1
            
        elif hasattr(module, 'get_loftq_stats'):
            # LoftQ module
            loftq_stats = module.get_loftq_stats()
            info["quantized_modules"] += 1
            scheme = loftq_stats["quantization_scheme"]
            info["quantization_types"][scheme] = info["quantization_types"].get(scheme, 0) + 1
        
        # Count parameters
        if isinstance(module, nn.Linear):
            module_params = module.in_features * module.out_features
            if module.bias is not None:
                module_params += module.out_features
            total_params += module_params
            
            # Check if quantized
            if hasattr(module, 'get_memory_footprint') or hasattr(module, 'get_compression_info'):
                quantized_params += module_params
    
    info["parameter_breakdown"] = {
        "total_parameters": total_params,
        "quantized_parameters": quantized_params,
        "full_precision_parameters": total_params - quantized_params,
        "quantization_coverage": quantized_params / total_params * 100 if total_params > 0 else 0
    }
    
    return info


def estimate_quantized_memory(
    model: nn.Module,
    batch_size: int = 1,
    sequence_length: int = 512
) -> Dict[str, float]:
    """Estimate memory usage for quantized model"""
    
    memory_breakdown = {
        "quantized_weights_mb": 0,
        "full_precision_weights_mb": 0,
        "lora_weights_mb": 0,
        "activations_mb": 0,
        "gradients_mb": 0,
        "total_mb": 0
    }
    
    for module in model.modules():
        if hasattr(module, 'get_memory_footprint'):
            # BnB module
            footprint = module.get_memory_footprint()
            memory_breakdown["quantized_weights_mb"] += footprint["weight_memory_mb"]
            
        elif hasattr(module, 'get_compression_info'):
            # AQLM module
            comp_info = module.get_compression_info()
            memory_breakdown["quantized_weights_mb"] += comp_info["compressed_size_mb"]
            
        elif hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # LoRA module
            lora_params = module.lora_A.numel() + module.lora_B.numel()
            memory_breakdown["lora_weights_mb"] += lora_params * 4 / (1024 * 1024)
            
        elif isinstance(module, nn.Linear):
            # Regular linear layer
            params = module.weight.numel()
            if module.bias is not None:
                params += module.bias.numel()
            memory_breakdown["full_precision_weights_mb"] += params * 4 / (1024 * 1024)
    
    # Estimate activations (rough)
    memory_breakdown["activations_mb"] = batch_size * sequence_length * 1024 * 4 / (1024 * 1024)
    
    # Estimate gradients (only for LoRA parameters)
    memory_breakdown["gradients_mb"] = memory_breakdown["lora_weights_mb"]  # Same size as LoRA weights
    
    # Total
    memory_breakdown["total_mb"] = sum(memory_breakdown.values()) - memory_breakdown["total_mb"]  # Avoid double counting
    
    return memory_breakdown


def compare_quantization_methods(
    model: nn.Module,
    test_input: torch.Tensor,
    quantization_configs: Dict[str, QuantizationConfig]
) -> Dict[str, Dict[str, Any]]:
    """Compare different quantization methods"""
    
    results = {}
    original_output = model(test_input)
    
    for method_name, config in quantization_configs.items():
        print(f"🔍 Testing {method_name} quantization...")
        
        # Create quantized version
        quantized_model = _create_quantized_model(model, config)
        
        # Timing test
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            quantized_output = quantized_model(test_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        # Quality metrics
        mse_error = torch.nn.functional.mse_loss(quantized_output, original_output).item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            quantized_output.flatten(), 
            original_output.flatten(),
            dim=0
        ).item()
        
        # Memory analysis
        memory_info = estimate_quantized_memory(quantized_model)
        quant_info = get_quantization_info(quantized_model)
        
        results[method_name] = {
            "inference_time_ms": (end_time - start_time) * 1000,
            "mse_error": mse_error,
            "cosine_similarity": cosine_sim,
            "memory_mb": memory_info["total_mb"],
            "compression_ratio": _calculate_compression_ratio(model, quantized_model),
            "quantization_coverage": quant_info["parameter_breakdown"]["quantization_coverage"],
            "config": config
        }
    
    return results


def quantization_quality_metrics(
    original_output: torch.Tensor,
    quantized_output: torch.Tensor
) -> Dict[str, float]:
    """Compute quality metrics for quantized model outputs"""
    
    # Flatten tensors for some metrics
    orig_flat = original_output.flatten()
    quant_flat = quantized_output.flatten()
    
    # Mean squared error
    mse = torch.nn.functional.mse_loss(quantized_output, original_output).item()
    
    # Mean absolute error
    mae = torch.nn.functional.l1_loss(quantized_output, original_output).item()
    
    # Cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(orig_flat, quant_flat, dim=0).item()
    
    # Pearson correlation
    orig_centered = orig_flat - orig_flat.mean()
    quant_centered = quant_flat - quant_flat.mean()
    correlation = (orig_centered * quant_centered).sum() / (
        torch.sqrt((orig_centered ** 2).sum()) * torch.sqrt((quant_centered ** 2).sum())
    ).item()
    
    # Signal-to-noise ratio (in dB)
    signal_power = (orig_flat ** 2).mean()
    noise_power = ((orig_flat - quant_flat) ** 2).mean()
    snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()
    
    # Relative error
    relative_error = (torch.norm(original_output - quantized_output) / torch.norm(original_output)).item()
    
    return {
        "mse": mse,
        "mae": mae,
        "cosine_similarity": cosine_sim,
        "pearson_correlation": correlation,
        "snr_db": snr_db,
        "relative_error": relative_error,
        "quality_score": (cosine_sim + correlation) / 2 - relative_error  # Combined metric
    }


def _create_quantized_model(model: nn.Module, config: QuantizationConfig) -> nn.Module:
    """Create quantized model based on config type"""
    model_copy = type(model)(**model.init_kwargs) if hasattr(model, 'init_kwargs') else model
    
    if isinstance(config, BnBConfig):
        from .bnb_integration import create_bnb_model
        return create_bnb_model(model_copy, config)
    elif isinstance(config, AQLMConfig):
        from .aqlm_integration import create_aqlm_model
        return create_aqlm_model(model_copy, config)
    elif isinstance(config, LoftQConfig):
        # LoftQ requires special handling - placeholder
        return model_copy
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")


def _calculate_compression_ratio(original_model: nn.Module, quantized_model: nn.Module) -> float:
    """Calculate compression ratio between original and quantized models"""
    
    # Rough estimate based on parameter counts
    orig_params = sum(p.numel() for p in original_model.parameters())
    
    # For quantized model, estimate based on quantization types
    quant_info = get_quantization_info(quantized_model)
    
    # Assume average compression based on quantization coverage
    coverage = quant_info["parameter_breakdown"]["quantization_coverage"] / 100
    
    # Rough compression ratios for different methods
    avg_compression = 1.0
    for quant_type, count in quant_info["quantization_types"].items():
        if "4-bit" in quant_type:
            avg_compression = 8.0  # fp32 -> 4bit
        elif "8-bit" in quant_type:
            avg_compression = 4.0  # fp32 -> 8bit
        elif "2-bit" in quant_type:
            avg_compression = 16.0  # fp32 -> 2bit
    
    # Weighted compression ratio
    effective_compression = coverage * avg_compression + (1 - coverage) * 1.0
    
    return effective_compression
