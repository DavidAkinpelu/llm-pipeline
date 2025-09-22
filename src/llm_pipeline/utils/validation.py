"""Input validation utilities."""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Union
from ..core.config import LoRAConfig, DoRAConfig, RSLoRAConfig
from ..core.registry import ModelRegistry


def validate_config(config: LoRAConfig) -> List[str]:
    """Validate LoRA configuration and return list of issues"""
    issues = []
    
    # Check rank
    if config.r <= 0:
        issues.append(f"Rank must be positive, got {config.r}")
    if config.r > 256:
        issues.append(f"Rank {config.r} is unusually high, consider lower values")
    
    # Check alpha
    if config.alpha <= 0:
        issues.append(f"Alpha must be positive, got {config.alpha}")
    
    # Check dropout
    if not 0 <= config.dropout <= 1:
        issues.append(f"Dropout must be in [0, 1], got {config.dropout}")
    
    # Check target modules
    if not config.target_modules:
        issues.append("No target modules specified")
    
    # Check bias setting
    valid_bias = ["none", "all", "lora_only"]
    if config.bias not in valid_bias:
        issues.append(f"Invalid bias setting '{config.bias}', must be one of {valid_bias}")
    
    # Check initialization
    valid_init = ["gaussian", "kaiming", "xavier"]
    if config.init_lora_weights not in valid_init:
        issues.append(f"Invalid initialization '{config.init_lora_weights}', must be one of {valid_init}")
    
    # DoRA-specific validation
    if isinstance(config, DoRAConfig):
        valid_mag_init = ["ones", "random", "kaiming"]
        if config.magnitude_init not in valid_mag_init:
            issues.append(f"Invalid magnitude_init '{config.magnitude_init}', must be one of {valid_mag_init}")
    
    return issues


def validate_model_compatibility(
    model: nn.Module,
    config: LoRAConfig,
    model_type: Optional[str] = None
) -> Dict[str, Any]:
    """Validate model compatibility with LoRA configuration"""
    
    results = {
        "compatible": True,
        "issues": [],
        "warnings": [],
        "target_modules_found": [],
        "target_modules_missing": [],
        "model_info": {}
    }
    
    # Detect model type if not provided
    if model_type is None and model is not None:
        model_type = ModelRegistry.detect_model_type(model.__class__.__name__)
        if model_type is None:
            results["warnings"].append("Could not auto-detect model type")
            model_type = "unknown"
    elif model_type is None:
        results["warnings"].append("No model provided and no model type specified")
        model_type = "unknown"
    
    results["model_info"]["detected_type"] = model_type
    
    # Check if target modules exist in model
    if model is not None:
        model_modules = {name.split('.')[-1] for name, _ in model.named_modules()}
    else:
        model_modules = set()
    
    if config is not None and config.target_modules is not None:
        for target in config.target_modules:
            if target in model_modules:
                results["target_modules_found"].append(target)
            else:
                results["target_modules_missing"].append(target)
    
    # Check for issues
    if not results["target_modules_found"]:
        results["compatible"] = False
        results["issues"].append("No target modules found in model")
    
    if len(results["target_modules_missing"]) > len(results["target_modules_found"]):
        results["warnings"].append(f"More target modules missing than found")
    
    # Count linear layers
    if model is not None:
        linear_layers = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
        results["model_info"]["total_linear_layers"] = len(linear_layers)
        results["model_info"]["total_parameters"] = sum(p.numel() for p in model.parameters())
        
        # Estimate LoRA parameters
        lora_params = 0
        if config is not None and config.target_modules is not None:
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    module_name = name.split('.')[-1]
                    if module_name in config.target_modules:
                        # LoRA adds r * (in_features + out_features) parameters
                        lora_params += config.r * (module.in_features + module.out_features)
    else:
        linear_layers = []
        results["model_info"]["total_linear_layers"] = 0
        results["model_info"]["total_parameters"] = 0
        lora_params = 0
    
    results["model_info"]["estimated_lora_parameters"] = lora_params
    results["model_info"]["parameter_efficiency"] = (
        lora_params / results["model_info"]["total_parameters"] * 100
        if results["model_info"]["total_parameters"] > 0 else 0
    )
    
    return results


def check_adapter_compatibility(
    adapter_configs: Dict[str, LoRAConfig]
) -> Dict[str, Any]:
    """Check compatibility between multiple adapter configurations"""
    
    results = {
        "compatible": True,
        "issues": [],
        "warnings": [],
        "config_comparison": {}
    }
    
    if len(adapter_configs) < 2:
        results["warnings"].append("Need at least 2 configs to compare")
        return results
    
    # Compare configurations
    configs = list(adapter_configs.values())
    first_config = configs[0]
    
    for i, config in enumerate(configs[1:], 1):
        comparison_key = f"config_{i}_vs_base"
        
        # Check target modules overlap
        overlap = set(first_config.target_modules) & set(config.target_modules)
        unique_first = set(first_config.target_modules) - set(config.target_modules)
        unique_second = set(config.target_modules) - set(first_config.target_modules)
        
        results["config_comparison"][comparison_key] = {
            "target_overlap": list(overlap),
            "unique_to_first": list(unique_first),
            "unique_to_second": list(unique_second),
            "rank_difference": abs(first_config.r - config.r),
            "alpha_difference": abs(first_config.alpha - config.alpha)
        }
        
        # Check for potential conflicts
        if not overlap:
            results["warnings"].append(f"No target module overlap between configs")
        
        if abs(first_config.r - config.r) > 32:
            results["warnings"].append(f"Large rank difference: {first_config.r} vs {config.r}")
        
        if abs(first_config.alpha - config.alpha) > 16:
            results["warnings"].append(f"Large alpha difference: {first_config.alpha} vs {config.alpha}")
    
    return results


def validate_input_tensor(
    tensor: torch.Tensor,
    expected_shape: Optional[tuple] = None,
    expected_dtype: Optional[torch.dtype] = None,
    expected_device: Optional[torch.device] = None
) -> List[str]:
    """Validate input tensor properties"""
    issues = []
    
    if not isinstance(tensor, torch.Tensor):
        issues.append(f"Expected torch.Tensor, got {type(tensor)}")
        return issues
    
    if expected_shape is not None:
        if tensor.shape != expected_shape:
            issues.append(f"Shape mismatch: expected {expected_shape}, got {tensor.shape}")
    
    if expected_dtype is not None:
        if tensor.dtype != expected_dtype:
            issues.append(f"Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
    
    if expected_device is not None:
        if tensor.device != expected_device:
            issues.append(f"Device mismatch: expected {expected_device}, got {tensor.device}")
    
    # Check for common issues
    if torch.isnan(tensor).any():
        issues.append("Tensor contains NaN values")
    
    if torch.isinf(tensor).any():
        issues.append("Tensor contains infinite values")
    
    if tensor.numel() == 0:
        issues.append("Tensor is empty")
    
    return issues