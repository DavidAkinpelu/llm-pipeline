"""AQLM 2-bit quantization integration."""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any, List
import warnings

try:
    import aqlm
    AQLM_AVAILABLE = True
except ImportError:
    AQLM_AVAILABLE = False
    # Warning will be handled conditionally based on config

from ...core.config import LoRAConfig  
from ...core.base_module import BaseLoRAModule
from ..configs import AQLMConfig


def _setup_aqlm_logger(config: AQLMConfig) -> logging.Logger:
    """Setup logger based on AQLM configuration."""
    logger = logging.getLogger("aqlm")
    logger.setLevel(getattr(logging, config.log_level))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


class AQLMQuantizedLinear(nn.Module):
    """AQLM 2-bit quantized linear layer"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: AQLMConfig,
        bias: bool = True
    ):
        super().__init__()
        
        if not AQLM_AVAILABLE:
            raise ImportError("AQLM is required for 2-bit quantization")
        
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        
        # Setup logging
        self.logger = _setup_aqlm_logger(config) if config.enable_logging else None
        
        # Create AQLM quantized layer
        self.quantized_layer = aqlm.QuantizedLinear(
            in_features=in_features,
            out_features=out_features,
            in_group_size=config.in_group_size,
            out_group_size=config.out_group_size,
            num_codebooks=config.num_codebooks,
            nbits_per_codebook=config.nbits_per_codebook,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AQLM quantized layer"""
        return self.quantized_layer(x)
    
    def get_compression_info(self) -> Dict[str, Any]:
        """Get compression information"""
        # Calculate bit sizes using configurable values
        num_params = self.in_features * self.out_features
        original_bits = num_params * self.config.baseline_bits
        compressed_bits = num_params * self.config.compressed_bits
        
        # Calculate memory sizes using configurable factors
        bytes_per_bit = 8  # 8 bits per byte
        memory_factor = self.config.memory_unit_factor ** 2  # Convert to MB (1024^2)
        
        return {
            "original_size_mb": original_bits / (bytes_per_bit * memory_factor),
            "compressed_size_mb": compressed_bits / (bytes_per_bit * memory_factor), 
            "compression_ratio": original_bits / compressed_bits,
            "bits_per_parameter": self.config.compressed_bits,
            "baseline_bits_per_parameter": self.config.baseline_bits,
            "num_codebooks": self.config.num_codebooks,
            "group_size": self.config.in_group_size
        }


class AQLMQuantizedLoRA(nn.Module):
    """LoRA adapter for AQLM quantized models"""
    
    def __init__(
        self,
        quantized_layer: AQLMQuantizedLinear,
        lora_config: LoRAConfig,
        adapter_name: str = "default"
    ):
        super().__init__()
        
        self.quantized_layer = quantized_layer
        self.lora_config = lora_config
        self.adapter_name = adapter_name
        
        # Setup logging
        self.logger = _setup_aqlm_logger(quantized_layer.config) if quantized_layer.config.enable_logging else None
        
        # LoRA parameters
        self.lora_adapters = nn.ModuleDict()
        self.active_adapters = set()
        
        # Add default adapter
        self.add_adapter(adapter_name, lora_config)
    
    def add_adapter(self, adapter_name: str, config: LoRAConfig):
        """Add LoRA adapter to AQLM quantized layer"""
        lora_module = BaseLoRAModule(
            self.quantized_layer.in_features,
            self.quantized_layer.out_features,
            config,
            adapter_name
        )
        
        self.lora_adapters[adapter_name] = lora_module
        self.active_adapters.add(adapter_name)
    
    def set_active_adapters(self, adapter_names: List[str]):
        """Set active LoRA adapters"""
        self.active_adapters = set(adapter_names) & set(self.lora_adapters.keys())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: AQLM quantized base + LoRA adaptations"""
        # AQLM quantized base layer
        result = self.quantized_layer(x)
        
        # Add LoRA adaptations  
        for adapter_name in self.active_adapters:
            if adapter_name in self.lora_adapters:
                lora_output = self.lora_adapters[adapter_name](x)
                result += lora_output
        
        return result
    
    def get_extreme_compression_stats(self) -> Dict[str, float]:
        """Get extreme compression statistics"""
        base_info = self.quantized_layer.get_compression_info()
        config = self.quantized_layer.config
        
        # LoRA parameters (using configurable precision)
        lora_params = sum(adapter.get_parameter_count() for adapter in self.lora_adapters.values())
        memory_factor = config.memory_unit_factor ** 2  # Convert to MB (1024^2)
        lora_size_mb = lora_params * config.bytes_per_element / memory_factor
        
        total_size = base_info["compressed_size_mb"] + lora_size_mb
        original_size = base_info["original_size_mb"]
        
        return {
            "aqlm_compressed_mb": base_info["compressed_size_mb"],
            "lora_size_mb": lora_size_mb,
            "total_size_mb": total_size,
            "original_size_mb": original_size,
            "total_compression_ratio": original_size / total_size,
            "memory_savings_percent": (1 - total_size / original_size) * 100
        }


def create_aqlm_model(
    model: nn.Module,
    aqlm_config: AQLMConfig,
    lora_config: Optional[LoRAConfig] = None,
    target_modules: Optional[List[str]] = None
) -> nn.Module:
    """Create AQLM 2-bit quantized model with optional LoRA"""
    
    # Handle import warning based on config
    if not AQLM_AVAILABLE:
        if not aqlm_config.suppress_import_warnings:
            warnings.warn("AQLM not available. Install from: https://github.com/Vahe1994/AQLM")
        raise ImportError("AQLM not available")
    
    # Setup logging
    logger = _setup_aqlm_logger(aqlm_config) if aqlm_config.enable_logging else None
    
    # Use configurable target modules
    target_modules = target_modules or aqlm_config.default_target_modules
    quantized_modules = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module_name = name.split('.')[-1]
            
            # Check if this module should be quantized
            if any(target in module_name for target in target_modules):
                # Create AQLM quantized replacement
                quantized_linear = AQLMQuantizedLinear(
                    module.in_features,
                    module.out_features,
                    aqlm_config,
                    bias=(module.bias is not None)
                )
                
                # Add LoRA if requested
                if lora_config is not None:
                    quantized_linear = AQLMQuantizedLoRA(
                        quantized_linear,
                        lora_config,
                        adapter_name="default"
                    )
                
                quantized_modules[name] = quantized_linear
    
    # Replace modules
    for name, new_module in quantized_modules.items():
        _replace_module(model, name, new_module)
    
    # Log results using configurable logging
    message = f"Applied AQLM {aqlm_config.compressed_bits}-bit quantization to {len(quantized_modules)} modules"
    if logger:
        logger.info(message)
    elif aqlm_config.enable_logging:
        print(f"✅ {message}")
    
    return model


def _replace_module(model: nn.Module, module_name: str, new_module: nn.Module):
    """Replace a module in the model hierarchy"""
    # Same implementation as BnB version
    module_path = module_name.split('.')
    parent = model
    
    for part in module_path[:-1]:
        if hasattr(parent, part):
            parent = getattr(parent, part)
        else:
            try:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = parent[part]
            except (KeyError, IndexError, TypeError):
                raise AttributeError(f"Cannot navigate to {module_name}")
    
    final_name = module_path[-1]
    if hasattr(parent, final_name):
        setattr(parent, final_name, new_module)
    else:
        try:
            if final_name.isdigit():
                parent[int(final_name)] = new_module
            else:
                parent[final_name] = new_module
        except (KeyError, IndexError, TypeError):
            raise AttributeError(f"Cannot replace module {module_name}")
