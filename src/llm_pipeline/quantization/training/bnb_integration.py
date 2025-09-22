"""BitsAndBytes integration for 4-bit and 8-bit quantization."""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any, List
import warnings

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    # Warning will be handled conditionally based on config

from ...core.config import LoRAConfig
from ...core.base_module import BaseLoRAModule
from ..configs import BnBConfig


def _setup_bnb_logger(config: BnBConfig) -> logging.Logger:
    """Setup logger based on BnB configuration."""
    logger = logging.getLogger("bnb")
    logger.setLevel(getattr(logging, config.log_level))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def _initialize_weights(weight_tensor: torch.Tensor, config: BnBConfig) -> torch.Tensor:
    """Initialize weights based on configuration."""
    if config.weight_init_method == "random":
        nn.init.normal_(weight_tensor, std=config.weight_init_std)
    elif config.weight_init_method == "zeros":
        nn.init.zeros_(weight_tensor)
    elif config.weight_init_method == "ones":
        nn.init.ones_(weight_tensor)
    elif config.weight_init_method == "xavier":
        nn.init.xavier_uniform_(weight_tensor)
    elif config.weight_init_method == "kaiming":
        nn.init.kaiming_uniform_(weight_tensor)
    
    return weight_tensor


class BnBQuantizedLinear(nn.Module):
    """BitsAndBytes quantized linear layer"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: BnBConfig,
        bias: bool = True
    ):
        super().__init__()
        
        if not BNB_AVAILABLE:
            raise ImportError("BitsAndBytes is required for quantized layers")
        
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        
        # Setup logging
        self.logger = _setup_bnb_logger(config) if config.enable_logging else None
        
        # Create quantized layer based on config
        if config.load_in_4bit:
            self.quantized_layer = Linear4bit(
                input_features=in_features,
                output_features=out_features,
                bias=bias,
                compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
                compress_statistics=config.bnb_4bit_use_double_quant,
                quant_type=config.bnb_4bit_quant_type
            )
            # Initialize quantization state with configurable method
            if hasattr(self.quantized_layer, 'weight'):
                weight_tensor = torch.randn(out_features, in_features)
                self.quantized_layer.weight.data = _initialize_weights(weight_tensor, config)
        elif config.load_in_8bit:
            self.quantized_layer = Linear8bitLt(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                has_fp16_weights=config.llm_int8_has_fp16_weight,
                threshold=config.llm_int8_threshold
            )
        else:
            raise ValueError("Must enable either 4-bit or 8-bit quantization")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantized layer"""
        return self.quantized_layer(x)
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """Get memory footprint of quantized layer"""
        num_params = self.in_features * self.out_features
        
        if self.config.load_in_4bit:
            # Use configurable bytes per 4-bit parameter
            weight_memory = num_params * self.config.bytes_per_4bit_param
            compression_ratio = self.config.compression_ratio_4bit
            quantization_type = "4-bit"
        else:
            # Use configurable bytes per 8-bit parameter
            weight_memory = num_params * self.config.bytes_per_8bit_param
            compression_ratio = self.config.compression_ratio_8bit
            quantization_type = "8-bit"
        
        # Use configurable memory unit conversion
        memory_factor = self.config.memory_unit_factor ** 2  # Convert to MB (1024^2)
        
        return {
            "weight_memory_mb": weight_memory / memory_factor,
            "quantization_type": quantization_type,
            "compression_ratio": compression_ratio,
            "bytes_per_param": self.config.bytes_per_4bit_param if self.config.load_in_4bit else self.config.bytes_per_8bit_param
        }


class BnBQuantizedLoRA(nn.Module):
    """LoRA adapter for BitsAndBytes quantized models"""
    
    def __init__(
        self,
        quantized_layer: BnBQuantizedLinear,
        lora_config: LoRAConfig,
        adapter_name: str = "default"
    ):
        super().__init__()
        
        self.quantized_layer = quantized_layer
        self.lora_config = lora_config
        self.adapter_name = adapter_name
        
        # Setup logging
        self.logger = _setup_bnb_logger(quantized_layer.config) if quantized_layer.config.enable_logging else None
        
        # LoRA parameters
        self.lora_adapters = nn.ModuleDict()
        self.active_adapters = set()
        
        # Add default adapter
        self.add_adapter(adapter_name, lora_config)
    
    def add_adapter(self, adapter_name: str, config: LoRAConfig):
        """Add LoRA adapter to quantized layer"""
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
        """Forward pass: quantized base + LoRA adaptations"""
        # Quantized base layer
        result = self.quantized_layer(x)
        
        # Add LoRA adaptations
        for adapter_name in self.active_adapters:
            if adapter_name in self.lora_adapters:
                lora_output = self.lora_adapters[adapter_name](x)
                result += lora_output
        
        return result
    
    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Get parameter count breakdown"""
        # Base quantized parameters
        base_params = self.quantized_layer.in_features * self.quantized_layer.out_features
        
        # LoRA parameters
        lora_params = sum(
            adapter.get_parameter_count() 
            for adapter in self.lora_adapters.values()
        )
        
        return {
            "base_parameters": base_params,
            "lora_parameters": lora_params,
            "total_parameters": base_params + lora_params,
            "lora_efficiency": lora_params / base_params * 100
        }


def create_bnb_model(
    model: nn.Module,
    bnb_config: BnBConfig,
    lora_config: Optional[LoRAConfig] = None
) -> nn.Module:
    """Create BnB quantized model with optional LoRA"""
    
    # Handle import warning based on config
    if not BNB_AVAILABLE:
        if not bnb_config.suppress_import_warnings:
            warnings.warn("BitsAndBytes not available. Install with: pip install bitsandbytes")
        raise ImportError("BitsAndBytes not available")
    
    # Setup logging
    logger = _setup_bnb_logger(bnb_config) if bnb_config.enable_logging else None
    
    # Replace linear layers with quantized versions
    quantized_modules = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Create quantized replacement
            quantized_linear = BnBQuantizedLinear(
                module.in_features,
                module.out_features,
                bnb_config,
                bias=(module.bias is not None)
            )
            
            # Add LoRA if requested
            if lora_config is not None:
                quantized_linear = BnBQuantizedLoRA(
                    quantized_linear,
                    lora_config,
                    adapter_name="default"
                )
            
            quantized_modules[name] = quantized_linear
    
    # Replace modules in model
    for name, new_module in quantized_modules.items():
        _replace_module(model, name, new_module)
    
    # Log results using configurable logging
    quantization_type = "4-bit" if bnb_config.load_in_4bit else "8-bit"
    message = f"Applied BnB {quantization_type} quantization to {len(quantized_modules)} linear layers"
    if logger:
        logger.info(message)
    elif bnb_config.enable_logging:
        print(f"✅ {message}")
    
    return model


def _replace_module(model: nn.Module, module_name: str, new_module: nn.Module):
    """Replace a module in the model hierarchy"""
    module_path = module_name.split('.')
    parent = model
    
    # Navigate to parent
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
    
    # Replace final module
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
