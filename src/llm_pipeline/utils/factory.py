"""Factory functions for creating models, configs, and adapters."""

import torch.nn as nn
from typing import Optional, List, Dict, Any
from ..core.config import LoRAConfig, DoRAConfig, RSLoRAConfig, MultiAdapterConfig
from ..core.model_wrapper import LoRAModelWrapper


def create_lora_model(
    model: nn.Module,
    config: LoRAConfig,
    model_type: Optional[str] = None
) -> LoRAModelWrapper:
    """Factory function to create LoRA-enhanced model"""
    return LoRAModelWrapper(model, config, model_type)


def get_lora_config(
    r: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    **kwargs
) -> LoRAConfig:
    """Factory function to create LoRA config"""
    return LoRAConfig(r=r, alpha=alpha, target_modules=target_modules, **kwargs)


def get_dora_config(
    r: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    magnitude_init: str = "ones",
    **kwargs
) -> DoRAConfig:
    """Factory function to create DoRA config"""
    return DoRAConfig(
        r=r, 
        alpha=alpha, 
        target_modules=target_modules,
        magnitude_init=magnitude_init,
        **kwargs
    )


def get_rslora_config(
    r: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    **kwargs
) -> RSLoRAConfig:
    """Factory function to create RSLoRA config"""
    return RSLoRAConfig(r=r, alpha=alpha, target_modules=target_modules, **kwargs)


def get_multi_adapter_config(
    max_adapters: int = 8,
    enable_routing: bool = False,
    routing_hidden_size: Optional[int] = None,
    adapter_fusion_method: str = "weighted_sum",
    **kwargs
) -> MultiAdapterConfig:
    """Factory function to create MultiAdapter config"""
    return MultiAdapterConfig(
        max_adapters=max_adapters,
        enable_routing=enable_routing,
        routing_hidden_size=routing_hidden_size,
        adapter_fusion_method=adapter_fusion_method,
        **kwargs
    )


def create_adapter_linear(
    base_layer: nn.Linear,
    adapter_type: str,
    config: LoRAConfig,
    adapter_name: str = "default"
) -> nn.Module:
    """Factory function to create adapter-enhanced linear layer"""
    
    if adapter_type == "lora":
        from ..adapters.lora import LoRALinear
        return LoRALinear(base_layer, config, adapter_name)
    
    elif adapter_type == "dora":
        from ..adapters.dora import DoRALinear
        if not isinstance(config, DoRAConfig):
            dora_config = DoRAConfig(**config.__dict__)
        else:
            dora_config = config
        return DoRALinear(base_layer, dora_config, adapter_name)
    
    elif adapter_type == "rslora":
        from ..adapters.rslora import RSLoRALinear
        if not isinstance(config, RSLoRAConfig):
            rslora_config = RSLoRAConfig(**config.__dict__)
        else:
            rslora_config = config
        return RSLoRALinear(base_layer, rslora_config, adapter_name)
    
    elif adapter_type == "multi":
        from ..adapters.adapter_manager import MultiAdapterLoRALinear
        return MultiAdapterLoRALinear(base_layer, config)
    
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


def create_config_from_dict(config_dict: Dict[str, Any], config_type: str = "lora") -> LoRAConfig:
    """Create config from dictionary"""
    if config_type == "lora":
        return LoRAConfig(**config_dict)
    elif config_type == "dora":
        return DoRAConfig(**config_dict)
    elif config_type == "rslora":
        return RSLoRAConfig(**config_dict)
    else:
        raise ValueError(f"Unknown config type: {config_type}")