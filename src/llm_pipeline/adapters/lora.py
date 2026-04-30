"""Standard LoRA implementation."""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Set
from ..core.base_module import BaseLoRAModule
from ..core.config import LoRAConfig


class LoRALinear(nn.Module):
    """LoRA-enhanced Linear layer"""
    def __init__(
        self,
        base_layer: nn.Linear,
        config: LoRAConfig,
        adapter_name: str = "default"
    ):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.adapter_name = adapter_name
        
        # Store original parameters
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # LoRA adapters (can have multiple)
        self.lora_adapters = nn.ModuleDict()
        self.active_adapters = set()
        
        # Add default adapter
        self.add_adapter(adapter_name, config)
        
        # Bias handling
        self.bias_enabled = self._should_handle_bias()
        if self.bias_enabled:
            self.lora_bias = nn.Parameter(torch.zeros(self.out_features))
    
    def _should_handle_bias(self) -> bool:
        """Determine if bias should be handled based on config"""
        if self.config.bias == "none":
            return False
        elif self.config.bias == "all":
            return True
        elif self.config.bias == "lora_only":
            return True
        return False
    
    def add_adapter(self, adapter_name: str, config: LoRAConfig):
        """Add a new LoRA adapter"""
        if adapter_name in self.lora_adapters:
            raise ValueError(f"Adapter {adapter_name} already exists")
        
        # Choose the correct module type based on config
        if hasattr(config, 'use_dora') and config.use_dora:
            from .dora import DoRAModule
            adapter_module = DoRAModule(self.in_features, self.out_features, config, adapter_name)
        elif hasattr(config, 'use_rslora') and config.use_rslora:
            from .rslora import RSLoRAModule
            adapter_module = RSLoRAModule(self.in_features, self.out_features, config, adapter_name)
        else:
            adapter_module = BaseLoRAModule(self.in_features, self.out_features, config, adapter_name)
            
        self.lora_adapters[adapter_name] = adapter_module
        self.active_adapters.add(adapter_name)
    
    def remove_adapter(self, adapter_name: str):
        """Remove a LoRA adapter"""
        if adapter_name in self.lora_adapters:
            del self.lora_adapters[adapter_name]
            self.active_adapters.discard(adapter_name)
    
    def set_active_adapters(self, adapter_names: List[str]):
        """Set which adapters are active"""
        valid_adapters = [name for name in adapter_names if name in self.lora_adapters]
        self.active_adapters = set(valid_adapters)
        
        if len(valid_adapters) != len(adapter_names):
            missing = set(adapter_names) - set(valid_adapters)
            print(f"Warning: Adapters not found: {missing}")
    
    def get_active_adapters(self) -> List[str]:
        """Get list of currently active adapters"""
        return list(self.active_adapters)
    
    def disable_adapters(self):
        """Temporarily disable all adapters"""
        self._temp_active = self.active_adapters.copy()
        self.active_adapters.clear()
    
    def enable_adapters(self):
        """Re-enable previously disabled adapters"""
        if hasattr(self, '_temp_active'):
            self.active_adapters = self._temp_active
            delattr(self, '_temp_active')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation"""
        # Base layer forward pass
        result = self.base_layer(x)
        
        # Add LoRA adaptations
        for adapter_name in self.active_adapters:
            if adapter_name in self.lora_adapters:
                # Pass base_weight for DoRA compatibility
                if hasattr(self.lora_adapters[adapter_name], 'forward') and 'base_weight' in self.lora_adapters[adapter_name].forward.__code__.co_varnames:
                    adapter_output = self.lora_adapters[adapter_name](x, base_weight=self.base_layer.weight)
                else:
                    adapter_output = self.lora_adapters[adapter_name](x)
                result += adapter_output
        
        # Add LoRA bias if enabled
        if self.bias_enabled and hasattr(self, 'lora_bias'):
            result += self.lora_bias
                
        return result
    
    def get_adapter_state_dict(self, adapter_name: str) -> Dict[str, torch.Tensor]:
        """Get state dict for a specific adapter"""
        if adapter_name not in self.lora_adapters:
            raise ValueError(f"Adapter {adapter_name} not found")
        return self.lora_adapters[adapter_name].state_dict()
    
    def load_adapter_state_dict(self, adapter_name: str, state_dict: Dict[str, torch.Tensor]):
        """Load state dict for a specific adapter"""
        if adapter_name not in self.lora_adapters:
            raise ValueError(f"Adapter {adapter_name} not found")
        self.lora_adapters[adapter_name].load_state_dict(state_dict)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown"""
        counts = {"base": self.base_layer.weight.numel()}
        if self.base_layer.bias is not None:
            counts["base"] += self.base_layer.bias.numel()
            
        for name, adapter in self.lora_adapters.items():
            counts[f"adapter_{name}"] = adapter.get_parameter_count()
            
        if self.bias_enabled and hasattr(self, 'lora_bias'):
            counts["lora_bias"] = self.lora_bias.numel()
            
        return counts
    
    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"adapters={list(self.lora_adapters.keys())}, "
                f"active={list(self.active_adapters)}")