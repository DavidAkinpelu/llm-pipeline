"""DoRA (Weight-Decomposed Low-Rank Adaptation) implementation."""

import torch
import torch.nn as nn
import math
from typing import Optional
from ..core.base_module import BaseLoRAModule
from ..core.config import DoRAConfig


class DoRAModule(BaseLoRAModule):
    """DoRA (Weight-Decomposed Low-Rank Adaptation) implementation"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: DoRAConfig,
        adapter_name: str = "default"
    ):
        super().__init__(in_features, out_features, config, adapter_name)
        
        # DoRA magnitude parameter
        self.magnitude = nn.Parameter(torch.ones(out_features))
        self._init_magnitude()
        
        # Cache for efficiency
        self._cached_norm = None
        self._cached_weight = None
    
    def _init_magnitude(self):
        """Initialize magnitude parameter."""
        if self.config.magnitude_init == "ones":
            nn.init.ones_(self.magnitude)
        elif self.config.magnitude_init == "random":
            nn.init.normal_(self.magnitude, mean=1.0, std=0.02)
        elif self.config.magnitude_init == "kaiming":
            # For 1D tensors, use normal initialization with kaiming variance
            nn.init.normal_(self.magnitude, mean=1.0, std=0.02)
        else:
            raise ValueError(f"Unknown magnitude init: {self.config.magnitude_init}")
    
    def forward(self, x: torch.Tensor, base_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for DoRA"""
        if base_weight is None:
            raise ValueError("DoRA requires base_weight parameter for forward pass")
        
        if self.rank == 0:
            # DoRA Rank=0: Apply only magnitude scaling to base weight
            # W = m * W0 / ||W0||
            weight_norm = torch.norm(base_weight, dim=1, keepdim=True)
            weight_norm = torch.clamp(weight_norm, min=1e-8)
            normalized_weight = base_weight / weight_norm
            final_weight = self.magnitude.unsqueeze(1) * normalized_weight
            
            return x @ final_weight.T
        
        # Standard DoRA with LoRA adaptation
        # Compute LoRA delta
        lora_result = x @ self.lora_A.T
        lora_result = self.dropout(lora_result)
        lora_delta = lora_result @ self.lora_B.T
        
        # DoRA decomposition: W = m * (W0 + ΔW) / ||W0 + ΔW||
        # Compute combined weight
        lora_weight = self.lora_B @ self.lora_A * self.scaling
        combined_weight = base_weight + lora_weight
        
        # Compute weight norm for each output dimension
        weight_norm = torch.norm(combined_weight, dim=1, keepdim=True)
        
        # Avoid division by zero
        weight_norm = torch.clamp(weight_norm, min=1e-8)
        
        # Normalize and apply magnitude scaling
        normalized_weight = combined_weight / weight_norm
        final_weight = self.magnitude.unsqueeze(1) * normalized_weight
        
        return x @ final_weight.T
    
    def get_decomposed_weight(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Get the decomposed weight matrix for analysis"""
        if self.rank == 0:
            # DoRA Rank=0: Only magnitude scaling applied to base weight
            weight_norm = torch.norm(base_weight, dim=1, keepdim=True)
            weight_norm = torch.clamp(weight_norm, min=1e-8)
            normalized_weight = base_weight / weight_norm
            return self.magnitude.unsqueeze(1) * normalized_weight
        
        # Standard DoRA with LoRA adaptation
        lora_weight = self.lora_B @ self.lora_A * self.scaling
        combined_weight = base_weight + lora_weight
        weight_norm = torch.norm(combined_weight, dim=1, keepdim=True)
        weight_norm = torch.clamp(weight_norm, min=1e-8)
        normalized_weight = combined_weight / weight_norm
        return self.magnitude.unsqueeze(1) * normalized_weight
    
    def get_magnitude_stats(self) -> dict:
        """Get statistics about magnitude parameters"""
        return {
            "mean": self.magnitude.mean().item(),
            "std": self.magnitude.std().item(),
            "min": self.magnitude.min().item(),
            "max": self.magnitude.max().item()
        }
    
    def get_parameter_count(self) -> int:
        """Get number of trainable parameters in this DoRA adapter"""
        lora_params = super().get_parameter_count()  # LoRA A and B matrices
        magnitude_params = self.magnitude.numel()    # Magnitude parameters
        return lora_params + magnitude_params
    
    def get_delta_weight(self, base_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get the delta weight matrix for DoRA"""
        if self.rank == 0:
            # For rank=0, return the magnitude-scaled base weight minus base weight
            if base_weight is None:
                return torch.zeros(self.out_features, self.in_features)
            
            weight_norm = torch.norm(base_weight, dim=1, keepdim=True)
            weight_norm = torch.clamp(weight_norm, min=1e-8)
            normalized_weight = base_weight / weight_norm
            scaled_weight = self.magnitude.unsqueeze(1) * normalized_weight
            
            return scaled_weight - base_weight
        
        # Standard LoRA delta for rank > 0
        return super().get_delta_weight()
    
    def extra_repr(self) -> str:
        base_repr = super().extra_repr()
        return f"{base_repr}, magnitude_init={self.config.magnitude_init}"


class DoRALinear(nn.Module):
    """DoRA-enhanced Linear layer"""
    def __init__(
        self,
        base_layer: nn.Linear,
        config: DoRAConfig,
        adapter_name: str = "default"
    ):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.adapter_name = adapter_name
        
        # Store dimensions
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # DoRA adapter
        self.dora_adapter = DoRAModule(
            self.in_features, self.out_features, config, adapter_name
        )
        
        # Track if adapter is active
        self.adapter_active = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with DoRA adaptation"""
        if not self.adapter_active:
            return self.base_layer(x)
        
        # Get base output
        base_output = self.base_layer(x)
        
        # Add DoRA adaptation
        dora_output = self.dora_adapter(x, self.base_layer.weight)
        
        return dora_output
    
    def disable_adapter(self):
        """Disable DoRA adapter"""
        self.adapter_active = False
    
    def enable_adapter(self):
        """Enable DoRA adapter"""
        self.adapter_active = True
    
    def get_magnitude_stats(self) -> dict:
        """Get magnitude statistics"""
        return self.dora_adapter.get_magnitude_stats()
