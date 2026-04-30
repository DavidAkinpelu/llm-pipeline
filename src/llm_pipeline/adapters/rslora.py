"""RSLoRA (Rank-Stabilized LoRA) implementation."""

import torch
import torch.nn as nn
import math
from typing import Optional
from ..core.base_module import BaseLoRAModule
from ..core.config import RSLoRAConfig


class RSLoRAModule(BaseLoRAModule):
    """RSLoRA (Rank-Stabilized LoRA) implementation.
    
    RSLoRA uses sqrt(r) scaling instead of r scaling for better stability.
    No additional parameters beyond standard LoRA.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: RSLoRAConfig,
        adapter_name: str = "default"
    ):
        # RSLoRA uses sqrt(r) scaling which is handled in config.scaling
        super().__init__(in_features, out_features, config, adapter_name)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with RSLoRA sqrt(r) scaling."""
        if self.rank == 0:
            # RSLoRA Rank=0: No adaptation (same as standard LoRA Rank=0)
            batch_size = x.size(0)
            seq_len = x.size(1) if x.dim() > 2 else 1
            return torch.zeros(batch_size, seq_len, self.out_features, device=x.device, dtype=x.dtype)
        
        # Standard LoRA computation with RSLoRA scaling (alpha / sqrt(r))
        result = x @ self.lora_A.T  # (batch, seq, rank)
        result = self.dropout(result)
        result = result @ self.lora_B.T  # (batch, seq, out_features)
        
        # Apply RSLoRA scaling: alpha / sqrt(r)
        return result * self.scaling
    
    def get_effective_scaling(self) -> float:
        """Get the effective scaling factor (alpha / sqrt(r) for RSLoRA)."""
        return self.scaling
    
    def extra_repr(self) -> str:
        base_repr = super().extra_repr()
        return f"{base_repr}, rslora_scaling={self.scaling:.3f}"


class RSLoRALinear(nn.Module):
    """RSLoRA-enhanced Linear layer"""
    def __init__(
        self,
        base_layer: nn.Linear,
        config: RSLoRAConfig,
        adapter_name: str = "default"
    ):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.adapter_name = adapter_name
        
        # Store dimensions
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # RSLoRA adapter
        self.rslora_adapter = RSLoRAModule(
            self.in_features, self.out_features, config, adapter_name
        )
        
        # Track if adapter is active
        self.adapter_active = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with RSLoRA adaptation"""
        # Base layer output
        result = self.base_layer(x)
        
        # Add RSLoRA adaptation if active
        if self.adapter_active:
            adapter_output = self.rslora_adapter(x)
            result += adapter_output
        
        return result
    
    def disable_adapter(self):
        """Disable RSLoRA adapter"""
        self.adapter_active = False
    
    def enable_adapter(self):
        """Enable RSLoRA adapter"""
        self.adapter_active = True
    
    def get_effective_scaling(self) -> float:
        """Get effective scaling factor"""
        return self.rslora_adapter.get_effective_scaling()
    
    def get_stabilizer_value(self) -> float:
        """Get stabilizer value"""
        return self.rslora_adapter.get_stabilizer_value()