"""Base LoRA module implementation."""

import torch
import torch.nn as nn
import math
from typing import Optional
from .config import LoRAConfig


class BaseLoRAModule(nn.Module):
    """Base LoRA module implementation"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        adapter_name: str = "default"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.adapter_name = adapter_name
        self.rank = config.r
        self.scaling = config.scaling
        
        # Only create parameters if rank > 0
        if config.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(config.r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, config.r))
        else:
            # Rank=0: No LoRA parameters needed (pure inference)
            self.lora_A = None
            self.lora_B = None
        
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
        self._initialize_weights()
        
        self.merged = False
        
    def _initialize_weights(self):
        """Initialize LoRA weights according to configuration"""
        if self.rank == 0:
            return  # No weights to initialize when rank=0
            
        if self.config.init_lora_weights == "gaussian":
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_B)
        elif self.config.init_lora_weights == "kaiming":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        elif self.config.init_lora_weights == "xavier":
            nn.init.xavier_uniform_(self.lora_A)
            nn.init.zeros_(self.lora_B)
        else:
            raise ValueError(f"Unknown initialization method: {self.config.init_lora_weights}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer"""
        if self.rank == 0:
            # Return zero contribution with correct output shape
            batch_size = x.size(0)
            seq_len = x.size(1) if x.dim() > 2 else 1
            return torch.zeros(batch_size, seq_len, self.out_features, device=x.device, dtype=x.dtype)
            
        # Standard LoRA computation: x @ A.T @ B.T * scaling
        result = x @ self.lora_A.T  # (batch, seq, rank)
        result = self.dropout(result)
        result = result @ self.lora_B.T  # (batch, seq, out_features)
        return result * self.scaling
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}"
    
    def reset_parameters(self):
        """Reset LoRA parameters to initial state"""
        self._initialize_weights()
        self.merged = False
    
    def get_delta_weight(self) -> torch.Tensor:
        """Get the delta weight matrix (B @ A)"""
        if self.rank == 0:
            return torch.zeros(self.out_features, self.in_features)
        return (self.lora_B @ self.lora_A) * self.scaling
    
    def get_parameter_count(self) -> int:
        """Get number of trainable parameters in this adapter"""
        if self.rank == 0:
            return 0
        return self.lora_A.numel() + self.lora_B.numel()