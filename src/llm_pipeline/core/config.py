"""Configuration classes for LoRA and other adapters."""

import torch
from dataclasses import dataclass, field
from typing import List, Optional
import math


@dataclass
class LoRAConfig:
    """Configuration for LoRA parameters."""
    
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    

    target_modules: Optional[List[str]] = None
    
    bias: str = "none"  
    
    # Task and initialization
    task_type: str = "CAUSAL_LM"
    init_lora_weights: str = "gaussian"  
    
    # Advanced variants
    use_rslora: bool = False
    use_dora: bool = False
    
    def __post_init__(self):
        """Validate and compute derived properties."""
        # Don't set default target_modules here - let model wrapper handle it
        # if self.target_modules is None:
        #     self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"Invalid bias setting: {self.bias}")
            
        if self.init_lora_weights not in ["gaussian", "kaiming", "xavier"]:
            raise ValueError(f"Invalid init method: {self.init_lora_weights}")
    
    @property
    def scaling(self) -> float:
        """Compute scaling factor based on variant."""
        if self.r == 0:
            return 0.0  # No scaling when rank=0 (pure inference)
        if self.use_rslora:
            return self.alpha / math.sqrt(self.r)
        else:
            return self.alpha / self.r


@dataclass 
class DoRAConfig(LoRAConfig):
    """Configuration for DoRA (Weight-Decomposed LoRA)."""
    
    use_dora: bool = field(default=True, init=False)
    magnitude_init: str = "ones"  
    
    def __post_init__(self):
        super().__post_init__()
        if self.magnitude_init not in ["ones", "random", "kaiming"]:
            raise ValueError(f"Invalid magnitude_init: {self.magnitude_init}")


@dataclass
class RSLoRAConfig(LoRAConfig):
    """Configuration for RSLoRA (Rank-Stabilized LoRA)."""
    
    use_rslora: bool = field(default=True, init=False)


@dataclass
class MultiAdapterConfig:
    """Configuration for multi-adapter management."""
    
    max_adapters: int = 8
    enable_routing: bool = False
    routing_hidden_size: Optional[int] = None
    adapter_fusion_method: str = "weighted_sum"
    
    # Attention-based routing configuration
    num_attention_heads: Optional[int] = None  # None = auto-determine
    attention_dropout: float = 0.1
    
    # Adapter weight initialization
    adapter_weight_init: str = "ones"  # "ones", "uniform", "normal", "learned"
    adapter_weight_init_std: float = 0.02
    
    # Device and loading configuration
    load_checkpoint_device: Optional[torch.device] = None  # None = auto-detect
    
    # Logging configuration
    enable_logging: bool = True
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    
    def __post_init__(self):
        if self.enable_routing and self.routing_hidden_size is None:
            raise ValueError("routing_hidden_size required when enable_routing=True")
            
        if self.adapter_fusion_method not in ["weighted_sum", "learned_routing", "attention"]:
            raise ValueError(f"Invalid fusion method: {self.adapter_fusion_method}")
            
        if self.adapter_weight_init not in ["ones", "uniform", "normal", "learned"]:
            raise ValueError(f"Invalid adapter_weight_init: {self.adapter_weight_init}")
            
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")
            
        # Auto-determine attention heads if not specified
        if self.num_attention_heads is None and self.enable_routing:
            # Default to hidden_size // 64, but at least 1 and at most 16
            if self.routing_hidden_size:
                self.num_attention_heads = max(1, min(16, self.routing_hidden_size // 64))
            else:
                self.num_attention_heads = 8  # Fallback default