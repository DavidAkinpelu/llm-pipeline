"""Configuration classes for LoRA and other adapters."""

import torch
from dataclasses import dataclass, field
from typing import List, Optional, Literal
import math


@dataclass
class LoRAConfig:
    """Configuration for LoRA parameters.
    
    Args:
        r: LoRA rank (1-64 recommended)
        alpha: LoRA alpha parameter (typically 2*r)
        dropout: Dropout rate (0.0-1.0)
        target_modules: List of module names to apply LoRA to
        bias: Bias type - "none", "all", or "lora_only"
        task_type: Task type - "CAUSAL_LM", "SEQ_2_SEQ_LM", or "QUESTION_ANS"
        init_lora_weights: Initialization method - "gaussian", "kaiming", or "xavier"
        use_rslora: Whether to use RSLoRA variant
        use_dora: Whether to use DoRA variant
    """
    
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    
    # Use Literal types for clear valid values
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: Literal["CAUSAL_LM", "SEQ_2_SEQ_LM", "QUESTION_ANS"] = "CAUSAL_LM"
    init_lora_weights: Literal["gaussian", "kaiming", "xavier"] = "gaussian"
    
    # Advanced variants
    use_rslora: bool = False
    use_dora: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate bias setting
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"Invalid bias setting: {self.bias}. Must be 'none', 'all', or 'lora_only'")
        
        # Validate task type
        if self.task_type not in ["CAUSAL_LM", "SEQ_2_SEQ_LM", "QUESTION_ANS"]:
            raise ValueError(f"Invalid task_type: {self.task_type}. Must be 'CAUSAL_LM', 'SEQ_2_SEQ_LM', or 'QUESTION_ANS'")
        
        # Validate initialization method
        if self.init_lora_weights not in ["gaussian", "kaiming", "xavier"]:
            raise ValueError(f"Invalid init_lora_weights: {self.init_lora_weights}. Must be 'gaussian', 'kaiming', or 'xavier'")
    
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
    """Configuration for DoRA (Weight-Decomposed LoRA).
    
    Args:
        magnitude_init: Magnitude initialization method - "ones", "random", or "kaiming"
        All other parameters inherited from LoRAConfig
    """
    
    use_dora: bool = field(default=True, init=False)
    magnitude_init: Literal["ones", "random", "kaiming"] = "ones"
    
    def __post_init__(self):
        """Validate DoRA-specific parameters."""
        super().__post_init__()  # Call parent validation
        
        # Validate magnitude initialization
        if self.magnitude_init not in ["ones", "random", "kaiming"]:
            raise ValueError(f"Invalid magnitude_init: {self.magnitude_init}. Must be 'ones', 'random', or 'kaiming'")


@dataclass
class RSLoRAConfig(LoRAConfig):
    """Configuration for RSLoRA (Rank-Stabilized LoRA).
    
    RSLoRA uses a different scaling factor: alpha / sqrt(r) instead of alpha / r.
    This provides better stability and performance for certain tasks.
    
    All parameters are inherited from LoRAConfig.
    """
    
    use_rslora: bool = field(default=True, init=False)


@dataclass
class MultiAdapterConfig:
    """Configuration for multi-adapter management.
    
    Args:
        max_adapters: Maximum number of adapters to manage
        enable_routing: Whether to enable adapter routing
        routing_hidden_size: Hidden size for routing (required if enable_routing=True)
        adapter_fusion_method: Fusion method - "weighted_sum", "learned_routing", or "attention"
        num_attention_heads: Number of attention heads (None = auto-determine)
        attention_dropout: Attention dropout rate
        adapter_weight_init: Weight initialization - "ones", "uniform", "normal", or "learned"
        adapter_weight_init_std: Standard deviation for weight initialization
        load_checkpoint_device: Device for loading checkpoints (None = auto-detect)
        enable_logging: Whether to enable logging
        log_level: Log level - "DEBUG", "INFO", "WARNING", or "ERROR"
    """
    
    max_adapters: int = 8
    enable_routing: bool = False
    routing_hidden_size: Optional[int] = None
    adapter_fusion_method: Literal["weighted_sum", "learned_routing", "attention"] = "weighted_sum"
    
    # Attention-based routing configuration
    num_attention_heads: Optional[int] = None  # None = auto-determine
    attention_dropout: float = 0.1
    
    # Adapter weight initialization
    adapter_weight_init: Literal["ones", "uniform", "normal", "learned"] = "ones"
    adapter_weight_init_std: float = 0.02
    
    # Device and loading configuration
    load_checkpoint_device: Optional[torch.device] = None  # None = auto-detect
    
    # Logging configuration
    enable_logging: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    def __post_init__(self):
        if self.enable_routing and self.routing_hidden_size is None:
            raise ValueError("routing_hidden_size required when enable_routing=True")
            
        # Auto-determine attention heads if not specified
        if self.num_attention_heads is None and self.enable_routing:
            # Default to hidden_size // 64, but at least 1 and at most 16
            if self.routing_hidden_size:
                self.num_attention_heads = max(1, min(16, self.routing_hidden_size // 64))
            else:
                self.num_attention_heads = 8  # Fallback default