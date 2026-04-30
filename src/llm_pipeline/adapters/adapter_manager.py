"""Multi-adapter management and routing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional, Dict, Any, Union
from ..core.base_module import BaseLoRAModule
from ..core.config import LoRAConfig, MultiAdapterConfig
from .dora import DoRAModule
from .rslora import RSLoRAModule


def _setup_logger(config: MultiAdapterConfig) -> logging.Logger:
    """Setup logger based on configuration."""
    logger = logging.getLogger("adapter_manager")
    logger.setLevel(getattr(logging, config.log_level))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


class AdapterRouter(nn.Module):
    """Router for selecting and combining multiple adapters"""
    def __init__(
        self,
        num_adapters: int,
        hidden_size: int,
        routing_method: str = "learned_routing",
        num_attention_heads: Optional[int] = None,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        self.num_adapters = num_adapters
        self.hidden_size = hidden_size
        self.routing_method = routing_method
        
        if routing_method == "learned_routing":
            self.router = nn.Linear(hidden_size, num_adapters)
        elif routing_method == "attention":
            # Auto-determine attention heads if not specified
            if num_attention_heads is None:
                num_attention_heads = max(1, min(16, hidden_size // 64))
            
            self.router = nn.MultiheadAttention(
                hidden_size, 
                num_heads=num_attention_heads,
                dropout=attention_dropout,
                batch_first=True
            )
            # Learnable adapter embeddings for attention
            self.adapter_embeddings = nn.Parameter(
                torch.randn(num_adapters, hidden_size)
            )
        else:
            raise ValueError(f"Unsupported routing method: {routing_method}")
    
    def forward(self, x: torch.Tensor, num_active_adapters: int) -> torch.Tensor:
        """Compute routing weights"""
        if self.routing_method == "learned_routing":
            # Simple learned routing
            routing_logits = self.router(x.mean(dim=1))  # Pool sequence dimension
            routing_weights = F.softmax(routing_logits[:, :num_active_adapters], dim=-1)
            return routing_weights
        
        elif self.routing_method == "attention":
            # Attention-based routing
            batch_size = x.size(0)
            
            # Use adapter embeddings as keys/values
            adapter_emb = self.adapter_embeddings[:num_active_adapters].unsqueeze(0)
            adapter_emb = adapter_emb.expand(batch_size, -1, -1)
            
            # Query is the input representation (pooled)
            query = x.mean(dim=1, keepdim=True)  # (batch, 1, hidden)
            
            # Compute attention
            attn_output, attn_weights = self.router(
                query, adapter_emb, adapter_emb
            )
            
            # Return attention weights
            return attn_weights.squeeze(1)  # (batch, num_adapters)
        
        else:
            raise ValueError(f"Unsupported routing method: {self.routing_method}")


class MultiAdapterLoRALinear(nn.Module):
    """LoRA Linear layer with multiple adapter support and routing"""
    def __init__(
        self,
        base_layer: nn.Linear,
        config: LoRAConfig,
        multi_config: Optional[MultiAdapterConfig] = None
    ):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.multi_config = multi_config or MultiAdapterConfig()
        
        # Store dimensions
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # Adapter storage
        self.adapters = nn.ModuleDict()
        self.adapter_weights = nn.ParameterDict()  # For weighted combination
        self.adapter_types = {}  # Track adapter types
        self.active_adapters = []
        
        # Setup logging
        self.logger = _setup_logger(self.multi_config) if self.multi_config.enable_logging else None
        
        # Routing mechanism
        self.router = None
        if self.multi_config.enable_routing:
            self._setup_routing()
    
    def _setup_routing(self):
        """Setup routing mechanism."""
        if self.multi_config.routing_hidden_size is None:
            raise ValueError("routing_hidden_size must be provided for routing")
        
        self.router = AdapterRouter(
            num_adapters=self.multi_config.max_adapters,
            hidden_size=self.multi_config.routing_hidden_size,
            routing_method=self.multi_config.adapter_fusion_method,
            num_attention_heads=self.multi_config.num_attention_heads,
            attention_dropout=self.multi_config.attention_dropout
        )
    
    def _initialize_adapter_weight(self, init_method: str, std: float = 0.02) -> torch.Tensor:
        """Initialize adapter weight based on configuration.
        
        Args:
            init_method: Initialization method ("ones", "uniform", "normal", "learned")
            std: Standard deviation for normal initialization
            
        Returns:
            Initialized weight tensor
        """
        if init_method == "ones":
            return torch.ones(1)
        elif init_method == "uniform":
            return torch.rand(1) * 2 - 1  # Uniform in [-1, 1]
        elif init_method == "normal":
            return torch.randn(1) * std + 1.0  # Normal around 1.0
        elif init_method == "learned":
            # Start with ones but allow full learning
            return nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
    
    def _get_device_for_loading(self) -> str:
        """Get appropriate device for loading checkpoints.
        
        Returns:
            Device string for loading
        """
        # Priority: config device > any parameter device > CPU
        if self.multi_config.load_checkpoint_device is not None:
            return str(self.multi_config.load_checkpoint_device)
            
        # Try to get device from any parameter
        for param in self.parameters():
            return str(param.device)
            
        # Fallback to CPU
        return 'cpu'
    
    def add_adapter(
        self, 
        adapter_name: str, 
        config: Optional[LoRAConfig] = None,
        adapter_type: str = "lora"
    ):
        """Add a new adapter"""
        if len(self.adapters) >= self.multi_config.max_adapters:
            raise ValueError(f"Maximum number of adapters ({self.multi_config.max_adapters}) reached")
            
        config = config or self.config
        
        # Create adapter based on type
        if adapter_type == "lora":
            adapter = BaseLoRAModule(self.in_features, self.out_features, config, adapter_name)
        elif adapter_type == "dora":
            if not hasattr(config, 'magnitude_init'):
                # Convert to DoRAConfig if needed
                from ..core.config import DoRAConfig
                config = DoRAConfig(**config.__dict__)
            adapter = DoRAModule(self.in_features, self.out_features, config, adapter_name)
        elif adapter_type == "rslora":
            if not hasattr(config, 'use_rslora') or not config.use_rslora:
                # Convert to RSLoRAConfig if needed
                from ..core.config import RSLoRAConfig
                config = RSLoRAConfig(**config.__dict__)
            adapter = RSLoRAModule(self.in_features, self.out_features, config, adapter_name)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
            
        self.adapters[adapter_name] = adapter
        self.adapter_weights[adapter_name] = nn.Parameter(
            self._initialize_adapter_weight(
                self.multi_config.adapter_weight_init,
                self.multi_config.adapter_weight_init_std
            )
        )
        self.adapter_types[adapter_name] = adapter_type
        
        if self.logger:
            self.logger.info(f"Added {adapter_type} adapter: {adapter_name}")
        elif self.multi_config.enable_logging:
            print(f"Added {adapter_type} adapter: {adapter_name}")
    
    def remove_adapter(self, adapter_name: str):
        """Remove an adapter"""
        if adapter_name in self.adapters:
            del self.adapters[adapter_name]
            del self.adapter_weights[adapter_name]
            del self.adapter_types[adapter_name]
            
            # Remove from active adapters if present
            if adapter_name in self.active_adapters:
                self.active_adapters.remove(adapter_name)
            
            if self.logger:
                self.logger.info(f"🗑️ Removed adapter: {adapter_name}")
            elif self.multi_config.enable_logging:
                print(f"🗑️ Removed adapter: {adapter_name}")
    
    def set_active_adapters(self, adapter_names: List[str], weights: Optional[List[float]] = None):
        """Set active adapters with optional weights"""
        # Validate adapter names
        valid_adapters = [name for name in adapter_names if name in self.adapters]
        if len(valid_adapters) != len(adapter_names):
            missing = set(adapter_names) - set(valid_adapters)
            if self.logger:
                self.logger.warning(f"Adapters not found: {missing}")
            elif self.multi_config.enable_logging:
                print(f"Warning: Adapters not found: {missing}")
        
        self.active_adapters = valid_adapters
        
        # Set weights if provided
        if weights:
            if len(weights) != len(valid_adapters):
                raise ValueError("Number of weights must match number of adapters")
            
            for name, weight in zip(valid_adapters, weights):
                self.adapter_weights[name].data = torch.tensor(weight, dtype=torch.float32)
    
    def get_active_adapters(self) -> List[str]:
        """Get list of currently active adapters"""
        return self.active_adapters.copy()
    
    def get_adapter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all adapters"""
        info = {}
        for name, adapter in self.adapters.items():
            info[name] = {
                "type": self.adapter_types[name],
                "rank": adapter.rank,
                "scaling": adapter.scaling,
                "parameters": adapter.get_parameter_count(),
                "active": name in self.active_adapters,
                "weight": self.adapter_weights[name].item()
            }
        return info
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-adapter support"""
        # Base layer output
        result = self.base_layer(x)
        
        if not self.active_adapters:
            return result
        
        # Compute adapter outputs
        adapter_outputs = []
        adapter_weights = []
        
        for adapter_name in self.active_adapters:
            if adapter_name in self.adapters:
                adapter = self.adapters[adapter_name]
                adapter_type = self.adapter_types[adapter_name]
                
                # Compute adapter output based on type
                if adapter_type == "dora":
                    adapter_output = adapter(x, self.base_layer.weight)
                else:
                    adapter_output = adapter(x)
                    
                adapter_outputs.append(adapter_output)
                adapter_weights.append(self.adapter_weights[adapter_name])
        
        # Combine adapter outputs
        if adapter_outputs:
            if self.multi_config.enable_routing and self.router is not None:
                # Use learned routing weights
                routing_weights = self.router(x, len(adapter_outputs))
                
                # Combine using routing weights
                combined_output = torch.zeros_like(adapter_outputs[0])
                for i, output in enumerate(adapter_outputs):
                    weight = routing_weights[:, i].unsqueeze(-1).unsqueeze(-1)
                    combined_output += weight * output
            else:
                # Use fixed weights
                combined_output = torch.zeros_like(adapter_outputs[0])
                total_weight = sum(weight.item() for weight in adapter_weights)
                
                for weight, output in zip(adapter_weights, adapter_outputs):
                    normalized_weight = weight / total_weight if total_weight > 0 else weight
                    combined_output += normalized_weight * output
            
            result += combined_output
            
        return result
    
    def get_routing_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get routing weights for analysis"""
        if self.multi_config.enable_routing and self.router is not None and self.active_adapters:
            return self.router(x, len(self.active_adapters))
        return None
    
    def save_adapter(self, adapter_name: str, path: str):
        """Save a specific adapter"""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter {adapter_name} not found")
        
        state_dict = {
            "adapter_state": self.adapters[adapter_name].state_dict(),
            "adapter_weight": self.adapter_weights[adapter_name].data,
            "adapter_type": self.adapter_types[adapter_name],
            "config": self.adapters[adapter_name].config
        }
        torch.save(state_dict, path)
    
    def load_adapter(self, adapter_name: str, path: str):
        """Load a specific adapter"""
        device = self._get_device_for_loading()
        checkpoint = torch.load(path, map_location=device)
        
        # Add adapter if it doesn't exist
        if adapter_name not in self.adapters:
            self.add_adapter(
                adapter_name, 
                checkpoint["config"], 
                checkpoint["adapter_type"]
            )
        
        # Load state
        self.adapters[adapter_name].load_state_dict(checkpoint["adapter_state"])
        self.adapter_weights[adapter_name].data = checkpoint["adapter_weight"]
    
    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"adapters={len(self.adapters)}, active={len(self.active_adapters)}, "
                f"routing={self.multi_config.enable_routing}")