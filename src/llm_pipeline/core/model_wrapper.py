"""Model wrapper for HuggingFace models with automatic LoRA injection."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Any
from .config import LoRAConfig
from .registry import ModelRegistry
# Import moved to avoid circular imports


class LoRAModelWrapper(nn.Module):
    """Wrapper for HuggingFace models with automatic LoRA injection"""
    
    def __init__(
        self,
        base_model: nn.Module,
        config: LoRAConfig,
        model_type: Optional[str] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.model_type = model_type or self._detect_model_type()
        
        # Get target modules
        if config.target_modules:
            self.target_modules = config.target_modules
        else:
            self.target_modules = ModelRegistry.get_target_modules(self.model_type)
        
        # LoRA module tracking
        self.lora_modules = {}
        self.modules_to_save = set(ModelRegistry.get_modules_to_save(self.model_type))
        
        # Model info
        self.model_info = ModelRegistry.get_model_info(self.model_type)
        
        # Apply LoRA
        self._inject_lora_layers()
        
        # Set requires_grad appropriately
        self._set_requires_grad()
        
        print(f"Applied LoRA to {self.model_type} model")
        print(f"Target modules: {self.target_modules}")
        print(f"Injected LoRA into {len(self.lora_modules)} modules")
    
    def _detect_model_type(self) -> str:
        """Auto-detect model type from class name"""
        class_name = self.base_model.__class__.__name__.lower()
        detected = ModelRegistry.detect_model_type(class_name)
        return detected or "unknown"
    
    def _should_replace_module(self, name: str, module: nn.Module) -> bool:
        """Check if a module should be replaced with LoRA version"""
        if not isinstance(module, nn.Linear):
            return False
            
        # Skip very small layers (embeddings, bias layers, etc.)
        if module.in_features < 8 or module.out_features < 8:
            return False
            
        # Check if module name matches target patterns
        module_name = name.split('.')[-1]  # Get last part of module name
        for target in self.target_modules:
            if target == module_name or target in module_name:
                return True
        return False
    
    def _inject_lora_layers(self):
        """Inject LoRA layers into the model"""
        # Import here to avoid circular imports
        from ..adapters.lora import LoRALinear
        
        replacements = []
        
        for name, module in self.base_model.named_modules():
            if self._should_replace_module(name, module):
                # Create LoRA-enhanced version
                lora_module = LoRALinear(module, self.config)
                replacements.append((name, lora_module))
        
        # Apply replacements
        for name, lora_module in replacements:
            self._replace_module(name, lora_module)
            self.lora_modules[name] = lora_module
            print(f"Injected LoRA into: {name}")
    
    def _replace_module(self, module_name: str, new_module: nn.Module):
        """Replace a module in the model hierarchy"""
        module_path = module_name.split('.')
        parent = self.base_model
        
        # Navigate to parent module
        for part in module_path[:-1]:
            if hasattr(parent, part):
                parent = getattr(parent, part)
            else:
                # Handle modules in ModuleList/ModuleDict
                try:
                    if part.isdigit():
                        parent = parent[int(part)]
                    else:
                        parent = parent[part]
                except (KeyError, IndexError, TypeError):
                    raise AttributeError(f"Cannot navigate to {module_name}")
        
        # Replace the target module
        final_name = module_path[-1]
        if hasattr(parent, final_name):
            setattr(parent, final_name, new_module)
        else:
            # Handle ModuleList/ModuleDict cases
            try:
                if final_name.isdigit():
                    parent[int(final_name)] = new_module
                else:
                    parent[final_name] = new_module
            except (KeyError, IndexError, TypeError):
                raise AttributeError(f"Cannot replace module {module_name}")
    
    def _set_requires_grad(self):
        """Set requires_grad for all parameters"""
        # First, freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Enable gradients for LoRA parameters
        for module in self.lora_modules.values():
            for param in module.parameters():
                if "lora_" in param._get_name() if hasattr(param, '_get_name') else True:
                    param.requires_grad = True
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model"""
        return self.base_model(*args, **kwargs)
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict containing only LoRA parameters"""
        lora_state_dict = {}
        for name, module in self.lora_modules.items():
            module_state = module.state_dict()
            for param_name, param in module_state.items():
                if "lora_" in param_name:
                    lora_state_dict[f"{name}.{param_name}"] = param.clone()
        return lora_state_dict
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """Load LoRA parameters from state dict"""
        missing_keys = []
        unexpected_keys = []
        
        for name, param in state_dict.items():
            try:
                module_name, param_name = name.rsplit('.', 1)
                if module_name in self.lora_modules:
                    module = self.lora_modules[module_name]
                    if hasattr(module, param_name):
                        getattr(module, param_name).data.copy_(param)
                    else:
                        unexpected_keys.append(name)
                else:
                    missing_keys.append(name)
            except ValueError:
                unexpected_keys.append(name)
        
        if strict and (missing_keys or unexpected_keys):
            error_msg = []
            if missing_keys:
                error_msg.append(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                error_msg.append(f"Unexpected keys: {unexpected_keys}")
            raise RuntimeError("Error loading LoRA state dict:\n" + "\n".join(error_msg))
        
        return missing_keys, unexpected_keys
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights to a file"""
        torch.save(self.get_lora_state_dict(), path)
    
    def load_lora_weights(self, path: str, strict: bool = True):
        """Load LoRA weights from a file"""
        state_dict = torch.load(path, map_location='cpu')
        return self.load_lora_state_dict(state_dict, strict=strict)
    
    def add_adapter(self, adapter_name: str, config: Optional[LoRAConfig] = None):
        """Add a new adapter to all LoRA modules"""
        config = config or self.config
        for module in self.lora_modules.values():
            if hasattr(module, 'add_adapter'):
                module.add_adapter(adapter_name, config)
    
    def set_active_adapters(self, adapter_names: List[str]):
        """Set active adapters for all LoRA modules"""
        for module in self.lora_modules.values():
            if hasattr(module, 'set_active_adapters'):
                module.set_active_adapters(adapter_names)
    
    def merge_adapters(
        self, 
        adapter_names: Optional[List[str]] = None,
        strategy: str = "linear",
        weights: Optional[List[float]] = None,
        alpha: float = 1.0,
        density: float = 0.5,
        majority_sign_method: str = "total",
        save_merged: bool = True
    ):
        """Merge LoRA adapters using specified strategy.
        
        Args:
            adapter_names: List of adapter names to merge (None = merge all active)
            strategy: Merging strategy ("linear", "ties", "dare_linear", "dare_ties", "additive", "weighted_sum", "scaled_additive", "maximum", "mean")
            weights: Weights for weighted strategies
            alpha: Scaling factor for scaled_additive strategy
            density: Fraction of weights to retain (TIES/DARE strategies)
            majority_sign_method: Sign consensus method ("total" or "frequency")
            save_merged: Whether to save merged adapter
            
        Returns:
            Name of merged adapter if save_merged=True
        """
        from ..adapters.merging import LoRAAdapterMerger, MergeStrategy
        
        # Convert strategy string to enum
        try:
            merge_strategy = MergeStrategy(strategy)
        except ValueError:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {[s.value for s in MergeStrategy]}")
        
        # Get active adapters if none specified
        if adapter_names is None:
            adapter_names = []
            for module in self.lora_modules.values():
                if hasattr(module, 'active_adapters'):
                    adapter_names.extend(module.active_adapters)
            adapter_names = list(set(adapter_names))  # Remove duplicates
            
        if not adapter_names:
            raise ValueError("No adapters to merge")
            
        # Create merger and perform merge
        merger = LoRAAdapterMerger(self)
        return merger.merge_adapters(
            adapter_names, merge_strategy, weights, alpha, 
            density, majority_sign_method, save_merged
        )
    
    def merge_with_base(
        self,
        adapter_names: Optional[List[str]] = None,
        strategy: str = "linear",
        weights: Optional[List[float]] = None,
        alpha: float = 1.0,
        density: float = 0.5,
        majority_sign_method: str = "total"
    ):
        """Merge LoRA adapters directly into base model weights.
        
        Args:
            adapter_names: List of adapter names to merge (None = merge all active)
            strategy: Merging strategy ("linear", "ties", "dare_linear", "dare_ties", "additive", "weighted_sum", "scaled_additive", "maximum", "mean")
            weights: Weights for weighted strategies
            alpha: Scaling factor for scaled_additive strategy
            density: Fraction of weights to retain (TIES/DARE strategies)
            majority_sign_method: Sign consensus method ("total" or "frequency")
        """
        from ..adapters.merging import LoRAAdapterMerger, MergeStrategy
        
        # Convert strategy string to enum
        try:
            merge_strategy = MergeStrategy(strategy)
        except ValueError:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {[s.value for s in MergeStrategy]}")
        
        # Get active adapters if none specified
        if adapter_names is None:
            adapter_names = []
            for module in self.lora_modules.values():
                if hasattr(module, 'active_adapters'):
                    adapter_names.extend(module.active_adapters)
            adapter_names = list(set(adapter_names))  # Remove duplicates
            
        if not adapter_names:
            raise ValueError("No adapters to merge")
            
        # Create merger and perform merge
        merger = LoRAAdapterMerger(self)
        merger.merge_with_base(adapter_names, merge_strategy, weights, alpha, density, majority_sign_method)
    
    def unmerge_adapters(self):
        """Unmerge LoRA adapters from base weights."""
        from ..adapters.merging import LoRAAdapterMerger
        
        merger = LoRAAdapterMerger(self)
        merger.unmerge_adapters()
    
    def print_trainable_parameters(self):
        """Print the number of trainable parameters"""
        trainable_params = 0
        all_param = 0
        lora_params = 0
        
        for name, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            
            if param.requires_grad:
                trainable_params += num_params
                if "lora_" in name:
                    lora_params += num_params
        
        print(f"Parameter Summary:")
        print(f"Trainable params: {trainable_params:,}")
        print(f"All params: {all_param:,}")
        print(f"LoRA params: {lora_params:,}")
        print(f"Trainable%: {100 * trainable_params / all_param:.2f}%")
        print(f"LoRA efficiency: {lora_params / trainable_params * 100:.1f}% of trainable params")
    
    def get_memory_footprint(
        self, 
        optimizer: Optional[torch.optim.Optimizer] = None,
        precision: str = "fp32"
    ) -> Dict[str, Any]:
        """Get memory footprint information with optional optimizer.
        
        Args:
            optimizer: Optional optimizer instance for accurate memory estimation
            precision: Model precision ("fp32", "fp16", "bf16")
            
        Returns:
            Dictionary with memory breakdown
        """
        from ..utils.memory import get_memory_footprint
        
        return get_memory_footprint(self, optimizer, precision)
    
    def estimate_training_memory(
        self,
        optimizer_type: str = "adam",
        batch_size: int = 1,
        sequence_length: int = 512,
        precision: str = "fp32"
    ) -> Dict[str, float]:
        """Estimate training memory with specific optimizer type.
        
        Args:
            optimizer_type: Type of optimizer ("adam", "sgd", "rmsprop", etc.)
            batch_size: Training batch size
            sequence_length: Input sequence length
            precision: Model precision
            
        Returns:
            Dictionary with memory estimates
        """
        from ..utils.optimizer_memory import estimate_training_memory_with_optimizer
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return estimate_training_memory_with_optimizer(
            trainable_params, optimizer_type, batch_size, sequence_length, precision
        )
    
    def compare_optimizer_memory(
        self,
        optimizer_types: Optional[list] = None,
        precision: str = "fp32"
    ) -> Dict[str, Dict[str, float]]:
        """Compare memory usage across different optimizer types.
        
        Args:
            optimizer_types: List of optimizer types to compare
            precision: Model precision
            
        Returns:
            Dictionary with memory usage for each optimizer
        """
        from ..utils.optimizer_memory import compare_optimizer_memory_usage
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return compare_optimizer_memory_usage(trainable_params, optimizer_types, precision)


def create_lora_model(
    model: nn.Module,
    config: LoRAConfig,
    model_type: Optional[str] = None
) -> LoRAModelWrapper:
    """Factory function to create LoRA-enhanced model"""
    return LoRAModelWrapper(model, config, model_type)