"""Adapter merging functionality for LoRA adapters."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum


class MergeStrategy(Enum):
    """Available adapter merging strategies."""
    # Basic strategies
    ADDITIVE = "additive"  # Simple addition of adapter weights
    WEIGHTED_SUM = "weighted_sum"  # Weighted combination of adapters
    SCALED_ADDITIVE = "scaled_additive"  # Scaled addition with alpha
    MAXIMUM = "maximum"  # Element-wise maximum
    MEAN = "mean"  # Element-wise mean
    
    # Advanced LoRA-specific strategies (from mergekit/LoRAX)
    TIES = "ties"  # TIES: Task-specific Interference Elimination
    DARE_LINEAR = "dare_linear"  # DARE with linear combination
    DARE_TIES = "dare_ties"  # DARE with TIES consensus


@dataclass
class MergeConfig:
    """Configuration for adapter merging."""
    strategy: MergeStrategy = MergeStrategy.WEIGHTED_SUM  # Default to WEIGHTED_SUM
    weights: Optional[List[float]] = None  # Weights for weighted strategies
    alpha: float = 1.0  # Scaling factor for scaled_additive
    save_merged_adapter: bool = True  # Whether to save merged adapter
    merged_adapter_name: str = "merged"  # Name for merged adapter
    preserve_originals: bool = True  # Whether to keep original adapters
    
    # Advanced strategy parameters
    density: float = 0.5  # Fraction of weights to retain (TIES/DARE)
    majority_sign_method: str = "total"  # Sign consensus method: "total" or "frequency"
    
    # Device and infrastructure settings
    device: Optional[torch.device] = None  # Device for merged adapters (None = auto-detect)
    
    # Merged adapter configuration
    merged_adapter_rank: Optional[int] = None  # Rank for merged adapter (None = auto-determine)
    merged_adapter_alpha: float = 1.0  # Alpha for merged adapter
    merged_adapter_dropout: float = 0.0  # Dropout for merged adapter
    merged_adapter_bias: str = "none"  # Bias setting for merged adapter
    allow_rank_zero: bool = True  # Whether to allow rank=0 merged adapters
    
    # DARE-specific settings
    dare_rescaling_factor: Optional[float] = None  # Custom rescaling factor (None = use density)
    dare_enable_rescaling: bool = True  # Whether to apply rescaling after pruning


class AdapterMerger:
    """Merger for LoRA adapters."""
    
    def __init__(self, config: Optional[MergeConfig] = None):
        """Initialize adapter merger.
        
        Args:
            config: Merging configuration
        """
        self.config = config or MergeConfig()
        
    def merge_adapters(
        self,
        adapters: Dict[str, torch.Tensor],
        strategy: Optional[MergeStrategy] = None,
        weights: Optional[List[float]] = None,
        alpha: Optional[float] = None,
        density: Optional[float] = None,
        majority_sign_method: Optional[str] = None
    ) -> torch.Tensor:
        """Merge multiple adapters using specified strategy.
        
        Args:
            adapters: Dictionary mapping adapter names to their weights
            strategy: Merging strategy to use
            weights: Weights for weighted strategies
            alpha: Scaling factor for scaled_additive strategy
            density: Fraction of weights to retain (TIES/DARE)
            majority_sign_method: Sign consensus method for TIES/DARE_TIES
            
        Returns:
            Merged adapter weights
        """
        strategy = self.config.strategy if strategy is None else strategy
        weights = self.config.weights if weights is None else weights
        alpha = self.config.alpha if alpha is None else alpha
        density = self.config.density if density is None else density
        majority_sign_method = (
            self.config.majority_sign_method
            if majority_sign_method is None
            else majority_sign_method
        )
        
        if not adapters:
            raise ValueError("No adapters provided for merging")
            
        adapter_list = list(adapters.values())
        adapter_names = list(adapters.keys())
        
        # Basic strategies
        if strategy == MergeStrategy.ADDITIVE:
            return self._merge_additive(adapter_list)
        elif strategy == MergeStrategy.WEIGHTED_SUM:
            return self._merge_weighted_sum(adapter_list, weights)
        elif strategy == MergeStrategy.SCALED_ADDITIVE:
            return self._merge_scaled_additive(adapter_list, alpha)
        elif strategy == MergeStrategy.MAXIMUM:
            return self._merge_maximum(adapter_list)
        elif strategy == MergeStrategy.MEAN:
            return self._merge_mean(adapter_list)
        
        # Advanced LoRA-specific strategies
        elif strategy == MergeStrategy.TIES:
            return self._merge_ties(adapter_list, weights, density, majority_sign_method)
        elif strategy == MergeStrategy.DARE_LINEAR:
            return self._merge_dare_linear(adapter_list, weights, density)
        elif strategy == MergeStrategy.DARE_TIES:
            return self._merge_dare_ties(adapter_list, weights, density, majority_sign_method)
        else:
            raise ValueError(f"Unknown merging strategy: {strategy}")
            
    def _merge_additive(self, adapters: List[torch.Tensor]) -> torch.Tensor:
        """Merge adapters by simple addition.
        
        Args:
            adapters: List of adapter weight tensors
            
        Returns:
            Merged adapter weights
        """
        if len(adapters) == 1:
            return adapters[0]
            
        merged = adapters[0].clone()
        for adapter in adapters[1:]:
            merged += adapter
            
        return merged
        
    def _merge_weighted_sum(
        self, 
        adapters: List[torch.Tensor], 
        weights: Optional[List[float]]
    ) -> torch.Tensor:
        """Merge adapters using weighted sum.
        
        Args:
            adapters: List of adapter weight tensors
            weights: List of weights for each adapter
            
        Returns:
            Merged adapter weights
        """
        if len(adapters) == 1:
            return adapters[0]
            
        if weights is None:
            # Equal weights if not specified
            weights = [1.0 / len(adapters)] * len(adapters)
        elif len(weights) != len(adapters):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of adapters ({len(adapters)})")
            
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero")
        weights = [w / total_weight for w in weights]
        
        merged = torch.zeros_like(adapters[0])
        for adapter, weight in zip(adapters, weights):
            merged += weight * adapter
            
        return merged
        
    def _merge_scaled_additive(
        self, 
        adapters: List[torch.Tensor], 
        alpha: float
    ) -> torch.Tensor:
        """Merge adapters using scaled addition.
        
        Args:
            adapters: List of adapter weight tensors
            alpha: Scaling factor
            
        Returns:
            Merged adapter weights
        """
        merged = self._merge_additive(adapters)
        return alpha * merged
        
    def _merge_maximum(self, adapters: List[torch.Tensor]) -> torch.Tensor:
        """Merge adapters using element-wise maximum.
        
        Args:
            adapters: List of adapter weight tensors
            
        Returns:
            Merged adapter weights
        """
        if len(adapters) == 1:
            return adapters[0]
            
        # Stack adapters and take maximum along first dimension
        stacked = torch.stack(adapters, dim=0)
        return torch.max(stacked, dim=0)[0]
        
    def _merge_mean(self, adapters: List[torch.Tensor]) -> torch.Tensor:
        """Merge adapters using element-wise mean.
        
        Args:
            adapters: List of adapter weight tensors
            
        Returns:
            Merged adapter weights
        """
        if len(adapters) == 1:
            return adapters[0]
            
        # Stack adapters and take mean along first dimension
        stacked = torch.stack(adapters, dim=0)
        return torch.mean(stacked, dim=0)
        
        
    def _merge_ties(
        self, 
        adapters: List[torch.Tensor], 
        weights: Optional[List[float]], 
        density: float,
        majority_sign_method: str
    ) -> torch.Tensor:
        """TIES merging: Task-specific Interference Elimination.
        
        Based on: https://arxiv.org/abs/2306.01708
        
        Args:
            adapters: List of adapter weight tensors (task vectors)
            weights: List of weights for each adapter
            density: Fraction of weights to retain after sparsification
            majority_sign_method: Method for sign consensus ("total" or "frequency")
            
        Returns:
            Merged adapter weights
        """
        if len(adapters) == 1:
            return adapters[0]
            
        if weights is None:
            weights = [1.0] * len(adapters)
        elif len(weights) != len(adapters):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of adapters ({len(adapters)})")
            
        # Step 1: Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero")
        weights = [w / total_weight for w in weights]
        
        # Step 2: Resolve sign conflicts and sparsify
        merged = self._resolve_interference(adapters, weights, density, majority_sign_method)
        
        return merged
        
    def _merge_dare_linear(
        self, 
        adapters: List[torch.Tensor], 
        weights: Optional[List[float]], 
        density: float
    ) -> torch.Tensor:
        """DARE (Linear): Drop And REscale with linear combination.
        
        Based on: https://arxiv.org/abs/2311.03099
        
        Args:
            adapters: List of adapter weight tensors
            weights: List of weights for each adapter
            density: Fraction of weights to retain after pruning
            
        Returns:
            Merged adapter weights
        """
        if len(adapters) == 1:
            return adapters[0]
            
        if weights is None:
            weights = [1.0] * len(adapters)
        elif len(weights) != len(adapters):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of adapters ({len(adapters)})")
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero")
        weights = [w / total_weight for w in weights]
        
        # Apply DARE pruning and rescaling to each adapter
        dare_adapters = []
        for adapter in adapters:
            pruned = self._dare_prune_and_rescale(adapter, density, self.config)
            dare_adapters.append(pruned)
            
        # Linear combination of DARE-processed adapters
        merged = torch.zeros_like(adapters[0])
        for adapter, weight in zip(dare_adapters, weights):
            merged += weight * adapter
            
        return merged
        
    def _merge_dare_ties(
        self, 
        adapters: List[torch.Tensor], 
        weights: Optional[List[float]], 
        density: float,
        majority_sign_method: str
    ) -> torch.Tensor:
        """DARE (TIES): DARE pruning with TIES sign consensus.
        
        Args:
            adapters: List of adapter weight tensors
            weights: List of weights for each adapter
            density: Fraction of weights to retain after pruning
            majority_sign_method: Method for sign consensus
            
        Returns:
            Merged adapter weights
        """
        if len(adapters) == 1:
            return adapters[0]
            
        if weights is None:
            weights = [1.0] * len(adapters)
        elif len(weights) != len(adapters):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of adapters ({len(adapters)})")
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero")
        weights = [w / total_weight for w in weights]
        
        # Apply DARE pruning and rescaling to each adapter
        dare_adapters = []
        for adapter in adapters:
            pruned = self._dare_prune_and_rescale(adapter, density, self.config)
            dare_adapters.append(pruned)
            
        # Apply TIES sign consensus to DARE-processed adapters
        merged = self._resolve_interference(dare_adapters, weights, 1.0, majority_sign_method)
        
        return merged
        
    def _dare_prune_and_rescale(self, adapter: torch.Tensor, density: float, config: MergeConfig) -> torch.Tensor:
        """Apply DARE pruning and rescaling to a single adapter.
        
        Args:
            adapter: Adapter weight tensor
            density: Fraction of weights to retain
            config: Merge configuration with DARE settings
            
        Returns:
            Pruned and rescaled adapter
        """
        # Random pruning mask
        mask = torch.rand_like(adapter) < density
        
        # Apply pruning
        pruned = adapter * mask
        
        # Apply rescaling if enabled
        if config.dare_enable_rescaling and density > 0:
            rescaling_factor = config.dare_rescaling_factor if config.dare_rescaling_factor is not None else density
            pruned = pruned / rescaling_factor
            
        return pruned
        
    def _resolve_interference(
        self, 
        adapters: List[torch.Tensor], 
        weights: List[float], 
        density: float,
        majority_sign_method: str
    ) -> torch.Tensor:
        """Resolve interference between adapters using sign consensus.
        
        Args:
            adapters: List of adapter weight tensors
            weights: List of weights for each adapter
            density: Fraction of weights to retain after sparsification
            majority_sign_method: Method for sign consensus
            
        Returns:
            Merged adapter weights with resolved interference
        """
        if len(adapters) == 1:
            return adapters[0]
            
        # Stack adapters for processing
        stacked = torch.stack(adapters, dim=0)  # Shape: (n_adapters, ...)
        
        # Compute sign consensus
        signs = torch.sign(stacked)
        
        if majority_sign_method == "total":
            # Use total magnitude for sign consensus
            magnitudes = torch.abs(stacked)
            weighted_magnitudes = magnitudes * torch.tensor(weights, device=stacked.device).view(-1, *([1] * (stacked.dim() - 1)))
            consensus_signs = torch.sign(torch.sum(signs * weighted_magnitudes, dim=0))
        elif majority_sign_method == "frequency":
            # Use frequency-based sign consensus
            positive_count = torch.sum(signs > 0, dim=0)
            negative_count = torch.sum(signs < 0, dim=0)
            consensus_signs = torch.where(positive_count >= negative_count, 1.0, -1.0)
        else:
            raise ValueError(f"Unknown majority_sign_method: {majority_sign_method}")
            
        # Compute weighted average magnitudes
        weighted_magnitudes = torch.abs(stacked) * torch.tensor(weights, device=stacked.device).view(-1, *([1] * (stacked.dim() - 1)))
        avg_magnitudes = torch.sum(weighted_magnitudes, dim=0)
        
        # Apply sign consensus
        merged = consensus_signs * avg_magnitudes
        
        # Apply density-based sparsification
        if density < 1.0:
            # Keep top-k% of weights by magnitude
            flat_merged = merged.flatten()
            k = int(density * flat_merged.numel())
            
            if k > 0:
                # Get top-k indices
                _, top_indices = torch.topk(torch.abs(flat_merged), k)
                
                # Create mask
                mask = torch.zeros_like(flat_merged)
                mask[top_indices] = 1.0
                mask = mask.reshape(merged.shape)
                
                # Apply mask
                merged = merged * mask
                
        return merged


class LoRAAdapterMerger:
    """High-level merger for LoRA model adapters."""
    
    def __init__(self, model_wrapper, config: Optional[MergeConfig] = None):
        """Initialize LoRA adapter merger.
        
        Args:
            model_wrapper: LoRAModelWrapper instance
            config: Merging configuration
        """
        self.model_wrapper = model_wrapper
        self.config = config or MergeConfig()
        self.merger = AdapterMerger(self.config)
        self.original_weights = {}  # Store original weights for unmerging
        
    def _get_device_for_tensor(self, adapter) -> torch.device:
        """Get appropriate device for tensor creation.
        
        Args:
            adapter: LoRA adapter module
            
        Returns:
            Device to use for tensor creation
        """
        # Priority: config device > adapter device > model device > CPU
        if self.config.device is not None:
            return self.config.device
            
        if hasattr(adapter, 'lora_A') and adapter.lora_A is not None:
            return adapter.lora_A.device
            
        # Try to get device from model wrapper
        if hasattr(self.model_wrapper, 'device'):
            return self.model_wrapper.device
            
        # Try to get device from any parameter in the model
        for param in self.model_wrapper.parameters():
            return param.device
            
        # Fallback to CPU
        return torch.device('cpu')
        
    def merge_adapters(
        self,
        adapter_names: List[str],
        strategy: Optional[MergeStrategy] = None,
        weights: Optional[List[float]] = None,
        alpha: Optional[float] = None,
        density: Optional[float] = None,
        majority_sign_method: Optional[str] = None,
        save_merged: Optional[bool] = None
    ) -> str:
        """Merge multiple adapters in the model.
        
        Args:
            adapter_names: List of adapter names to merge
            strategy: Merging strategy to use
            weights: Weights for weighted strategies
            alpha: Scaling factor for scaled_additive strategy
            density: Fraction of weights to retain (TIES/DARE strategies)
            majority_sign_method: Sign consensus method ("total" or "frequency")
            save_merged: Whether to save merged adapter
            
        Returns:
            Name of the merged adapter
        """
        save_merged = save_merged if save_merged is not None else self.config.save_merged_adapter
        
        if not adapter_names:
            raise ValueError("No adapter names provided")
            
        # Collect adapters from all LoRA modules
        merged_adapters = {}
        
        for module_name, lora_module in self.model_wrapper.lora_modules.items():
            if not hasattr(lora_module, 'lora_adapters'):
                continue
                
            # Get adapter weights for this module
            adapter_weights = {}
            for adapter_name in adapter_names:
                if adapter_name in lora_module.lora_adapters:
                    adapter = lora_module.lora_adapters[adapter_name]
                    # Get delta weights (B @ A)
                    if hasattr(adapter, 'get_delta_weight'):
                        adapter_weights[adapter_name] = adapter.get_delta_weight()
                    else:
                        # Fallback: manually compute B @ A
                        if adapter.lora_A is not None and adapter.lora_B is not None:
                            adapter_weights[adapter_name] = adapter.lora_B @ adapter.lora_A
                        else:
                            # Rank=0 case - zero contribution
                            device = self._get_device_for_tensor(adapter)
                            adapter_weights[adapter_name] = torch.zeros(
                                adapter.out_features, adapter.in_features,
                                device=device
                            )
                            
            if adapter_weights:
                # Merge adapters for this module
                merged_weight = self.merger.merge_adapters(
                    adapter_weights, strategy, weights, alpha,
                    self.config.density, self.config.majority_sign_method
                )
                merged_adapters[module_name] = merged_weight
                
        if not merged_adapters:
            raise ValueError("No adapters found to merge")
            
        # Create merged adapter name
        merged_name = self.config.merged_adapter_name
        if merged_name in [name for module in self.model_wrapper.lora_modules.values() 
                          if hasattr(module, 'lora_adapters') 
                          for name in module.lora_adapters.keys()]:
            # Add suffix if name already exists
            counter = 1
            while f"{merged_name}_{counter}" in [name for module in self.model_wrapper.lora_modules.values() 
                                               if hasattr(module, 'lora_adapters') 
                                               for name in module.lora_adapters.keys()]:
                counter += 1
            merged_name = f"{merged_name}_{counter}"
            
        # Add merged adapter to model
        if save_merged:
            self._add_merged_adapter(merged_adapters, merged_name)
            
        return merged_name
        
    def merge_with_base(
        self,
        adapter_names: List[str],
        strategy: Optional[MergeStrategy] = None,
        weights: Optional[List[float]] = None,
        alpha: Optional[float] = None,
        density: Optional[float] = None,
        majority_sign_method: Optional[str] = None
    ):
        """Merge adapters directly into base model weights.
        
        Args:
            adapter_names: List of adapter names to merge
            strategy: Merging strategy to use
            weights: Weights for weighted_sum strategy
            alpha: Scaling factor for scaled_additive strategy
        """
        # Store original weights if not already stored
        if not self.original_weights:
            self._store_original_weights()
            
        # Get merged adapter weights
        merged_adapters = {}
        
        for module_name, lora_module in self.model_wrapper.lora_modules.items():
            if not hasattr(lora_module, 'lora_adapters'):
                continue
                
            # Get adapter weights for this module
            adapter_weights = {}
            for adapter_name in adapter_names:
                if adapter_name in lora_module.lora_adapters:
                    adapter = lora_module.lora_adapters[adapter_name]
                    if hasattr(adapter, 'get_delta_weight'):
                        adapter_weights[adapter_name] = adapter.get_delta_weight()
                    else:
                        if adapter.lora_A is not None and adapter.lora_B is not None:
                            adapter_weights[adapter_name] = adapter.lora_B @ adapter.lora_A
                        else:
                            device = self._get_device_for_tensor(adapter)
                            adapter_weights[adapter_name] = torch.zeros(
                                adapter.out_features, adapter.in_features,
                                device=device
                            )
                            
            if adapter_weights:
                # Merge adapters for this module
                merged_weight = self.merger.merge_adapters(
                    adapter_weights, strategy, weights, alpha,
                    self.config.density, self.config.majority_sign_method
                )
                merged_adapters[module_name] = merged_weight
                
        # Apply merged weights to base model
        self._apply_merged_weights(merged_adapters)
        
    def unmerge_adapters(self):
        """Restore original base model weights."""
        if not self.original_weights:
            raise ValueError("No original weights stored. Cannot unmerge.")
            
        # Restore original weights
        for module_name, original_weight in self.original_weights.items():
            if module_name in self.model_wrapper.lora_modules:
                lora_module = self.model_wrapper.lora_modules[module_name]
                if hasattr(lora_module, 'base_layer'):
                    lora_module.base_layer.weight.data = original_weight.clone()
                    
        # Clear stored weights
        self.original_weights.clear()
        
    def _store_original_weights(self):
        """Store original base model weights."""
        for module_name, lora_module in self.model_wrapper.lora_modules.items():
            if hasattr(lora_module, 'base_layer'):
                self.original_weights[module_name] = lora_module.base_layer.weight.data.clone()
                
    def _apply_merged_weights(self, merged_adapters: Dict[str, torch.Tensor]):
        """Apply merged weights to base model.
        
        Args:
            merged_adapters: Dictionary mapping module names to merged weights
        """
        for module_name, merged_weight in merged_adapters.items():
            if module_name in self.model_wrapper.lora_modules:
                lora_module = self.model_wrapper.lora_modules[module_name]
                if hasattr(lora_module, 'base_layer'):
                    # Add merged adapter weights to base weights
                    lora_module.base_layer.weight.data += merged_weight
                    
    def _add_merged_adapter(
        self, 
        merged_adapters: Dict[str, torch.Tensor], 
        adapter_name: str
    ):
        """Add merged adapter to model.
        
        Args:
            merged_adapters: Dictionary mapping module names to merged weights
            adapter_name: Name for the merged adapter
        """
        # Create config for the merged adapter using configurable settings
        from ..core.config import LoRAConfig
        merged_config = LoRAConfig(
            r=self.config.merged_adapter_rank or 1,  # Use config or minimal rank
            alpha=self.config.merged_adapter_alpha,
            dropout=self.config.merged_adapter_dropout,
            bias=self.config.merged_adapter_bias
        )
        
        # Add merged adapter to each module
        for module_name, merged_weight in merged_adapters.items():
            if module_name in self.model_wrapper.lora_modules:
                lora_module = self.model_wrapper.lora_modules[module_name]
                
                # Decompose merged weight into A and B matrices
                # Use SVD to get low-rank approximation
                U, S, V = torch.svd(merged_weight)
                
                # Determine appropriate rank for merged adapter
                if self.config.merged_adapter_rank is not None:
                    rank = self.config.merged_adapter_rank
                else:
                    # Auto-determine rank from merged weight
                    rank = min(1, merged_weight.size(0), merged_weight.size(1))
                    
                # Allow rank=0 if configured and merged weight is effectively zero
                if self.config.allow_rank_zero and torch.allclose(merged_weight, torch.zeros_like(merged_weight)):
                    rank = 0
                
                # Create new config with appropriate rank
                adapter_config = LoRAConfig(
                    r=rank,
                    alpha=self.config.merged_adapter_alpha,
                    dropout=self.config.merged_adapter_dropout,
                    bias=self.config.merged_adapter_bias
                )
                
                # Add adapter to module
                if hasattr(lora_module, 'add_adapter'):
                    lora_module.add_adapter(adapter_name, adapter_config)
                    
                    # Set the weights
                    if adapter_name in lora_module.lora_adapters:
                        adapter = lora_module.lora_adapters[adapter_name]
                        if adapter.lora_A is not None and adapter.lora_B is not None:
                            # Set A and B from SVD decomposition
                            adapter.lora_A.data = V[:, :rank].T
                            adapter.lora_B.data = U[:, :rank] @ torch.diag(S[:rank])
                        
    def get_adapter_info(self) -> Dict[str, Any]:
        """Get information about available adapters.
        
        Returns:
            Dictionary with adapter information
        """
        info = {
            "adapters": {},
            "total_adapters": 0,
            "modules_with_adapters": 0
        }
        
        for module_name, lora_module in self.model_wrapper.lora_modules.items():
            if hasattr(lora_module, 'lora_adapters'):
                module_adapters = list(lora_module.lora_adapters.keys())
                info["adapters"][module_name] = module_adapters
                info["total_adapters"] += len(module_adapters)
                
        info["modules_with_adapters"] = len(info["adapters"])
        
        return info
