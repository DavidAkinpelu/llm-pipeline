"""Full precision merging strategy for QLoRA."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .base_merger import BaseMerger


class FullPrecisionMerger(BaseMerger):
    """Merge to full precision strategy for speed optimization."""
    
    def __init__(self, target_precision: str = "fp16"):
        """Initialize full precision merger.
        
        Args:
            target_precision: Target precision ("fp32", "fp16")
        """
        self.target_precision = target_precision
        
    def merge(
        self, 
        base_model, 
        lora_adapter, 
        target_precision: Optional[str] = None
    ):
        """Merge base model with LoRA adapter to full precision.
        
        Args:
            base_model: Base quantized model
            lora_adapter: LoRA adapter weights
            target_precision: Target precision (defaults to self.target_precision)
            
        Returns:
            Merged full precision model
        """
        if not self.validate_inputs(base_model, lora_adapter):
            raise ValueError("Invalid inputs for merging")
            
        target_precision = target_precision or self.target_precision
        
        # Get merged weights
        merged_weights = self._merge_weights(base_model, lora_adapter)
        
        # Convert to target precision
        precision_weights = self._convert_to_precision(merged_weights, target_precision)
        
        # Create merged model
        merged_model = self._create_merged_model(base_model, precision_weights)
        
        return merged_model
        
    def get_memory_usage(self, base_model, lora_adapter) -> float:
        """Estimate memory usage after merge (higher than base model).
        
        Args:
            base_model: Base quantized model
            lora_adapter: LoRA adapter weights
            
        Returns:
            Estimated memory usage in GB (higher than base model)
        """
        # Full precision merge uses more memory than quantized base
        base_memory = self._estimate_model_memory(base_model)
        
        # Estimate full precision memory (2x for FP16, 4x for FP32)
        if self.target_precision == "fp32":
            multiplier = 4.0
        else:  # fp16
            multiplier = 2.0
            
        return base_memory * multiplier
        
    def get_quality_estimate(self) -> float:
        """Get quality estimate for full precision merging.
        
        Returns:
            Quality score (1.0 - highest quality)
        """
        return 1.0
        
    def _merge_weights(self, base_model, lora_adapter) -> Dict[str, torch.Tensor]:
        """Merge base model weights with LoRA adapter weights.
        
        Args:
            base_model: Base quantized model
            lora_adapter: LoRA adapter weights
            
        Returns:
            Dictionary of merged weights
        """
        merged_weights = {}
        
        for layer_name, base_layer in base_model.named_modules():
            if hasattr(base_layer, 'weight') and isinstance(base_layer, nn.Linear):
                # Dequantize base weights
                base_weights = self._dequantize_weights(base_layer.weight)
                
                # Get LoRA weights for this layer
                lora_weight = self._get_lora_weight(lora_adapter, layer_name)
                
                # Merge weights: W_merged = W_base + W_lora
                merged_weight = base_weights + lora_weight
                merged_weights[layer_name] = merged_weight
                
        return merged_weights
        
    def _convert_to_precision(
        self, 
        weights: Dict[str, torch.Tensor], 
        target_precision: str
    ) -> Dict[str, torch.Tensor]:
        """Convert weights to target precision.
        
        Args:
            weights: Merged weights dictionary
            target_precision: Target precision
            
        Returns:
            Dictionary of weights in target precision
        """
        precision_weights = {}
        
        for layer_name, weight in weights.items():
            if target_precision == "fp32":
                precision_weight = weight.float()
            elif target_precision == "fp16":
                precision_weight = weight.half()
            else:
                raise ValueError(f"Unsupported target precision: {target_precision}")
                
            precision_weights[layer_name] = precision_weight
            
        return precision_weights
        
    def _dequantize_weights(self, quantized_weight) -> torch.Tensor:
        """Dequantize weights from quantized format.
        
        Args:
            quantized_weight: Quantized weight tensor
            
        Returns:
            Dequantized weight tensor
        """
        # This would depend on the actual quantization implementation
        # For now, assume weights are already in float format
        if hasattr(quantized_weight, 'data'):
            return quantized_weight.data.float()
        return quantized_weight.float()
        
    def _get_lora_weight(self, lora_adapter, layer_name: str) -> torch.Tensor:
        """Get LoRA weight for a specific layer.
        
        Args:
            lora_adapter: LoRA adapter weights
            layer_name: Name of the layer
            
        Returns:
            LoRA weight contribution
        """
        # Construct LoRA weight names
        lora_a_name = f"{layer_name}.lora_A"
        lora_b_name = f"{layer_name}.lora_B"
        
        if lora_a_name in lora_adapter and lora_b_name in lora_adapter:
            lora_a = lora_adapter[lora_a_name]
            lora_b = lora_adapter[lora_b_name]
            # LoRA contribution: W_lora = B @ A
            return torch.matmul(lora_b, lora_a)
        else:
            # No LoRA weights for this layer - return zeros
            # This is a placeholder - in practice, you'd need the actual weight shape
            return torch.zeros(100, 100)  # Placeholder shape
            
    def _create_merged_model(self, base_model, precision_weights):
        """Create merged model with full precision weights.
        
        Args:
            base_model: Original base model
            precision_weights: Full precision merged weights
            
        Returns:
            Merged model
        """
        # Create a copy of the base model
        merged_model = base_model.__class__(**base_model.config)
        
        # Load the merged weights
        for name, param in merged_model.named_parameters():
            if name in precision_weights:
                param.data = precision_weights[name]
                
        return merged_model
        
    def _estimate_model_memory(self, model) -> float:
        """Estimate model memory usage.
        
        Args:
            model: Model to estimate
            
        Returns:
            Estimated memory in GB
        """
        total_params = sum(p.numel() for p in model.parameters())
        # Assume 4-bit quantization for base model
        memory_bytes = total_params * 0.5  # 4 bits = 0.5 bytes
        return memory_bytes / (1024**3)  # Convert to GB
