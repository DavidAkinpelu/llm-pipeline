"""Quantized merging strategy for QLoRA."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .base_merger import BaseMerger


class QuantizedMerger(BaseMerger):
    """Merge and requantize strategy for memory efficiency."""
    
    def __init__(self, quantization_type: str = "nf4"):
        """Initialize quantized merger.
        
        Args:
            quantization_type: Type of quantization ("nf4", "fp4", "int8")
        """
        self.quantization_type = quantization_type
        
    def merge(
        self, 
        base_model, 
        lora_adapter, 
        target_precision: Optional[str] = None
    ):
        """Merge base model with LoRA adapter and requantize.
        
        Args:
            base_model: Base quantized model
            lora_adapter: LoRA adapter weights
            target_precision: Target precision (defaults to quantization_type)
            
        Returns:
            Merged and requantized model
        """
        if not self.validate_inputs(base_model, lora_adapter):
            raise ValueError("Invalid inputs for merging")
            
        target_precision = target_precision or self.quantization_type
        
        # Get merged weights
        merged_weights = self._merge_weights(base_model, lora_adapter)
        
        # Requantize merged weights
        quantized_weights = self._quantize_weights(merged_weights, target_precision)
        
        # Create merged model
        merged_model = self._create_merged_model(base_model, quantized_weights)
        
        return merged_model
        
    def get_memory_usage(self, base_model, lora_adapter) -> float:
        """Estimate memory usage after merge (same as base model).
        
        Args:
            base_model: Base quantized model
            lora_adapter: LoRA adapter weights
            
        Returns:
            Estimated memory usage in GB (same as base model)
        """
        # Quantized merge keeps same memory as base model
        return self._estimate_model_memory(base_model)
        
    def get_quality_estimate(self) -> float:
        """Get quality estimate for quantized merging.
        
        Returns:
            Quality score (0.8 - good quality with some quantization loss)
        """
        return 0.8
        
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
        
    def _quantize_weights(
        self, 
        weights: Dict[str, torch.Tensor], 
        target_precision: str
    ) -> Dict[str, torch.Tensor]:
        """Quantize merged weights.
        
        Args:
            weights: Merged weights dictionary
            target_precision: Target precision
            
        Returns:
            Dictionary of quantized weights
        """
        quantized_weights = {}
        
        for layer_name, weight in weights.items():
            if target_precision == "4bit":
                quantized_weight = self._quantize_4bit(weight)
            elif target_precision == "8bit":
                quantized_weight = self._quantize_8bit(weight)
            elif target_precision == "2bit":
                quantized_weight = self._quantize_2bit(weight)
            else:
                raise ValueError(f"Unsupported target precision: {target_precision}")
                
            quantized_weights[layer_name] = quantized_weight
            
        return quantized_weights
        
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
            # No LoRA weights for this layer
            return torch.zeros_like(layer_name)  # Placeholder
            
    def _quantize_4bit(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights to 4-bit.
        
        Args:
            weights: Input weights
            
        Returns:
            4-bit quantized weights
        """
        if self.quantization_type == "nf4":
            return self._quantize_nf4(weights)
        elif self.quantization_type == "fp4":
            return self._quantize_fp4(weights)
        else:
            return self._quantize_int4(weights)
            
    def _quantize_nf4(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize using NF4 (NormalFloat4) format."""
        # NF4 quantization levels
        nf4_levels = torch.tensor([
            -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
            0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1.0
        ], device=weights.device, dtype=weights.dtype)
        
        # Normalize weights
        absmax = torch.abs(weights).max()
        normalized = weights / absmax
        
        # Find closest NF4 levels
        quantized = torch.zeros_like(normalized)
        for i, level in enumerate(nf4_levels):
            mask = torch.abs(normalized - level) == torch.abs(normalized - nf4_levels).min(dim=-1, keepdim=True)[0]
            quantized[mask] = level
            
        return quantized * absmax
        
    def _quantize_fp4(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize using FP4 format."""
        # Simple FP4 quantization
        scale = torch.abs(weights).max() / 7.0  # FP4 range
        quantized = torch.round(weights / scale) * scale
        return quantized
        
    def _quantize_int4(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize using INT4 format."""
        # Simple INT4 quantization
        min_val, max_val = weights.min(), weights.max()
        scale = (max_val - min_val) / 15.0  # 4-bit range
        quantized = torch.round((weights - min_val) / scale) * scale + min_val
        return quantized
        
    def _quantize_8bit(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights to 8-bit."""
        # Simple 8-bit quantization
        min_val, max_val = weights.min(), weights.max()
        scale = (max_val - min_val) / 255.0  # 8-bit range
        quantized = torch.round((weights - min_val) / scale) * scale + min_val
        return quantized
        
    def _quantize_2bit(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights to 2-bit."""
        # Simple 2-bit quantization
        min_val, max_val = weights.min(), weights.max()
        scale = (max_val - min_val) / 3.0  # 2-bit range
        quantized = torch.round((weights - min_val) / scale) * scale + min_val
        return quantized
        
    def _create_merged_model(self, base_model, quantized_weights):
        """Create merged model with quantized weights.
        
        Args:
            base_model: Original base model
            quantized_weights: Quantized merged weights
            
        Returns:
            Merged model
        """
        # Create a copy of the base model
        merged_model = base_model.__class__(**base_model.config)
        
        # Load the merged weights
        for name, param in merged_model.named_parameters():
            if name in quantized_weights:
                param.data = quantized_weights[name]
                
        return merged_model
        
    def _estimate_model_memory(self, model) -> float:
        """Estimate model memory usage.
        
        Args:
            model: Model to estimate
            
        Returns:
            Estimated memory in GB
        """
        total_params = sum(p.numel() for p in model.parameters())
        # Assume 4-bit quantization
        memory_bytes = total_params * 0.5  # 4 bits = 0.5 bytes
        return memory_bytes / (1024**3)  # Convert to GB
