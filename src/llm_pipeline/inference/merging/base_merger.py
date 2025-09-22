"""Base merger interface for QLoRA merging strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch


class BaseMerger(ABC):
    """Base class for QLoRA merging strategies."""
    
    @abstractmethod
    def merge(
        self, 
        base_model, 
        lora_adapter, 
        target_precision: Optional[str] = None
    ) -> Any:
        """Merge base model with LoRA adapter.
        
        Args:
            base_model: Base quantized model
            lora_adapter: LoRA adapter weights
            target_precision: Target precision for merged model
            
        Returns:
            Merged model
        """
        pass
        
    @abstractmethod
    def get_memory_usage(
        self, 
        base_model, 
        lora_adapter
    ) -> float:
        """Estimate memory usage after merge.
        
        Args:
            base_model: Base quantized model
            lora_adapter: LoRA adapter weights
            
        Returns:
            Estimated memory usage in GB
        """
        pass
        
    @abstractmethod
    def get_quality_estimate(self) -> float:
        """Get quality estimate for this merging strategy.
        
        Returns:
            Quality score (0.0 to 1.0)
        """
        pass
        
    def validate_inputs(
        self, 
        base_model, 
        lora_adapter
    ) -> bool:
        """Validate inputs for merging.
        
        Args:
            base_model: Base model to validate
            lora_adapter: LoRA adapter to validate
            
        Returns:
            True if inputs are valid
        """
        if base_model is None:
            return False
        if lora_adapter is None:
            return False
        return True
        
    def get_merge_info(self) -> Dict[str, Any]:
        """Get information about this merger.
        
        Returns:
            Dictionary with merger information
        """
        return {
            "merger_type": self.__class__.__name__,
            "quality_estimate": self.get_quality_estimate(),
            "supports_precision": True,
            "requires_requantization": False
        }
