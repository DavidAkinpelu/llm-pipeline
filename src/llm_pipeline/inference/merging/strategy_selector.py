"""Strategy selector for choosing optimal merging approach."""

from typing import Dict, Any, Optional
from enum import Enum


class MergeStrategy(Enum):
    """Available merging strategies."""
    FULL_PRECISION_FP32 = "full_precision_fp32"
    FULL_PRECISION_FP16 = "full_precision_fp16"
    QUANTIZED_8BIT = "quantized_8bit"
    QUANTIZED_4BIT = "quantized_4bit"
    QUANTIZED_2BIT = "quantized_2bit"
    AUTO = "auto"


class MergeStrategySelector:
    """Select optimal merging strategy based on constraints."""
    
    def __init__(self):
        """Initialize strategy selector."""
        self.strategy_recommendations = {
            "high_quality": {
                "unlimited_memory": MergeStrategy.FULL_PRECISION_FP16,
                "8gb_memory": MergeStrategy.QUANTIZED_8BIT,
                "4gb_memory": MergeStrategy.QUANTIZED_4BIT,
                "2gb_memory": MergeStrategy.QUANTIZED_2BIT
            },
            "balanced": {
                "unlimited_memory": MergeStrategy.QUANTIZED_8BIT,
                "8gb_memory": MergeStrategy.QUANTIZED_4BIT,
                "4gb_memory": MergeStrategy.QUANTIZED_4BIT,
                "2gb_memory": MergeStrategy.QUANTIZED_2BIT
            },
            "memory_efficient": {
                "unlimited_memory": MergeStrategy.QUANTIZED_4BIT,
                "8gb_memory": MergeStrategy.QUANTIZED_4BIT,
                "4gb_memory": MergeStrategy.QUANTIZED_2BIT,
                "2gb_memory": MergeStrategy.QUANTIZED_2BIT
            }
        }
        
    def select_strategy(
        self,
        available_memory_gb: float,
        model_size_gb: float,
        quality_requirement: str = "balanced",
        num_adapters: int = 1
    ) -> MergeStrategy:
        """Select optimal merging strategy.
        
        Args:
            available_memory_gb: Available memory in GB
            model_size_gb: Model size in GB
            quality_requirement: Quality requirement ("high_quality", "balanced", "memory_efficient")
            num_adapters: Number of adapters to merge
            
        Returns:
            Selected merging strategy
        """
        # Determine memory category
        memory_category = self._categorize_memory(available_memory_gb, model_size_gb)
        
        # Get recommendation
        if quality_requirement not in self.strategy_recommendations:
            quality_requirement = "balanced"
            
        if memory_category not in self.strategy_recommendations[quality_requirement]:
            memory_category = "4gb_memory"  # Default fallback
            
        strategy = self.strategy_recommendations[quality_requirement][memory_category]
        
        # Adjust for multiple adapters
        if num_adapters > 1:
            strategy = self._adjust_for_multiple_adapters(strategy, num_adapters)
            
        return strategy
        
    def recommend_strategy(
        self,
        available_memory_gb: float,
        model_size_gb: float,
        quality_requirement: str = "balanced"
    ) -> Dict[str, Any]:
        """Get strategy recommendation with details.
        
        Args:
            available_memory_gb: Available memory in GB
            model_size_gb: Model size in GB
            quality_requirement: Quality requirement
            
        Returns:
            Recommendation details
        """
        strategy = self.select_strategy(available_memory_gb, model_size_gb, quality_requirement)
        
        return {
            "strategy": strategy,
            "quality_requirement": quality_requirement,
            "available_memory_gb": available_memory_gb,
            "model_size_gb": model_size_gb,
            "reasoning": self._get_reasoning(strategy, available_memory_gb, model_size_gb)
        }
        
    def _categorize_memory(self, available_memory_gb: float, model_size_gb: float) -> str:
        """Categorize available memory."""
        if available_memory_gb >= model_size_gb * 4:
            return "unlimited_memory"
        elif available_memory_gb >= 8.0:
            return "8gb_memory"
        elif available_memory_gb >= 4.0:
            return "4gb_memory"
        else:
            return "2gb_memory"
            
    def _adjust_for_multiple_adapters(self, strategy: MergeStrategy, num_adapters: int) -> MergeStrategy:
        """Adjust strategy for multiple adapters."""
        # Multiple adapters require more memory, so choose more aggressive quantization
        if num_adapters > 4:
            if strategy == MergeStrategy.FULL_PRECISION_FP16:
                return MergeStrategy.QUANTIZED_8BIT
            elif strategy == MergeStrategy.QUANTIZED_8BIT:
                return MergeStrategy.QUANTIZED_4BIT
            elif strategy == MergeStrategy.QUANTIZED_4BIT:
                return MergeStrategy.QUANTIZED_2BIT
                
        return strategy
        
    def _get_reasoning(self, strategy: MergeStrategy, available_memory_gb: float, model_size_gb: float) -> str:
        """Get reasoning for strategy selection."""
        reasoning_map = {
            MergeStrategy.FULL_PRECISION_FP32: "High memory availability allows full precision",
            MergeStrategy.FULL_PRECISION_FP16: "Good memory availability with quality optimization",
            MergeStrategy.QUANTIZED_8BIT: "Balanced memory usage and quality",
            MergeStrategy.QUANTIZED_4BIT: "Memory efficient with acceptable quality",
            MergeStrategy.QUANTIZED_2BIT: "Maximum memory efficiency for constrained environments"
        }
        
        return reasoning_map.get(strategy, "Automatic strategy selection")
