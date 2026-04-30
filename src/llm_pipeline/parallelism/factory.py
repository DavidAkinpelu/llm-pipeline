"""Factory for creating parallelism strategies."""

from typing import List, Optional
from .base import BaseParallelism, ParallelismConfig
from .registry import parallelism_registry


class ParallelismFactory:
    """Factory for creating parallelism strategies."""
    
    @staticmethod
    def create_parallelism(config: ParallelismConfig) -> BaseParallelism:
        """Create parallelism strategy based on config."""
        
        strategy_class = parallelism_registry.get_strategy(config.parallelism_type)
        return strategy_class(config)
    
    @staticmethod
    def create_multi_parallelism(configs: List[ParallelismConfig]) -> List[BaseParallelism]:
        """Create multiple parallelism strategies."""
        return [ParallelismFactory.create_parallelism(config) for config in configs]
    
    @staticmethod
    def create_strategy_by_name(name: str, **kwargs) -> BaseParallelism:
        """Create parallelism strategy by name with parameters."""
        
        strategy_class = parallelism_registry.get_strategy(name)
        
        # Create appropriate config based on strategy type
        if name == "tensor_parallel":
            from .base import TensorParallelConfig
            config = TensorParallelConfig(**kwargs)
        elif name == "pipeline_parallel":
            from .base import PipelineParallelConfig
            config = PipelineParallelConfig(**kwargs)
        elif name == "data_parallel":
            from .base import DataParallelConfig
            config = DataParallelConfig(**kwargs)
        else:
            config = ParallelismConfig(parallelism_type=name, **kwargs)
        
        return strategy_class(config)
    
    @staticmethod
    def list_available_strategies() -> List[str]:
        """List all available parallelism strategies."""
        return parallelism_registry.list_strategies()
