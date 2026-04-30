"""Registry for parallelism strategies."""

from typing import Dict, Type, List
from .base import BaseParallelism


class ParallelismRegistry:
    """Registry for all parallelism strategies."""
    
    def __init__(self):
        self._strategies: Dict[str, Type[BaseParallelism]] = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default Qwen3 parallelism strategies."""
        try:
            from .strategies.tensor_parallel import Qwen3TensorParallelismStrategy
            self.register("tensor_parallel", Qwen3TensorParallelismStrategy)
        except ImportError:
            pass  # Strategies will be registered when imported
    
    def register(self, name: str, strategy_class: Type[BaseParallelism]):
        """Register a parallelism strategy."""
        self._strategies[name] = strategy_class
        print(f"Registered parallelism strategy: {name}")
    
    def get_strategy(self, name: str) -> Type[BaseParallelism]:
        """Get a parallelism strategy by name."""
        strategy = self._strategies.get(name)
        if strategy is None:
            raise ValueError(f"Unknown parallelism strategy: {name}. Available: {list(self._strategies.keys())}")
        return strategy
    
    def list_strategies(self) -> List[str]:
        """List all available strategies."""
        return list(self._strategies.keys())
    
    def is_registered(self, name: str) -> bool:
        """Check if a strategy is registered."""
        return name in self._strategies


# Global registry instance
parallelism_registry = ParallelismRegistry()
