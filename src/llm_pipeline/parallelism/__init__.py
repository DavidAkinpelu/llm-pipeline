"""Parallelism framework for Qwen3 models."""

from .base import BaseParallelism, ParallelismConfig, TensorParallelConfig
from .registry import ParallelismRegistry, parallelism_registry
from .factory import ParallelismFactory
from .tensor_parallel import Qwen3TensorParallelism
from .expert_parallel import ExpertParallelConfig, ExpertParallelMoE

# Import strategies to register them
try:
    from .strategies.tensor_parallel import Qwen3TensorParallelismStrategy
except ImportError:
    pass

__all__ = [
    "BaseParallelism",
    "ParallelismConfig",
    "TensorParallelConfig", 
    "ParallelismRegistry",
    "parallelism_registry",
    "ParallelismFactory",
    "Qwen3TensorParallelism",
    "ExpertParallelConfig",
    "ExpertParallelMoE",
]
