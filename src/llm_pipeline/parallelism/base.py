"""Base parallelism framework for Qwen3 models."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ParallelismConfig:
    """Base configuration for parallelism strategies."""
    
    parallelism_type: str = "tensor"
    rank: int = 0
    world_size: int = 1
    backend: str = "nccl"
    process_group: Optional[Any] = None
    
    # Strategy-specific parameters
    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    pipeline_parallel_size: int = 1
    pipeline_parallel_rank: int = 0
    data_parallel_size: int = 1
    data_parallel_rank: int = 0


class BaseParallelism(ABC):
    """Base class for all parallelism strategies."""
    
    def __init__(self, config: ParallelismConfig):
        self.config = config
        self.rank = config.rank
        self.world_size = config.world_size
        self.device = f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for parallelism."""
        pass
    
    @abstractmethod
    def setup_distributed(self):
        """Setup distributed environment."""
        pass
    
    @abstractmethod
    def cleanup_distributed(self):
        """Cleanup distributed environment."""
        pass
    
    @abstractmethod
    def can_apply(self, model: nn.Module) -> bool:
        """Check if parallelism can be applied to model."""
        pass


class TensorParallelConfig(ParallelismConfig):
    """Configuration for tensor parallelism."""
    
    def __init__(self, 
                 tensor_parallel_size: int = 1,
                 tensor_parallel_rank: int = 0,
                 split_dimension: str = "column",
                 **kwargs):
        kwargs.setdefault("world_size", tensor_parallel_size)
        kwargs.setdefault("rank", tensor_parallel_rank)
        super().__init__(parallelism_type="tensor_parallel", **kwargs)
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_rank = tensor_parallel_rank
        self.split_dimension = split_dimension


class PipelineParallelConfig(ParallelismConfig):
    """Configuration for pipeline parallelism."""
    
    def __init__(self,
                 pipeline_parallel_size: int = 1,
                 pipeline_parallel_rank: int = 0,
                 micro_batch_size: int = 1,
                 **kwargs):
        super().__init__(parallelism_type="pipeline", **kwargs)
        self.pipeline_parallel_size = pipeline_parallel_size
        self.pipeline_parallel_rank = pipeline_parallel_rank
        self.micro_batch_size = micro_batch_size


class DataParallelConfig(ParallelismConfig):
    """Configuration for data parallelism."""
    
    def __init__(self,
                 data_parallel_size: int = 1,
                 data_parallel_rank: int = 0,
                 gradient_accumulation_steps: int = 1,
                 **kwargs):
        super().__init__(parallelism_type="data", **kwargs)
        self.data_parallel_size = data_parallel_size
        self.data_parallel_rank = data_parallel_rank
        self.gradient_accumulation_steps = gradient_accumulation_steps
