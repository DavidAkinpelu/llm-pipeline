"""Tensor parallelism strategy for Qwen3 models."""

import torch.nn as nn
from ..base import BaseParallelism, TensorParallelConfig
from ..tensor_parallel import Qwen3TensorParallelism


class Qwen3TensorParallelismStrategy(BaseParallelism):
    """Qwen3-specific tensor parallelism strategy."""
    
    def __init__(self, config: TensorParallelConfig):
        super().__init__(config)
        self.tensor_parallelism = Qwen3TensorParallelism(config)
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for tensor parallelism."""
        return self.tensor_parallelism.wrap_model(model)
    
    def setup_distributed(self):
        """Setup distributed environment."""
        self.tensor_parallelism.setup_distributed()
    
    def cleanup_distributed(self):
        """Cleanup distributed environment."""
        self.tensor_parallelism.cleanup_distributed()
    
    def can_apply(self, model: nn.Module) -> bool:
        """Check if tensor parallelism can be applied to Qwen3 model."""
        return self.tensor_parallelism.can_apply(model)
