"""Communication primitives for parallelism."""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torch.distributed as dist
from .patterns import CommunicationPattern


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


class CommunicationPrimitive(ABC):
    """Base class for communication primitives."""
    
    def __init__(self, world_size: int, rank: int, process_group=None):
        self.world_size = world_size
        self.rank = rank
        self.process_group = process_group
    
    @abstractmethod
    def execute(self, tensors: List[torch.Tensor], **kwargs):
        """Execute communication primitive."""
        pass
    
    @abstractmethod
    def get_pattern(self) -> CommunicationPattern:
        """Get communication pattern."""
        pass


class AllReducePrimitive(CommunicationPrimitive):
    """All-reduce communication primitive."""
    
    def execute(self, tensors: List[torch.Tensor], **kwargs):
        """Execute all-reduce operation."""
        if self.world_size > 1 and _is_dist():
            for tensor in tensors:
                dist.all_reduce(tensor, group=self.process_group)
        return tensors
    
    def get_pattern(self) -> CommunicationPattern:
        return CommunicationPattern.ALL_REDUCE


class AllGatherPrimitive(CommunicationPrimitive):
    """All-gather communication primitive."""
    
    def execute(self, tensors: List[torch.Tensor], dim: int = 0, **kwargs):
        """Execute all-gather operation."""
        if self.world_size > 1 and _is_dist():
            results = []
            for tensor in tensors:
                gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
                dist.all_gather(gathered, tensor, group=self.process_group)
                results.append(torch.cat(gathered, dim=dim))
            return results
        return tensors
    
    def get_pattern(self) -> CommunicationPattern:
        return CommunicationPattern.ALL_GATHER


class ReduceScatterPrimitive(CommunicationPrimitive):
    """Reduce-scatter communication primitive."""
    
    def execute(self, tensors: List[torch.Tensor], dim: int = 0, **kwargs):
        """Execute reduce-scatter operation."""
        if self.world_size > 1 and _is_dist():
            results = []
            for tensor in tensors:
                if tensor.size(dim) % self.world_size != 0:
                    raise ValueError(
                        f"tensor dimension {tensor.size(dim)} not divisible by world_size={self.world_size}"
                    )
                chunks = list(torch.chunk(tensor, self.world_size, dim=dim))
                
                # Reduce-scatter
                output = torch.zeros_like(chunks[self.rank])
                dist.reduce_scatter(output, chunks, group=self.process_group)
                results.append(output)
            return results
        return tensors
    
    def get_pattern(self) -> CommunicationPattern:
        return CommunicationPattern.REDUCE_SCATTER
