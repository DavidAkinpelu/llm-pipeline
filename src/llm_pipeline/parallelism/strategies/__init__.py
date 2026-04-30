"""Parallelism strategies for Qwen3 models."""

from .tensor_parallel import Qwen3TensorParallelismStrategy

__all__ = [
    "Qwen3TensorParallelismStrategy"
]
