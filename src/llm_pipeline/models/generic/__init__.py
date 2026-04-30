"""Generic transformers-based model wrapper for non-Qwen3 architectures."""

from .transformers_wrapper import (
    GenericFactory,
    load_causal_lm,
)

__all__ = ["GenericFactory", "load_causal_lm"]
