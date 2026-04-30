"""Qwen3 model implementation.

Two public model classes:

- ``Qwen3Model``: the bare transformer (embeddings + decoder layers + final
  norm). Returns a dict with ``last_hidden_state`` and optional
  ``past_key_values``. Use this when you want to add a custom head.
- ``Qwen3ForCausalLM``: the full causal-LM, with tied lm_head on top of
  ``Qwen3Model``. This is what ``Qwen3Factory`` and ``Qwen3InferenceEngine``
  load.
"""

from .custom_builder import Qwen3Model, Qwen3ForCausalLM
from .config import Qwen3Config
from .factory import Qwen3Factory

__all__ = [
    "Qwen3Model",
    "Qwen3ForCausalLM",
    "Qwen3Config",
    "Qwen3Factory",
]
