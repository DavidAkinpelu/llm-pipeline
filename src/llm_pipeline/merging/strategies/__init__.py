"""Model-level merging strategies (operate on full state_dicts).

These are distinct from `llm_pipeline.adapters.merging`, which merges
LoRA adapters within a single base model.
"""

from .linear import linear_merge
from .task_arithmetic import task_arithmetic
from .ties import ties_merge
from .dare import dare_merge

__all__ = ["linear_merge", "task_arithmetic", "ties_merge", "dare_merge"]
