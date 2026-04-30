"""Model-level merging (full state_dict merges).

For LoRA adapter merging, see ``llm_pipeline.adapters.merging``.
"""

from .strategies import linear_merge, task_arithmetic, ties_merge, dare_merge
from .quality import perplexity, MergeQualityReport
from .composition import sequential_merge

__all__ = [
    "linear_merge",
    "task_arithmetic",
    "ties_merge",
    "dare_merge",
    "perplexity",
    "MergeQualityReport",
    "sequential_merge",
]
