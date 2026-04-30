"""Best-of-N / Rejection-Sampling Fine-Tuning (RFT) dataset construction.

Recipe
------

1. For each prompt, generate ``N`` candidate completions from the model
   (typically a temperature > 0 sample from the SFT-tuned base).
2. Score every candidate with a reward model (or any
   ``scorer(prompt, completion) -> float`` callable).
3. Keep the best-of-N — the highest-scoring completion per prompt — as
   a new (prompt, completion) pair.
4. Fine-tune the model on the resulting "self-distilled" dataset with
   plain SFT.

Why this works: the reward model identifies the *best* answer the base
model is already capable of producing; SFT-on-best-of-N then concentrates
probability mass on that mode. Several rounds compound. Same idea as
the WebGPT / RLHF data flywheel, minus the PPO step.

This module provides the dataset-construction half. The actual SFT step
plugs into the existing ``SFTTrainer``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass
class RFTRecord:
    """One (prompt, kept_completion, score) triple."""

    prompt: str
    completion: str
    score: float
    rejected: List[Tuple[str, float]] = field(default_factory=list)


# Type aliases.
GeneratorFn = Callable[[str, int], List[str]]    # (prompt, n) -> N completions
ScorerFn = Callable[[str, str], float]            # (prompt, completion) -> reward


@dataclass
class BestOfNSampler:
    """Helper that produces ``N`` completions per prompt and picks the best.

    >>> sampler = BestOfNSampler(generator=gen, scorer=reward_fn, n=8)
    >>> records = sampler.run(prompts=["...", "..."])
    >>> # records is a list[RFTRecord] sorted in input order.

    The ``generator`` callable is responsible for sampling — temperature,
    top-k / top-p settings live there, not in the sampler. ``scorer``
    returns a scalar reward; higher is better. Tied top scores break
    by insertion order (first sample wins).
    """

    generator: GeneratorFn
    scorer: ScorerFn
    n: int = 8
    keep_rejected: bool = False

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"n must be ≥ 1; got {self.n}")

    def run(self, prompts: Sequence[str]) -> List[RFTRecord]:
        out: List[RFTRecord] = []
        for prompt in prompts:
            candidates = self.generator(prompt, self.n)
            if not candidates:
                continue
            scored = [(c, float(self.scorer(prompt, c))) for c in candidates]
            best_idx = max(range(len(scored)), key=lambda i: scored[i][1])
            best_c, best_s = scored[best_idx]
            rejected = (
                [s for i, s in enumerate(scored) if i != best_idx]
                if self.keep_rejected else []
            )
            out.append(RFTRecord(
                prompt=prompt, completion=best_c, score=best_s, rejected=rejected,
            ))
        return out


def rejection_sampling_finetune_dataset(
    prompts: Sequence[str],
    generator: GeneratorFn,
    scorer: ScorerFn,
    n: int = 8,
    score_threshold: Optional[float] = None,
) -> List[RFTRecord]:
    """One-call entry point: best-of-N over a prompt list, optionally
    filtered by a minimum reward.

    ``score_threshold`` lets you drop prompts where even the best
    completion scored too low — the reward model's signal that the base
    model can't produce a good answer here, regardless of which sample
    we'd pick. Common practice in production RFT pipelines.
    """
    sampler = BestOfNSampler(generator=generator, scorer=scorer, n=n)
    records = sampler.run(prompts)
    if score_threshold is not None:
        records = [r for r in records if r.score >= score_threshold]
    return records
