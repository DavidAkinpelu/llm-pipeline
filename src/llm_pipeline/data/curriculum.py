"""Curriculum learning: difficulty-scored datasets with paced exposure.

Three composable building blocks:

- **Scorers** rank examples by difficulty (lower = easier). Stateless
  callables ``(example) -> float`` so they're trivial to compose.
- **Pacing functions** map training step → fraction of the
  difficulty-sorted dataset to expose at that step. Bengio et al. 2009
  showed that growing the visible set monotonically (rather than
  switching cleanly between buckets) gives the smoothest training curve.
- **`CurriculumDataset`** wraps any indexable dataset, pre-scores +
  sorts at construction, and exposes only the easiest ``pacing(step)``
  fraction at any given step. The trainer calls ``set_step(global_step)``
  each iteration via the ``CurriculumStepHook`` callback.

Plus **`SelfPacedSampler`** (Kumar et al. 2010): the model itself
decides which examples to include based on its current loss. The
loss-threshold pacing function relaxes over training so harder
examples get included as the model improves.

References
----------

- Bengio, Louradour, Collobert, Weston, "Curriculum Learning", ICML 2009
- Kumar, Packer, Koller, "Self-Paced Learning for Latent Variable Models", NeurIPS 2010
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #


class DifficultyScorer(ABC):
    """Score an example for difficulty. Lower scores = easier examples."""

    @abstractmethod
    def __call__(self, example: Any) -> float:
        ...


class LengthScorer(DifficultyScorer):
    """Difficulty = sequence length. Cheapest scorer; works surprisingly
    well for autoregressive pretraining (longer sequences = more positions
    to predict = harder loss target).

    ``field`` selects the attribute / key to measure. For dict-style
    examples (HF datasets), use ``"input_ids"``. For raw token lists,
    pass a callable in `field_fn` instead.
    """

    def __init__(self, field: str = "input_ids", field_fn: Optional[Callable[[Any], Any]] = None):
        self.field = field
        self.field_fn = field_fn

    def __call__(self, example: Any) -> float:
        if self.field_fn is not None:
            value = self.field_fn(example)
        elif isinstance(example, dict):
            value = example[self.field]
        else:
            value = getattr(example, self.field)
        return float(len(value))


class MetadataScorer(DifficultyScorer):
    """Difficulty = a precomputed score on the example.

    For datasets with curated difficulty labels (Evol-Instruct evolution
    chains, MATH problem levels, GSM8K rated splits). The field's value
    is cast to ``float``.
    """

    def __init__(self, field: str = "difficulty"):
        self.field = field

    def __call__(self, example: Any) -> float:
        if isinstance(example, dict):
            return float(example[self.field])
        return float(getattr(example, self.field))


class PerplexityScorer(DifficultyScorer):
    """Difficulty = per-example loss from a frozen reference model.

    Standard "intrinsic" difficulty signal — examples the reference
    model finds confusing are flagged hard. The reference is held in
    eval mode and ``no_grad`` throughout; only one forward per example
    at construction time of the wrapping ``CurriculumDataset``.

    Example contract: a dict with ``input_ids`` (and optionally
    ``labels``; if absent we use ``input_ids`` shifted by one for the
    causal LM objective).
    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def __call__(self, example: Any) -> float:
        if isinstance(example, dict):
            ids = example["input_ids"]
            labels = example.get("labels", ids)
        else:
            ids = example.input_ids
            labels = getattr(example, "labels", ids)
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        labels_t = torch.as_tensor(labels, dtype=torch.long, device=self.device).unsqueeze(0)
        out = self.model(input_ids=ids_t, labels=labels_t)
        loss = out.loss if hasattr(out, "loss") else out[0]
        return float(loss.item())


@dataclass
class CompositeScorer(DifficultyScorer):
    """Weighted sum of multiple scorers.

    ``weights`` may be omitted (uniform). Useful when no single signal
    captures difficulty — e.g. length × perplexity for code datasets.
    """

    scorers: List[DifficultyScorer]
    weights: Optional[List[float]] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = [1.0] * len(self.scorers)
        if len(self.weights) != len(self.scorers):
            raise ValueError(
                f"weights count ({len(self.weights)}) ≠ scorers count ({len(self.scorers)})"
            )

    def __call__(self, example: Any) -> float:
        return sum(w * s(example) for w, s in zip(self.weights, self.scorers))


# --------------------------------------------------------------------------- #
# Pacing functions
# --------------------------------------------------------------------------- #


class PacingFunction(ABC):
    """Map training step → fraction of dataset to expose, in [0, 1]."""

    @abstractmethod
    def __call__(self, step: int) -> float:
        ...


@dataclass
class LinearPacing(PacingFunction):
    """``frac(step) = lerp(start_frac, end_frac, step / total_steps)``,
    clamped to ``[start_frac, end_frac]`` outside the range.
    """

    start_frac: float = 0.1
    end_frac: float = 1.0
    total_steps: int = 10_000

    def __post_init__(self):
        _validate_fracs(self.start_frac, self.end_frac, self.total_steps)

    def __call__(self, step: int) -> float:
        if step <= 0:
            return self.start_frac
        if step >= self.total_steps:
            return self.end_frac
        t = step / self.total_steps
        return self.start_frac + (self.end_frac - self.start_frac) * t


@dataclass
class SqrtPacing(PacingFunction):
    """``frac(step) = start + (end - start) · sqrt(step / total_steps)``.

    Bengio et al.'s default: fast initial expansion (the model needs to
    see the diversity early) then plateau as we approach full data.
    """

    start_frac: float = 0.1
    end_frac: float = 1.0
    total_steps: int = 10_000

    def __post_init__(self):
        _validate_fracs(self.start_frac, self.end_frac, self.total_steps)

    def __call__(self, step: int) -> float:
        if step <= 0:
            return self.start_frac
        if step >= self.total_steps:
            return self.end_frac
        t = math.sqrt(step / self.total_steps)
        return self.start_frac + (self.end_frac - self.start_frac) * t


@dataclass
class ExponentialPacing(PacingFunction):
    """``frac(step) = start · (end/start)^(step / total_steps)``.

    Slow start, accelerates near the end. Useful when "easy" data is
    very easy and the model overfits quickly without it.
    """

    start_frac: float = 0.1
    end_frac: float = 1.0
    total_steps: int = 10_000

    def __post_init__(self):
        _validate_fracs(self.start_frac, self.end_frac, self.total_steps)
        if self.start_frac <= 0:
            raise ValueError("ExponentialPacing requires start_frac > 0")

    def __call__(self, step: int) -> float:
        if step <= 0:
            return self.start_frac
        if step >= self.total_steps:
            return self.end_frac
        t = step / self.total_steps
        return self.start_frac * (self.end_frac / self.start_frac) ** t


@dataclass
class StepPacing(PacingFunction):
    """Discrete checkpointed pacing.

    ``checkpoints``: ``[(step_0, frac_0), (step_1, frac_1), ...]`` —
    holds at each ``frac_i`` until ``step_{i+1}``. Useful for "warm up
    on the easy 10% for 1000 steps, then 50% for 5000 steps, then full".
    """

    checkpoints: List[Tuple[int, float]]

    def __post_init__(self):
        if not self.checkpoints:
            raise ValueError("StepPacing needs at least one checkpoint")
        # Sort defensively.
        self.checkpoints = sorted(self.checkpoints, key=lambda p: p[0])
        for s, f in self.checkpoints:
            if not 0.0 <= f <= 1.0:
                raise ValueError(f"checkpoint fraction {f} out of [0, 1]")

    def __call__(self, step: int) -> float:
        active = self.checkpoints[0][1]
        for s, f in self.checkpoints:
            if step >= s:
                active = f
            else:
                break
        return active


def _validate_fracs(start: float, end: float, total: int) -> None:
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError(f"start/end fractions must be in [0, 1]; got {start}, {end}")
    if start > end:
        raise ValueError(f"start_frac ({start}) > end_frac ({end})")
    if total <= 0:
        raise ValueError(f"total_steps must be > 0; got {total}")


# --------------------------------------------------------------------------- #
# CurriculumDataset
# --------------------------------------------------------------------------- #


class CurriculumDataset:
    """Wraps a sequence-style dataset; exposes only the easiest fraction
    per the pacing function at the current step.

    Build once (one full pass to score every example), then call
    ``set_step(step)`` each training iteration.

    >>> ds = CurriculumDataset(
    ...     base=my_hf_dataset,
    ...     scorer=LengthScorer(),
    ...     pacing=SqrtPacing(start_frac=0.1, end_frac=1.0, total_steps=10000),
    ...     shuffle_seed=42,
    ... )
    >>> ds.set_step(0)
    >>> len(ds)              # 10% of base, sorted easy-first then shuffled
    >>> ds.set_step(10000)
    >>> len(ds)              # full base
    """

    def __init__(
        self,
        base: Sequence[Any],
        scorer: DifficultyScorer,
        pacing: PacingFunction,
        shuffle_seed: int = 0,
    ):
        self.base = base
        self.scorer = scorer
        self.pacing = pacing
        self.shuffle_seed = shuffle_seed
        # Pre-score every example. ``scores[i]`` is the difficulty of base[i].
        scores = [scorer(base[i]) for i in range(len(base))]
        # Sorted indices: easiest first.
        self._sorted_indices = sorted(range(len(base)), key=lambda i: scores[i])
        self._scores = scores
        self._step = 0
        self._exposed_indices: List[int] = self._compute_exposed(self._step)

    def set_step(self, step: int) -> None:
        """Advance the curriculum's notion of training step.

        Re-derives the exposed subset; cheap (just a shuffle of the prefix).
        """
        self._step = step
        self._exposed_indices = self._compute_exposed(step)

    def _compute_exposed(self, step: int) -> List[int]:
        frac = self.pacing(step)
        n_exposed = max(1, int(round(frac * len(self.base))))
        n_exposed = min(n_exposed, len(self.base))
        exposed = list(self._sorted_indices[:n_exposed])
        # Deterministic per-step shuffle so the trainer doesn't see the
        # same example order on every epoch.
        rng = random.Random(self.shuffle_seed + step)
        rng.shuffle(exposed)
        return exposed

    def __len__(self) -> int:
        return len(self._exposed_indices)

    def __getitem__(self, idx: int) -> Any:
        base_idx = self._exposed_indices[idx]
        return self.base[base_idx]

    def __iter__(self) -> Iterator[Any]:
        for idx in self._exposed_indices:
            yield self.base[idx]

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def current_fraction(self) -> float:
        return self.pacing(self._step)

    @property
    def difficulty_scores(self) -> List[float]:
        """The full per-example score table, indexed by base position."""
        return list(self._scores)


# --------------------------------------------------------------------------- #
# Self-paced learning
# --------------------------------------------------------------------------- #


@dataclass
class SelfPacedSampler:
    """Self-paced learning (Kumar et al. 2010) — model decides what's easy.

    Each ``evaluate(model)`` call re-scores every example using the
    current model's loss; the threshold (a function of step) determines
    which examples are admitted to the training set on the next round.

    Threshold rises over time: starts low (only the very easy examples
    pass), relaxes as the model improves to admit harder ones.

    Wraps any indexable dataset. ``set_step(step)`` updates the active
    set; ``evaluate(model)`` triggers a re-scoring pass.

    >>> sampler = SelfPacedSampler(
    ...     base=my_dataset,
    ...     example_loss_fn=lambda model, ex: forward_and_loss(model, ex),
    ...     threshold_fn=LinearPacing(start_frac=0.5, end_frac=5.0, total_steps=10000),
    ... )
    >>> sampler.evaluate(model)
    >>> sampler.set_step(0)
    >>> # iterate over the included examples
    """

    base: Sequence[Any]
    example_loss_fn: Callable[[torch.nn.Module, Any], float]
    threshold_fn: Callable[[int], float]
    re_evaluate_every_n_steps: int = 1000

    def __post_init__(self):
        self._losses: Optional[List[float]] = None
        self._step = 0
        self._last_eval_step = -1
        self._exposed_indices: List[int] = []

    def evaluate(self, model: torch.nn.Module) -> None:
        """Re-score every example with the current model. Sets ``_losses``
        and updates the exposed subset based on the current threshold.
        """
        with torch.no_grad():
            self._losses = [self.example_loss_fn(model, self.base[i]) for i in range(len(self.base))]
        self._last_eval_step = self._step
        self._refresh_exposed()

    def set_step(self, step: int) -> None:
        self._step = step
        if self._losses is None:
            return
        # Re-evaluate cadence is enforced by the caller — they decide when
        # to invoke ``evaluate(model)`` based on this step.
        self._refresh_exposed()

    def should_re_evaluate(self) -> bool:
        if self._losses is None:
            return True
        return (self._step - self._last_eval_step) >= self.re_evaluate_every_n_steps

    def _refresh_exposed(self) -> None:
        threshold = self.threshold_fn(self._step)
        self._exposed_indices = [
            i for i, loss in enumerate(self._losses) if loss <= threshold
        ]

    def __len__(self) -> int:
        return len(self._exposed_indices)

    def __getitem__(self, idx: int) -> Any:
        return self.base[self._exposed_indices[idx]]

    def __iter__(self) -> Iterator[Any]:
        for i in self._exposed_indices:
            yield self.base[i]


# --------------------------------------------------------------------------- #
# Trainer integration
# --------------------------------------------------------------------------- #


class CurriculumStepHook:
    """Trainer callback that calls ``dataset.set_step(global_step)`` each
    iteration. Compatible with any object that exposes ``set_step``.

    The Trainer's ``on_step_begin`` callback signature is
    ``(trainer) -> None``; this hook reads ``trainer.global_step`` and
    forwards it.
    """

    def __init__(self, curriculum: Any):
        if not hasattr(curriculum, "set_step"):
            raise TypeError("curriculum object must expose set_step(step)")
        self.curriculum = curriculum

    def __call__(self, trainer: Any) -> None:
        step = getattr(trainer, "global_step", 0)
        self.curriculum.set_step(step)
