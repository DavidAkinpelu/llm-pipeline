"""Quality-regression callback for the Trainer.

Runs an evaluator at a configurable cadence; tracks the metric history;
warns when the metric drops more than a configurable threshold from its
best value (catches training divergence early without aborting the run).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class RegressionWarning(UserWarning):
    """Raised when an evaluation metric drops below the regression threshold."""


@dataclass
class EvaluationHistory:
    """Records metric values over training steps for plotting + analysis."""

    steps: List[int] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def record(self, step: int, metrics: Dict[str, float]) -> None:
        self.steps.append(step)
        for k, v in metrics.items():
            self.metrics.setdefault(k, []).append(float(v))

    def best(self, name: str, mode: str = "min") -> Optional[float]:
        values = self.metrics.get(name, [])
        if not values:
            return None
        return min(values) if mode == "min" else max(values)


@dataclass
class EvaluationCallback:
    """Trainer callback that evaluates periodically and tracks regression.

    Parameters
    ----------
    evaluator : object
        Anything with ``evaluate(...) -> dict``. Typically a
        ``PerplexityEvaluator`` or one of the benchmark evaluators.
    every_n_steps : int
        Cadence. Evaluator runs at ``trainer.global_step % every_n_steps == 0``.
    metric_name : str
        Which key from the evaluator's output dict to track for regression.
    mode : "min" | "max"
        Whether lower or higher is better. ``"min"`` for perplexity / loss;
        ``"max"`` for accuracy / win rate.
    regression_threshold : float
        Relative drop from the best-seen value that triggers
        ``RegressionWarning``. Default 1% (0.01).
    evaluator_kwargs : dict
        Extra kwargs passed to ``evaluator.evaluate(**kwargs)``.
    """

    evaluator: Any
    every_n_steps: int = 1000
    metric_name: str = "perplexity"
    mode: str = "min"
    regression_threshold: float = 0.01
    evaluator_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max'; got {self.mode}")
        if self.every_n_steps < 1:
            raise ValueError(f"every_n_steps must be ≥ 1; got {self.every_n_steps}")
        self.history = EvaluationHistory()

    def __call__(self, trainer: Any) -> None:
        step = getattr(trainer, "global_step", 0)
        if step == 0 or step % self.every_n_steps != 0:
            return
        metrics = self.evaluator.evaluate(**self.evaluator_kwargs)
        self.history.record(step, metrics)
        self._check_regression(step, metrics)

    def _check_regression(self, step: int, metrics: Dict[str, float]) -> None:
        if self.metric_name not in metrics:
            return
        current = metrics[self.metric_name]
        best = self.history.best(self.metric_name, mode=self.mode)
        if best is None or best == 0:
            return
        if self.mode == "min":
            # "drop" = increase past best by threshold.
            relative_change = (current - best) / abs(best)
            if relative_change > self.regression_threshold:
                warnings.warn(
                    f"step {step}: {self.metric_name} regressed from best "
                    f"{best:.4f} to {current:.4f} (+{relative_change:.2%})",
                    RegressionWarning, stacklevel=2,
                )
        else:
            # "drop" = decrease past best by threshold.
            relative_change = (best - current) / abs(best)
            if relative_change > self.regression_threshold:
                warnings.warn(
                    f"step {step}: {self.metric_name} regressed from best "
                    f"{best:.4f} to {current:.4f} (-{relative_change:.2%})",
                    RegressionWarning, stacklevel=2,
                )
