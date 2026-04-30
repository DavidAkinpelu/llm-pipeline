"""Hyperparameter sweep wrapper: grid + random + Optuna soft-import.

Three sweep strategies sharing one runner:

- **`GridSweep`**: Cartesian product over named lists. Deterministic
  order (sorted by key). The default for small parameter spaces.
- **`RandomSweep`**: uniform sample from a search-space dict
  (``{"lr": ("loguniform", 1e-5, 1e-2), ...}``). Always available.
- **`OptunaSweep`**: soft-imports Optuna; falls through to ``RandomSweep``
  with a warning when Optuna isn't installed.

The ``SweepRunner`` runs trials sequentially, calling a user-supplied
``train_fn(params) -> metrics_dict`` per trial. Trial records go to a
JSONL log so you can grep / load with pandas without extra deps.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple


@dataclass
class TrialRecord:
    """One sweep trial — params in, metrics out, plus timing + status."""

    trial_id: int
    params: Dict[str, Any]
    metrics: Dict[str, float]
    started_at: float
    duration_s: float
    status: str = "completed"                    # "completed" | "failed"
    error: Optional[str] = None


class Sweep(ABC):
    """Base sweep — yields parameter dicts."""

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]: ...

    @abstractmethod
    def __len__(self) -> int: ...


@dataclass
class GridSweep(Sweep):
    """Cartesian product over named lists.

    >>> sweep = GridSweep({"lr": [1e-4, 1e-3], "batch_size": [16, 32]})
    >>> list(sweep)
    [{"batch_size": 16, "lr": 0.0001}, {"batch_size": 16, "lr": 0.001}, ...]
    """

    params: Dict[str, Sequence[Any]]

    def __post_init__(self):
        if not self.params:
            raise ValueError("GridSweep needs at least one parameter")
        for k, v in self.params.items():
            if len(list(v)) == 0:
                raise ValueError(f"parameter {k!r} has empty value list")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        keys = sorted(self.params)
        value_lists = [list(self.params[k]) for k in keys]
        for combo in itertools.product(*value_lists):
            yield dict(zip(keys, combo))

    def __len__(self) -> int:
        n = 1
        for v in self.params.values():
            n *= len(list(v))
        return n


@dataclass
class RandomSweep(Sweep):
    """Random sampling from a structured search space.

    Each entry of ``search_space`` is one of:
      - ``("uniform", lo, hi)``: uniform float in ``[lo, hi]``.
      - ``("loguniform", lo, hi)``: log-uniform float (good for LR / WD).
      - ``("int", lo, hi)``: uniform int in ``[lo, hi]`` (inclusive).
      - ``("choice", [a, b, c])``: discrete choice.

    >>> sweep = RandomSweep(
    ...     search_space={"lr": ("loguniform", 1e-5, 1e-2),
    ...                   "batch_size": ("choice", [16, 32, 64])},
    ...     n_trials=20, seed=0,
    ... )
    """

    search_space: Dict[str, Tuple[Any, ...]]
    n_trials: int = 32
    seed: int = 0

    def __post_init__(self):
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be ≥ 1; got {self.n_trials}")
        for k, spec in self.search_space.items():
            if not isinstance(spec, tuple) or len(spec) < 2:
                raise ValueError(f"search_space[{k!r}] must be a tuple")
            kind = spec[0]
            if kind not in {"uniform", "loguniform", "int", "choice"}:
                raise ValueError(f"unknown distribution: {kind!r}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = random.Random(self.seed)
        for _ in range(self.n_trials):
            params = {}
            for k, spec in self.search_space.items():
                params[k] = self._sample(spec, rng)
            yield params

    def __len__(self) -> int:
        return self.n_trials

    @staticmethod
    def _sample(spec: Tuple[Any, ...], rng: random.Random) -> Any:
        kind = spec[0]
        if kind == "uniform":
            return rng.uniform(spec[1], spec[2])
        if kind == "loguniform":
            log_lo, log_hi = math.log(spec[1]), math.log(spec[2])
            return math.exp(rng.uniform(log_lo, log_hi))
        if kind == "int":
            return rng.randint(spec[1], spec[2])
        if kind == "choice":
            return rng.choice(spec[1])
        raise ValueError(f"unknown distribution: {kind!r}")


class OptunaSweep(Sweep):
    """Optuna-driven Bayesian search. Falls back to ``RandomSweep`` when
    Optuna isn't installed.

    ``search_space_fn(trial)`` is the standard Optuna pattern:

    >>> def search(trial):
    ...     return {
    ...         "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    ...         "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
    ...     }
    """

    def __init__(
        self,
        search_space_fn: Callable[[Any], Dict[str, Any]],
        n_trials: int = 32,
        direction: str = "minimize",
        objective_metric: str = "loss",
    ):
        self.search_space_fn = search_space_fn
        self.n_trials = n_trials
        self.direction = direction
        self.objective_metric = objective_metric
        self._study = None
        self._pending_trials = deque()
        try:
            import optuna  # type: ignore[import-not-found]
            self._optuna = optuna
        except ImportError:
            warnings.warn(
                "Optuna not installed; OptunaSweep falls back to a RandomSweep "
                "stub. ``pip install optuna`` to enable Bayesian search.",
                UserWarning, stacklevel=2,
            )
            self._optuna = None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self._optuna is None:
            # Without Optuna we can't introspect the search space; the
            # caller's `search_space_fn` only works inside an Optuna trial.
            # Best we can do is run it n_trials times against a stub trial
            # that calls ``suggest_*`` with reasonable defaults.
            for i in range(self.n_trials):
                stub = _StubTrial(seed=i)
                yield self.search_space_fn(stub)
            return
        self._study = self._optuna.create_study(direction=self.direction)
        self._pending_trials.clear()
        for _ in range(self.n_trials):
            trial = self._study.ask()
            self._pending_trials.append(trial)
            yield self.search_space_fn(trial)

    def __len__(self) -> int:
        return self.n_trials

    def report_trial_result(
        self,
        metrics: Dict[str, float],
        objective_metric: Optional[str] = None,
    ) -> None:
        if self._optuna is None or self._study is None or not self._pending_trials:
            return
        trial = self._pending_trials.popleft()
        metric_name = objective_metric or self.objective_metric
        if metric_name not in metrics:
            self._study.tell(
                trial, state=self._optuna.trial.TrialState.FAIL,
            )
            return
        self._study.tell(trial, float(metrics[metric_name]))

    def report_trial_failure(self) -> None:
        if self._optuna is None or self._study is None or not self._pending_trials:
            return
        trial = self._pending_trials.popleft()
        self._study.tell(trial, state=self._optuna.trial.TrialState.FAIL)


class _StubTrial:
    """Minimal Optuna-trial stand-in for the no-Optuna fallback path.

    Implements ``suggest_float``, ``suggest_int``, ``suggest_categorical``
    by sampling from a seeded RNG. Doesn't do Bayesian optimisation —
    the warning emitted at OptunaSweep init makes the fallback explicit.
    """

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def suggest_float(self, name, low, high, log=False):
        if log:
            return math.exp(self.rng.uniform(math.log(low), math.log(high)))
        return self.rng.uniform(low, high)

    def suggest_int(self, name, low, high):
        return self.rng.randint(low, high)

    def suggest_categorical(self, name, choices):
        return self.rng.choice(choices)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


@dataclass
class SweepResult:
    trials: List[TrialRecord]
    best_trial: Optional[TrialRecord]
    sweep_name: str

    def to_pandas(self):
        """Return a DataFrame; soft-imports pandas."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas required for to_pandas") from e
        return pd.DataFrame([asdict(t) for t in self.trials])


class SweepRunner:
    """Runs a sweep sequentially; logs each trial to a JSONL file.

    >>> def train(params):
    ...     return {"loss": params["lr"] ** 2}     # toy
    >>> runner = SweepRunner(GridSweep({"lr": [0.01, 0.1, 1.0]}), train,
    ...                      objective_metric="loss", direction="min")
    >>> result = runner.run(log_path="sweep.jsonl")
    >>> result.best_trial.params
    {"lr": 0.01}
    """

    def __init__(
        self,
        sweep: Sweep,
        train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        objective_metric: str = "loss",
        direction: str = "min",
    ):
        if direction not in ("min", "max"):
            raise ValueError(f"direction must be 'min' or 'max'; got {direction}")
        self.sweep = sweep
        self.train_fn = train_fn
        self.objective_metric = objective_metric
        self.direction = direction

    def run(
        self,
        log_path: Optional[str | Path] = None,
        sweep_name: str = "sweep",
    ) -> SweepResult:
        trials: List[TrialRecord] = []
        log_file = None
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w")
        try:
            for i, params in enumerate(self.sweep):
                started = time.time()
                try:
                    metrics = self.train_fn(params)
                    if not isinstance(metrics, dict):
                        raise TypeError(
                            f"train_fn must return a dict; got {type(metrics)}"
                        )
                    rec = TrialRecord(
                        trial_id=i, params=params, metrics=metrics,
                        started_at=started, duration_s=time.time() - started,
                    )
                except Exception as e:
                    rec = TrialRecord(
                        trial_id=i, params=params, metrics={},
                        started_at=started, duration_s=time.time() - started,
                        status="failed", error=str(e),
                    )
                trials.append(rec)
                if rec.status == "completed":
                    reporter = getattr(self.sweep, "report_trial_result", None)
                    if callable(reporter):
                        reporter(rec.metrics, objective_metric=self.objective_metric)
                else:
                    reporter = getattr(self.sweep, "report_trial_failure", None)
                    if callable(reporter):
                        reporter()
                if log_file is not None:
                    log_file.write(json.dumps(asdict(rec)) + "\n")
                    log_file.flush()
        finally:
            if log_file is not None:
                log_file.close()

        best = self._best(trials)
        return SweepResult(trials=trials, best_trial=best, sweep_name=sweep_name)

    def _best(self, trials: List[TrialRecord]) -> Optional[TrialRecord]:
        eligible = [t for t in trials if t.status == "completed" and self.objective_metric in t.metrics]
        if not eligible:
            return None
        if self.direction == "min":
            return min(eligible, key=lambda t: t.metrics[self.objective_metric])
        return max(eligible, key=lambda t: t.metrics[self.objective_metric])
