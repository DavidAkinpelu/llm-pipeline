"""Resource and cost tracking for training runs.

Accumulates the metrics that matter for ops + budget:

- Wall time (total + by-phase if you ``mark_phase``)
- GPU-time (sum of per-rank wall × n_gpus)
- Peak GPU memory per device
- Tokens/sec, samples/sec

``estimate_cost(gpu_hourly_rate)`` returns dollars from accumulated
GPU-hours. Rates aren't hardcoded — cloud prices change weekly; the
caller passes their own rate.

Production stacks pipe this into Prometheus / DataDog. The lightweight
in-process version here is deliberate: it integrates with the existing
``Trainer`` callbacks without any extra infra.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class PhaseTiming:
    name: str
    started_at: float
    duration_s: float = 0.0


class RunTracker:
    """Accumulates resource + throughput stats across a training run.

    Typical use:

    >>> tracker = RunTracker(n_gpus=torch.cuda.device_count() or 1)
    >>> tracker.start()
    >>> # In training loop:
    >>> tracker.record_step(n_samples=batch_size, n_tokens=batch_tokens)
    >>> # Optionally mark phases:
    >>> with tracker.phase("data_loading"):
    ...     batch = next(loader)
    >>> tracker.stop()
    >>> summary = tracker.summary()        # dict
    >>> dollars = tracker.estimate_cost(gpu_hourly_rate=2.50)
    """

    def __init__(self, n_gpus: int = 1):
        if n_gpus < 1:
            raise ValueError(f"n_gpus must be ≥ 1; got {n_gpus}")
        self.n_gpus = n_gpus
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None
        self._steps = 0
        self._samples = 0
        self._tokens = 0
        self._phases: List[PhaseTiming] = []
        self._peak_memory_gb: Dict[int, float] = {}

    def start(self) -> None:
        self._start_time = time.time()
        self._stop_time = None
        self._peak_memory_gb = {}
        self._reset_peak_memory()
        self._update_peak_memory()

    def stop(self) -> None:
        self._stop_time = time.time()
        self._update_peak_memory()

    def record_step(self, n_samples: int = 0, n_tokens: int = 0) -> None:
        """Call once per training step."""
        self._steps += 1
        self._samples += int(n_samples)
        self._tokens += int(n_tokens)
        self._update_peak_memory()

    def phase(self, name: str):
        """Context manager that times a labelled phase."""
        return _PhaseCM(self, name)

    @property
    def wall_time_s(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._stop_time if self._stop_time is not None else time.time()
        return end - self._start_time

    @property
    def gpu_hours(self) -> float:
        return self.wall_time_s * self.n_gpus / 3600.0

    @property
    def tokens_per_sec(self) -> float:
        wall = self.wall_time_s
        return self._tokens / wall if wall > 0 else 0.0

    @property
    def samples_per_sec(self) -> float:
        wall = self.wall_time_s
        return self._samples / wall if wall > 0 else 0.0

    def estimate_cost(self, gpu_hourly_rate: float) -> float:
        """Dollar estimate from accumulated GPU-hours × rate."""
        return self.gpu_hours * gpu_hourly_rate

    def summary(self) -> Dict[str, Any]:
        """Headline metrics + per-phase breakdown."""
        return {
            "wall_time_s": round(self.wall_time_s, 2),
            "gpu_hours": round(self.gpu_hours, 4),
            "n_gpus": self.n_gpus,
            "n_steps": self._steps,
            "n_samples": self._samples,
            "n_tokens": self._tokens,
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "samples_per_sec": round(self.samples_per_sec, 2),
            "peak_memory_gb": dict(self._peak_memory_gb),
            "phases": [
                {"name": p.name, "duration_s": round(p.duration_s, 3)}
                for p in self._phases
            ],
        }

    def _update_peak_memory(self) -> None:
        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            try:
                peak_b = torch.cuda.max_memory_allocated(i)
                gb = peak_b / (1024 ** 3)
                if gb > self._peak_memory_gb.get(i, 0.0):
                    self._peak_memory_gb[i] = round(gb, 3)
            except (RuntimeError, AssertionError):
                pass

    def _reset_peak_memory(self) -> None:
        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(i)
            except (RuntimeError, AssertionError):
                pass


class _PhaseCM:
    """Context manager that records a phase timing on the tracker."""

    def __init__(self, tracker: RunTracker, name: str):
        self.tracker = tracker
        self.name = name
        self._t0: Optional[float] = None

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        duration = time.time() - self._t0
        self.tracker._phases.append(PhaseTiming(
            name=self.name, started_at=self._t0, duration_s=duration,
        ))
        return False
