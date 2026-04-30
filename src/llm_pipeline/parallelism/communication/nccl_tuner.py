"""NCCL collective tuning harness.

Times all-reduce variants under different buffer sizes / algorithms so
the user can pick a config that matches their cluster topology. Real
benchmarks need ≥2 GPU NCCL host; the API + bookkeeping here ships
unblocked, with a clear "needs multi-GPU" note when invoked single-rank.

Two metrics:
- **Bandwidth** (GB/s) — total bytes moved / wall time
- **Latency** (us) — wall time per collective

The tuner sweeps buffer sizes from 1 KB to 256 MB by powers of 2 and
records both. The result lets the surrounding training stack pick a
``NCCL_BUFFSIZE`` env var that minimises end-to-end iter time.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class TuningResult:
    """One measurement at one buffer size."""

    buffer_bytes: int
    latency_us: float
    bandwidth_gbps: float


@dataclass
class NCCLTunerConfig:
    """Knobs for ``NCCLTuner.run``.

    Attributes
    ----------
    sizes_kb : list[int]
        Buffer sizes to sweep, in kibibytes.
    n_warmup : int
        Warmup iterations per size (NCCL has substantial first-call overhead).
    n_iter : int
        Timed iterations per size; the tuner reports the median.
    """

    sizes_kb: List[int] = field(default_factory=lambda: [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144])
    n_warmup: int = 5
    n_iter: int = 20


class NCCLTuner:
    """Time NCCL all-reduce at a sweep of buffer sizes.

    Single-rank invocation runs locally for API testing; the actual
    bandwidth measurement requires a multi-rank distributed group.
    """

    def __init__(self, config: Optional[NCCLTunerConfig] = None):
        self.config = config or NCCLTunerConfig()

    def run(self, device: Optional[torch.device] = None) -> List[TuningResult]:
        """Sweep buffer sizes and time NCCL all-reduce. Returns one
        ``TuningResult`` per size.

        Behaviour:
          - If ``torch.distributed`` is not initialised, returns an
            empty list and prints a clear "needs multi-GPU" note.
          - Otherwise, runs the sweep on the current process group.
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            print(
                "[NCCLTuner] torch.distributed not initialised; "
                "skipping benchmark — needs ≥2 GPU NCCL host. "
                "Use ``torchrun --nproc-per-node=N`` to launch.",
            )
            return []

        device = device or torch.device(f"cuda:{torch.distributed.get_rank() % torch.cuda.device_count()}")
        results: List[TuningResult] = []
        for size_kb in self.config.sizes_kb:
            n_floats = size_kb * 1024 // 4              # fp32
            buf = torch.randn(n_floats, device=device)

            # Warmup.
            for _ in range(self.config.n_warmup):
                torch.distributed.all_reduce(buf)
            torch.cuda.synchronize(device)

            # Timed.
            t0 = time.perf_counter()
            for _ in range(self.config.n_iter):
                torch.distributed.all_reduce(buf)
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - t0) / self.config.n_iter

            buffer_bytes = n_floats * 4
            bandwidth = buffer_bytes / max(elapsed, 1e-9) / 1e9
            results.append(TuningResult(
                buffer_bytes=buffer_bytes,
                latency_us=elapsed * 1e6,
                bandwidth_gbps=bandwidth,
            ))
        return results

    @staticmethod
    def best_buffer_size(results: List[TuningResult]) -> Optional[int]:
        """Return the buffer size (bytes) that maxed out bandwidth."""
        if not results:
            return None
        return max(results, key=lambda r: r.bandwidth_gbps).buffer_bytes
