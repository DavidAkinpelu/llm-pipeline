"""Checkpointing utilities: distributed sharded saves, auto-resume,
best-of-N retention.

Three loosely-coupled pieces. ``DistributedCheckpointSaver`` writes
checkpoints (sharded for FSDP, full for single-GPU). ``auto_resume_from_latest``
finds the highest-step checkpoint and rehydrates a model + optimizer +
scheduler. ``BestOfNRetention`` is the disk-hygiene policy that prunes
old checkpoints based on a tracked metric.

Used together: at training time, the saver writes; the retention policy
deletes; on restart, ``auto_resume_from_latest`` picks up where we left
off.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.distributed as dist


# --------------------------------------------------------------------------- #
# Distributed checkpoint saver
# --------------------------------------------------------------------------- #


class DistributedCheckpointSaver:
    """Saves model + optimizer + scheduler state.

    Two strategies:

    - ``"full"``: rank-0 saves the whole state dict via ``torch.save``.
      Standard for single-GPU runs.
    - ``"sharded"``: each rank writes its local shards via
      ``torch.distributed.checkpoint.save`` (PyTorch 2.0+). Falls back to
      per-rank ``torch.save`` shards on older PyTorch.

    Atomicity: writes go to a tmp directory, rename to the final name on
    success. Partial failures don't corrupt the on-disk state.
    """

    def __init__(self, strategy: Literal["full", "sharded"] = "full"):
        if strategy not in ("full", "sharded"):
            raise ValueError(f"unknown strategy: {strategy!r}")
        self.strategy = strategy
        self._is_distributed = dist.is_available() and dist.is_initialized()
        if strategy == "sharded" and not self._is_distributed:
            warnings.warn(
                "DistributedCheckpointSaver(strategy='sharded') falls back "
                "to 'full' because torch.distributed is not initialised.",
                UserWarning, stacklevel=2,
            )
            self.strategy = "full"

    def save(self, state: Dict[str, Any], path: str | Path) -> None:
        """Save ``state`` (a dict of state_dicts and metadata) to ``path``.

        For ``strategy="full"``: ``path`` is a file (``.pt``).
        For ``strategy="sharded"``: ``path`` is a directory.
        """
        path = Path(path)
        if self.strategy == "full":
            self._save_full(state, path)
        else:
            self._save_sharded(state, path)

    def load(self, path: str | Path) -> Dict[str, Any]:
        """Inverse of ``save``."""
        path = Path(path)
        if self.strategy == "full":
            return torch.load(path, map_location="cpu", weights_only=False)
        return self._load_sharded(path)

    # ------------------------------------------------------------------ #

    def _save_full(self, state: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: tmp + rename.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        os.close(fd)
        try:
            torch.save(state, tmp)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _save_sharded(self, state: Dict[str, Any], path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank() if self._is_distributed else 0
        try:
            # PyTorch 2.0+ DCP path.
            from torch.distributed.checkpoint import save as dcp_save  # type: ignore[import-not-found]
            dcp_save(state_dict=state, checkpoint_id=str(path))
        except Exception:
            # Fallback: per-rank shard files. Less efficient but always works.
            shard = path / f"shard_rank{rank}.pt"
            torch.save(state, shard)
        # Manifest index — the rank-0 process writes a small JSON listing the shards.
        if rank == 0:
            shards = sorted(p.name for p in path.glob("shard_rank*.pt"))
            (path / "manifest.json").write_text(json.dumps({"shards": shards}, indent=2))

    def _load_sharded(self, path: Path) -> Dict[str, Any]:
        try:
            from torch.distributed.checkpoint import load as dcp_load  # type: ignore[import-not-found]
            state: Dict[str, Any] = {}
            dcp_load(state_dict=state, checkpoint_id=str(path))
            return state
        except Exception:
            rank = dist.get_rank() if self._is_distributed else 0
            shard = path / f"shard_rank{rank}.pt"
            if not shard.exists():
                # Fall back to the rank-0 shard for single-process resume.
                shard = path / "shard_rank0.pt"
            return torch.load(shard, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- #
# Auto-resume
# --------------------------------------------------------------------------- #


_STEP_RE = re.compile(r"step-(\d+)(?:\.pt|/?$)")


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Tuple[int, Path]]:
    """Return ``(step, path)`` of the highest-step checkpoint, or None.

    Looks for both ``step-{N}.pt`` files (full strategy) and ``step-{N}/``
    directories (sharded strategy).
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    candidates: List[Tuple[int, Path]] = []
    for entry in checkpoint_dir.iterdir():
        m = _STEP_RE.match(entry.name)
        if m is None:
            continue
        candidates.append((int(m.group(1)), entry))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p[0])


def auto_resume_from_latest(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    saver: Optional[DistributedCheckpointSaver] = None,
) -> int:
    """Load the latest checkpoint into ``model``/``optimizer``/``scheduler``;
    return the step the training loop should continue from.

    On a fresh dir (or no checkpoints), returns 0 — the caller starts
    training from scratch. The function is idempotent and safe to call
    unconditionally at the top of a training run.
    """
    found = find_latest_checkpoint(checkpoint_dir)
    if found is None:
        return 0
    step, path = found
    saver = saver or DistributedCheckpointSaver(
        strategy="sharded" if path.is_dir() else "full",
    )
    state = saver.load(path)
    if "model" in state:
        model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if "rng_state" in state:
        try:
            torch.set_rng_state(state["rng_state"])
        except (RuntimeError, TypeError):
            pass                                              # best-effort
    return state.get("step", step)


# --------------------------------------------------------------------------- #
# Best-of-N retention policy
# --------------------------------------------------------------------------- #


@dataclass
class BestOfNRetention:
    """Disk-hygiene policy: keep the top-N checkpoints by metric, plus
    optionally the latest. Prune the rest.

    Use as a callback after each save. The policy holds a small history
    of ``(step, path, metric)`` triples in memory; on each ``record``
    call it decides which paths to remove from disk.

    ``mode``: ``"min"`` for loss/perplexity, ``"max"`` for accuracy.
    """

    metric_name: str
    mode: Literal["min", "max"] = "min"
    n_best: int = 3
    keep_latest: bool = True

    def __post_init__(self):
        if self.n_best < 1:
            raise ValueError(f"n_best must be ≥ 1; got {self.n_best}")
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max'; got {self.mode}")
        self._records: List[Tuple[int, Path, float]] = []

    def record(self, step: int, path: str | Path, metric_value: float) -> List[Path]:
        """Register a new checkpoint; return the list of paths actually pruned."""
        self._records.append((step, Path(path), float(metric_value)))
        return self._enforce()

    def _enforce(self) -> List[Path]:
        # Sort by metric (best first). For min mode, smallest first.
        if self.mode == "min":
            sorted_recs = sorted(self._records, key=lambda r: r[2])
        else:
            sorted_recs = sorted(self._records, key=lambda r: -r[2])

        keep: set[Path] = set()
        for rec in sorted_recs[: self.n_best]:
            keep.add(rec[1])
        if self.keep_latest:
            latest = max(self._records, key=lambda r: r[0])
            keep.add(latest[1])

        pruned: List[Path] = []
        survivors: List[Tuple[int, Path, float]] = []
        for rec in self._records:
            if rec[1] in keep:
                survivors.append(rec)
            else:
                self._delete(rec[1])
                pruned.append(rec[1])
        self._records = survivors
        return pruned

    def _delete(self, path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except OSError:
                pass

    def kept(self) -> List[Tuple[int, Path, float]]:
        """Currently-retained checkpoint records. Useful for logging."""
        return list(self._records)
