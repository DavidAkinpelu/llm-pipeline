"""Run manifest — captures everything needed to reproduce a training run.

Without a manifest, "the trained model was good" isn't reproducible —
you can't re-create the run because the config, data, env, and code
state aren't pinned together. This module captures all of that into
one JSON sidecar.

Captured fields:

- Identity: ``run_id``, ``started_at``
- Code state: ``git_sha``, ``git_dirty`` (uncommitted changes flag)
- Versions: ``python_version``, ``torch_version``, ``cuda_version``
- Hardware: ``hostname``, ``gpu_info`` (per-device name + memory)
- Config: full training config dump
- Data: ``dataset_fingerprint`` (length + hash of first/last K examples)
- Reproducibility: ``seed``, filtered ``env_vars``

Honest scope: ``git_sha`` capture uses subprocess + timeout; on
non-git checkouts or systems without git, falls back to ``None``.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


_ENV_VAR_ALLOW_PREFIXES = (
    "LLM_PIPELINE_",
    "CUDA_",
    "NCCL_",
    "TORCH_",
    "PYTORCH_",
)
_ENV_VAR_ALLOW_NAMES = {
    "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM",
}


@dataclass
class RunManifest:
    """Everything we need to reproduce a training run.

    All fields are JSON-serialisable. ``save_to(path)`` / ``load_from(path)``
    round-trip via the standard ``json`` module — no extra deps.
    """

    run_id: str
    started_at: float
    git_sha: Optional[str]
    git_dirty: bool
    python_version: str
    torch_version: str
    cuda_version: Optional[str]
    hostname: str
    user: str
    gpu_info: List[Dict[str, Any]]
    config: Dict[str, Any]
    dataset_fingerprint: Optional[Dict[str, Any]]
    seed: Optional[int]
    env_vars: Dict[str, str]
    notes: str = ""

    def save_to(self, path: str | Path) -> None:
        """Write the manifest as pretty-printed JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True, default=str)

    @classmethod
    def load_from(cls, path: str | Path) -> "RunManifest":
        with Path(path).open("r") as f:
            data = json.load(f)
        return cls(**data)


def _git_sha(repo_dir: Optional[str | Path] = None) -> tuple[Optional[str], bool]:
    """Return ``(sha, dirty)``. ``sha=None`` if git unavailable / non-git."""
    cwd = str(repo_dir) if repo_dir else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=2.0,
        )
        if sha.returncode != 0:
            return None, False
        head = sha.stdout.strip()
        # Dirty = any uncommitted changes.
        diff = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=2.0,
        )
        dirty = bool(diff.stdout.strip())
        return head, dirty
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None, False


def _gpu_info() -> List[Dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info.append({
            "index": i,
            "name": props.name,
            "memory_gb": round(props.total_memory / (1024 ** 3), 2),
            "compute_capability": f"{props.major}.{props.minor}",
        })
    return info


def _filtered_env() -> Dict[str, str]:
    """Capture env vars relevant to LLM training.

    Filtered to:
      - Project-prefixed: ``LLM_PIPELINE_*``
      - Common ML env: ``CUDA_*``, ``NCCL_*``, ``TORCH_*``, ``PYTORCH_*``
      - Specifically-named: PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG, etc.

    Skips PATH, HOME, LD_LIBRARY_PATH, secrets — anything that's either
    too noisy or potentially sensitive.
    """
    out = {}
    for key, value in os.environ.items():
        if key in _ENV_VAR_ALLOW_NAMES or any(key.startswith(p) for p in _ENV_VAR_ALLOW_PREFIXES):
            out[key] = value
    return out


def _dataset_fingerprint(
    dataset: Optional[Iterable[Any]] = None, n_sample: int = 16,
) -> Optional[Dict[str, Any]]:
    """Hash a sample of the dataset's first / last N records.

    Cheap reproducibility check — different versions of the same
    dataset produce different fingerprints. Doesn't require iterating
    the whole dataset when the dataset is sized + indexable. Iterable-
    only / streaming datasets are skipped to avoid consuming them.
    """
    if dataset is None:
        return None
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        # Streaming / iterable-only datasets are intentionally skipped.
        return {"length": None, "hash": None}
    n = len(dataset)
    sample_idxs = list(range(min(n_sample, n))) + list(range(max(n - n_sample, 0), n))
    sample_idxs = sorted(set(sample_idxs))
    sample = [str(dataset[i]) for i in sample_idxs]
    h = hashlib.blake2b("\n".join(sample).encode("utf-8"), digest_size=16).hexdigest()
    return {"length": n, "hash": h, "sample_size": len(sample_idxs)}


def capture_run_environment(
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    dataset: Optional[Iterable[Any]] = None,
    repo_dir: Optional[str | Path] = None,
    notes: str = "",
) -> RunManifest:
    """Build a ``RunManifest`` from the current process state.

    Call this once at the start of training; pass the resulting
    manifest's ``save_to`` to write it next to the checkpoint dir.
    """
    git_sha, git_dirty = _git_sha(repo_dir)
    cuda_version = torch.version.cuda if torch.cuda.is_available() else None
    return RunManifest(
        run_id=str(uuid.uuid4()),
        started_at=time.time(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        hostname=socket.gethostname(),
        user=getpass.getuser(),
        gpu_info=_gpu_info(),
        config=dict(config or {}),
        dataset_fingerprint=_dataset_fingerprint(dataset),
        seed=seed,
        env_vars=_filtered_env(),
        notes=notes,
    )
