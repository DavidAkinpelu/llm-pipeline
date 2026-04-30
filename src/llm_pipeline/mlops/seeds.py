"""Deterministic seeding for reproducible runs.

One call to ``set_global_seed`` covers Python's ``random``, NumPy,
PyTorch (CPU + CUDA), and the ``PYTHONHASHSEED`` env var. The
``deterministic`` flag toggles cuDNN's deterministic mode (slower but
bit-reproducible).

**Caveat surfaced explicitly**: even with all of these, true bit-level
determinism on GPU requires ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` to be
set in the environment **before** Python imports torch. We can't do
that for you (the env var is read at CUDA-context init); we document it
and check it in ``set_global_seed``, emitting a warning when missing.
"""

from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class SeedState:
    """Snapshot of the seed configuration. Pass to ``restore_seed_state``
    to roll back to this point — useful for tests that mutate global
    seed state.
    """

    seed: int
    python_hash_seed: Optional[str]
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cublas_workspace_config: Optional[str]
    deterministic_algorithms: Optional[bool]


def set_global_seed(seed: int, deterministic: bool = True) -> SeedState:
    """Seed every RNG the project touches and return a ``SeedState``
    describing what was set.

    Parameters
    ----------
    seed : int
        Master seed. The same value is fed to all RNGs; the project's
        few places that need independent streams derive sub-seeds from
        this (e.g., ``Generator().manual_seed(seed + offset)``).
    deterministic : bool
        If True, force cuDNN into deterministic mode (no benchmarking,
        no kernel auto-tuning). 10-20% slower on conv-heavy workloads;
        much smaller penalty on transformer training. **Set
        ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in your shell** before launch
        for full bit-reproducibility on GPU; ``set_global_seed`` warns
        if it's missing.
    """
    if seed < 0:
        raise ValueError(f"seed must be ≥ 0; got {seed}")

    state = SeedState(
        seed=seed,
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        deterministic_algorithms=(
            torch.are_deterministic_algorithms_enabled()
            if hasattr(torch, "are_deterministic_algorithms_enabled")
            else None
        ),
    )

    # Python.
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # NumPy.
    np.random.seed(seed)
    # Torch.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # On older torch (<1.12) ``use_deterministic_algorithms`` may not exist.
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        if torch.cuda.is_available() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            warnings.warn(
                "set_global_seed(deterministic=True): full GPU "
                "bit-determinism needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in "
                "the environment before Python imports torch. Set it in "
                "your launch script.",
                UserWarning, stacklevel=2,
            )
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass
    return state


def restore_seed_state(state: SeedState) -> None:
    """Reverse a previous ``set_global_seed`` call.

    Restores the cudnn flags + the env vars to whatever they were before
    the seed call. RNG state itself is reset to the snapshot's ``seed``
    (we can't restore the *consumed* RNG state without saving the full
    Mersenne Twister state, which is overkill for the test-isolation use
    case this targets).
    """
    if state.python_hash_seed is None:
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        os.environ["PYTHONHASHSEED"] = state.python_hash_seed
    if state.cublas_workspace_config is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = state.cublas_workspace_config
    torch.backends.cudnn.deterministic = state.cudnn_deterministic
    torch.backends.cudnn.benchmark = state.cudnn_benchmark
    if (
        state.deterministic_algorithms is not None
        and hasattr(torch, "use_deterministic_algorithms")
    ):
        try:
            torch.use_deterministic_algorithms(state.deterministic_algorithms)
        except Exception:
            pass
    random.seed(state.seed)
    np.random.seed(state.seed)
    torch.manual_seed(state.seed)
