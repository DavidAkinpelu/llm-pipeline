"""Model lifecycle registry — track which weight versions exist on disk.

The serving stack needs a system of record for "which weights are good
to serve right now": names, versions, paths, training metadata, and a
retire/deprecate pointer so old weights get pulled from rotation safely.

Persistence: a JSONL file. One line per registered version. Atomic
operations via ``os.replace`` on a tmp file so concurrent registrations
from independent processes never see a torn file. No SQLite dep — for
the scale this typically operates at (hundreds-to-low-thousands of
versions), JSONL is plenty fast and is human-debuggable.

Operations
----------

- ``register(name, version, path, metadata)`` — append a new version.
- ``list(name=None)`` — all versions, optionally filtered by name.
- ``get(name, version="latest")`` — fetch a specific entry; ``"latest"``
  returns the most recently registered non-deprecated version.
- ``deprecate(name, version)`` — mark a version unsafe to serve.
- ``prune_deprecated()`` — physically remove deprecated entries from
  the index (keeps the file small).

Concurrency: single-machine multi-process is supported via the
atomic-rename pattern; multi-machine writes need an external lock.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from typing import ClassVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows-only fallback
    fcntl = None
    import msvcrt


class WeightVersionError(Exception):
    """Raised on duplicate registration or "latest" lookup with no candidates."""


@dataclass
class WeightVersion:
    """One registered weight version.

    Attributes
    ----------
    name : str
        The logical model name (e.g. ``qwen3-0.6b-fine-tuned``).
    version : str
        Caller-assigned version string. Conventional but not enforced:
        SemVer or ``YYYYMMDD-N``.
    path : str
        On-disk path to the weights. Caller's responsibility to keep
        valid; the registry doesn't dereference it on read.
    metadata : dict
        Arbitrary JSON-serialisable metadata (training hash, eval scores,
        promotion notes).
    registered_at : float
        Unix timestamp; auto-set if omitted at registration.
    deprecated : bool
        ``True`` once retired — excluded from "latest" lookups but still
        retrievable by exact (name, version) for audit / rollback.
    """

    name: str
    version: str
    path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = 0.0
    deprecated: bool = False


class ModelLifecycleRegistry:
    """JSONL-backed catalogue of weight versions.

    >>> reg = ModelLifecycleRegistry("/var/lib/llm-pipeline/registry.jsonl")
    >>> reg.register("qwen-fine-1", "v1", "/weights/qwen-v1", {"eval_loss": 1.2})
    >>> reg.register("qwen-fine-1", "v2", "/weights/qwen-v2", {"eval_loss": 1.1})
    >>> reg.get("qwen-fine-1", "latest").version
    'v2'
    >>> reg.deprecate("qwen-fine-1", "v2")
    >>> reg.get("qwen-fine-1", "latest").version
    'v1'
    """

    _inproc_guard: ClassVar[threading.Lock] = threading.Lock()
    _inproc_locks: ClassVar[Dict[Path, threading.Lock]] = {}

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        # Bootstrap an empty index so the first read doesn't FileNotFoundError.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        with self._inproc_guard:
            self._lock = self._inproc_locks.setdefault(self.path.resolve(), threading.Lock())

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    def list(self, name: Optional[str] = None, include_deprecated: bool = True) -> List[WeightVersion]:
        with self._locked_io():
            entries = self._load_all()
        if name is not None:
            entries = [e for e in entries if e.name == name]
        if not include_deprecated:
            entries = [e for e in entries if not e.deprecated]
        return entries

    def get(self, name: str, version: str = "latest") -> WeightVersion:
        with self._locked_io():
            entries = [e for e in self._load_all() if e.name == name]
        if version == "latest":
            non_dep = [e for e in entries if not e.deprecated]
            if not non_dep:
                raise WeightVersionError(
                    f"no non-deprecated versions for {name!r}",
                )
            return max(non_dep, key=lambda e: e.registered_at)
        for e in entries:
            if e.version == version:
                return e
        raise WeightVersionError(f"no entry for {name!r}@{version!r}")

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def register(
        self,
        name: str,
        version: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        registered_at: Optional[float] = None,
    ) -> WeightVersion:
        """Append a new ``(name, version)`` entry.

        Re-registering an existing ``(name, version)`` updates its path
        and metadata (idempotent for clients that retry on partial
        failures).
        """
        import time as _time
        ts = registered_at if registered_at is not None else _time.time()
        new = WeightVersion(
            name=name, version=version, path=path,
            metadata=dict(metadata or {}),
            registered_at=ts, deprecated=False,
        )
        with self._locked_io():
            entries = self._load_all()
            # Replace any existing (name, version) entry — idempotent.
            entries = [e for e in entries if not (e.name == name and e.version == version)]
            entries.append(new)
            self._atomic_save(entries)
        return new

    def deprecate(self, name: str, version: str) -> None:
        with self._locked_io():
            entries = self._load_all()
            found = False
            for e in entries:
                if e.name == name and e.version == version:
                    e.deprecated = True
                    found = True
                    break
            if not found:
                raise WeightVersionError(f"no entry for {name!r}@{version!r}")
            self._atomic_save(entries)

    def prune_deprecated(self) -> int:
        """Physically remove all deprecated entries; return the number pruned."""
        with self._locked_io():
            entries = self._load_all()
            kept = [e for e in entries if not e.deprecated]
            n = len(entries) - len(kept)
            self._atomic_save(kept)
            return n

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #

    def _load_all(self) -> List[WeightVersion]:
        out: List[WeightVersion] = []
        with self.path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                out.append(WeightVersion(**obj))
        return out

    def _atomic_save(self, entries: List[WeightVersion]) -> None:
        # tmp file + rename. ``os.replace`` is atomic on POSIX and Windows.
        fd, tmp_path = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                for e in entries:
                    f.write(json.dumps(asdict(e), separators=(",", ":")) + "\n")
            os.replace(tmp_path, self.path)
        except Exception:
            # Best-effort cleanup of the stranded tmp file.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @contextmanager
    def _locked_io(self):
        with self._lock:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock_path.open("a+b") as lock_file:
                self._acquire_file_lock(lock_file)
                try:
                    yield
                finally:
                    self._release_file_lock(lock_file)

    @staticmethod
    def _acquire_file_lock(lock_file) -> None:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        else:  # pragma: no cover - Windows-only fallback
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)

    @staticmethod
    def _release_file_lock(lock_file) -> None:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        else:  # pragma: no cover - Windows-only fallback
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
