"""Per-request LoRA adapter routing for the serving layer.

The ``MultiAdapterLoRALinear`` module already supports holding N LoRA
adapters and switching between them at the module level via
``set_active_adapters([...])``. What's left for serving is plumbing:

- Load N adapters at server startup, register them with the model.
- On each request, look at a header / param (``X-LoRA-Adapter`` or
  ``adapter`` field in the JSON body) and decide which adapter(s) to
  activate before forward.
- Restore the default adapter set after the response so concurrent
  requests don't poison each other.

This module implements that as a context manager:

>>> with router.activate("medical-fine-tune"):
...     output = engine.generate(prompt, ...)

Plus a ``LoRARouter.from_config(model, paths_dict)`` helper that loads
adapters from disk at construction time.

Concurrency note
----------------

LoRA routing as implemented here mutates **module-level state** on the
shared model — two concurrent requests with different adapters will
race. The router's ``activate`` context manager takes a per-router
asyncio lock; the serving layer should use the async API and serialise
adapter switches. For true concurrent multi-LoRA inference, the model
needs per-request adapter masks (the "S-LoRA" approach), which is a
substantially bigger refactor and is roadmapped separately.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import torch.nn as nn

from ...adapters.adapter_manager import MultiAdapterLoRALinear


@dataclass
class LoRARouter:
    """Picks which LoRA adapter(s) to activate per request.

    Attributes
    ----------
    model : nn.Module
        The wrapped model. Must contain at least one
        ``MultiAdapterLoRALinear`` somewhere in its module tree.
    default_adapters : list[str]
        Adapters to activate when no per-request override is given.
        Empty list means "no adapter active" (raw base model).
    """

    model: nn.Module
    default_adapters: List[str] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def known_adapters(self) -> List[str]:
        """Return the union of adapter names present on any
        ``MultiAdapterLoRALinear`` module in the model tree.
        """
        seen: set[str] = set()
        for m in self.model.modules():
            if isinstance(m, MultiAdapterLoRALinear):
                seen.update(m.get_adapter_info().keys())
        return sorted(seen)

    @contextlib.contextmanager
    def activate(self, adapter_names: Optional[Sequence[str]] = None) -> Iterator[None]:
        """Temporarily activate ``adapter_names`` (or ``default_adapters`` if
        None) on every ``MultiAdapterLoRALinear`` in the model. Restores the
        previous active set on exit, even if the wrapped block raises.
        """
        names = list(adapter_names) if adapter_names is not None else list(self.default_adapters)
        self._validate(names)

        # Snapshot the current active adapters per module so we can restore.
        snapshots: List[tuple[MultiAdapterLoRALinear, List[str]]] = []
        for m in self.model.modules():
            if isinstance(m, MultiAdapterLoRALinear):
                snapshots.append((m, list(m.get_active_adapters())))
        try:
            for m, _ in snapshots:
                m.set_active_adapters(names)
            yield
        finally:
            for m, prev in snapshots:
                m.set_active_adapters(prev)

    @contextlib.asynccontextmanager
    async def activate_async(self, adapter_names: Optional[Sequence[str]] = None):
        """Async-friendly version of ``activate`` that serialises adapter
        switches via the router's lock.
        """
        async with self._lock:
            with self.activate(adapter_names):
                yield

    def _validate(self, names: Sequence[str]) -> None:
        if not names:
            return  # empty is fine — disables LoRA entirely
        known = set(self.known_adapters())
        for n in names:
            if n not in known:
                raise KeyError(f"unknown adapter: {n!r}; known: {sorted(known)}")

    @classmethod
    def from_paths(
        cls,
        model: nn.Module,
        adapter_paths: Dict[str, str | Path],
        default_adapters: Optional[Sequence[str]] = None,
    ) -> "LoRARouter":
        """Construct a router by loading adapters from on-disk paths.

        For each ``(name, path)``, calls ``module.load_adapter(name, path)``
        on every ``MultiAdapterLoRALinear`` module in the tree. The router
        then exposes those names through ``activate``.
        """
        for m in model.modules():
            if isinstance(m, MultiAdapterLoRALinear):
                for name, path in adapter_paths.items():
                    m.load_adapter(name, str(path))
        return cls(model=model, default_adapters=list(default_adapters or []))
