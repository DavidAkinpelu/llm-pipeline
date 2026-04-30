"""Multi-model routing for the serving layer.

Lets one server expose multiple LLMs and pick which to use per request
via the ``model`` field in OpenAI-compatible chat / completion bodies.

Two collaborating pieces:

- ``MultiModelRegistry`` — name → engine map with lazy load on first
  request and optional LRU eviction once a memory budget is hit.
- ``select_engine_for_request(registry, requested_model)`` — small
  routing helper that the existing ``server.py`` routes call to pick
  which engine handles a request (with fallback to a default).

The engine type is intentionally a duck — anything with ``generate(...)``
fits, including the existing ``Qwen3InferenceEngine`` and the generic
HF wrapper. The registry holds **factories** (callables that return an
engine), not eagerly-loaded engines, so registering 20 models at startup
doesn't OOM the GPU.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


# Type alias: a model factory yields an engine on first call.
EngineFactory = Callable[[], Any]


@dataclass(frozen=True)
class ModelEntry:
    """A registered model — its name, factory, and optional metadata."""

    name: str
    factory: EngineFactory
    metadata: Dict[str, Any]


class MultiModelRegistry:
    """Name → engine map with lazy load + optional LRU eviction.

    >>> registry = MultiModelRegistry(max_loaded=2)
    >>> registry.register("qwen-0.6b", lambda: build_engine("Qwen/Qwen3-0.6B"))
    >>> registry.register("llama-1b",  lambda: build_engine("meta-llama/Llama-3.2-1B"))
    >>> engine = registry.get("qwen-0.6b")    # loads on first access
    >>> engine = registry.get("qwen-0.6b")    # cached

    With ``max_loaded`` set, the registry evicts the least-recently-used
    engine when a new model would push past the budget. Use ``None`` for
    "load everything, never evict".
    """

    def __init__(self, max_loaded: Optional[int] = None, default_model: Optional[str] = None):
        if max_loaded is not None and max_loaded < 1:
            raise ValueError(f"max_loaded must be ≥ 1 or None; got {max_loaded}")
        self.max_loaded = max_loaded
        self.default_model = default_model
        self._entries: Dict[str, ModelEntry] = {}
        self._loaded: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        factory: EngineFactory,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a model. Idempotent — re-registering replaces the entry
        and evicts any cached engine under that name.
        """
        with self._lock:
            self._entries[name] = ModelEntry(
                name=name, factory=factory, metadata=dict(metadata or {}),
            )
            self._loaded.pop(name, None)
            if self.default_model is None:
                self.default_model = name

    def unregister(self, name: str) -> bool:
        with self._lock:
            removed = self._entries.pop(name, None) is not None
            self._loaded.pop(name, None)
            if self.default_model == name:
                self.default_model = next(iter(self._entries), None)
            return removed

    def list_models(self) -> List[str]:
        with self._lock:
            return list(self._entries.keys())

    def get_metadata(self, name: str) -> Dict[str, Any]:
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise KeyError(name)
            return dict(entry.metadata)

    def get(self, name: str) -> Any:
        """Return the loaded engine for ``name``, loading on first access.

        Raises ``KeyError`` if no model with that name is registered.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise KeyError(f"unknown model: {name!r}; registered: {list(self._entries)}")
            cached = self._loaded.get(name)
            if cached is not None:
                # Mark as MRU.
                self._loaded.move_to_end(name)
                return cached

            # LRU eviction before loading the new engine.
            while self.max_loaded is not None and len(self._loaded) >= self.max_loaded:
                evicted_name, _ = self._loaded.popitem(last=False)
                # The evicted engine is just dropped; if its destructor frees
                # GPU memory (typical for our engines) this is enough.
                del evicted_name

            engine = entry.factory()
            self._loaded[name] = engine
            return engine

    def is_loaded(self, name: str) -> bool:
        with self._lock:
            return name in self._loaded

    def loaded_models(self) -> List[str]:
        """Currently-resident engines, in LRU order (oldest first)."""
        with self._lock:
            return list(self._loaded.keys())


# --------------------------------------------------------------------------- #
# Routing helper
# --------------------------------------------------------------------------- #


def select_engine_for_request(
    registry: MultiModelRegistry,
    requested_model: Optional[str],
) -> tuple[str, Any]:
    """Pick which engine handles a request.

    Returns ``(model_name, engine)``. Falls back to ``registry.default_model``
    when ``requested_model`` is None or empty. Raises ``KeyError`` if the
    requested model isn't registered (route should map this to HTTP 404).
    """
    name = requested_model or registry.default_model
    if name is None:
        raise KeyError("no models registered and no default set")
    engine = registry.get(name)
    return name, engine
