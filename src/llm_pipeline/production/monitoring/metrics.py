"""Prometheus metrics for the serving layer.

If prometheus_client is unavailable, falls back to in-memory counters so
unit tests and lightweight deployments keep working — the /metrics endpoint
just returns a stub payload in that case.
"""

from contextlib import contextmanager
from time import perf_counter
from typing import Tuple

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )
    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False

    class Counter:  # type: ignore[no-redef]
        def __init__(self, *_a, **_k): self._v = 0
        def inc(self, n: float = 1.0): self._v += n
        def labels(self, **_k): return self

    class CollectorRegistry:  # type: ignore[no-redef]
        pass

    class Histogram:  # type: ignore[no-redef]
        def __init__(self, *_a, **_k): self._obs = []
        def observe(self, v: float): self._obs.append(v)
        def labels(self, **_k): return self

    def generate_latest(*_a, **_k) -> bytes:  # type: ignore[no-redef]
        return b"# prometheus_client not installed\n"

    CONTENT_TYPE_LATEST = "text/plain"  # type: ignore[assignment]


class MetricsRegistry:
    """Holds the standard metric set for the server."""

    def __init__(self) -> None:
        self._registry = CollectorRegistry() if _HAS_PROM else None
        self.requests_total = Counter(
            "llm_pipeline_requests_total",
            "Total inference requests by endpoint and status",
            ["endpoint", "status"],
            registry=self._registry,
        )
        self.request_latency = Histogram(
            "llm_pipeline_request_latency_seconds",
            "Request latency in seconds",
            ["endpoint"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=self._registry,
        )
        self.completion_tokens = Counter(
            "llm_pipeline_completion_tokens_total",
            "Total tokens generated",
            registry=self._registry,
        )


@contextmanager
def timed_request(registry: MetricsRegistry, endpoint: str):
    t0 = perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        registry.request_latency.labels(endpoint=endpoint).observe(perf_counter() - t0)
        registry.requests_total.labels(endpoint=endpoint, status=status).inc()


def metrics_response(_registry: MetricsRegistry) -> Tuple[bytes, str]:
    """Render a Prometheus exposition. Returns (body, content_type)."""
    if _HAS_PROM:
        return generate_latest(_registry._registry), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST
