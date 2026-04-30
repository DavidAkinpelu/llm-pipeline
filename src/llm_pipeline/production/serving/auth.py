"""Authentication, API keys, and rate limiting for the serving layer.

Three composable pieces:

- ``APIKeyStore`` — interface + ``InMemoryAPIKeyStore`` /
  ``JSONFileAPIKeyStore`` implementations. Each key carries its own
  scopes, expiry, and per-key rate limit.
- ``verify_api_key(authorization_header, store)`` — parses the standard
  ``Authorization: Bearer <key>`` header, looks the key up, and returns
  the ``APIKey`` record (or raises ``AuthError``).
- ``TokenBucketLimiter`` — per-key token-bucket admission control.
  Thread-safe via a single global lock; the contention is fine for the
  serving layer's request rate.

These are intentionally framework-agnostic — the FastAPI dependencies
that wire them into request handling live alongside ``server.py`` and
opt-in: ``build_app`` callers without an ``auth_store`` keyword keep
today's open behaviour.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Optional


class AuthError(Exception):
    """Raised on missing / invalid / expired keys, or scope mismatch.

    Routes that catch this should return HTTP 401 with the message in
    the response body — no internal stack trace.
    """


# --------------------------------------------------------------------------- #
# Key records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class APIKey:
    """One issued API key + its policy."""

    key: str                        # the secret string clients send
    name: str                       # human-readable label
    scopes: FrozenSet[str]          # e.g. {"chat", "completions"}
    expires_at: Optional[float] = None       # unix timestamp; None = no expiry
    rate_limit_rps: float = 10.0    # tokens added per second
    rate_limit_burst: float = 50.0  # bucket capacity

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.expires_at is None:
            return False
        return (now if now is not None else time.time()) >= self.expires_at


# --------------------------------------------------------------------------- #
# Stores
# --------------------------------------------------------------------------- #


class APIKeyStore:
    """Read-only key lookup interface.

    Subclasses implement ``get(key)`` returning the ``APIKey`` or None.
    Concrete classes here are CPU-only and synchronous — production
    setups would back this with Redis or a DB.
    """

    def get(self, key: str) -> Optional[APIKey]:
        raise NotImplementedError


class InMemoryAPIKeyStore(APIKeyStore):
    """Dict-backed store. Use for tests and small deployments."""

    def __init__(self, keys: Iterable[APIKey] = ()):
        self._keys: Dict[str, APIKey] = {k.key: k for k in keys}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[APIKey]:
        with self._lock:
            return self._keys.get(key)

    def add(self, api_key: APIKey) -> None:
        with self._lock:
            self._keys[api_key.key] = api_key

    def remove(self, key: str) -> bool:
        with self._lock:
            return self._keys.pop(key, None) is not None


class JSONFileAPIKeyStore(InMemoryAPIKeyStore):
    """Loads keys from a JSON file at construction time.

    File schema: a list of objects with the same fields as ``APIKey``
    (``scopes`` as a JSON array; converted to ``frozenset`` on load).
    Missing optional fields fall back to the defaults.
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            super().__init__([])
            return
        raw = json.loads(path.read_text())
        keys = []
        for entry in raw:
            keys.append(APIKey(
                key=entry["key"],
                name=entry.get("name", ""),
                scopes=frozenset(entry.get("scopes", ())),
                expires_at=entry.get("expires_at"),
                rate_limit_rps=float(entry.get("rate_limit_rps", 10.0)),
                rate_limit_burst=float(entry.get("rate_limit_burst", 50.0)),
            ))
        super().__init__(keys)


# --------------------------------------------------------------------------- #
# Header verification
# --------------------------------------------------------------------------- #


def verify_api_key(
    authorization: Optional[str],
    store: APIKeyStore,
    required_scope: Optional[str] = None,
) -> APIKey:
    """Parse + validate ``Authorization: Bearer <key>`` against ``store``.

    Raises ``AuthError`` on missing header, malformed header, unknown
    key, expired key, or missing scope. Returns the ``APIKey`` record on
    success so the route can use the key's metadata downstream.
    """
    if not authorization:
        raise AuthError("missing Authorization header")
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise AuthError("malformed Authorization header (expected 'Bearer <key>')")
    api_key = store.get(parts[1])
    if api_key is None:
        raise AuthError("unknown API key")
    if api_key.is_expired():
        raise AuthError("API key expired")
    if required_scope is not None and required_scope not in api_key.scopes:
        raise AuthError(f"key lacks required scope: {required_scope!r}")
    return api_key


# --------------------------------------------------------------------------- #
# Rate limiting (token bucket)
# --------------------------------------------------------------------------- #


@dataclass
class _Bucket:
    capacity: float
    rate: float                                    # tokens per second
    tokens: float
    last_refill: float


class TokenBucketLimiter:
    """Per-key token bucket.

    Semantics: each key starts with ``capacity`` tokens; tokens refill at
    ``rate`` per second up to ``capacity``. Each request consumes one
    token. ``check(key)`` returns ``(allowed, retry_after_seconds)``.

    Thread-safety: a single ``threading.Lock`` guards the entire bucket
    table. For the request rates a serving stack actually sees this is
    fine — the critical section is ~50 ns of dict lookup + arithmetic.
    """

    def __init__(self, default_rps: float = 10.0, default_burst: float = 50.0):
        self.default_rps = default_rps
        self.default_burst = default_burst
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(
        self,
        key: str,
        rps: Optional[float] = None,
        burst: Optional[float] = None,
        cost: float = 1.0,
        now: Optional[float] = None,
    ) -> tuple[bool, float]:
        """Try to consume ``cost`` tokens for ``key``.

        Returns ``(allowed, retry_after)`` — ``retry_after`` is 0 when
        allowed; otherwise the seconds until enough tokens are available.
        """
        now = now if now is not None else time.monotonic()
        rate = rps if rps is not None else self.default_rps
        cap = burst if burst is not None else self.default_burst

        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(
                    capacity=cap, rate=rate, tokens=cap, last_refill=now,
                )
                self._buckets[key] = bucket
            # Update rate / capacity if the policy changed mid-flight (no clamp).
            bucket.capacity = cap
            bucket.rate = rate
            # Refill.
            elapsed = max(now - bucket.last_refill, 0.0)
            bucket.tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.rate)
            bucket.last_refill = now

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True, 0.0
            deficit = cost - bucket.tokens
            retry_after = deficit / bucket.rate if bucket.rate > 0 else float("inf")
            return False, retry_after

    def reset(self, key: str) -> None:
        """Clear ``key``'s bucket — full refill on next check."""
        with self._lock:
            self._buckets.pop(key, None)
