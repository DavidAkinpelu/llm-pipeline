"""Priority queue + backpressure for the serving layer.

Lets the serving stack admit, reorder, and reject inference requests
based on a priority class so latency-sensitive traffic doesn't get
starved by batch / background traffic. Three design choices worth
flagging up front:

1. **Async-native**. Backed by ``asyncio.Condition`` + ``heapq``. The
   serving routes are already ``async def``, so no thread crossings.

2. **Reject-or-wait policy**. When the queue is full the caller picks
   between immediate rejection (HTTP 503) or wait-with-timeout
   (queues up to ``wait_timeout`` seconds, returns 503 if still full).

3. **Strict FIFO within a priority class**. The heap key is
   ``(priority, monotonic_counter)`` — lower priority value goes first,
   ties broken by insertion order. The counter prevents tie-breaks from
   sorting on the payload (which would be slow and undefined).
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


class Priority(IntEnum):
    """Three default priority classes. Lower value = served first."""

    CRITICAL = 0
    NORMAL = 1
    BATCH = 2


class BackpressurePolicy(IntEnum):
    REJECT_IMMEDIATELY = 0
    WAIT_WITH_TIMEOUT = 1


@dataclass
class BackpressureConfig:
    """Knobs for the priority queue's admission policy.

    Attributes
    ----------
    max_concurrent : int
        Maximum number of requests in flight at once. Higher → more
        concurrency, more GPU memory pressure. Default 1 (strict
        serial admission, the safe choice for single-GPU serving).
    max_queue_depth : int
        Maximum total system depth (in-flight + waiting). Once hit,
        new arrivals are rejected (or wait, per ``policy``).
    policy : BackpressurePolicy
        REJECT_IMMEDIATELY (return 503 + Retry-After) or
        WAIT_WITH_TIMEOUT (wait up to ``wait_timeout`` seconds).
    wait_timeout : float
        Only used when policy == WAIT_WITH_TIMEOUT.
    retry_after_seconds : float
        Hint sent back in the HTTP 503 response.
    """

    max_concurrent: int = 1
    max_queue_depth: int = 256
    policy: BackpressurePolicy = BackpressurePolicy.REJECT_IMMEDIATELY
    wait_timeout: float = 1.0
    retry_after_seconds: float = 1.0


class QueueOverflow(Exception):
    """Raised when admission fails. The route handler should map this to
    HTTP 503 with ``Retry-After: <retry_after_seconds>``.
    """

    def __init__(self, retry_after: float):
        super().__init__(f"queue full; retry after {retry_after:.2f}s")
        self.retry_after = retry_after


@dataclass(order=True)
class _Slot:
    """Heap entry — ordering uses ``(priority, counter)`` only."""

    priority: int
    counter: int
    payload: Any = field(compare=False)


class PriorityRequestQueue:
    """Async priority queue with admission backpressure.

    Typical use inside a route handler:

    >>> queue = PriorityRequestQueue(BackpressureConfig(max_queue_depth=64))
    >>> # In the route:
    >>> try:
    ...     async with queue.admit(priority=Priority.NORMAL, payload=req):
    ...         response = await engine.generate(req)
    ... except QueueOverflow as e:
    ...     raise HTTPException(503, headers={"Retry-After": str(e.retry_after)})

    The ``admit`` context manager handles the heap-ordering + dequeue-on-
    exit bookkeeping; the route just runs its work inside the ``with``.
    """

    def __init__(self, config: Optional[BackpressureConfig] = None):
        self.config = config or BackpressureConfig()
        if self.config.max_queue_depth < 1:
            raise ValueError(
                f"max_queue_depth must be ≥ 1; got {self.config.max_queue_depth}"
            )
        if self.config.max_concurrent < 1:
            raise ValueError(
                f"max_concurrent must be ≥ 1; got {self.config.max_concurrent}"
            )
        if self.config.max_concurrent > self.config.max_queue_depth:
            raise ValueError(
                f"max_concurrent ({self.config.max_concurrent}) must be ≤ "
                f"max_queue_depth ({self.config.max_queue_depth})"
            )
        self._heap: list[_Slot] = []
        self._counter = itertools.count()
        self._cond = asyncio.Condition()
        self._active: int = 0                  # in-flight slots (admitted, not yet released)

    @property
    def depth(self) -> int:
        """Current count of admitted-but-not-yet-released slots."""
        return self._active

    @property
    def pending(self) -> int:
        """Number of waiters parked in the heap (not yet at the head)."""
        return len(self._heap)

    async def acquire(self, priority: int, payload: Any = None) -> int:
        """Admit one request; returns the slot's monotonic counter id.

        On overflow under ``REJECT_IMMEDIATELY`` policy, raises
        ``QueueOverflow`` immediately. Under ``WAIT_WITH_TIMEOUT``, waits
        up to ``wait_timeout`` seconds for a free slot, then raises if
        still full.
        """
        cfg = self.config
        my_id = next(self._counter)
        slot = _Slot(priority=int(priority), counter=my_id, payload=payload)

        async with self._cond:
            # Total system depth = in-flight + already-waiting + this new arrival.
            if self._active + len(self._heap) + 1 > cfg.max_queue_depth:
                if cfg.policy == BackpressurePolicy.REJECT_IMMEDIATELY:
                    raise QueueOverflow(cfg.retry_after_seconds)
                # WAIT_WITH_TIMEOUT for system to drain. Don't enqueue yet —
                # we'd violate the depth cap. Loop on cond + timeout.
                deadline = asyncio.get_running_loop().time() + cfg.wait_timeout
                while self._active + len(self._heap) + 1 > cfg.max_queue_depth:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise QueueOverflow(cfg.retry_after_seconds)
                    try:
                        await asyncio.wait_for(self._cond.wait(), remaining)
                    except asyncio.TimeoutError:
                        raise QueueOverflow(cfg.retry_after_seconds)

            heapq.heappush(self._heap, slot)
            try:
                while True:
                    head = self._heap[0]
                    is_my_turn = head.counter == my_id
                    has_concurrent_slot = self._active < cfg.max_concurrent
                    if is_my_turn and has_concurrent_slot:
                        heapq.heappop(self._heap)
                        self._active += 1
                        return my_id

                    # Either someone higher-priority is at head, or all
                    # concurrent slots are taken. Wait for a notify.
                    await self._cond.wait()
            except BaseException:
                # Clean up on cancellation / unexpected exit.
                try:
                    self._heap.remove(slot)
                    heapq.heapify(self._heap)
                except ValueError:
                    pass
                raise

    async def release(self) -> None:
        """Free a previously-acquired slot. Wakes any pending waiters."""
        async with self._cond:
            if self._active > 0:
                self._active -= 1
            self._cond.notify_all()

    def admit(self, priority: int, payload: Any = None):
        """Async context manager wrapper around acquire/release.

        Usage:

            async with queue.admit(Priority.NORMAL, payload=req):
                ...
        """
        queue = self

        class _Admit:
            async def __aenter__(self_inner):
                self_inner._slot_id = await queue.acquire(priority, payload)
                return self_inner._slot_id

            async def __aexit__(self_inner, exc_type, exc, tb):
                await queue.release()
                return False

        return _Admit()
