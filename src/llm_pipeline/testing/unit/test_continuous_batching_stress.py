"""Stress test for the continuous-batching code path.

Validates the correctness invariants of `ContinuousBatch` under
high-concurrency request submission — without needing a live GPU
inference engine. Uses the existing `ContinuousBatch` request-management
state machine directly (admission, eviction, completion bookkeeping).
This is the test that catches deadlocks and race conditions in the queue
machinery; it doesn't validate generation throughput (which needs a
working model + GPU).
"""

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from llm_pipeline.inference.continuous_batching import (
    ContinuousBatch,
    ContinuousBatcher,
    ContinuousBatchConfig,
    RequestState,
)


def _make_request(rid: str, n_tokens: int = 16) -> RequestState:
    rs = RequestState(rid, prompt=f"prompt-{rid}", max_tokens=n_tokens)
    rs.input_tokens = list(range(n_tokens))
    rs.current_length = n_tokens
    return rs


# --------------------------------------------------------------------------- #
# Single-thread sanity: the basic state-machine operations work.
# --------------------------------------------------------------------------- #


def test_batch_admits_under_max_batch_size():
    cfg = ContinuousBatchConfig(max_batch_size=4)
    batch = ContinuousBatch(cfg)
    for i in range(4):
        assert batch.add_request(_make_request(f"r{i}"))
    # 5th request should fail to admit (max_batch_size).
    assert not batch.add_request(_make_request("r5"))


# --------------------------------------------------------------------------- #
# Concurrent admission: many threads pushing requests must not corrupt state.
# --------------------------------------------------------------------------- #


def test_concurrent_admission_does_not_exceed_max_batch_size():
    """Spawn N threads, each trying to admit a request. The total admitted
    count must not exceed ``max_batch_size`` — proves the admission gate
    is correctly synchronised.
    """
    cfg = ContinuousBatchConfig(max_batch_size=8)
    batch = ContinuousBatch(cfg)

    n_threads = 32
    admitted = []
    admitted_lock = threading.Lock()
    barrier = threading.Barrier(n_threads)

    def _worker(tid: int):
        barrier.wait()                          # release all threads at once
        ok = batch.add_request(_make_request(f"req-{tid}"))
        if ok:
            with admitted_lock:
                admitted.append(tid)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(admitted) <= cfg.max_batch_size
    # At least *something* was admitted (the gate isn't deadlocked).
    assert len(admitted) > 0


def test_concurrent_admission_preserves_unique_request_ids():
    """No two threads should see the same request_id committed."""
    cfg = ContinuousBatchConfig(max_batch_size=64)
    batch = ContinuousBatch(cfg)

    n_threads = 32
    barrier = threading.Barrier(n_threads)

    def _worker(tid: int):
        barrier.wait()
        batch.add_request(_make_request(f"req-{tid}"))

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All admitted request IDs are unique. ``active_requests`` may be a
    # dict (id → state) or list of states depending on the implementation;
    # handle both.
    if isinstance(batch.active_requests, dict):
        rids = list(batch.active_requests.keys())
    else:
        rids = [r.request_id for r in batch.active_requests]
    assert len(rids) == len(set(rids))


def test_no_deadlock_under_admit_remove_loop():
    """Hammer the batch with concurrent admit + remove operations and
    ensure the workers all finish within a reasonable timeout.
    """
    cfg = ContinuousBatchConfig(max_batch_size=4)
    batch = ContinuousBatch(cfg)

    stop = threading.Event()

    def _admitter():
        i = 0
        while not stop.is_set():
            batch.add_request(_make_request(f"a-{i}"))
            i += 1

    def _remover():
        while not stop.is_set():
            ar = batch.active_requests
            if not ar:
                continue
            try:
                rid = next(iter(ar)) if isinstance(ar, dict) else ar[0].request_id
                batch.remove_request(rid)
            except (KeyError, IndexError, RuntimeError):
                # Lost a race with admit/remove — that's fine, we're
                # testing for deadlock-free convergence, not perfect
                # serialisation.
                pass

    workers = [
        threading.Thread(target=_admitter),
        threading.Thread(target=_admitter),
        threading.Thread(target=_remover),
    ]
    for w in workers:
        w.start()

    time.sleep(0.5)                              # let them race for a bit
    stop.set()
    for w in workers:
        w.join(timeout=2.0)

    for w in workers:
        assert not w.is_alive(), "deadlocked worker"


def test_batcher_get_results_processes_requests_synchronously():
    """The blocking results API should work without a background thread."""
    class _Tokenizer:
        eos_token_id = 7
        pad_token_id = 0

        def encode(self, prompt, add_special_tokens=True):
            return [1, 2]

        def decode(self, token_ids, skip_special_tokens=True):
            return " ".join(str(t) for t in token_ids)

    class _Sampler:
        def sample_token(self, logits, kwargs):
            return torch.full((logits.shape[0],), 7, dtype=torch.long)

    class _Model:
        def __call__(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
            batch, seq = input_ids.shape
            logits = torch.zeros(batch, seq, 8)
            return SimpleNamespace(logits=logits, past_key_values=None)

    engine = SimpleNamespace(
        device=torch.device("cpu"),
        tokenizer=_Tokenizer(),
        sampler=_Sampler(),
        model=_Model(),
        config=SimpleNamespace(use_paged_attention=False),
    )
    batcher = ContinuousBatcher(
        engine,
        ContinuousBatchConfig(max_batch_size=2, enable_kv_cache=False),
    )

    request_ids = [
        batcher.add_request("req-0", "prompt-0", max_tokens=1),
        batcher.add_request("req-1", "prompt-1", max_tokens=1),
    ]

    results = batcher.get_results(request_ids, timeout_s=1.0)

    assert results == ["7", "7"]
