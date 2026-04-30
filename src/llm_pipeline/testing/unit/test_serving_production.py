"""Tests for the serving/production layer: auth, multi-model, lifecycle,
deployment manifests, and the priority queue.

CPU-only; the K8s manifest tests just YAML-parse the files (a real
cluster validation is roadmapped on the cloud-validation queue).
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_pipeline.production.serving import (
    APIKey,
    AuthError,
    BackpressureConfig,
    BackpressurePolicy,
    InMemoryAPIKeyStore,
    JSONFileAPIKeyStore,
    ModelEntry,
    MultiModelRegistry,
    Priority,
    PriorityRequestQueue,
    QueueOverflow,
    TokenBucketLimiter,
    build_app,
    format_tools_as_system_prompt,
    select_engine_for_request,
    verify_api_key,
)
from llm_pipeline.production.monitoring.metrics import MetricsRegistry, _HAS_PROM
from llm_pipeline.production.registry import (
    ModelLifecycleRegistry,
    WeightVersionError,
)


# --------------------------------------------------------------------------- #
# Auth + API keys
# --------------------------------------------------------------------------- #


def _key(k: str = "secret-1", scopes=("chat",), expires=None, rps=10.0, burst=50.0) -> APIKey:
    return APIKey(
        key=k, name="test", scopes=frozenset(scopes),
        expires_at=expires, rate_limit_rps=rps, rate_limit_burst=burst,
    )


def test_verify_valid_key_returns_record():
    store = InMemoryAPIKeyStore([_key()])
    api_key = verify_api_key("Bearer secret-1", store, required_scope="chat")
    assert api_key.key == "secret-1"


def test_verify_missing_header_raises():
    store = InMemoryAPIKeyStore([_key()])
    with pytest.raises(AuthError, match="missing"):
        verify_api_key(None, store)


def test_verify_malformed_header_raises():
    store = InMemoryAPIKeyStore([_key()])
    with pytest.raises(AuthError, match="malformed"):
        verify_api_key("Token abc", store)
    with pytest.raises(AuthError, match="malformed"):
        verify_api_key("Bearer", store)


def test_verify_unknown_key_raises():
    store = InMemoryAPIKeyStore([_key()])
    with pytest.raises(AuthError, match="unknown"):
        verify_api_key("Bearer not-a-real-key", store)


def test_verify_expired_key_raises():
    store = InMemoryAPIKeyStore([_key(expires=time.time() - 1)])
    with pytest.raises(AuthError, match="expired"):
        verify_api_key("Bearer secret-1", store)


def test_verify_scope_mismatch_raises():
    store = InMemoryAPIKeyStore([_key(scopes=("models",))])
    with pytest.raises(AuthError, match="scope"):
        verify_api_key("Bearer secret-1", store, required_scope="chat")


def test_json_file_store_loads_keys():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "keys.json"
        path.write_text(json.dumps([
            {"key": "k1", "name": "alice", "scopes": ["chat"]},
            {"key": "k2", "name": "bob", "scopes": ["chat", "models"], "rate_limit_rps": 5.0},
        ]))
        store = JSONFileAPIKeyStore(path)
        k1 = store.get("k1")
        assert k1 is not None and k1.scopes == frozenset({"chat"})
        k2 = store.get("k2")
        assert k2 is not None and k2.rate_limit_rps == 5.0


# --------------------------------------------------------------------------- #
# Token bucket rate limiter
# --------------------------------------------------------------------------- #


def test_token_bucket_allows_within_burst():
    limiter = TokenBucketLimiter(default_rps=1.0, default_burst=5.0)
    now = 0.0
    for _ in range(5):
        ok, _ = limiter.check("k", now=now)
        assert ok


def test_token_bucket_rejects_after_burst_exhausted():
    limiter = TokenBucketLimiter(default_rps=1.0, default_burst=3.0)
    now = 0.0
    for _ in range(3):
        ok, _ = limiter.check("k", now=now)
        assert ok
    ok, retry = limiter.check("k", now=now)
    assert not ok
    assert retry > 0


def test_token_bucket_refills_over_time():
    limiter = TokenBucketLimiter(default_rps=1.0, default_burst=2.0)
    now = 0.0
    limiter.check("k", now=now)
    limiter.check("k", now=now)
    ok_now, _ = limiter.check("k", now=now)
    assert not ok_now
    # 2.5 seconds later → 2.5 tokens refilled, capped at burst=2.
    ok_later, _ = limiter.check("k", now=now + 2.5)
    assert ok_later


def test_token_bucket_per_key_independent():
    limiter = TokenBucketLimiter(default_rps=1.0, default_burst=1.0)
    now = 0.0
    assert limiter.check("alice", now=now)[0]
    assert limiter.check("bob", now=now)[0]              # bob has his own bucket
    assert not limiter.check("alice", now=now)[0]        # alice exhausted


# --------------------------------------------------------------------------- #
# Multi-model registry
# --------------------------------------------------------------------------- #


def test_multi_model_lazy_load_on_first_access():
    load_count = [0]

    def factory():
        load_count[0] += 1
        return f"engine-{load_count[0]}"

    reg = MultiModelRegistry()
    reg.register("a", factory)
    assert load_count[0] == 0                             # not loaded yet
    engine = reg.get("a")
    assert engine == "engine-1"
    assert load_count[0] == 1
    # Second access uses cache.
    reg.get("a")
    assert load_count[0] == 1


def test_multi_model_lru_evicts_when_budget_hit():
    load_count = [0]

    def factory():
        load_count[0] += 1
        return f"engine-{load_count[0]}"

    reg = MultiModelRegistry(max_loaded=2)
    reg.register("a", factory)
    reg.register("b", factory)
    reg.register("c", factory)
    reg.get("a")                                          # load a
    reg.get("b")                                          # load b (a, b loaded)
    reg.get("c")                                          # would be 3 — evict a
    assert "a" not in reg.loaded_models()
    assert sorted(reg.loaded_models()) == ["b", "c"]


def test_multi_model_unknown_model_raises():
    reg = MultiModelRegistry()
    with pytest.raises(KeyError, match="unknown model"):
        reg.get("nope")


def test_multi_model_default_model_picked_when_request_omits_it():
    reg = MultiModelRegistry()
    reg.register("a", lambda: "engine-a")
    name, engine = select_engine_for_request(reg, requested_model=None)
    assert name == "a" and engine == "engine-a"


def test_multi_model_unregister_clears_default():
    reg = MultiModelRegistry()
    reg.register("a", lambda: "engine-a")
    reg.unregister("a")
    with pytest.raises(KeyError):
        select_engine_for_request(reg, None)


def test_multi_model_register_idempotent_and_replaces_engine():
    reg = MultiModelRegistry()
    reg.register("a", lambda: "old")
    reg.get("a")
    assert reg.is_loaded("a")
    reg.register("a", lambda: "new")                      # re-register
    assert not reg.is_loaded("a")                          # cached engine evicted
    assert reg.get("a") == "new"


# --------------------------------------------------------------------------- #
# Model lifecycle registry
# --------------------------------------------------------------------------- #


def _tmp_registry(td) -> ModelLifecycleRegistry:
    return ModelLifecycleRegistry(Path(td) / "registry.jsonl")


def test_lifecycle_register_and_get_round_trip():
    with tempfile.TemporaryDirectory() as td:
        reg = _tmp_registry(td)
        reg.register("m", "v1", "/path/v1", {"loss": 1.0})
        v = reg.get("m", "v1")
        assert v.path == "/path/v1"
        assert v.metadata["loss"] == 1.0


def test_lifecycle_latest_returns_most_recent_non_deprecated():
    with tempfile.TemporaryDirectory() as td:
        reg = _tmp_registry(td)
        reg.register("m", "v1", "/p1", registered_at=100.0)
        reg.register("m", "v2", "/p2", registered_at=200.0)
        assert reg.get("m", "latest").version == "v2"


def test_lifecycle_deprecate_excludes_from_latest():
    with tempfile.TemporaryDirectory() as td:
        reg = _tmp_registry(td)
        reg.register("m", "v1", "/p1", registered_at=100.0)
        reg.register("m", "v2", "/p2", registered_at=200.0)
        reg.deprecate("m", "v2")
        assert reg.get("m", "latest").version == "v1"
        # But exact lookup still works.
        assert reg.get("m", "v2").deprecated is True


def test_lifecycle_idempotent_re_register():
    with tempfile.TemporaryDirectory() as td:
        reg = _tmp_registry(td)
        reg.register("m", "v1", "/p1", {"a": 1})
        reg.register("m", "v1", "/p2", {"a": 2})
        v = reg.get("m", "v1")
        assert v.path == "/p2"
        assert v.metadata["a"] == 2
        # No duplicate entries.
        assert len(reg.list("m")) == 1


def test_lifecycle_prune_deprecated_returns_count():
    with tempfile.TemporaryDirectory() as td:
        reg = _tmp_registry(td)
        reg.register("m", "v1", "/p1")
        reg.register("m", "v2", "/p2")
        reg.deprecate("m", "v1")
        n = reg.prune_deprecated()
        assert n == 1
        assert len(reg.list("m")) == 1


def test_lifecycle_unknown_lookup_raises():
    with tempfile.TemporaryDirectory() as td:
        reg = _tmp_registry(td)
        with pytest.raises(WeightVersionError):
            reg.get("nope", "v1")
        reg.register("m", "v1", "/p")
        reg.deprecate("m", "v1")
        with pytest.raises(WeightVersionError, match="non-deprecated"):
            reg.get("m", "latest")


def test_lifecycle_atomic_write_survives_concurrent_registrations():
    """Two threads racing to register; the file ends up consistent (no
    JSONL parse errors) and both entries are present.
    """
    import threading
    with tempfile.TemporaryDirectory() as td:
        reg = _tmp_registry(td)

        def _worker(prefix: str):
            for i in range(20):
                reg.register("m", f"{prefix}-{i}", f"/p/{prefix}/{i}")

        t1 = threading.Thread(target=_worker, args=("a",))
        t2 = threading.Thread(target=_worker, args=("b",))
        t1.start(); t2.start()
        t1.join(); t2.join()
        all_versions = [v.version for v in reg.list("m")]
        assert len(all_versions) == 40
        assert len(set(all_versions)) == 40


def test_lifecycle_survives_concurrent_registrations_across_instances():
    import threading
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "registry.jsonl"
        reg1 = ModelLifecycleRegistry(path)
        reg2 = ModelLifecycleRegistry(path)

        def _worker(reg: ModelLifecycleRegistry, prefix: str):
            for i in range(20):
                reg.register("m", f"{prefix}-{i}", f"/p/{prefix}/{i}")

        t1 = threading.Thread(target=_worker, args=(reg1, "a"))
        t2 = threading.Thread(target=_worker, args=(reg2, "b"))
        t1.start(); t2.start()
        t1.join(); t2.join()

        all_versions = [v.version for v in ModelLifecycleRegistry(path).list("m")]
        assert len(all_versions) == 40
        assert len(set(all_versions)) == 40


# --------------------------------------------------------------------------- #
# Priority queue + backpressure
# --------------------------------------------------------------------------- #


def test_priority_queue_higher_priority_dequeued_first():
    """CRITICAL admitted before NORMAL when both wait behind a saturated queue.

    With ``max_concurrent=1``, only one slot is in flight at a time — two
    later arrivals must wait. Priority ordering kicks in there: even though
    NORMAL arrived first, CRITICAL drains first when the slot frees.
    """
    cfg = BackpressureConfig(
        max_concurrent=1, max_queue_depth=10,
        policy=BackpressurePolicy.WAIT_WITH_TIMEOUT,
        wait_timeout=2.0,
    )
    queue = PriorityRequestQueue(cfg)

    async def _run():
        order = []
        # An explicit start-event makes the test deterministic regardless of
        # the event-loop scheduling order — first_holder definitely takes
        # the slot before either queued worker enqueues.
        first_admitted = asyncio.Event()

        async def first_holder():
            async with queue.admit(Priority.NORMAL):
                first_admitted.set()
                await asyncio.sleep(0.1)

        async def queued(prio: int, label: str):
            await first_admitted.wait()
            async with queue.admit(prio):
                order.append(label)

        # Give the queued workers a definite enqueue order: n first, then c.
        async def driver():
            holder = asyncio.create_task(first_holder())
            await first_admitted.wait()
            n_task = asyncio.create_task(queued(Priority.NORMAL, "n"))
            await asyncio.sleep(0.01)                  # let n enqueue
            c_task = asyncio.create_task(queued(Priority.CRITICAL, "c"))
            await asyncio.gather(holder, n_task, c_task)

        await driver()
        return order

    order = asyncio.run(_run())
    # When the holder releases, CRITICAL (later arrival) jumps NORMAL.
    assert order.index("c") < order.index("n"), order


def test_priority_queue_rejects_when_full_under_immediate_policy():
    cfg = BackpressureConfig(
        max_concurrent=2, max_queue_depth=2,
        policy=BackpressurePolicy.REJECT_IMMEDIATELY,
    )
    queue = PriorityRequestQueue(cfg)

    async def _run():
        async def hold():
            async with queue.admit(Priority.NORMAL):
                await asyncio.sleep(0.05)

        # Two slots in flight (max_concurrent=2 == max_queue_depth=2 →
        # zero waiters allowed); a third should be rejected.
        h1 = asyncio.create_task(hold())
        h2 = asyncio.create_task(hold())
        await asyncio.sleep(0.01)
        with pytest.raises(QueueOverflow) as exc_info:
            async with queue.admit(Priority.NORMAL):
                pass
        assert exc_info.value.retry_after > 0
        await asyncio.gather(h1, h2)

    asyncio.run(_run())


def test_priority_queue_wait_with_timeout_eventually_admits():
    cfg = BackpressureConfig(
        max_queue_depth=1, policy=BackpressurePolicy.WAIT_WITH_TIMEOUT,
        wait_timeout=0.5,
    )
    queue = PriorityRequestQueue(cfg)

    async def _run():
        results = []

        async def task(label: str, hold_for: float):
            async with queue.admit(Priority.NORMAL):
                results.append(label)
                await asyncio.sleep(hold_for)

        await asyncio.gather(
            task("first", 0.05),
            task("second", 0.0),                     # waits, then runs
        )
        return results

    out = asyncio.run(_run())
    assert out == ["first", "second"]


def test_priority_queue_wait_with_timeout_rejects_after_timeout():
    cfg = BackpressureConfig(
        max_queue_depth=1, policy=BackpressurePolicy.WAIT_WITH_TIMEOUT,
        wait_timeout=0.05,
    )
    queue = PriorityRequestQueue(cfg)

    async def _run():
        async def hold():
            async with queue.admit(Priority.NORMAL):
                await asyncio.sleep(0.5)

        h = asyncio.create_task(hold())
        await asyncio.sleep(0.01)
        with pytest.raises(QueueOverflow):
            async with queue.admit(Priority.NORMAL):
                pass
        h.cancel()
        try:
            await h
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())


def test_priority_queue_invalid_depth_rejected_at_construction():
    with pytest.raises(ValueError, match="max_queue_depth"):
        PriorityRequestQueue(BackpressureConfig(max_queue_depth=0))
    with pytest.raises(ValueError, match="max_concurrent"):
        PriorityRequestQueue(BackpressureConfig(max_concurrent=0, max_queue_depth=4))
    with pytest.raises(ValueError, match="max_concurrent"):
        PriorityRequestQueue(BackpressureConfig(max_concurrent=10, max_queue_depth=4))


# --------------------------------------------------------------------------- #
# K8s manifests + Helm chart
# --------------------------------------------------------------------------- #


_DEPLOY_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "deployment"


def test_dockerfile_exists_and_is_non_empty():
    f = _DEPLOY_DIR / "Dockerfile"
    assert f.exists()
    text = f.read_text()
    assert "FROM" in text and "EXPOSE" in text


@pytest.mark.parametrize("manifest", [
    "k8s/deployment.yaml", "k8s/service.yaml", "k8s/configmap.yaml",
    "k8s/hpa.yaml", "k8s/pvc.yaml",
])
def test_k8s_manifests_are_valid_yaml(manifest):
    """Each manifest file parses as YAML and exposes a ``kind`` field —
    cheap structural smoke test that doesn't need a cluster.
    """
    yaml = pytest.importorskip("yaml")
    f = _DEPLOY_DIR / manifest
    assert f.exists(), f"{manifest} missing"
    docs = list(yaml.safe_load_all(f.read_text()))
    assert all(d.get("kind") for d in docs if d), f"{manifest} missing 'kind'"


def test_helm_chart_yaml_present_and_complete():
    yaml = pytest.importorskip("yaml")
    chart_yaml = _DEPLOY_DIR / "helm" / "llm-pipeline" / "Chart.yaml"
    values_yaml = _DEPLOY_DIR / "helm" / "llm-pipeline" / "values.yaml"
    assert chart_yaml.exists() and values_yaml.exists()
    chart = yaml.safe_load(chart_yaml.read_text())
    assert chart["apiVersion"] == "v2"
    assert chart["name"] == "llm-pipeline"
    assert chart["type"] == "application"
    values = yaml.safe_load(values_yaml.read_text())
    # Required keys for the templates below to render.
    for k in ("image", "modelPath", "service", "resources"):
        assert k in values, f"values.yaml missing {k!r}"


def test_helm_templates_present():
    """All four templates referenced by ``Chart.yaml`` exist."""
    tdir = _DEPLOY_DIR / "helm" / "llm-pipeline" / "templates"
    for name in ("deployment.yaml", "service.yaml", "_helpers.tpl",
                 "pvc.yaml", "hpa.yaml"):
        assert (tdir / name).exists(), f"missing {name}"


# --------------------------------------------------------------------------- #
# Metrics + server behavior
# --------------------------------------------------------------------------- #


def test_metrics_registry_can_be_constructed_twice():
    first = MetricsRegistry()
    second = MetricsRegistry()
    assert first is not second


def test_format_tools_tool_choice_is_human_readable():
    text = format_tools_as_system_prompt([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ], tool_choice="required")
    assert "must call exactly one tool" in text
    assert ".get(tool_choice" not in text


def test_server_forwards_sampling_args_and_validates_model_and_options():
    fastapi = pytest.importorskip("fastapi")
    testclient = pytest.importorskip("fastapi.testclient")

    class _DummyTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return type("Tok", (), {"input_ids": [1, 2, 3]})()

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "prompt"

    class _DummyEngine:
        def __init__(self):
            self.tokenizer = _DummyTokenizer()
            self.calls = []

        def generate(self, prompt, **kwargs):
            self.calls.append(("generate", prompt, kwargs))
            return "done"

        def stream_generate(self, prompt, **kwargs):
            self.calls.append(("stream_generate", prompt, kwargs))
            yield "chunk"

    engine = _DummyEngine()
    with patch("llm_pipeline.inference.Qwen3InferenceEngine", return_value=engine), \
         patch("llm_pipeline.inference.Qwen3InferenceConfig", side_effect=lambda **k: k):
        app = build_app("model-a", device="cpu", dtype="fp32")

    client = testclient.TestClient(app)

    ok = client.post("/v1/completions", json={
        "model": "model-a",
        "prompt": "hi",
        "max_tokens": 7,
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 3,
    })
    assert ok.status_code == 200
    assert engine.calls[-1] == (
        "generate",
        "hi",
        {"max_tokens": 7, "temperature": 0.5, "top_k": 3, "top_p": 0.2},
    )

    wrong_model = client.post("/v1/completions", json={"model": "other", "prompt": "hi"})
    assert wrong_model.status_code == 404

    unsupported_n = client.post("/v1/completions", json={"model": "model-a", "prompt": "hi", "n": 2})
    assert unsupported_n.status_code == 400

    unsupported_stop = client.post(
        "/v1/chat/completions",
        json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}], "stop": "END"},
    )
    assert unsupported_stop.status_code == 400


def test_server_streaming_completions_use_sse_and_stream_generate():
    pytest.importorskip("fastapi")
    testclient = pytest.importorskip("fastapi.testclient")

    class _DummyTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return type("Tok", (), {"input_ids": [1, 2]})()

    class _DummyEngine:
        def __init__(self):
            self.tokenizer = _DummyTokenizer()
            self.calls = []

        def generate(self, prompt, **kwargs):
            self.calls.append(("generate", prompt, kwargs))
            return "done"

        def stream_generate(self, prompt, **kwargs):
            self.calls.append(("stream_generate", prompt, kwargs))
            yield "a"
            yield "b"

    engine = _DummyEngine()
    with patch("llm_pipeline.inference.Qwen3InferenceEngine", return_value=engine), \
         patch("llm_pipeline.inference.Qwen3InferenceConfig", side_effect=lambda **k: k):
        app = build_app("model-a", device="cpu", dtype="fp32")

    client = testclient.TestClient(app)
    response = client.post(
        "/v1/completions",
        json={"model": "model-a", "prompt": "hi", "stream": True, "top_k": 9, "top_p": 0.7},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text
    assert engine.calls[-1] == (
        "stream_generate",
        "hi",
        {"max_tokens": 256, "temperature": 1.0, "top_k": 9, "top_p": 0.7},
    )
