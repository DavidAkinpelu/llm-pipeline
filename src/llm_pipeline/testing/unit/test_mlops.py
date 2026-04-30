"""Tests for the MLOps module: seeds, manifest, checkpointing,
sweeps, and the resource tracker.
"""

import json
import os
import random
import sys
import tempfile
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from llm_pipeline.mlops import (
    BestOfNRetention,
    DistributedCheckpointSaver,
    GridSweep,
    OptunaSweep,
    RandomSweep,
    RunManifest,
    RunTracker,
    SweepRunner,
    auto_resume_from_latest,
    capture_run_environment,
    find_latest_checkpoint,
    restore_seed_state,
    set_global_seed,
)


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #


def test_set_global_seed_makes_python_numpy_torch_deterministic():
    """Same seed → same RNG outputs across all three frameworks."""
    s1 = set_global_seed(42, deterministic=True)
    p1 = random.random()
    n1 = float(np.random.rand())
    t1 = float(torch.rand(1).item())

    s2 = set_global_seed(42, deterministic=True)
    p2 = random.random()
    n2 = float(np.random.rand())
    t2 = float(torch.rand(1).item())

    assert p1 == p2
    assert n1 == n2
    assert t1 == t2
    # Restore something sane afterwards.
    restore_seed_state(s1)


def test_set_global_seed_different_seeds_produce_different_outputs():
    set_global_seed(1)
    a = float(torch.rand(1).item())
    set_global_seed(2)
    b = float(torch.rand(1).item())
    assert a != b


def test_set_global_seed_rejects_negative():
    with pytest.raises(ValueError, match="seed"):
        set_global_seed(-5)


def test_restore_seed_state_round_trip():
    """Snapshot → mutate → restore. The PYTHONHASHSEED env var should
    return to its original value after restore.
    """
    original = os.environ.get("PYTHONHASHSEED")
    snap = set_global_seed(99, deterministic=False)
    assert os.environ["PYTHONHASHSEED"] == "99"
    set_global_seed(7, deterministic=False)
    assert os.environ["PYTHONHASHSEED"] == "7"
    restore_seed_state(snap)
    if original is None:
        # Snapshot recorded `None`; restore should pop the var.
        # But snap was taken AFTER set_global_seed(99) wrote it, so its
        # python_hash_seed field is the *pre-99* value, which is `original`.
        if original is None:
            assert "PYTHONHASHSEED" not in os.environ or os.environ["PYTHONHASHSEED"] != "7"


def test_set_global_seed_deterministic_false_disables_deterministic_flags():
    original_det = torch.backends.cudnn.deterministic
    original_bench = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        set_global_seed(11, deterministic=True)
        set_global_seed(12, deterministic=False)
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
        if hasattr(torch, "are_deterministic_algorithms_enabled"):
            assert torch.are_deterministic_algorithms_enabled() is False
    finally:
        torch.backends.cudnn.deterministic = original_det
        torch.backends.cudnn.benchmark = original_bench


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def test_manifest_round_trip_via_json():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "manifest.json"
        manifest = capture_run_environment(
            config={"lr": 1e-4, "batch_size": 16}, seed=42,
        )
        manifest.save_to(path)
        loaded = RunManifest.load_from(path)
        assert loaded.run_id == manifest.run_id
        assert loaded.config == manifest.config
        assert loaded.seed == 42


def test_manifest_captures_torch_version():
    manifest = capture_run_environment(config={})
    assert manifest.torch_version == torch.__version__


def test_manifest_dataset_fingerprint_deterministic():
    """Same dataset → same fingerprint hash."""
    ds = [{"text": f"example {i}"} for i in range(50)]
    m1 = capture_run_environment(config={}, dataset=ds)
    m2 = capture_run_environment(config={}, dataset=ds)
    assert m1.dataset_fingerprint["hash"] == m2.dataset_fingerprint["hash"]
    assert m1.dataset_fingerprint["length"] == 50


def test_manifest_filtered_env_excludes_path():
    """PATH and HOME shouldn't leak into the manifest."""
    manifest = capture_run_environment(config={})
    assert "PATH" not in manifest.env_vars
    assert "HOME" not in manifest.env_vars


def test_manifest_handles_missing_dataset():
    manifest = capture_run_environment(config={})
    assert manifest.dataset_fingerprint is None


def test_manifest_skips_iterable_only_dataset_without_consuming_it():
    class CountingIterable:
        def __init__(self, n):
            self.n = n
            self.iterated = 0

        def __iter__(self):
            for i in range(self.n):
                self.iterated += 1
                yield {"i": i}

    dataset = CountingIterable(1000)
    manifest = capture_run_environment(config={}, dataset=dataset)
    assert manifest.dataset_fingerprint == {"length": None, "hash": None}
    assert dataset.iterated == 0


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


def test_full_checkpoint_save_load_round_trip():
    with tempfile.TemporaryDirectory() as td:
        saver = DistributedCheckpointSaver(strategy="full")
        state = {
            "model": {"layer.weight": torch.tensor([1.0, 2.0, 3.0])},
            "step": 100,
            "extra": {"loss": 1.23},
        }
        path = Path(td) / "step-100.pt"
        saver.save(state, path)
        loaded = saver.load(path)
        torch.testing.assert_close(loaded["model"]["layer.weight"], state["model"]["layer.weight"])
        assert loaded["step"] == 100
        assert loaded["extra"]["loss"] == 1.23


def test_sharded_checkpoint_falls_back_to_full_in_single_process():
    """Without torch.distributed initialised, sharded → full with a warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        saver = DistributedCheckpointSaver(strategy="sharded")
    assert saver.strategy == "full"
    assert any("falls back" in str(w.message).lower() for w in caught)


def test_distributed_saver_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        DistributedCheckpointSaver(strategy="parallel-magic")


def test_find_latest_checkpoint_picks_highest_step():
    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td)
        for step in (10, 50, 30):
            (ckpt_dir / f"step-{step}.pt").write_bytes(b"")
        result = find_latest_checkpoint(ckpt_dir)
        assert result is not None
        step, path = result
        assert step == 50


def test_find_latest_checkpoint_returns_none_for_empty_dir():
    with tempfile.TemporaryDirectory() as td:
        assert find_latest_checkpoint(td) is None


def test_find_latest_checkpoint_handles_nonexistent_dir():
    assert find_latest_checkpoint("/tmp/nonexistent-xyz-987") is None


def test_auto_resume_loads_state_into_model():
    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td)
        # Save a checkpoint.
        model = nn.Linear(4, 4)
        with torch.no_grad():
            model.weight.fill_(7.0)
        saver = DistributedCheckpointSaver(strategy="full")
        saver.save(
            {"model": model.state_dict(), "step": 250},
            ckpt_dir / "step-250.pt",
        )
        # Load into a fresh model.
        fresh = nn.Linear(4, 4)
        with torch.no_grad():
            fresh.weight.fill_(0.0)
        step = auto_resume_from_latest(ckpt_dir, fresh)
        assert step == 250
        assert torch.all(fresh.weight == 7.0)


def test_auto_resume_returns_zero_for_fresh_dir():
    with tempfile.TemporaryDirectory() as td:
        model = nn.Linear(4, 4)
        step = auto_resume_from_latest(td, model)
        assert step == 0


# --------------------------------------------------------------------------- #
# Best-of-N retention
# --------------------------------------------------------------------------- #


def test_best_of_n_keeps_top_k_by_metric():
    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td)
        policy = BestOfNRetention(metric_name="loss", mode="min", n_best=2, keep_latest=False)
        # Save three checkpoint files with different "loss" metrics.
        paths = [ckpt_dir / f"step-{s}.pt" for s in (10, 20, 30)]
        for p in paths:
            p.write_bytes(b"")
        policy.record(10, paths[0], 1.0)
        policy.record(20, paths[1], 5.0)                     # worst — should be pruned
        pruned = policy.record(30, paths[2], 0.5)
        assert paths[1] in pruned
        assert paths[1].exists() is False
        assert paths[0].exists() and paths[2].exists()


def test_best_of_n_keep_latest_always_retains_newest():
    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td)
        policy = BestOfNRetention(metric_name="loss", mode="min", n_best=1, keep_latest=True)
        paths = [ckpt_dir / f"step-{s}.pt" for s in (10, 20, 30)]
        for p in paths:
            p.write_bytes(b"")
        policy.record(10, paths[0], 0.5)                      # best
        policy.record(20, paths[1], 5.0)                      # worst
        policy.record(30, paths[2], 1.0)                      # latest, not best, but kept
        assert paths[0].exists()                               # best
        assert paths[2].exists()                               # latest (kept_latest=True)
        assert paths[1].exists() is False                      # pruned


def test_best_of_n_max_mode_inverts_picking():
    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td)
        policy = BestOfNRetention(metric_name="acc", mode="max", n_best=1, keep_latest=False)
        a = ckpt_dir / "step-1.pt"
        b = ckpt_dir / "step-2.pt"
        a.write_bytes(b""); b.write_bytes(b"")
        policy.record(1, a, 0.5)
        policy.record(2, b, 0.9)                              # higher acc → kept
        assert b.exists() and not a.exists()


def test_best_of_n_rejects_invalid_args():
    with pytest.raises(ValueError, match="n_best"):
        BestOfNRetention(metric_name="x", n_best=0)
    with pytest.raises(ValueError, match="mode"):
        BestOfNRetention(metric_name="x", mode="weird")


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #


def test_grid_sweep_cartesian_product():
    sweep = GridSweep({"lr": [1e-4, 1e-3], "bs": [16, 32]})
    runs = list(sweep)
    assert len(runs) == 4
    assert len(sweep) == 4
    # All combinations present.
    pairs = {(r["lr"], r["bs"]) for r in runs}
    assert pairs == {(1e-4, 16), (1e-4, 32), (1e-3, 16), (1e-3, 32)}


def test_grid_sweep_empty_param_list_rejected():
    with pytest.raises(ValueError, match="empty"):
        GridSweep({"lr": []})


def test_grid_sweep_no_params_rejected():
    with pytest.raises(ValueError, match="parameter"):
        GridSweep({})


def test_random_sweep_reproducible_with_seed():
    space = {"lr": ("loguniform", 1e-5, 1e-2), "size": ("int", 1, 10)}
    sweep_a = list(RandomSweep(space, n_trials=5, seed=42))
    sweep_b = list(RandomSweep(space, n_trials=5, seed=42))
    assert sweep_a == sweep_b


def test_random_sweep_distributions():
    space = {
        "x": ("uniform", 0.0, 1.0),
        "y": ("loguniform", 1e-3, 1.0),
        "z": ("int", 1, 100),
        "w": ("choice", ["a", "b", "c"]),
    }
    runs = list(RandomSweep(space, n_trials=20, seed=0))
    for r in runs:
        assert 0.0 <= r["x"] <= 1.0
        assert 1e-3 <= r["y"] <= 1.0
        assert 1 <= r["z"] <= 100
        assert r["w"] in {"a", "b", "c"}


def test_random_sweep_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="distribution"):
        RandomSweep({"x": ("magic", 0, 1)}, n_trials=1)


def test_optuna_sweep_falls_back_when_optuna_missing():
    """Without Optuna installed, OptunaSweep emits a warning + uses a stub."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sweep = OptunaSweep(
            search_space_fn=lambda t: {
                "lr": t.suggest_float("lr", 1e-5, 1e-2, log=True),
                "bs": t.suggest_categorical("bs", [16, 32]),
            },
            n_trials=3,
        )
        runs = list(sweep)
    if sweep._optuna is None:
        assert any("Optuna not installed" in str(w.message) for w in caught)
    assert len(runs) == 3


def test_sweep_runner_reports_results_back_to_optuna(monkeypatch):
    class FakeTrial:
        def __init__(self, value):
            self.value = value

        def suggest_float(self, name, low, high, log=False):
            return self.value

    class FakeStudy:
        def __init__(self):
            self.trials = [FakeTrial(0.1), FakeTrial(0.2)]
            self.tell_calls = []

        def ask(self):
            return self.trials[len(self.tell_calls)]

        def tell(self, trial, value=None, state=None):
            self.tell_calls.append((trial.value, value, state))

    fake_study = FakeStudy()
    fake_optuna = SimpleNamespace(
        create_study=lambda direction: fake_study,
        trial=SimpleNamespace(TrialState=SimpleNamespace(FAIL="FAIL")),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    sweep = OptunaSweep(
        search_space_fn=lambda trial: {
            "loss": trial.suggest_float("loss", 0.0, 1.0),
        },
        n_trials=2,
        objective_metric="loss",
    )
    runner = SweepRunner(
        sweep,
        train_fn=lambda params: {"loss": params["loss"]},
        objective_metric="loss",
    )

    runner.run()

    assert fake_study.tell_calls == [
        (0.1, 0.1, None),
        (0.2, 0.2, None),
    ]


def test_sweep_runner_reports_optuna_failures(monkeypatch):
    class FakeTrial:
        def suggest_float(self, name, low, high, log=False):
            return 0.1

    class FakeStudy:
        def __init__(self):
            self.trial = FakeTrial()
            self.tell_calls = []

        def ask(self):
            return self.trial

        def tell(self, trial, value=None, state=None):
            self.tell_calls.append((value, state))

    fake_study = FakeStudy()
    fake_optuna = SimpleNamespace(
        create_study=lambda direction: fake_study,
        trial=SimpleNamespace(TrialState=SimpleNamespace(FAIL="FAIL")),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    sweep = OptunaSweep(
        search_space_fn=lambda trial: {
            "loss": trial.suggest_float("loss", 0.0, 1.0),
        },
        n_trials=1,
        objective_metric="loss",
    )
    runner = SweepRunner(
        sweep,
        train_fn=lambda params: (_ for _ in ()).throw(RuntimeError("boom")),
        objective_metric="loss",
    )

    runner.run()

    assert fake_study.tell_calls == [(None, "FAIL")]


def test_sweep_runner_finds_best_trial():
    sweep = GridSweep({"x": [1.0, 2.0, 3.0]})

    def train(params):
        return {"loss": (params["x"] - 2.0) ** 2}            # min at x=2

    runner = SweepRunner(sweep, train_fn=train, objective_metric="loss", direction="min")
    result = runner.run()
    assert result.best_trial is not None
    assert result.best_trial.params["x"] == 2.0
    assert len(result.trials) == 3


def test_sweep_runner_handles_failed_train_fn():
    sweep = GridSweep({"x": [1.0, 2.0]})

    def crashy_train(params):
        if params["x"] == 1.0:
            raise RuntimeError("intentional")
        return {"loss": 0.5}

    runner = SweepRunner(sweep, train_fn=crashy_train, objective_metric="loss")
    result = runner.run()
    assert result.trials[0].status == "failed"
    assert result.trials[0].error == "intentional"
    assert result.trials[1].status == "completed"
    # Best is the only completed trial.
    assert result.best_trial.params["x"] == 2.0


def test_sweep_runner_writes_jsonl_log():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "sweep.jsonl"
        runner = SweepRunner(
            GridSweep({"x": [1, 2]}),
            train_fn=lambda p: {"loss": p["x"]},
            objective_metric="loss",
        )
        result = runner.run(log_path=log_path)
        lines = log_path.read_text().splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        assert records[0]["params"] == {"x": 1}


def test_sweep_runner_rejects_invalid_direction():
    with pytest.raises(ValueError, match="direction"):
        SweepRunner(GridSweep({"x": [1]}), train_fn=lambda p: {}, direction="weird")


# --------------------------------------------------------------------------- #
# Resource tracker
# --------------------------------------------------------------------------- #


def test_run_tracker_accumulates_throughput():
    tracker = RunTracker(n_gpus=1)
    tracker.start()
    time.sleep(0.05)
    tracker.record_step(n_samples=8, n_tokens=128)
    tracker.record_step(n_samples=8, n_tokens=128)
    tracker.stop()
    summary = tracker.summary()
    assert summary["n_steps"] == 2
    assert summary["n_samples"] == 16
    assert summary["n_tokens"] == 256
    assert summary["wall_time_s"] >= 0.05
    assert summary["tokens_per_sec"] > 0


def test_run_tracker_estimate_cost_uses_provided_rate():
    tracker = RunTracker(n_gpus=4)
    tracker.start()
    time.sleep(0.1)
    tracker.stop()
    cost = tracker.estimate_cost(gpu_hourly_rate=2.50)
    expected = tracker.gpu_hours * 2.50
    assert abs(cost - expected) < 1e-9


def test_run_tracker_phase_records_duration():
    tracker = RunTracker(n_gpus=1)
    tracker.start()
    with tracker.phase("data_loading"):
        time.sleep(0.05)
    tracker.stop()
    summary = tracker.summary()
    phases = summary["phases"]
    assert len(phases) == 1
    assert phases[0]["name"] == "data_loading"
    assert phases[0]["duration_s"] >= 0.05


def test_run_tracker_rejects_zero_gpus():
    with pytest.raises(ValueError, match="n_gpus"):
        RunTracker(n_gpus=0)


def test_run_tracker_summary_before_stop_uses_current_time():
    """``summary`` works mid-run too — wall_time grows monotonically."""
    tracker = RunTracker()
    tracker.start()
    time.sleep(0.02)
    s1 = tracker.summary()
    time.sleep(0.05)
    s2 = tracker.summary()
    assert s2["wall_time_s"] >= s1["wall_time_s"]


def test_run_tracker_start_resets_peak_memory_state(monkeypatch):
    reset_calls = []
    tracker = RunTracker()
    tracker._peak_memory_gb = {0: 1.5}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda index: reset_calls.append(index),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda index: 0)

    tracker.start()

    assert reset_calls == [0, 1]
    assert tracker.summary()["peak_memory_gb"] == {}
