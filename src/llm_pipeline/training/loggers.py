"""Pluggable logging backends for ``Trainer``.

The default ``Trainer`` logs to ``print`` when no ``on_log`` callable is
supplied. This module provides drop-in adapters for common experiment
trackers, plus a ``MultiLogger`` that fans out to several backends at
once. Each backend exposes a ``__call__(record: dict) -> None`` so it
plugs straight into ``Trainer.train(on_log=...)``.

Example::

    from llm_pipeline.training.loggers import WandbLogger, ConsoleLogger, MultiLogger

    logger = MultiLogger([
        ConsoleLogger(),
        WandbLogger(project="qwen3-sft", config={"lr": 5e-5}),
    ])
    trainer.train(on_log=logger)

If a backend's optional dependency (wandb / mlflow / tensorboard) is not
installed, that backend raises a clear ``ImportError`` at construction
time — never silently — so missing dependencies surface during dev rather
than mid-training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Console (default-equivalent; explicit so users can compose)
# --------------------------------------------------------------------------- #


class ConsoleLogger:
    """Print one ``[step N] key=val key=val ...`` line per record."""

    def __init__(self, fmt_floats: str = ".4f"):
        self.fmt_floats = fmt_floats

    def __call__(self, record: Dict[str, Any]) -> None:
        step = record.get("step", "?")
        parts = [f"[step {step}]"]
        for k, v in record.items():
            if k == "step":
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:{self.fmt_floats}}")
            else:
                parts.append(f"{k}={v}")
        print(" ".join(parts))


# --------------------------------------------------------------------------- #
# W&B
# --------------------------------------------------------------------------- #


class WandbLogger:
    """Forward records to ``wandb.log``.

    The first call initializes a run if one isn't already active. Subsequent
    calls just log. Pass ``project``, ``run_name``, ``config`` etc. through
    the constructor — they're forwarded to ``wandb.init``.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **init_kwargs: Any,
    ):
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb not installed. Install with `pip install wandb`."
            ) from e
        self._wandb = wandb
        self._project = project
        self._run_name = run_name
        self._config = config or {}
        self._init_kwargs = init_kwargs
        self._run = None

    def _ensure_run(self) -> None:
        if self._run is not None:
            return
        if self._wandb.run is not None:
            self._run = self._wandb.run
        else:
            self._run = self._wandb.init(
                project=self._project,
                name=self._run_name,
                config=self._config,
                **self._init_kwargs,
            )

    def __call__(self, record: Dict[str, Any]) -> None:
        self._ensure_run()
        # ``step`` is special in wandb: pass it as the named ``step`` argument
        # so the panel x-axis lines up across logs.
        step = record.get("step")
        payload = {k: v for k, v in record.items() if k != "step"}
        self._wandb.log(payload, step=step)

    def finish(self) -> None:
        if self._run is not None and hasattr(self._run, "finish"):
            self._run.finish()
            self._run = None


# --------------------------------------------------------------------------- #
# MLflow
# --------------------------------------------------------------------------- #


class MLflowLogger:
    """Forward records to ``mlflow.log_metrics``.

    Starts a run on first call (if none active), tagged with the supplied
    ``run_name`` and ``experiment_name``. Non-numeric values in records
    are skipped (mlflow metrics must be numeric).
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "mlflow not installed. Install with `pip install mlflow`."
            ) from e
        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        self._run_name = run_name
        self._params = params or {}
        self._run_id: Optional[str] = None

    def _ensure_run(self) -> None:
        if self._run_id is not None:
            return
        active = self._mlflow.active_run()
        if active is None:
            run = self._mlflow.start_run(run_name=self._run_name)
            self._run_id = run.info.run_id
            for k, v in self._params.items():
                self._mlflow.log_param(k, v)
        else:
            self._run_id = active.info.run_id

    def __call__(self, record: Dict[str, Any]) -> None:
        self._ensure_run()
        step = record.get("step")
        metrics = {k: float(v) for k, v in record.items() if k != "step" and isinstance(v, (int, float))}
        if metrics:
            self._mlflow.log_metrics(metrics, step=step)

    def finish(self) -> None:
        if self._run_id is not None:
            self._mlflow.end_run()
            self._run_id = None


# --------------------------------------------------------------------------- #
# TensorBoard
# --------------------------------------------------------------------------- #


class TensorBoardLogger:
    """Write scalar records via ``torch.utils.tensorboard.SummaryWriter``."""

    def __init__(self, log_dir: Optional[str] = None, **writer_kwargs: Any):
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "tensorboard not installed. Install with `pip install tensorboard`."
            ) from e
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=log_dir, **writer_kwargs)

    def __call__(self, record: Dict[str, Any]) -> None:
        step = int(record.get("step", 0))
        for k, v in record.items():
            if k == "step":
                continue
            if isinstance(v, (int, float)):
                self._writer.add_scalar(k, float(v), global_step=step)

    def finish(self) -> None:
        self._writer.flush()
        self._writer.close()


# --------------------------------------------------------------------------- #
# MultiLogger
# --------------------------------------------------------------------------- #


class MultiLogger:
    """Fan out records to multiple backends.

    Order is preserved. Each backend is called in turn; if one raises, the
    exception propagates (training stops). Use this rather than wrapping
    each backend yourself when you want, say, console + wandb at the same
    time.
    """

    def __init__(self, backends: Sequence[Any]):
        self.backends: List[Any] = list(backends)

    def __call__(self, record: Dict[str, Any]) -> None:
        for b in self.backends:
            b(record)

    def finish(self) -> None:
        for b in self.backends:
            if hasattr(b, "finish"):
                b.finish()
