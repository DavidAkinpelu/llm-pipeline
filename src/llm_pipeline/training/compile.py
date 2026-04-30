"""``torch.compile`` integration for the training loop.

Wraps a model with ``torch.compile`` in a way that's safe for the
``Trainer`` — dynamic shapes (variable seq length), no CUDA-graph
capture (gradient checkpointing + compile interact poorly otherwise),
and a soft-fallback when ``torch.compile`` isn't available (older
PyTorch, very old CUDA).

Usage::

    from llm_pipeline.training import compile_model_for_training
    model = compile_model_for_training(model, mode="default")
    trainer = Trainer(model=model, ...)
    trainer.train()

The compiled model has the same forward signature; the trainer doesn't
need to know.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def compile_model_for_training(
    model: nn.Module,
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool = True,
) -> nn.Module:
    """Compile ``model`` for training. Safe defaults; fails open.

    Parameters
    ----------
    mode : str
        Passed straight to ``torch.compile``. ``"default"`` is the
        balanced choice; ``"reduce-overhead"`` enables CUDA graphs
        (NOT compatible with gradient checkpointing); ``"max-autotune"``
        is for inference.
    fullgraph : bool
        If True, refuse to fall back to eager on graph breaks. Almost
        always False for training (HF models have plenty of breaks).
    dynamic : bool
        Allow variable shapes (sequence length, batch size). Strongly
        recommended for any non-trivial training stack — fixed-shape
        compilation triggers a recompile every batch boundary.

    Behaviour: if ``torch.compile`` isn't available (PyTorch < 2.0),
    returns the model unchanged with a one-shot warning.
    """
    if not hasattr(torch, "compile"):
        import warnings
        warnings.warn(
            "torch.compile is not available (PyTorch < 2.0); "
            "skipping compilation",
            RuntimeWarning, stacklevel=2,
        )
        return model

    return torch.compile(model, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
