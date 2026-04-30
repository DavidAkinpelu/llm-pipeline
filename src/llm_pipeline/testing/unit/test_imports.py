"""Smoke test: every public package imports cleanly without GPU.

Catches the kind of regression introduced when packages are renamed or
deleted but their re-exports linger (the recent inference.core/.generation
deletions had no test like this to flag dangling imports).
"""

import importlib

import pytest


PUBLIC_MODULES = [
    "llm_pipeline",
    "llm_pipeline.core",
    "llm_pipeline.adapters",
    "llm_pipeline.inference",
    "llm_pipeline.parallelism",
    "llm_pipeline.merging",
    "llm_pipeline.training",
    "llm_pipeline.training.optimization",
    "llm_pipeline.training.modes",
    "llm_pipeline.production",
    "llm_pipeline.production.monitoring",
    "llm_pipeline.utils",
    "llm_pipeline.models.qwen3",
    "llm_pipeline.models.generic",
    "llm_pipeline.cli",
]


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_quantization_imports_when_optional_deps_present():
    """Quantization re-exports bnb/aqlm — only assert if those are installed."""
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        pytest.skip("bitsandbytes not installed")
    importlib.import_module("llm_pipeline.quantization")


def test_serving_module_loads_lazily():
    """Serving server must import even without fastapi (lazy import)."""
    importlib.import_module("llm_pipeline.production.serving.server")
