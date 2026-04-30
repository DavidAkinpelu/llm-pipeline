"""Tests for the per-request LoRA adapter router."""

import asyncio
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from llm_pipeline.adapters.adapter_manager import MultiAdapterLoRALinear
from llm_pipeline.core.config import LoRAConfig, MultiAdapterConfig
from llm_pipeline.production.serving import LoRARouter


def _build_model_with_two_adapters() -> nn.Module:
    """Tiny model: a single MultiAdapterLoRALinear with two registered adapters."""
    cfg = LoRAConfig(r=4, alpha=8.0, dropout=0.0, target_modules=["x"])
    multi = MultiAdapterConfig(max_adapters=4, enable_logging=False)
    layer = MultiAdapterLoRALinear(
        base_layer=nn.Linear(8, 8, bias=False),
        config=cfg,
        multi_config=multi,
    )
    layer.add_adapter(adapter_name="medical", config=cfg)
    layer.add_adapter(adapter_name="legal", config=cfg)

    model = nn.Sequential(layer)
    return model


# --------------------------------------------------------------------------- #
# known_adapters
# --------------------------------------------------------------------------- #


def test_router_lists_known_adapters():
    model = _build_model_with_two_adapters()
    router = LoRARouter(model=model)
    assert sorted(router.known_adapters()) == ["legal", "medical"]


def test_router_with_no_multi_adapter_returns_empty():
    router = LoRARouter(model=nn.Linear(4, 4))
    assert router.known_adapters() == []


# --------------------------------------------------------------------------- #
# activate context manager
# --------------------------------------------------------------------------- #


def test_router_activates_and_restores_adapters():
    model = _build_model_with_two_adapters()
    multi = next(m for m in model.modules() if isinstance(m, MultiAdapterLoRALinear))
    multi.set_active_adapters(["medical"])
    router = LoRARouter(model=model)

    assert multi.get_active_adapters() == ["medical"]
    with router.activate(["legal"]):
        assert multi.get_active_adapters() == ["legal"]
    # After context exit, restored to previous.
    assert multi.get_active_adapters() == ["medical"]


def test_router_default_adapters_used_when_none_specified():
    model = _build_model_with_two_adapters()
    multi = next(m for m in model.modules() if isinstance(m, MultiAdapterLoRALinear))
    router = LoRARouter(model=model, default_adapters=["medical"])

    with router.activate():                           # no override → use default
        assert multi.get_active_adapters() == ["medical"]


def test_router_rejects_unknown_adapter():
    router = LoRARouter(model=_build_model_with_two_adapters())
    with pytest.raises(KeyError, match="unknown adapter"):
        with router.activate(["nonexistent"]):
            pass


def test_router_restores_adapters_on_exception():
    """Even if the body inside ``activate`` raises, prior state is restored."""
    model = _build_model_with_two_adapters()
    multi = next(m for m in model.modules() if isinstance(m, MultiAdapterLoRALinear))
    multi.set_active_adapters(["medical"])
    router = LoRARouter(model=model)

    with pytest.raises(RuntimeError):
        with router.activate(["legal"]):
            raise RuntimeError("boom")
    assert multi.get_active_adapters() == ["medical"]


def test_router_empty_list_disables_lora():
    """Empty list is valid — turns off all adapters for the request."""
    model = _build_model_with_two_adapters()
    multi = next(m for m in model.modules() if isinstance(m, MultiAdapterLoRALinear))
    multi.set_active_adapters(["medical"])
    router = LoRARouter(model=model)
    with router.activate([]):
        assert multi.get_active_adapters() == []


# --------------------------------------------------------------------------- #
# async path
# --------------------------------------------------------------------------- #


def test_router_async_activation():
    """Async wrapper holds the router lock for the duration of the block."""
    model = _build_model_with_two_adapters()
    multi = next(m for m in model.modules() if isinstance(m, MultiAdapterLoRALinear))
    router = LoRARouter(model=model)

    async def _run():
        async with router.activate_async(["legal"]):
            assert multi.get_active_adapters() == ["legal"]

    asyncio.run(_run())
    # After completion, prior state restored (it was empty in this fixture).
