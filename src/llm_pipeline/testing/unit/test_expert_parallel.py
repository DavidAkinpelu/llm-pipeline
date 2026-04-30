"""Tests for expert-parallel MoE.

The multi-rank path needs NCCL + ≥2 GPUs and is committed but not
end-to-end-validated locally per the project's hardware policy. The
single-rank path (world_size=1) IS validated here against the reference
``MoEFeedForward``.
"""

import pytest
import torch

from llm_pipeline.models.moe import MoEConfig, MoEFeedForward
from llm_pipeline.parallelism.expert_parallel import (
    ExpertParallelConfig,
    ExpertParallelMoE,
)


def _moe_cfg(num_experts: int = 4, **overrides) -> MoEConfig:
    base = dict(
        hidden_size=16, intermediate_size=32,
        num_experts=num_experts, num_experts_per_token=2,
    )
    base.update(overrides)
    return MoEConfig(**base)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #


def test_ep_config_rejects_invalid_world_size():
    with pytest.raises(ValueError, match="world_size"):
        ExpertParallelConfig(world_size=0, rank=0)


def test_ep_config_rejects_out_of_range_rank():
    with pytest.raises(ValueError, match="rank"):
        ExpertParallelConfig(world_size=4, rank=4)
    with pytest.raises(ValueError, match="rank"):
        ExpertParallelConfig(world_size=4, rank=-1)


def test_layer_rejects_indivisible_expert_count():
    cfg = _moe_cfg(num_experts=5)
    with pytest.raises(ValueError, match="divisible"):
        ExpertParallelMoE(cfg, ExpertParallelConfig(world_size=2, rank=0))


# --------------------------------------------------------------------------- #
# Single-rank path: must match reference MoEFeedForward
# --------------------------------------------------------------------------- #


def test_world_size_one_matches_reference_moe():
    """``world_size=1`` should be a perfect drop-in for ``MoEFeedForward``."""
    cfg = _moe_cfg(num_experts=4, num_experts_per_token=2)
    torch.manual_seed(0)
    ref = MoEFeedForward(cfg)
    ep = ExpertParallelMoE(cfg, ExpertParallelConfig(world_size=1, rank=0))

    # Copy router + per-expert weights from ref → ep so the math is identical.
    with torch.no_grad():
        ep.router.gate.weight.copy_(ref.router.gate.weight)
        for i, expert in enumerate(ep.local_experts):
            expert.gate_proj.weight.copy_(ref.experts[i].gate_proj.weight)
            expert.up_proj.weight.copy_(ref.experts[i].up_proj.weight)
            expert.down_proj.weight.copy_(ref.experts[i].down_proj.weight)

    ref.eval()
    ep.eval()
    x = torch.randn(2, 3, cfg.hidden_size)
    torch.testing.assert_close(ep(x), ref(x), atol=1e-5, rtol=1e-5)


def test_world_size_one_only_holds_local_experts():
    """Sanity: per-rank module materialises only ``num_experts // world_size``
    experts. With world_size=1 that's all of them; with higher world_size
    each rank holds a strict subset.
    """
    cfg = _moe_cfg(num_experts=8)
    ep1 = ExpertParallelMoE(cfg, ExpertParallelConfig(world_size=1, rank=0))
    assert len(ep1.local_experts) == 8
    ep4 = ExpertParallelMoE(cfg, ExpertParallelConfig(world_size=4, rank=2))
    assert len(ep4.local_experts) == 2
    assert ep4.local_start == 2 * 2                  # rank 2 owns experts [4, 5)


def test_router_logits_stashed_for_aux_loss():
    cfg = _moe_cfg()
    ep = ExpertParallelMoE(cfg, ExpertParallelConfig(world_size=1, rank=0))
    x = torch.randn(2, 3, cfg.hidden_size)
    ep(x)
    assert ep.last_router_logits is not None
    assert ep.last_router_logits.shape == (6, cfg.num_experts)
    assert ep.last_expert_mask.shape == (6, cfg.num_experts)


def test_gradient_flows_through_local_experts_and_router():
    cfg = _moe_cfg()
    ep = ExpertParallelMoE(cfg, ExpertParallelConfig(world_size=1, rank=0))
    x = torch.randn(2, 3, cfg.hidden_size, requires_grad=True)
    ep(x).sum().backward()
    assert ep.router.gate.weight.grad is not None
    assert ep.router.gate.weight.grad.abs().sum() > 0
    grad_count = sum(
        1 for e in ep.local_experts
        if e.gate_proj.weight.grad is not None and e.gate_proj.weight.grad.abs().sum() > 0
    )
    assert grad_count > 0
