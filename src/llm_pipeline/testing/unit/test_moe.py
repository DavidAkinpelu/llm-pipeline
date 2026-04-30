"""Tests for the MoE building blocks."""

import pytest
import torch
import torch.nn.functional as F

from llm_pipeline.models.moe import (
    MoEConfig,
    MoEFeedForward,
    TopKRouter,
    collect_moe_aux_loss,
    compute_load_balancing_loss,
    compute_router_z_loss,
)


def _cfg(**overrides) -> MoEConfig:
    base = dict(hidden_size=64, intermediate_size=128, num_experts=4, num_experts_per_token=2)
    base.update(overrides)
    return MoEConfig(**base)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #


def test_config_rejects_invalid_topk():
    with pytest.raises(ValueError, match="num_experts_per_token"):
        MoEConfig(hidden_size=8, intermediate_size=16, num_experts=4, num_experts_per_token=5)


def test_config_rejects_zero_experts():
    with pytest.raises(ValueError, match="num_experts"):
        MoEConfig(hidden_size=8, intermediate_size=16, num_experts=0, num_experts_per_token=1)


def test_config_rejects_unknown_activation():
    with pytest.raises(ValueError, match="activation"):
        MoEConfig(
            hidden_size=8, intermediate_size=16, num_experts=2,
            num_experts_per_token=1, activation="relu",
        )


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


def test_router_picks_top_k_indices():
    torch.manual_seed(0)
    cfg = _cfg(num_experts=4, num_experts_per_token=2)
    router = TopKRouter(cfg)
    h = torch.randn(7, cfg.hidden_size)
    idx, gates, logits = router(h)
    assert idx.shape == (7, cfg.num_experts_per_token)
    assert gates.shape == (7, cfg.num_experts_per_token)
    assert logits.shape == (7, cfg.num_experts)
    # Indices in valid range, distinct per row.
    assert idx.min() >= 0 and idx.max() < cfg.num_experts
    for r in range(7):
        assert len(idx[r].unique()) == cfg.num_experts_per_token


def test_router_norm_topk_gates_sum_to_one():
    cfg = _cfg(norm_topk_prob=True)
    router = TopKRouter(cfg)
    h = torch.randn(5, cfg.hidden_size)
    _, gates, _ = router(h)
    torch.testing.assert_close(gates.sum(dim=-1), torch.ones(5), atol=1e-5, rtol=1e-5)


def test_router_no_norm_keeps_softmax_topk():
    cfg = _cfg(norm_topk_prob=False)
    router = TopKRouter(cfg)
    h = torch.randn(5, cfg.hidden_size)
    _, gates, logits = router(h)
    full_probs = F.softmax(logits.float(), dim=-1)
    # Sum of top-K *unnormalised* gates should match topk of full probs.
    expected_sums = full_probs.topk(cfg.num_experts_per_token, dim=-1).values.sum(dim=-1)
    torch.testing.assert_close(gates.sum(dim=-1).float(), expected_sums, atol=1e-5, rtol=1e-5)


def test_router_jitter_only_active_in_training():
    cfg = _cfg(router_jitter=0.5)
    router = TopKRouter(cfg)
    h = torch.randn(5, cfg.hidden_size)
    router.eval()
    _, _, l1 = router(h)
    _, _, l2 = router(h)
    torch.testing.assert_close(l1, l2)  # deterministic in eval
    router.train()
    _, _, l3 = router(h)
    _, _, l4 = router(h)
    assert not torch.allclose(l3, l4)   # noisy in training


# --------------------------------------------------------------------------- #
# MoEFeedForward
# --------------------------------------------------------------------------- #


def test_moe_forward_shape_preserved():
    cfg = _cfg()
    moe = MoEFeedForward(cfg)
    x = torch.randn(2, 5, cfg.hidden_size)
    y = moe(x)
    assert y.shape == x.shape


def test_moe_top1_with_one_expert_matches_dense_mlp():
    """K=1, E=1 ⇒ MoE is just a dense MLP with the gate==1 short-circuited."""
    cfg = _cfg(num_experts=1, num_experts_per_token=1, norm_topk_prob=True)
    moe = MoEFeedForward(cfg)
    x = torch.randn(3, 7, cfg.hidden_size)
    expected = moe.experts[0](x.reshape(-1, cfg.hidden_size)).reshape_as(x)
    actual = moe(x)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_moe_dispatch_is_correct_when_routing_forced(monkeypatch):
    """If we manually route all tokens to expert 0, the MoE output should
    equal expert 0's MLP output on those tokens (gates collapse to 1.0
    after norm because K=1).
    """
    cfg = _cfg(num_experts=4, num_experts_per_token=1, norm_topk_prob=True)
    moe = MoEFeedForward(cfg)

    # Patch the router to deterministically pick expert 0 with gate=1.0 for
    # every token. This isolates the dispatch logic from the router's input
    # sensitivity (which depends on x's sign).
    orig_forward = moe.router.forward

    def force_expert_0(h):
        T = h.shape[0]
        idx = torch.zeros(T, 1, dtype=torch.int64, device=h.device)
        gates = torch.ones(T, 1, device=h.device)
        # Logits with expert 0 dominating, so softmax → ≈ one-hot on expert 0.
        logits = torch.zeros(T, cfg.num_experts, device=h.device)
        logits[:, 0] = 100.0
        return idx, gates, logits

    monkeypatch.setattr(moe.router, "forward", force_expert_0)

    x = torch.randn(2, 4, cfg.hidden_size)
    expected = moe.experts[0](x.reshape(-1, cfg.hidden_size)).reshape_as(x)
    torch.testing.assert_close(moe(x), expected, atol=1e-5, rtol=1e-5)
    # Restore for cleanliness; monkeypatch undoes it on test exit anyway.
    moe.router.forward = orig_forward


def test_moe_naive_loop_matches_reference_dispatch():
    """Independently re-implement the dispatch: explicitly run every expert
    on every token, mask by routing, sum. Must match the optimised gather
    loop inside MoEFeedForward.
    """
    torch.manual_seed(0)
    cfg = _cfg(num_experts=4, num_experts_per_token=2)
    moe = MoEFeedForward(cfg)
    x = torch.randn(3, 5, cfg.hidden_size)

    h = x.reshape(-1, cfg.hidden_size)
    topk_idx, topk_gates, _ = moe.router(h)

    # Reference: run every expert on every token, accumulate weighted by
    # whether that expert was in the top-K.
    ref = torch.zeros_like(h)
    for t in range(h.shape[0]):
        for k in range(cfg.num_experts_per_token):
            e = topk_idx[t, k].item()
            ref[t] = ref[t] + topk_gates[t, k] * moe.experts[e](h[t : t + 1]).squeeze(0)

    # Run the actual module with the same router by patching the router to
    # be deterministic (re-routes the same way).
    moe.router.eval()
    actual = moe(x).reshape(-1, cfg.hidden_size)
    torch.testing.assert_close(actual, ref, atol=1e-5, rtol=1e-5)


def test_moe_router_logits_stashed_after_forward():
    cfg = _cfg()
    moe = MoEFeedForward(cfg)
    x = torch.randn(2, 3, cfg.hidden_size)
    assert moe.last_router_logits is None
    moe(x)
    assert moe.last_router_logits is not None
    assert moe.last_router_logits.shape == (6, cfg.num_experts)
    assert moe.last_expert_mask.shape == (6, cfg.num_experts)


def test_moe_gradient_flows_through_router_and_experts():
    cfg = _cfg()
    moe = MoEFeedForward(cfg)
    x = torch.randn(2, 3, cfg.hidden_size, requires_grad=True)
    y = moe(x)
    y.sum().backward()
    # Router gradient is non-zero.
    assert moe.router.gate.weight.grad is not None
    assert moe.router.gate.weight.grad.abs().sum() > 0
    # At least one expert receives gradient (top-K routing means some experts
    # may be skipped on a small batch, but not all).
    grad_count = sum(
        1 for e in moe.experts
        if e.gate_proj.weight.grad is not None and e.gate_proj.weight.grad.abs().sum() > 0
    )
    assert grad_count > 0, "no expert received gradient"


# --------------------------------------------------------------------------- #
# Aux losses
# --------------------------------------------------------------------------- #


def test_load_balance_loss_uniform_routing_hits_floor():
    """Perfectly uniform routing → loss = 1.0 (the lower bound)."""
    E, T = 4, 1024
    # Uniform router probs on each token.
    logits = torch.zeros(T, E)
    # Hard mask uniform too: cycle assignment.
    mask = torch.zeros(T, E, dtype=torch.int32)
    for t in range(T):
        mask[t, t % E] = 1
    loss = compute_load_balancing_loss(logits, mask, E)
    torch.testing.assert_close(loss, torch.tensor(1.0), atol=1e-3, rtol=1e-3)


def test_load_balance_loss_imbalanced_routing_grows():
    """Routing everything to expert 0 maximises the loss (= E)."""
    E, T = 4, 64
    # Force soft probs to expert 0.
    logits = torch.zeros(T, E)
    logits[:, 0] = 100.0
    mask = torch.zeros(T, E, dtype=torch.int32)
    mask[:, 0] = 1
    loss = compute_load_balancing_loss(logits, mask, E)
    # f_0 = 1, P_0 ≈ 1, others ~ 0  ⇒  E * (1*1 + 0 + 0 + 0) = E.
    torch.testing.assert_close(loss, torch.tensor(float(E)), atol=1e-3, rtol=1e-3)


def test_load_balance_loss_has_router_gradient():
    """The loss must produce a gradient on router_logits (the soft path)."""
    logits = torch.randn(16, 4, requires_grad=True)
    mask = torch.zeros(16, 4, dtype=torch.int32)
    mask[:, 0] = 1
    loss = compute_load_balancing_loss(logits, mask, 4)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_router_z_loss_matches_formula():
    logits = torch.randn(8, 4)
    expected = torch.logsumexp(logits, dim=-1).pow(2).mean()
    actual = compute_router_z_loss(logits)
    torch.testing.assert_close(actual, expected)


def test_collect_moe_aux_loss_returns_zero_for_dense_model():
    dense = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU())
    aux = collect_moe_aux_loss(dense)
    torch.testing.assert_close(aux, torch.tensor(0.0))


def test_collect_moe_aux_loss_aggregates_across_layers():
    """Two MoE layers in a sequence → aux loss is computed against the
    concatenated router stats from both.
    """
    cfg = _cfg()
    model = torch.nn.ModuleList([MoEFeedForward(cfg), MoEFeedForward(cfg)])
    x = torch.randn(2, 3, cfg.hidden_size)
    for m in model:
        x = m(x) + x
    aux = collect_moe_aux_loss(model, load_balance_coef=0.01, z_loss_coef=1e-3)
    assert aux.numel() == 1
    assert aux.item() > 0
