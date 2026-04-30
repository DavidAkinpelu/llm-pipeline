"""Unit tests for GRPO loss math.

End-to-end sampling/training is exercised in the smoke script, not here.
"""

import math

import pytest
import torch

from llm_pipeline.training.modes.grpo import (
    _per_token_kl,
    compute_grpo_loss,
)


def test_kl_estimator_zero_when_distributions_match():
    """If log_pi == log_ref, the unbiased KL estimator is exactly 0."""
    lp = torch.randn(2, 5)
    out = _per_token_kl(lp, lp)
    torch.testing.assert_close(out, torch.zeros_like(lp))


def test_kl_estimator_nonneg_for_random_distributions():
    """exp(d) - d - 1 ≥ 0 for all real d (Bregman divergence)."""
    torch.manual_seed(0)
    lp = torch.randn(8, 16)
    lr = torch.randn(8, 16)
    kl = _per_token_kl(lp, lr)
    assert (kl >= -1e-6).all()


def test_grpo_loss_zero_when_advantage_zero_and_kl_zero():
    """All advantages = 0 and policy = ref ⇒ surrogate = 0, KL = 0 ⇒ loss = 0."""
    log_pi = torch.zeros(4, 6)
    log_pi_old = torch.zeros(4, 6)
    log_ref = torch.zeros(4, 6)
    adv = torch.zeros(4)
    mask = torch.ones(4, 6)
    loss, _ = compute_grpo_loss(log_pi, log_pi_old, log_ref, adv, mask, clip_eps=0.2, beta_kl=0.04)
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_grpo_loss_decreases_when_policy_moves_toward_high_advantage():
    """Sanity: positive advantage + ratio > 1 ⇒ surrogate is positive ⇒ loss is negative."""
    log_pi_old = torch.zeros(2, 4)
    log_ref = torch.zeros(2, 4)
    mask = torch.ones(2, 4)
    adv = torch.tensor([1.0, 1.0])

    # Policy raises log-prob → ratio > 1 → positive surrogate → loss decreases.
    loss_neutral, _ = compute_grpo_loss(torch.zeros(2, 4), log_pi_old, log_ref, adv, mask)
    loss_better, _ = compute_grpo_loss(0.5 * torch.ones(2, 4), log_pi_old, log_ref, adv, mask)
    assert loss_better.item() < loss_neutral.item()


def test_grpo_clip_caps_unbounded_growth():
    """ratio = exp(huge) but clip caps the contribution at clip_eps."""
    log_pi_old = torch.zeros(1, 1)
    log_ref = torch.zeros(1, 1)
    mask = torch.ones(1, 1)
    adv = torch.tensor([1.0])
    eps = 0.2

    # Clipped surrogate at very large ratio: clip(1+huge) * adv = (1+eps) * adv.
    loss_huge, _ = compute_grpo_loss(torch.tensor([[10.0]]), log_pi_old, log_ref, adv, mask, clip_eps=eps, beta_kl=0.0)
    expected = -(1.0 + eps) * 1.0
    assert math.isclose(loss_huge.item(), expected, rel_tol=1e-5)


def test_grpo_loss_respects_response_mask():
    """Tokens outside the response mask must not affect the loss."""
    log_pi_old = torch.zeros(2, 6)
    log_ref = torch.zeros(2, 6)
    adv = torch.tensor([1.0, 1.0])

    # Same log_pi values inside the response, garbage outside; mask zero outside.
    log_pi_a = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                             [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    loss_a, _ = compute_grpo_loss(log_pi_a, log_pi_old, log_ref, adv, mask)

    # Now scramble the masked-out region; loss must be unchanged.
    log_pi_b = log_pi_a.clone()
    log_pi_b[:, 2:] = 99.0
    loss_b, _ = compute_grpo_loss(log_pi_b, log_pi_old, log_ref, adv, mask)
    torch.testing.assert_close(loss_a, loss_b)


def test_grpo_metrics_stay_finite_for_singleton_batch():
    """Singleton batches should not emit NaN diagnostics."""
    loss, metrics = compute_grpo_loss(
        log_pi=torch.tensor([[0.0]]),
        log_pi_old=torch.tensor([[0.0]]),
        log_ref=torch.tensor([[0.0]]),
        advantages=torch.tensor([1.0]),
        response_mask=torch.tensor([[1.0]]),
    )

    torch.testing.assert_close(loss, torch.tensor(-1.0))
    assert torch.isfinite(metrics["grpo/advantage_std"])
    torch.testing.assert_close(metrics["grpo/advantage_std"], torch.tensor(0.0))
