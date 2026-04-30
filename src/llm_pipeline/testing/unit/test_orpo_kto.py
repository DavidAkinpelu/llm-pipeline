"""Unit tests for ORPO + KTO loss math."""

import math

import pytest
import torch

from llm_pipeline.training.modes.orpo import (
    _log1mexp,
    _seq_log_prob_mean,
    compute_orpo_loss,
)
from llm_pipeline.training.modes.kto import compute_kto_loss
from llm_pipeline.training.modes.sft import IGNORE_INDEX


# --------------------------------------------------------------------------- #
# log1mexp
# --------------------------------------------------------------------------- #


def test_log1mexp_matches_direct_formula_far_from_zero():
    """For x = -2 (e^x = 0.135), log(1 - 0.135) = log(0.865) ≈ -0.1448."""
    x = torch.tensor([-2.0])
    out = _log1mexp(x)
    expected = math.log(1.0 - math.exp(-2.0))
    assert math.isclose(out.item(), expected, rel_tol=1e-6)


def test_log1mexp_matches_direct_formula_near_zero():
    """Stability check near x = 0 where naive log(1 - exp(x)) loses precision."""
    x = torch.tensor([-1e-6])
    out = _log1mexp(x).item()
    # log(1 - exp(-1e-6)) ≈ log(1e-6) = -ln(1e6) ≈ -13.8155
    assert math.isclose(out, math.log(-math.expm1(-1e-6)), rel_tol=1e-3)


# --------------------------------------------------------------------------- #
# _seq_log_prob_mean
# --------------------------------------------------------------------------- #


def test_seq_log_prob_mean_uniform_distribution():
    """Uniform vocab (logits = 0) ⇒ log(1/V) per position, regardless of label."""
    V = 4
    logits = torch.zeros(1, 5, V)
    labels = torch.tensor([[1, 2, 3, 0, 1]])
    out = _seq_log_prob_mean(logits, labels)
    expected = math.log(1.0 / V)
    assert math.isclose(out.item(), expected, rel_tol=1e-6)


def test_seq_log_prob_mean_ignores_masked_tokens():
    V = 4
    logits = torch.zeros(1, 5, V)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 1, 2, 3]])  # only 2 valid (after shift)
    out = _seq_log_prob_mean(logits, labels)
    expected = math.log(1.0 / V)
    assert math.isclose(out.item(), expected, rel_tol=1e-6)


# --------------------------------------------------------------------------- #
# ORPO
# --------------------------------------------------------------------------- #


def test_orpo_loss_or_term_zero_when_chosen_equals_rejected():
    """If logits and labels are identical, log_odds_w == log_odds_l ⇒ OR-loss = -log σ(0) = log 2."""
    torch.manual_seed(0)
    V = 8
    logits = torch.randn(2, 4, V)
    labels = torch.randint(0, V, (2, 4))
    loss, m = compute_orpo_loss(logits, labels, logits, labels, lambda_or=1.0)
    # At chosen == rejected, OR loss = -log σ(0) = log 2.
    expected_or = math.log(2.0)
    assert math.isclose(m["orpo/or_loss"].item(), expected_or, rel_tol=1e-5)


def test_orpo_loss_decreases_with_chosen_preference():
    """Boosting log-prob of chosen vs rejected should reduce the OR-loss."""
    torch.manual_seed(0)
    V = 8
    base_logits = torch.randn(1, 4, V)
    labels = torch.tensor([[1, 2, 3, 4]])
    chosen_logits = base_logits.clone()
    rejected_logits = base_logits.clone()

    _, m_neutral = compute_orpo_loss(chosen_logits, labels, rejected_logits, labels, lambda_or=1.0)

    # Boost the logit of the *chosen* token so chosen log-prob > rejected log-prob.
    boosted = base_logits.clone()
    for t in range(3):
        boosted[0, t, labels[0, t + 1].item()] += 4.0
    _, m_better = compute_orpo_loss(boosted, labels, rejected_logits, labels, lambda_or=1.0)
    assert m_better["orpo/or_loss"].item() < m_neutral["orpo/or_loss"].item()


def test_orpo_total_loss_combines_sft_and_or():
    """Total loss = SFT(chosen) + λ · OR. Verify with λ=0 and λ=1."""
    torch.manual_seed(0)
    V = 8
    logits_w = torch.randn(1, 4, V)
    logits_l = torch.randn(1, 4, V)
    labels_w = torch.tensor([[1, 2, 3, 4]])
    labels_l = torch.tensor([[1, 2, 3, 4]])

    loss0, m0 = compute_orpo_loss(logits_w, labels_w, logits_l, labels_l, lambda_or=0.0)
    loss1, m1 = compute_orpo_loss(logits_w, labels_w, logits_l, labels_l, lambda_or=1.0)
    # With λ=0, total == SFT.
    assert math.isclose(loss0.item(), m0["orpo/sft_loss"].item(), rel_tol=1e-6)
    # With λ=1, total == SFT + OR.
    expected = m1["orpo/sft_loss"].item() + m1["orpo/or_loss"].item()
    assert math.isclose(loss1.item(), expected, rel_tol=1e-6)


# --------------------------------------------------------------------------- #
# KTO
# --------------------------------------------------------------------------- #


def test_kto_loss_zero_when_policy_matches_ref_and_labels_balanced():
    """If policy == ref, rewards = 0 → z_ref = 0 → both losses = 1 - σ(0) = 0.5 each."""
    pol = torch.zeros(4)
    ref = torch.zeros(4)
    labels = torch.tensor([1, 0, 1, 0])
    loss, _ = compute_kto_loss(pol, ref, labels, beta=0.1)
    # 4 samples, each loses 0.5; mean = 0.5.
    assert math.isclose(loss.item(), 0.5, rel_tol=1e-6)


def test_kto_loss_decreases_when_policy_boosts_desirable():
    """Increasing the implicit reward on desirable samples should drop the loss."""
    ref = torch.zeros(4)
    labels = torch.tensor([1, 1, 0, 0])

    base = compute_kto_loss(torch.zeros(4), ref, labels, beta=0.1)[0].item()
    # Push desirable (idx 0,1) reward up; undesirable (idx 2,3) reward down.
    boosted = torch.tensor([2.0, 2.0, -2.0, -2.0])
    moved = compute_kto_loss(boosted, ref, labels, beta=0.1)[0].item()
    assert moved < base


def test_kto_loss_respects_class_weights():
    """desirable_weight=0 should make the loss contribution from desirable samples zero."""
    pol = torch.tensor([5.0, 5.0, -5.0, -5.0])
    ref = torch.zeros(4)
    labels = torch.tensor([1, 1, 0, 0])
    loss_full = compute_kto_loss(pol, ref, labels, beta=0.1,
                                 desirable_weight=1.0, undesirable_weight=1.0)[0].item()
    loss_no_desir = compute_kto_loss(pol, ref, labels, beta=0.1,
                                     desirable_weight=0.0, undesirable_weight=1.0)[0].item()
    assert loss_no_desir < loss_full
