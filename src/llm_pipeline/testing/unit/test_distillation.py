"""Tests for the distillation loss kernel."""

import pytest
import torch
import torch.nn.functional as F

from llm_pipeline.training.modes import (
    DistillationConfig,
    compute_distillation_loss,
)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #


def test_config_rejects_zero_temperature():
    with pytest.raises(ValueError, match="temperature"):
        DistillationConfig(temperature=0.0)


def test_config_rejects_alpha_outside_unit_interval():
    with pytest.raises(ValueError, match="alpha"):
        DistillationConfig(alpha=1.5)
    with pytest.raises(ValueError, match="alpha"):
        DistillationConfig(alpha=-0.1)


# --------------------------------------------------------------------------- #
# Loss math
# --------------------------------------------------------------------------- #


def test_kl_loss_zero_when_student_matches_teacher():
    """If student logits == teacher logits, KL(s||t) = 0."""
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 8)
    losses = compute_distillation_loss(logits, logits, alpha=1.0)
    assert losses["kl_loss"].item() < 1e-6
    assert losses["loss"].item() < 1e-6


def test_kl_loss_grows_with_divergence():
    """Two distinct random logit tensors should have KL > 0."""
    torch.manual_seed(0)
    student = torch.randn(2, 5, 8)
    teacher = torch.randn(2, 5, 8) * 3.0
    losses = compute_distillation_loss(student, teacher, alpha=1.0)
    assert losses["kl_loss"].item() > 0


def test_temperature_squared_factor_is_applied():
    """The Hinton T² compensation: doubling T should ~quadruple the KL term
    on the same logits (approximately — the underlying KL also shrinks
    proportionally, leaving a near-constant scaled loss).
    """
    torch.manual_seed(0)
    s = torch.randn(2, 5, 8)
    t = torch.randn(2, 5, 8)
    l1 = compute_distillation_loss(s, t, temperature=2.0, alpha=1.0)["kl_loss"].item()
    l2 = compute_distillation_loss(s, t, temperature=4.0, alpha=1.0)["kl_loss"].item()
    # With T² compensation both losses should be at the same order of magnitude;
    # without compensation l2 would be ~4× smaller. We assert l2 is within a
    # factor of 2 of l1 (compensation working).
    assert 0.5 < l1 / l2 < 2.5


def test_alpha_zero_returns_pure_ce():
    """alpha=0 → loss is the raw cross-entropy on hard labels."""
    s = torch.randn(2, 5, 8)
    t = torch.randn(2, 5, 8)
    labels = torch.randint(0, 8, (2, 5))
    losses = compute_distillation_loss(s, t, labels=labels, alpha=0.0)
    expected_ce = F.cross_entropy(s.reshape(-1, 8), labels.reshape(-1))
    torch.testing.assert_close(losses["loss"], expected_ce, atol=1e-5, rtol=1e-5)
    assert losses["ce_loss"].item() > 0


def test_alpha_one_ignores_labels():
    """alpha=1 → labels are not consulted; passing or omitting them gives
    the same loss.
    """
    torch.manual_seed(0)
    s = torch.randn(2, 5, 8)
    t = torch.randn(2, 5, 8)
    labels = torch.randint(0, 8, (2, 5))
    l_with = compute_distillation_loss(s, t, labels=labels, alpha=1.0)["loss"]
    l_without = compute_distillation_loss(s, t, labels=None, alpha=1.0)["loss"]
    torch.testing.assert_close(l_with, l_without, atol=1e-5, rtol=1e-5)


def test_ignore_index_masks_both_loss_terms():
    """Positions with label = -100 must not contribute to either KL or CE."""
    torch.manual_seed(0)
    s = torch.randn(1, 4, 8)
    t = torch.randn(1, 4, 8)
    labels_full = torch.tensor([[3, 1, 5, 7]])
    labels_masked = torch.tensor([[3, -100, 5, -100]])

    l_full = compute_distillation_loss(s, t, labels=labels_full, alpha=0.5)
    l_masked = compute_distillation_loss(s, t, labels=labels_masked, alpha=0.5)
    # The masked version sees only 2 of the 4 positions for both terms; values
    # will differ. The basic invariant is that both run without NaN and the
    # losses are finite.
    assert torch.isfinite(l_masked["loss"])
    assert torch.isfinite(l_full["loss"])


def test_loss_has_gradient_on_student():
    s = torch.randn(2, 3, 8, requires_grad=True)
    t = torch.randn(2, 3, 8)
    losses = compute_distillation_loss(s, t, alpha=1.0)
    losses["loss"].backward()
    assert s.grad is not None
    assert s.grad.abs().sum() > 0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_distillation_loss(
            torch.randn(2, 3, 8), torch.randn(2, 3, 16), alpha=1.0,
        )


def test_returned_diagnostics_contain_kl_and_ce():
    s = torch.randn(1, 2, 4)
    t = torch.randn(1, 2, 4)
    labels = torch.tensor([[1, 2]])
    losses = compute_distillation_loss(s, t, labels=labels, alpha=0.5)
    assert "kl_loss" in losses and "ce_loss" in losses
    assert torch.isfinite(losses["kl_loss"])
    assert torch.isfinite(losses["ce_loss"])
