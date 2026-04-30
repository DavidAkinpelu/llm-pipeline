"""Unit tests for the Bradley-Terry reward-model loss + scoring."""

import math

import pytest
import torch
import torch.nn as nn

from llm_pipeline.training.modes.reward_model import (
    RewardModel,
    compute_bradley_terry_loss,
)


def test_bradley_terry_loss_zero_when_chosen_dominates():
    """As (r_w - r_l) → +∞, -log σ(margin) → 0."""
    chosen = torch.tensor([10.0, 10.0, 10.0])
    rejected = torch.tensor([-10.0, -10.0, -10.0])
    loss, m = compute_bradley_terry_loss(chosen, rejected)
    assert loss.item() < 1e-6
    assert math.isclose(m["rm/accuracy"].item(), 1.0)


def test_bradley_terry_loss_log_two_when_equal():
    """If r_w == r_l, margin = 0 ⇒ -log σ(0) = log 2."""
    chosen = torch.zeros(4)
    rejected = torch.zeros(4)
    loss, m = compute_bradley_terry_loss(chosen, rejected)
    assert math.isclose(loss.item(), math.log(2.0), rel_tol=1e-6)
    assert math.isclose(m["rm/accuracy"].item(), 0.0)  # margin > 0 fails for ties


def test_bradley_terry_loss_decreases_with_margin():
    """Loss is monotonically decreasing in (r_w - r_l)."""
    losses = []
    for delta in [0.0, 1.0, 2.0, 5.0]:
        loss, _ = compute_bradley_terry_loss(torch.tensor([delta]), torch.tensor([0.0]))
        losses.append(loss.item())
    for a, b in zip(losses, losses[1:]):
        assert b < a


def test_reward_model_picks_last_non_pad_token():
    """RewardModel.forward should score the last non-pad position, not the absolute last."""
    class FakeBody(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"hidden_size": 4})()

        def forward(self, input_ids, attention_mask, output_hidden_states=False):
            B, T = input_ids.shape
            # Hidden state encodes the position index so we can verify which one was selected.
            h = torch.zeros(B, T, 4)
            for b in range(B):
                for t in range(T):
                    h[b, t, 0] = float(t)
            return type("O", (), {"hidden_states": [h]})()

    rm = RewardModel(FakeBody(), hidden_size=4)
    # Make the score head trivial: just read out feature 0 (= position index).
    with torch.no_grad():
        rm.score_head.weight.zero_()
        rm.score_head.weight[0, 0] = 1.0
        rm.score_head.bias.zero_()

    ids = torch.zeros(2, 5, dtype=torch.long)
    mask = torch.tensor([
        [1, 1, 1, 0, 0],   # last non-pad at idx 2
        [1, 1, 1, 1, 1],   # last non-pad at idx 4
    ])
    out = rm(ids, mask)
    torch.testing.assert_close(out, torch.tensor([2.0, 4.0]))
