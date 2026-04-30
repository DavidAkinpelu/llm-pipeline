"""Unit tests for DPO loss + data collator.

These tests don't need a GPU or a real model; they verify the math directly
against hand-computed expectations.
"""

import math

import pytest
import torch

from llm_pipeline.training.modes.dpo import (
    DPOConfig,
    DPODataCollator,
    _selected_logprobs,
    compute_dpo_loss,
)
from llm_pipeline.training.modes.sft import IGNORE_INDEX


# --------------------------------------------------------------------------- #
# compute_dpo_loss
# --------------------------------------------------------------------------- #


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def test_dpo_loss_zero_when_policy_equals_ref():
    """If policy logps equal ref logps, the DPO loss is exactly -log(0.5)."""
    pc = torch.tensor([-1.0, -2.0])
    pr = torch.tensor([-3.0, -4.0])
    rc, rr = pc.clone(), pr.clone()
    loss, _ = compute_dpo_loss(pc, pr, rc, rr, beta=0.1)
    assert torch.allclose(loss, torch.tensor(-math.log(_sigmoid(0.0))), atol=1e-6)


def test_dpo_loss_decreases_when_policy_prefers_chosen():
    """Loss should drop monotonically as the policy starts preferring chosen."""
    rc, rr = torch.tensor([-3.0]), torch.tensor([-3.0])
    losses = []
    for delta in [0.0, 0.5, 1.0, 2.0]:
        pc = torch.tensor([-3.0 + delta])
        pr = torch.tensor([-3.0 - delta])
        loss, _ = compute_dpo_loss(pc, pr, rc, rr, beta=0.1)
        losses.append(loss.item())
    for a, b in zip(losses, losses[1:]):
        assert b < a, f"DPO loss did not decrease: {losses}"


def test_dpo_loss_metrics_accuracy():
    """When policy prefers chosen, accuracy is 1; when it prefers rejected, 0."""
    rc, rr = torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])
    pc = torch.tensor([1.0, -1.0])  # 1st prefers chosen, 2nd prefers rejected
    pr = torch.tensor([-1.0, 1.0])
    _, m = compute_dpo_loss(pc, pr, rc, rr, beta=0.1)
    assert torch.equal(m["rewards/accuracy"], torch.tensor([1.0, 0.0]))


def test_dpo_loss_label_smoothing_gives_finite_value():
    pc, pr = torch.tensor([0.5]), torch.tensor([-0.5])
    rc, rr = torch.tensor([0.0]), torch.tensor([0.0])
    loss, _ = compute_dpo_loss(pc, pr, rc, rr, beta=0.1, label_smoothing=0.1)
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# _selected_logprobs
# --------------------------------------------------------------------------- #


def test_selected_logprobs_matches_manual_sum():
    """Hand-compute the shifted log-prob sum and verify."""
    torch.manual_seed(0)
    vocab = 5
    seq = 4
    logits = torch.randn(1, seq, vocab)
    labels = torch.tensor([[1, 2, 3, 4]])  # all valid

    got = _selected_logprobs(logits, labels)
    # Manual: shift_logits[t] predicts shift_labels[t] = labels[t+1].
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    expected = sum(log_probs[0, t, labels[0, t + 1]].item() for t in range(seq - 1))
    assert math.isclose(got.item(), expected, rel_tol=1e-6)


def test_selected_logprobs_ignores_index():
    """Tokens labeled IGNORE_INDEX must not contribute to the sum."""
    logits = torch.zeros(1, 3, 4)  # uniform → log_prob = log(1/4)
    labels = torch.tensor([[IGNORE_INDEX, 1, IGNORE_INDEX]])
    out = _selected_logprobs(logits, labels)
    # After shifting: shift_labels = [1, IGNORE]. Only first contributes.
    expected = math.log(0.25)
    assert math.isclose(out.item(), expected, rel_tol=1e-6)


# --------------------------------------------------------------------------- #
# DPODataCollator
# --------------------------------------------------------------------------- #


def test_collator_pads_and_masks_prompt():
    cfg = DPOConfig(max_length=8, pad_token_id=0)
    col = DPODataCollator(cfg)
    batch = [
        {
            "chosen_input_ids":   [10, 11, 12, 13, 14],  # prompt = [10,11], chosen = [12,13,14]
            "rejected_input_ids": [10, 11, 20, 21],      # prompt = [10,11], rejected = [20,21]
            "prompt_len": 2,
        },
        {
            "chosen_input_ids":   [30, 31, 32],
            "rejected_input_ids": [30, 31, 40, 41, 42],
            "prompt_len": 2,
        },
    ]
    out = col(batch)
    # Padded length is max sequence in batch (5).
    assert out["chosen_input_ids"].shape == (2, 5)
    assert out["rejected_input_ids"].shape == (2, 5)

    # Prompt positions (first 2) -> IGNORE_INDEX in labels.
    assert (out["chosen_labels"][:, :2] == IGNORE_INDEX).all()
    assert (out["rejected_labels"][:, :2] == IGNORE_INDEX).all()

    # Pad positions also IGNORE_INDEX (e.g. last entries of shorter sequence).
    assert out["chosen_labels"][1, -1].item() == IGNORE_INDEX
    assert out["rejected_attention_mask"][0, -1].item() == 0


def test_collator_with_real_tokens():
    """Sanity: shapes match across chosen/rejected."""
    cfg = DPOConfig(max_length=16, pad_token_id=0)
    col = DPODataCollator(cfg)
    batch = [
        {"chosen_input_ids": list(range(6)), "rejected_input_ids": list(range(4)), "prompt_len": 2},
        {"chosen_input_ids": list(range(10)), "rejected_input_ids": list(range(8)), "prompt_len": 3},
    ]
    out = col(batch)
    for k in ("chosen_input_ids", "chosen_attention_mask", "chosen_labels",
              "rejected_input_ids", "rejected_attention_mask", "rejected_labels"):
        assert out[k].shape[0] == 2
