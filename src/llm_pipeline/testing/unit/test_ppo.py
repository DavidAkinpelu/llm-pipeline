"""Unit tests for PPO loss + GAE math."""

import math
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from llm_pipeline.training.modes.ppo import (
    PPOConfig,
    PPOTrainer,
    ValueHead,
    compute_gae,
    compute_ppo_loss,
)
from llm_pipeline.training.trainer import TrainerConfig


# --------------------------------------------------------------------------- #
# GAE
# --------------------------------------------------------------------------- #


def test_gae_lambda_one_recovers_monte_carlo_returns():
    """With λ = 1 and a known reward sequence, returns equal cumulative discounted reward."""
    # One sample, three steps, rewards = [1, 1, 1], all values = 0, gamma = 1.
    rewards = torch.tensor([[1.0, 1.0, 1.0]])
    values = torch.zeros(1, 4)
    mask = torch.ones(1, 3)
    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    # Returns at each step = [3, 2, 1].
    torch.testing.assert_close(ret, torch.tensor([[3.0, 2.0, 1.0]]))
    # With V=0 everywhere, advantages = returns.
    torch.testing.assert_close(adv, ret)


def test_gae_zero_when_perfect_value():
    """If V(s) = E[returns] (here zeros for zero rewards), advantage = 0."""
    rewards = torch.zeros(1, 4)
    values = torch.zeros(1, 5)
    mask = torch.ones(1, 4)
    adv, _ = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
    torch.testing.assert_close(adv, torch.zeros_like(adv))


def test_gae_respects_response_mask():
    """Positions with mask=0 must have advantage = 0."""
    rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    values = torch.zeros(1, 5)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert (adv[:, 2:] == 0).all()
    assert (ret[:, 2:] == 0).all()


# --------------------------------------------------------------------------- #
# PPO loss
# --------------------------------------------------------------------------- #


def test_ppo_loss_zero_when_advantage_zero_and_returns_match_values():
    """All zeros → no PG signal, no value loss; entropy term still subtracts."""
    log_pi = torch.zeros(2, 4)
    log_pi_old = torch.zeros(2, 4)
    entropy = torch.zeros(2, 4)
    values = torch.zeros(2, 4)
    old_values = torch.zeros(2, 4)
    advantages = torch.zeros(2, 4)
    returns = torch.zeros(2, 4)
    mask = torch.ones(2, 4)
    loss, _ = compute_ppo_loss(
        log_pi, log_pi_old, entropy, values, old_values, advantages, returns, mask,
        clip_eps=0.2, value_clip_eps=0.2, vf_coef=0.5, entropy_coef=0.0,
    )
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_ppo_loss_decreases_when_policy_moves_toward_positive_advantage():
    log_pi_old = torch.zeros(1, 4)
    entropy = torch.zeros(1, 4)
    values = torch.zeros(1, 4)
    old_values = torch.zeros(1, 4)
    advantages = torch.ones(1, 4)
    returns = torch.zeros(1, 4)
    mask = torch.ones(1, 4)

    neutral, _ = compute_ppo_loss(
        torch.zeros(1, 4), log_pi_old, entropy, values, old_values, advantages, returns, mask,
    )
    moved, _ = compute_ppo_loss(
        0.5 * torch.ones(1, 4), log_pi_old, entropy, values, old_values, advantages, returns, mask,
    )
    assert moved.item() < neutral.item()


def test_ppo_clip_caps_unbounded_growth():
    """ratio = exp(huge) but clip caps surrogate at (1+eps)*A; with A=1, eps=0.2, pg_loss = -1.2."""
    log_pi_old = torch.zeros(1, 1)
    entropy = torch.zeros(1, 1)
    values = torch.zeros(1, 1)
    old_values = torch.zeros(1, 1)
    advantages = torch.ones(1, 1)
    returns = torch.zeros(1, 1)
    mask = torch.ones(1, 1)
    loss, _ = compute_ppo_loss(
        torch.tensor([[10.0]]), log_pi_old, entropy, values, old_values, advantages, returns, mask,
        clip_eps=0.2, value_clip_eps=0.0, vf_coef=0.0, entropy_coef=0.0,
    )
    assert math.isclose(loss.item(), -1.2, rel_tol=1e-5)


def test_value_loss_is_clipped():
    """value_clip_eps caps how far values can deviate from old_values within the loss."""
    log_pi = torch.zeros(1, 1)
    log_pi_old = torch.zeros(1, 1)
    entropy = torch.zeros(1, 1)
    advantages = torch.zeros(1, 1)
    returns = torch.tensor([[10.0]])
    old_values = torch.zeros(1, 1)
    mask = torch.ones(1, 1)

    # Without clip: v_loss = 0.5 * (0 - 10)^2 = 50, scaled by vf_coef.
    new_values_inside = torch.zeros(1, 1)  # values == old_values, no clip difference
    loss_unclipped, _ = compute_ppo_loss(
        log_pi, log_pi_old, entropy, new_values_inside, old_values, advantages, returns, mask,
        value_clip_eps=0.0, vf_coef=1.0, entropy_coef=0.0,
    )
    # 0.5 * (0 - 10)^2 = 50.
    assert math.isclose(loss_unclipped.item(), 50.0, rel_tol=1e-5)

    # With clip: same setup, value moves outside the clip range. Take max(unclipped, clipped) ⇒ unclipped wins.
    new_values_outside = torch.tensor([[1.0]])
    loss_clipped, _ = compute_ppo_loss(
        log_pi, log_pi_old, entropy, new_values_outside, old_values, advantages, returns, mask,
        value_clip_eps=0.2, vf_coef=1.0, entropy_coef=0.0,
    )
    # Clipped v = clamp(1-0, -.2, .2) = 0.2. clipped_loss = 0.5*(0.2-10)^2 = 48.02. unclipped = 0.5*(1-10)^2=40.5.
    # max = 48.02. So loss with clip should be > 40.5.
    assert loss_clipped.item() > 40.5


# --------------------------------------------------------------------------- #
# ValueHead
# --------------------------------------------------------------------------- #


def test_value_head_shapes():
    head = ValueHead(hidden_size=16)
    h = torch.randn(2, 5, 16)
    out = head(h)
    assert out.shape == (2, 5)


class _ToyPolicy(nn.Module):
    def __init__(self, vocab_size: int = 8, hidden_size: int = 4):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, output_hidden_states=False, attention_mask=None):
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        if output_hidden_states:
            return type("Output", (), {"logits": logits, "hidden_states": [hidden]})()
        return type("Output", (), {"logits": logits})()


class _ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 7

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        return type("Encoded", (), {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)})()

    def decode(self, ids, skip_special_tokens=True):
        return "ok"


def test_ppo_trainer_inner_epochs_take_multiple_optimizer_steps():
    """Configured inner PPO epochs should produce repeated optimizer updates."""
    trainer = PPOTrainer(
        model=_ToyPolicy(),
        prompt_dataloader=DataLoader([{"prompts": ["prompt"]}], batch_size=None),
        tokenizer=_ToyTokenizer(),
        reward_fn=lambda prompt, response: 1.0,
        config=TrainerConfig(
            output_dir=tempfile.mkdtemp(),
            num_epochs=1,
            log_every=0,
            save_every=0,
        ),
        ppo_config=PPOConfig(inner_epochs=3, max_new_tokens=1),
    )

    result = trainer.train()

    assert result["global_step"] == 3
    assert trainer.global_step == 3
