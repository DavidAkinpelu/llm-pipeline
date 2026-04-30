"""Reward-model training (Bradley-Terry on preference pairs).

A reward model is a causal LM with the LM head replaced by a scalar head:
``r_θ(x, y) = w · h_T(x, y) + b`` where ``h_T`` is the last-hidden-state of
the final non-padding token of ``(x, y)`` concatenated. Training uses the
Bradley-Terry log-likelihood over preference pairs:

    L = -E[log σ( r(x, y_w) - r(x, y_l) )]

Once trained, ``RewardModel`` exposes a callable
``score(prompt, response) -> float`` that GRPO/PPO can use as their
``reward_fn``.

Usage:

    rm = RewardModel.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
    rm.to("cuda:0")
    trainer = RewardModelTrainer(rm, preference_loader, ...)
    trainer.train()
    # Now use rm.score in GRPO/PPO:
    grpo = GRPOTrainer(policy, prompts, tok, reward_fn=rm.score, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dpo import DPODataCollator, DPOConfig, tokenize_preference
from ..trainer import Trainer, TrainerConfig


# --------------------------------------------------------------------------- #
# Reward model
# --------------------------------------------------------------------------- #


class RewardModel(nn.Module):
    """Causal LM body + a scalar reward head over the last non-pad token.

    The body is any HuggingFace ``AutoModelForCausalLM`` (we only use its
    hidden states). The head is a single ``Linear(hidden, 1)``.
    """

    def __init__(self, body: nn.Module, hidden_size: Optional[int] = None, tokenizer=None):
        super().__init__()
        self.body = body
        if hidden_size is None:
            hidden_size = getattr(getattr(body, "config", None), "hidden_size", None)
            if hidden_size is None:
                raise ValueError("Could not infer hidden_size; pass it explicitly.")
        self.score_head = nn.Linear(hidden_size, 1)
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **load_kwargs) -> "RewardModel":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name_or_path)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        body = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        return cls(body, tokenizer=tok)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return one scalar reward per sequence ``[B]``."""
        out = self.body(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = out.hidden_states[-1]                                    # [B, T, H]
        # Cast to the head's dtype — the body might be bf16 while the head
        # stays in fp32 for numerical stability.
        hidden = hidden.to(self.score_head.weight.dtype)
        scores = self.score_head(hidden).squeeze(-1)                      # [B, T]
        last_idx = attention_mask.sum(dim=-1).clamp(min=1) - 1            # [B]
        b = torch.arange(input_ids.size(0), device=input_ids.device)
        return scores[b, last_idx]

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> float:
        """Plain ``(prompt, response) -> float`` callable for use in GRPO/PPO."""
        if self.tokenizer is None:
            raise RuntimeError("RewardModel.score requires a tokenizer. Pass one to __init__ or use from_pretrained.")
        device = next(self.parameters()).device
        text = prompt + response
        enc = self.tokenizer(text, return_tensors="pt", truncation=True)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        return float(self.forward(ids, mask).item())


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def compute_bradley_terry_loss(
    chosen_rewards: torch.Tensor,    # [B]
    rejected_rewards: torch.Tensor,  # [B]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Standard preference-data loss: ``-log σ(r_w - r_l)``."""
    margin = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(margin).mean()
    metrics = {
        "rm/margin_mean": margin.detach().mean(),
        "rm/accuracy": (margin > 0).float().detach().mean(),
    }
    return loss, metrics


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


@dataclass
class RewardModelConfig:
    pad_token_id: int = 0


class RewardModelTrainer(Trainer):
    """Trainer that fits a ``RewardModel`` on preference pairs.

    Expects batches in the same format as ``DPOTrainer`` —
    ``DPODataCollator`` works unchanged.
    """

    def __init__(
        self,
        reward_model: RewardModel,
        train_dataloader: DataLoader,
        config: Optional[TrainerConfig] = None,
        eval_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(reward_model, train_dataloader, config, eval_dataloader, device)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        c_ids = batch["chosen_input_ids"]
        c_mask = batch["chosen_attention_mask"]
        r_ids = batch["rejected_input_ids"]
        r_mask = batch["rejected_attention_mask"]
        chosen_r = self.model(c_ids, c_mask)
        rejected_r = self.model(r_ids, r_mask)
        loss, _m = compute_bradley_terry_loss(chosen_r, rejected_r)
        return loss
