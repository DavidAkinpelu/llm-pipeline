"""Odds Ratio Preference Optimization (ORPO).

Reference: Hong et al. "ORPO: Monolithic Preference Optimization without
Reference Model" (2024).

ORPO is a single-stage algorithm that combines supervised fine-tuning with
preference learning, without needing a reference policy. The total loss is

    L = L_SFT + λ · L_OR

where ``L_SFT`` is the standard next-token cross-entropy on the *chosen*
response, and ``L_OR`` is the odds-ratio preference term

    L_OR = -log σ( log_odds(y_w|x) - log_odds(y_l|x) )
    log_odds(y|x) = log P(y|x) - log(1 - P(y|x))

``P(y|x)`` is the *geometric-mean* token probability — i.e. ``exp`` of the
mean log-probability over the response tokens. We use ``log1mexp`` for the
``log(1 - exp)`` term so things stay numerically stable.

Data shape is the same as DPO: each example has ``chosen_input_ids`` and
``rejected_input_ids`` plus ``prompt_len``. ``DPODataCollator`` works
unchanged for ORPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dpo import _selected_logprobs
from .sft import IGNORE_INDEX
from ..trainer import Trainer, TrainerConfig


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class ORPOConfig:
    lambda_or: float = 0.1   # weight on the odds-ratio term


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(x)) for x <= 0.

    Uses the standard two-branch trick to avoid catastrophic cancellation
    near x = 0 and overflow far from it.
    """
    # Clamp to slightly below 0 to keep log(1 - exp(x)) defined.
    x = x.clamp(max=-1e-12)
    return torch.where(
        x > -0.6931471805599453,  # x > log(0.5)
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


def _seq_log_prob_mean(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean log-prob, ignoring positions where label == IGNORE_INDEX.

    Returns shape ``(batch,)`` — the geometric-mean log-prob per sequence,
    which equals ``log P(y|x)`` when ``P`` is treated as the geometric mean
    of token probabilities.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask = (shift_labels != IGNORE_INDEX).to(log_probs.dtype)
    safe_labels = shift_labels.clamp(min=0)
    gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    sum_logp = (gathered * mask).sum(dim=-1)
    n = mask.sum(dim=-1).clamp(min=1.0)
    return sum_logp / n


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def compute_orpo_loss(
    chosen_logits: torch.Tensor,    # [B, T, V]
    chosen_labels: torch.Tensor,    # [B, T]
    rejected_logits: torch.Tensor,  # [B, T, V]
    rejected_labels: torch.Tensor,  # [B, T]
    lambda_or: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """ORPO loss = SFT(chosen) + λ · OR-loss(chosen, rejected)."""
    # 1) SFT cross-entropy on chosen.
    shift_logits = chosen_logits[..., :-1, :].contiguous()
    shift_labels = chosen_labels[..., 1:].contiguous()
    sft_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )

    # 2) Geometric-mean log-probs per sequence.
    log_p_w = _seq_log_prob_mean(chosen_logits, chosen_labels)
    log_p_l = _seq_log_prob_mean(rejected_logits, rejected_labels)

    # 3) Log-odds: log p - log(1 - p) where log p is in log space.
    log_odds_w = log_p_w - _log1mexp(log_p_w)
    log_odds_l = log_p_l - _log1mexp(log_p_l)

    # 4) Odds-ratio preference loss.
    or_loss = -F.logsigmoid(log_odds_w - log_odds_l).mean()

    loss = sft_loss + lambda_or * or_loss
    metrics = {
        "orpo/sft_loss": sft_loss.detach(),
        "orpo/or_loss": or_loss.detach(),
        "orpo/log_odds_diff": (log_odds_w - log_odds_l).detach().mean(),
    }
    return loss, metrics


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class ORPOTrainer(Trainer):
    """Trainer with ORPO loss. No reference model required.

    Expects batches in the same format as ``DPOTrainer`` — the
    ``DPODataCollator`` works unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        config: Optional[TrainerConfig] = None,
        orpo_config: Optional[ORPOConfig] = None,
        eval_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, train_dataloader, config, eval_dataloader, device)
        self.orpo_config = orpo_config or ORPOConfig()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        c_ids = batch["chosen_input_ids"]
        c_mask = batch["chosen_attention_mask"]
        c_lbl = batch["chosen_labels"]
        r_ids = batch["rejected_input_ids"]
        r_mask = batch["rejected_attention_mask"]
        r_lbl = batch["rejected_labels"]

        chosen_out = self.model(input_ids=c_ids, attention_mask=c_mask)
        rejected_out = self.model(input_ids=r_ids, attention_mask=r_mask)
        chosen_logits = getattr(chosen_out, "logits", chosen_out)
        rejected_logits = getattr(rejected_out, "logits", rejected_out)

        loss, _m = compute_orpo_loss(
            chosen_logits, c_lbl, rejected_logits, r_lbl,
            lambda_or=self.orpo_config.lambda_or,
        )
        return loss
