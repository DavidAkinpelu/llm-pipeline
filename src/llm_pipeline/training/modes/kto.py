"""Kahneman-Tversky Optimization (KTO).

Reference: Ethayarajh et al. "KTO: Model Alignment as Prospect Theoretic
Optimization" (2024).

KTO needs only **single-sample binary feedback** (each example is labeled
``desirable`` or ``undesirable``), not chosen/rejected pairs. The loss is
inspired by Kahneman-Tversky prospect theory:

    For desirable y:    L = 1 - σ( β · (r_θ(x,y) - z_ref) )
    For undesirable y:  L = 1 - σ( β · (z_ref - r_θ(x,y)) )

where ``r_θ(x,y) = log π(y|x) - log π_ref(y|x)`` is the implicit reward and
``z_ref`` is a per-batch reference value. The original paper estimates
``z_ref`` as a max-truncated KL — we use the simpler in-batch mean of
``r_θ`` over the *opposite* label group, which approximates the same
intent without needing a separate KL pass.

Data shape:
  Each example carries ``input_ids``, ``attention_mask``, ``labels`` (with
  prompt tokens masked to IGNORE_INDEX), and ``label`` ∈ {0, 1}: 1 means
  desirable, 0 means undesirable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dpo import _selected_logprobs
from .sft import IGNORE_INDEX
from ..trainer import Trainer, TrainerConfig


# --------------------------------------------------------------------------- #
# Config & data
# --------------------------------------------------------------------------- #


@dataclass
class KTOConfig:
    beta: float = 0.1
    desirable_weight: float = 1.0    # λ_D
    undesirable_weight: float = 1.0  # λ_U
    max_length: int = 2048
    pad_token_id: int = 0


def tokenize_kto(
    tokenizer,
    prompt: str,
    response: str,
    desirable: bool,
    max_length: int = 2048,
    add_eos: bool = True,
) -> Dict[str, Any]:
    """Tokenize one (prompt, response, label) example for KTO."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    resp_ids = tokenizer(response, add_special_tokens=False).input_ids
    if add_eos and tokenizer.eos_token_id is not None:
        resp_ids = resp_ids + [tokenizer.eos_token_id]
    full = (prompt_ids + resp_ids)[:max_length]
    return {
        "input_ids": full,
        "prompt_len": min(len(prompt_ids), len(full)),
        "label": 1 if desirable else 0,
    }


class KTODataCollator:
    """Pads a batch of (prompt+response, label) examples for KTO."""

    def __init__(self, config: Optional[KTOConfig] = None):
        self.config = config or KTOConfig()

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        cfg = self.config
        max_len = min(cfg.max_length, max(len(ex["input_ids"]) for ex in batch))
        pad = cfg.pad_token_id
        bsz = len(batch)
        ids = torch.full((bsz, max_len), pad, dtype=torch.long)
        mask = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long)
        kto_labels = torch.zeros(bsz, dtype=torch.long)

        for i, ex in enumerate(batch):
            seq = ex["input_ids"][: max_len]
            n = len(seq)
            ids[i, :n] = torch.tensor(seq, dtype=torch.long)
            mask[i, :n] = 1
            p = min(int(ex.get("prompt_len", 0)), n)
            labels[i, p:n] = ids[i, p:n]
            kto_labels[i] = int(ex["label"])

        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": labels,
            "kto_labels": kto_labels,
        }


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def compute_kto_loss(
    policy_logps: torch.Tensor,    # [B] sum of response log-probs under policy
    ref_logps: torch.Tensor,       # [B] same, under frozen reference
    kto_labels: torch.Tensor,      # [B] in {0, 1}
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """KTO loss with simple in-batch ``z_ref``.

    ``z_ref`` is approximated by the mean implicit reward of the *opposite*
    label group within the same batch. If the batch is single-class, falls
    back to the mean implicit reward overall.
    """
    rewards = beta * (policy_logps - ref_logps)  # [B]
    is_desir = kto_labels == 1
    is_undes = kto_labels == 0

    # In-batch z_ref per element (gradient-free reference baseline).
    desir_mean = rewards[is_desir].detach().mean() if is_desir.any() else rewards.detach().mean()
    undes_mean = rewards[is_undes].detach().mean() if is_undes.any() else rewards.detach().mean()
    z_ref = torch.where(is_desir, undes_mean, desir_mean)

    desir_loss = 1.0 - torch.sigmoid(rewards - z_ref)
    undes_loss = 1.0 - torch.sigmoid(z_ref - rewards)

    weight = torch.where(
        is_desir,
        torch.full_like(rewards, desirable_weight),
        torch.full_like(rewards, undesirable_weight),
    )
    per_sample = torch.where(is_desir, desir_loss, undes_loss) * weight
    loss = per_sample.mean()

    metrics = {
        "kto/reward_mean": rewards.detach().mean(),
        "kto/reward_desir": (rewards[is_desir].detach().mean() if is_desir.any() else torch.tensor(0.0)),
        "kto/reward_undes": (rewards[is_undes].detach().mean() if is_undes.any() else torch.tensor(0.0)),
    }
    return loss, metrics


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class KTOTrainer(Trainer):
    """Trainer for KTO. Holds a frozen reference model on the same device."""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        config: Optional[TrainerConfig] = None,
        kto_config: Optional[KTOConfig] = None,
        ref_model: Optional[nn.Module] = None,
        eval_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, train_dataloader, config, eval_dataloader, device)
        self.kto_config = kto_config or KTOConfig()

        if ref_model is None:
            ref_model = copy.deepcopy(self._unwrapped_model())
        ref_model.to(self.device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        self.ref_model = ref_model

    def _seq_logp(self, model: nn.Module, ids: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        outputs = model(input_ids=ids, attention_mask=mask)
        logits = getattr(outputs, "logits", outputs)
        return _selected_logprobs(logits, labels)  # [B] sum of response log-probs

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        labels = batch["labels"]
        kto_labels = batch["kto_labels"]

        policy_logps = self._seq_logp(self.model, ids, mask, labels)
        with torch.no_grad():
            ref_logps = self._seq_logp(self.ref_model, ids, mask, labels)

        loss, _m = compute_kto_loss(
            policy_logps, ref_logps, kto_labels,
            beta=self.kto_config.beta,
            desirable_weight=self.kto_config.desirable_weight,
            undesirable_weight=self.kto_config.undesirable_weight,
        )
        return loss
