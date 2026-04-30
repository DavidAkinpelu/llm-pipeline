"""Direct Preference Optimization (DPO).

Reference: Rafailov et al., "Direct Preference Optimization: Your Language
Model is Secretly a Reward Model" (NeurIPS 2023).

DPO replaces the SFT cross-entropy loss with a contrastive objective over
preference pairs ``(prompt, chosen, rejected)``. The loss is:

    L_DPO = -log σ( β · ((log π(chosen) - log π_ref(chosen))
                       - (log π(rejected) - log π_ref(rejected))) )

where ``π`` is the policy being trained and ``π_ref`` is a frozen reference
(typically the SFT-checkpoint that initialized π). β controls how much the
policy is allowed to drift from π_ref; common values are 0.1–0.5.

This module provides:
  * ``PreferenceExample`` — light dict shape for one (prompt, chosen, rejected) tuple.
  * ``DPODataCollator`` — pads/masks chosen and rejected sequences side-by-side.
  * ``compute_dpo_loss`` — given policy + ref logits, returns the scalar DPO loss.
  * ``DPOTrainer`` — thin subclass of ``Trainer`` that overrides ``compute_loss``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .sft import IGNORE_INDEX
from ..trainer import Trainer, TrainerConfig


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass
class DPOConfig:
    max_length: int = 2048
    pad_token_id: int = 0
    beta: float = 0.1
    label_smoothing: float = 0.0  # cDPO; 0 disables.


def tokenize_preference(
    tokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    max_length: int = 2048,
    add_eos: bool = True,
) -> Dict[str, Any]:
    """Tokenize one preference example into the shape expected by ``DPODataCollator``."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    chosen_ids = tokenizer(chosen, add_special_tokens=False).input_ids
    rejected_ids = tokenizer(rejected, add_special_tokens=False).input_ids
    if add_eos and tokenizer.eos_token_id is not None:
        chosen_ids = chosen_ids + [tokenizer.eos_token_id]
        rejected_ids = rejected_ids + [tokenizer.eos_token_id]

    chosen_input = (prompt_ids + chosen_ids)[:max_length]
    rejected_input = (prompt_ids + rejected_ids)[:max_length]
    return {
        "chosen_input_ids": chosen_input,
        "rejected_input_ids": rejected_input,
        "prompt_len": min(len(prompt_ids), len(chosen_input), len(rejected_input)),
    }


class DPODataCollator:
    """Pads chosen and rejected sequences from a batch of preference examples."""

    def __init__(self, config: Optional[DPOConfig] = None):
        self.config = config or DPOConfig()

    def _pad(self, sequences: Sequence[List[int]], pad_id: int, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = len(sequences)
        ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((bsz, max_len), dtype=torch.long)
        for i, s in enumerate(sequences):
            n = min(len(s), max_len)
            ids[i, :n] = torch.tensor(s[:n], dtype=torch.long)
            mask[i, :n] = 1
        return ids, mask

    def _labels(self, ids: torch.Tensor, prompt_lens: Sequence[int]) -> torch.Tensor:
        labels = ids.clone()
        for i, p in enumerate(prompt_lens):
            labels[i, :int(p)] = IGNORE_INDEX
        # Pad positions are IGNORE too — ids was filled with pad_token_id but the
        # mask says it's pad, so use the attention mask separately. Here we
        # rely on attention_mask + IGNORE for prompt; pad positions get
        # IGNORE via masked_fill in compute_dpo_loss.
        return labels

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        cfg = self.config
        max_len = min(
            cfg.max_length,
            max(max(len(ex["chosen_input_ids"]), len(ex["rejected_input_ids"])) for ex in batch),
        )
        pad = cfg.pad_token_id
        chosen_ids, chosen_mask = self._pad([ex["chosen_input_ids"] for ex in batch], pad, max_len)
        rejected_ids, rejected_mask = self._pad([ex["rejected_input_ids"] for ex in batch], pad, max_len)
        prompt_lens = [int(ex.get("prompt_len", 0)) for ex in batch]

        chosen_labels = self._labels(chosen_ids, prompt_lens)
        rejected_labels = self._labels(rejected_ids, prompt_lens)
        # Pad positions -> IGNORE_INDEX so they don't count toward log-prob.
        chosen_labels = chosen_labels.masked_fill(chosen_mask == 0, IGNORE_INDEX)
        rejected_labels = rejected_labels.masked_fill(rejected_mask == 0, IGNORE_INDEX)

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
            "rejected_labels": rejected_labels,
        }


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def _selected_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log-probabilities of the labels under logits, ignoring IGNORE_INDEX.

    Returns a tensor of shape ``(batch,)``.
    """
    # Shift so token n predicts token n+1 (standard causal-LM convention).
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask = (shift_labels != IGNORE_INDEX).float()
    safe_labels = shift_labels.clamp(min=0)
    gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=-1)


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the DPO loss and a few diagnostic tensors.

    ``label_smoothing`` > 0 enables conservative DPO (cDPO).
    """
    pi_logratio = policy_chosen_logps - policy_rejected_logps
    ref_logratio = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratio - ref_logratio)

    if label_smoothing == 0.0:
        loss = -F.logsigmoid(logits).mean()
    else:
        loss = (
            -(1 - label_smoothing) * F.logsigmoid(logits)
            - label_smoothing * F.logsigmoid(-logits)
        ).mean()

    metrics = {
        "rewards/chosen": beta * (policy_chosen_logps - ref_chosen_logps).detach(),
        "rewards/rejected": beta * (policy_rejected_logps - ref_rejected_logps).detach(),
        "rewards/margin": (logits / max(beta, 1e-8)).detach(),
        "rewards/accuracy": (logits > 0).float().detach(),
    }
    return loss, metrics


# --------------------------------------------------------------------------- #
# Trainer integration
# --------------------------------------------------------------------------- #


class DPOTrainer(Trainer):
    """Trainer with DPO loss. Holds a frozen reference model on the same device.

    By default the reference is a deep copy of the policy at trainer
    construction time. Pass ``ref_model`` to use a different reference
    (typical workflow: pass the SFT checkpoint).
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        config: Optional[TrainerConfig] = None,
        ref_model: Optional[nn.Module] = None,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        eval_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, train_dataloader, config, eval_dataloader, device)
        self.beta = beta
        self.label_smoothing = label_smoothing

        if ref_model is None:
            ref_model = copy.deepcopy(self.model)
        ref_model.to(self.device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        self.ref_model = ref_model

    def _forward_logps(self, model: nn.Module, ids: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        outputs = model(input_ids=ids, attention_mask=mask)
        logits = getattr(outputs, "logits", outputs)
        return _selected_logprobs(logits, labels)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        c_ids = batch["chosen_input_ids"]
        c_mask = batch["chosen_attention_mask"]
        c_lbl = batch["chosen_labels"]
        r_ids = batch["rejected_input_ids"]
        r_mask = batch["rejected_attention_mask"]
        r_lbl = batch["rejected_labels"]

        policy_chosen = self._forward_logps(self.model, c_ids, c_mask, c_lbl)
        policy_rejected = self._forward_logps(self.model, r_ids, r_mask, r_lbl)
        with torch.no_grad():
            ref_chosen = self._forward_logps(self.ref_model, c_ids, c_mask, c_lbl)
            ref_rejected = self._forward_logps(self.ref_model, r_ids, r_mask, r_lbl)

        loss, _metrics = compute_dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected,
            beta=self.beta, label_smoothing=self.label_smoothing,
        )
        return loss
