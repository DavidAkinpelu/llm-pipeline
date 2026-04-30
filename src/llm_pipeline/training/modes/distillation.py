"""Knowledge distillation: train a small student to match a frozen teacher's logits.

The classical Hinton et al. (2015) recipe — soft-target KL divergence between
``softmax(student / T)`` and ``softmax(teacher / T)``, optionally combined
with a hard-label cross-entropy term:

.. code-block:: text

    L = α · T² · KL(softmax(s/T) ‖ softmax(t/T)) + (1 − α) · CE(s, y_hard)

The ``T²`` factor compensates for the fact that the soft-target gradient
scales as ``1/T²`` — without it, raising ``T`` flattens the loss and slows
training. The two-term combo lets you mix "match the teacher's full
distribution" with "still get the right answer", weighted by ``α``.

This module provides:

- ``compute_distillation_loss(student_logits, teacher_logits, labels=None,
  temperature=4.0, alpha=0.9, ignore_index=-100)`` — the loss kernel.
- ``DistillationTrainer`` — a thin Trainer subclass that loads the teacher
  in eval mode, runs both forwards, and adds the distillation loss to
  the standard CE loss.

Use cases
---------

- Compress a 7B teacher into a 1B student that fits a phone.
- Match the soft logit distribution of a stronger teacher to bootstrap
  fine-tuning quality without paying the inference cost.
- Speculative-decoding training: distill the target's logits into a draft
  model so it predicts the right tokens (high acceptance rate).

Notes
-----

- Both models must share the same tokenizer / vocab — the loss compares
  positions of softmax distributions.
- Teacher is held in eval mode and ``no_grad``; only the student trains.
- Causal-LM padding: pass ``labels`` with ``-100`` at masked positions and
  the same mask is honoured by both terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..trainer import Trainer, TrainerConfig


@dataclass
class DistillationConfig:
    """Hyperparameters for ``compute_distillation_loss`` / ``DistillationTrainer``.

    Attributes
    ----------
    temperature : float
        ``T`` in the soft-target softmax. Higher T → softer distribution and
        more weight on the teacher's "non-argmax" preferences. Default 4.0
        is the original Hinton paper value; 2.0–8.0 is the typical range.
    alpha : float
        Weight on the KL loss. ``(1 − alpha)`` weights the hard-label CE.
        With ``alpha=1.0`` you get pure distillation (no labels needed).
    ignore_index : int
        Label value to ignore in both loss terms (HF convention: ``-100``).
    """

    temperature: float = 4.0
    alpha: float = 0.9
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {self.temperature}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {self.alpha}")


def compute_distillation_loss(
    student_logits: torch.Tensor,             # [B, T, V]
    teacher_logits: torch.Tensor,             # [B, T, V]
    labels: Optional[torch.Tensor] = None,    # [B, T] hard labels (or None for pure KL)
    temperature: float = 4.0,
    alpha: float = 0.9,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """Soft-target KL + optional hard-label CE distillation loss.

    Returns a dict with ``loss`` (the combined scalar to backprop on) plus
    ``kl_loss`` and ``ce_loss`` for logging. If ``labels`` is None or
    ``alpha == 1.0``, the CE term is zero.

    Shape contract: ``student_logits`` and ``teacher_logits`` must agree
    exactly. They're the **next-token** logits — i.e. position ``t``
    predicts ``labels[t]``. The caller is responsible for any shifting.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"shape mismatch: student {tuple(student_logits.shape)} "
            f"vs teacher {tuple(teacher_logits.shape)}"
        )

    # Build the position mask: True where we should count the loss.
    if labels is not None:
        valid = labels != ignore_index                      # [B, T]
    else:
        valid = torch.ones(student_logits.shape[:2], dtype=torch.bool, device=student_logits.device)
    n_valid = valid.sum().clamp_min(1)

    # KL(soft_student || soft_teacher), temperature-scaled.
    s_log = F.log_softmax(student_logits / temperature, dim=-1)
    t_soft = F.softmax(teacher_logits / temperature, dim=-1)
    kl_per_token = F.kl_div(s_log, t_soft, reduction="none").sum(dim=-1)   # [B, T]
    kl_loss = (kl_per_token * valid.to(kl_per_token.dtype)).sum() / n_valid
    # Compensate for the 1/T² gradient scaling.
    kl_loss = kl_loss * (temperature ** 2)

    # Hard-label CE (optional).
    if labels is not None and alpha < 1.0:
        ce_loss = F.cross_entropy(
            student_logits.reshape(-1, student_logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=ignore_index,
            reduction="mean",
        )
    else:
        ce_loss = torch.zeros((), device=student_logits.device)

    loss = alpha * kl_loss + (1.0 - alpha) * ce_loss
    return {"loss": loss, "kl_loss": kl_loss, "ce_loss": ce_loss}


class DistillationTrainer(Trainer):
    """Trainer that adds a teacher-student distillation loss to the standard SFT loop.

    The teacher must already be on the right device (typically the same as
    the student); we set it to eval mode and freeze it.

    >>> teacher = AutoModelForCausalLM.from_pretrained("big-model")
    >>> student = AutoModelForCausalLM.from_pretrained("tiny-model")
    >>> dist_cfg = DistillationConfig(temperature=4.0, alpha=0.9)
    >>> trainer = DistillationTrainer(
    ...     model=student, teacher_model=teacher, dataloader=dl,
    ...     config=TrainerConfig(...), distill_config=dist_cfg,
    ... )
    >>> trainer.train()
    """

    def __init__(
        self,
        *args,
        teacher_model: nn.Module,
        distill_config: Optional[DistillationConfig] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.distill_config = distill_config or DistillationConfig()

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Override the base trainer's loss to add the distillation term."""
        # Student forward (gradient on).
        student_out = self.model(**{k: v for k, v in batch.items() if k != "labels"})
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]

        # Teacher forward (no gradient).
        with torch.no_grad():
            teacher_out = self.teacher(**{k: v for k, v in batch.items() if k != "labels"})
            teacher_logits = teacher_out.logits if hasattr(teacher_out, "logits") else teacher_out[0]

        labels = batch.get("labels")
        losses = compute_distillation_loss(
            student_logits, teacher_logits, labels=labels,
            temperature=self.distill_config.temperature,
            alpha=self.distill_config.alpha,
            ignore_index=self.distill_config.ignore_index,
        )
        # Stash diagnostic terms for logging hooks.
        self._last_distill_metrics = {
            "kl_loss": losses["kl_loss"].detach(),
            "ce_loss": losses["ce_loss"].detach(),
        }
        return losses["loss"]
