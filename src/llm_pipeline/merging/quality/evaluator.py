"""Quality evaluation for merged models.

Currently exposes a perplexity helper that takes a model + a list of
tokenized texts (or a DataLoader yielding `input_ids` / `attention_mask`).
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import math

import torch
import torch.nn as nn


@dataclass
class MergeQualityReport:
    """Aggregate result from comparing baseline(s) and a merged model."""
    perplexities: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@torch.no_grad()
def perplexity(
    model: nn.Module,
    batches: Iterable[Dict[str, torch.Tensor]],
    device: Optional[torch.device] = None,
) -> float:
    """Compute corpus-level perplexity.

    Each batch is expected to provide `input_ids` and (optionally)
    `attention_mask`. Token-level cross-entropy is summed across the corpus
    and divided by the total number of label tokens (next-token prediction).
    """
    device = device or next(model.parameters()).device
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)

    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask.to(device) if attention_mask is not None else None)
        logits = getattr(outputs, "logits", outputs)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        if attention_mask is not None:
            shift_mask = attention_mask.to(device)[..., 1:].contiguous()
            shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)

        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        n_tokens = (shift_labels != -100).sum().item()
        total_nll += loss.item()
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)
