"""Sliding-window perplexity evaluator.

Standard recipe for fixed-context models: tokenise the corpus into one
long stream, slide a context window with a stride, and only score the
*last* ``stride`` tokens of each window (the prefix is just context for
the model). That way every token is scored exactly once even though
overlapping windows feed the model.

Output: dict with ``perplexity`` (the headline metric), ``loss`` (sum of
per-token NLL / total scored tokens), and ``tokens_scored``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch


@dataclass
class PerplexityEvaluator:
    """Sliding-window perplexity over an iterable of texts.

    Attributes
    ----------
    model : torch.nn.Module
        Causal-LM with the standard ``forward(input_ids, labels)`` →
        ``output.loss`` interface (HF causal LMs and the project's
        Qwen3 / Qwen3.5 engines all match).
    tokenizer : Any
        Tokenizer with ``encode(text)`` returning a ``list[int]``.
    max_length : int
        Window size. Must not exceed the model's context length.
    stride : int
        Number of *new* tokens scored per window. ``stride < max_length``
        means each window has a context prefix of ``max_length - stride``
        tokens that's already been scored in a previous window — those
        positions are masked out of the loss.
    device : torch.device | None
        Where to run the forward pass. Defaults to the model's device.
    """

    model: Any
    tokenizer: Any
    max_length: int = 2048
    stride: int = 1024
    device: Optional[torch.device] = None

    def __post_init__(self):
        if self.stride < 1:
            raise ValueError(f"stride must be ≥ 1; got {self.stride}")
        if self.max_length < self.stride:
            raise ValueError(
                f"max_length ({self.max_length}) must be ≥ stride ({self.stride})"
            )

    @torch.no_grad()
    def evaluate(
        self, dataset: Iterable[Any], text_field: str = "text",
    ) -> Dict[str, float]:
        """Compute perplexity on every example. Returns aggregated metrics."""
        device = self.device or next(self.model.parameters()).device
        self.model.eval()

        # Concatenate all token IDs into one stream; perplexity is reported
        # on the whole-corpus aggregate, which is the standard convention.
        all_tokens: list[int] = []
        for example in dataset:
            text = example[text_field] if isinstance(example, dict) else example
            ids = self.tokenizer.encode(text)
            if isinstance(ids, dict):
                ids = ids["input_ids"]
            all_tokens.extend(ids)

        if not all_tokens:
            return {"perplexity": float("nan"), "loss": 0.0, "tokens_scored": 0}

        nll_sum = 0.0
        n_scored = 0
        prev_end = 0

        for begin in range(0, len(all_tokens), self.stride):
            end = min(begin + self.max_length, len(all_tokens))
            chunk = torch.tensor(
                all_tokens[begin:end], dtype=torch.long, device=device,
            ).unsqueeze(0)

            # Loss only on tokens not already scored — the suffix from
            # ``prev_end`` onward.
            target_len = end - max(prev_end, begin)
            if target_len <= 0:
                if end == len(all_tokens):
                    break
                continue
            labels = chunk.clone()
            # Mask out the prefix (already-scored) positions with -100.
            labels[0, :-target_len] = -100

            out = self.model(input_ids=chunk, labels=labels)
            loss = out.loss if hasattr(out, "loss") else out[0]
            nll_sum += float(loss.item()) * target_len
            n_scored += target_len
            prev_end = end
            if end == len(all_tokens):
                break

        avg_loss = nll_sum / max(n_scored, 1)
        return {
            "perplexity": math.exp(avg_loss) if n_scored > 0 else float("nan"),
            "loss": avg_loss,
            "tokens_scored": n_scored,
        }
