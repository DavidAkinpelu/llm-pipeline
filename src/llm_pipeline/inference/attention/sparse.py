"""Sparse attention patterns: sink tokens + dilated.

Two well-studied patterns for very-long-context models:

- **Attention sinks** (Xiao et al. 2023): the first K tokens are always
  attended to regardless of distance. Compensates for the high attention
  weights LLMs assign to the first few positions even when they're not
  semantically relevant.
- **Dilated attention**: each position attends to every Δ-th past
  position plus a window. Captures long-range structure with sparsity.

Both implemented as boolean masks composed with SDPA. Custom kernels for
batched sparse attention are a follow-up — these are the correctness
references.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def attention_sink_mask(
    seq_len: int, n_sinks: int = 4, window: Optional[int] = None, device=None,
) -> torch.Tensor:
    """``mask[i, j] = True`` if j must NOT be attended by i.

    Allowed positions:
      - Causal predecessors within ``window`` (or full causal if window=None).
      - The first ``n_sinks`` positions, always.
    """
    if n_sinks < 0:
        raise ValueError(f"n_sinks must be ≥ 0; got {n_sinks}")
    i = torch.arange(seq_len, device=device).unsqueeze(-1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_violation = j > i
    if window is not None:
        too_far = (i - j) >= window
    else:
        too_far = torch.zeros_like(causal_violation)
    is_sink = j < n_sinks
    blocked = causal_violation | (too_far & ~is_sink)
    return blocked


def dilated_mask(
    seq_len: int, dilation: int = 4, window: int = 32, device=None,
) -> torch.Tensor:
    """Causal + dilated attention.

    Position i can attend j iff:
      - j ≤ i (causality), AND
      - (i - j) < window  OR  (i - j) % dilation == 0
    """
    if dilation < 1 or window < 1:
        raise ValueError(f"dilation and window must be ≥ 1; got {dilation}, {window}")
    i = torch.arange(seq_len, device=device).unsqueeze(-1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_violation = j > i
    distance = i - j
    in_window = (distance >= 0) & (distance < window)
    on_dilation = (distance >= 0) & (distance % dilation == 0)
    allowed = in_window | on_dilation
    return causal_violation | ~allowed


def sparse_attention(
    q: torch.Tensor,                     # [B, H, T, D]
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    n_sinks: Optional[int] = None,
    window: Optional[int] = None,
    dilation: Optional[int] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Sparse self-attention via SDPA + an explicit boolean mask.

    Pick exactly one of: ``n_sinks`` (sinks + optional window),
    ``dilation`` (dilated + window), or both ``n_sinks`` and ``dilation``
    (intersection).
    """
    T = q.shape[-2]
    parts = []
    if n_sinks is not None:
        parts.append(attention_sink_mask(T, n_sinks=n_sinks, window=window, device=q.device))
    if dilation is not None:
        d_window = window if window is not None else 32
        parts.append(dilated_mask(T, dilation=dilation, window=d_window, device=q.device))
    if not parts:
        raise ValueError("specify at least one of n_sinks or dilation")
    # Combined mask: blocked if blocked in ANY pattern (intersection of allowed).
    blocked = parts[0]
    for p in parts[1:]:
        blocked = blocked | p

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=~blocked, scale=softmax_scale, is_causal=False,
    )
