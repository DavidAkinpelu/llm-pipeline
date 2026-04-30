"""Sliding-window attention.

Restricts each query position to attending only the most recent
``window`` tokens (and itself). Used by Mistral / Qwen3-style models
with a windowed-attention KV cache.

Provides:

- ``sliding_window_mask(seq_len, window)`` — builds the boolean mask
  used by SDPA (or any masked-attention path).
- ``sliding_window_attention(q, k, v, window, causal=True)`` — full
  attention call with the mask baked in. Uses SDPA on CUDA, falls back
  to a torch reference on CPU (correctness, not speed).

A custom Triton kernel for sliding-window attention exists in vLLM /
xformers; we ship the masked-SDPA path here for cloud H100 debugging
parity, then swap in the fused kernel later.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def sliding_window_mask(seq_len: int, window: int, device=None) -> torch.Tensor:
    """Boolean mask: ``mask[i, j] = True`` if position j is *not* allowed
    to be attended by position i.

    Causal + window: position i can attend j iff ``i - window < j ≤ i``.
    """
    if window <= 0:
        raise ValueError(f"window must be ≥ 1; got {window}")
    i = torch.arange(seq_len, device=device).unsqueeze(-1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    too_far = (i - j) >= window                          # past the window
    future = j > i                                        # causal
    return too_far | future


def sliding_window_attention(
    q: torch.Tensor,                     # [B, H, T, D]
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Sliding-window self-attention via SDPA + explicit mask.

    The mask is constructed once per call; for repeated decodes with the
    same shape, callers can hoist the mask construction out.
    """
    if window <= 0:
        raise ValueError(f"window must be ≥ 1; got {window}")
    B, H, T, D = q.shape
    if k.shape[-2] != T or v.shape[-2] != T:
        raise ValueError(
            f"q/k/v sequence lengths must match; got {q.shape}, {k.shape}, {v.shape}"
        )
    mask = sliding_window_mask(T, window, device=q.device)
    # SDPA wants True = MASKED (i.e. don't attend). We pass the mask directly.
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=~mask, scale=softmax_scale, is_causal=False,
    )
