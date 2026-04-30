"""FlashAttention-3 wrapper.

Soft-imports ``flash_attn_3`` (the Hopper-only FA3 variant from Dao-AILab)
and falls through to FA2 / standard attention when unavailable. Same
gating pattern as the FlashQLA adapter — Hopper users get the kernel for
free at runtime; Ampere/CPU users land on the existing FA2 / SDPA path
silently.

API::

    from llm_pipeline.inference.attention.flash_attention_3 import flash_attn_3_func

    out = flash_attn_3_func(q, k, v, causal=True)

The function signature mirrors FA2 so callers can swap in by name.
"""

from __future__ import annotations

from typing import Optional

import torch


def _import_fa3():
    """Try to import FA3. Returns the function or None."""
    try:
        from flash_attn_3 import flash_attn_func as _fn      # type: ignore[import-not-found]
        return _fn
    except Exception:
        return None


def _import_fa2():
    try:
        from flash_attn import flash_attn_func as _fn         # type: ignore[import-not-found]
        return _fn
    except Exception:
        return None


def flash_attn_3_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Drop-in self-attention kernel.

    Dispatch order:
      1. FA3 if importable and CUDA tensor.
      2. FA2 if importable and CUDA tensor.
      3. Pure-PyTorch SDPA (always available).

    Inputs Q/K/V shape: ``[B, T, H, head_dim]`` — matches FA2's expected
    layout. Returns ``[B, T, H, head_dim]``.
    """
    if q.is_cuda:
        fa3 = _import_fa3()
        if fa3 is not None:
            return fa3(
                q, k, v, causal=causal,
                softmax_scale=softmax_scale, dropout_p=dropout_p,
            )
        fa2 = _import_fa2()
        if fa2 is not None:
            return fa2(
                q, k, v, causal=causal,
                softmax_scale=softmax_scale, dropout_p=dropout_p,
            )

    # Fallback: SDPA. Reshape from FA2's [B, T, H, D] to SDPA's [B, H, T, D].
    q_sd = q.transpose(1, 2)
    k_sd = k.transpose(1, 2)
    v_sd = v.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_sd, k_sd, v_sd,
        is_causal=causal,
        dropout_p=dropout_p,
        scale=softmax_scale,
    )
    return out.transpose(1, 2)
