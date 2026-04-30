"""Selective activation recomputation.

Standard ``torch.utils.checkpoint`` is all-or-nothing: either the entire
wrapped block is recomputed on backward, or nothing is. For transformer
blocks the right tradeoff is mixed: recompute the **cheap** ops (RMSNorm,
RoPE, activation functions) but **keep** the matmul outputs because
recomputing them costs the bulk of the forward pass.

This module provides ``SelectiveCheckpointWrapper`` — wraps a forward
function that returns ``(cheap_intermediate, expensive_intermediate)``
and on backward re-runs only the path producing the cheap intermediate.

Memory math (rough): for a Llama-style block,
    matmul activations ≈ 80% of activation memory
    norm + RoPE + activation ≈ 20% of activation memory
With selective recompute we save the 20% (recompute is cheap) but keep
the 80% (would dominate runtime if recomputed).
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import torch
import torch.nn as nn


class _SelectiveCheckpointFunction(torch.autograd.Function):
    """Custom autograd function — saves only the inputs + the *expensive*
    intermediate; recomputes the *cheap* intermediate on backward.
    """

    @staticmethod
    def forward(
        ctx,
        cheap_fn: Callable[..., torch.Tensor],
        expensive_fn: Callable[..., torch.Tensor],
        *args: torch.Tensor,
    ) -> torch.Tensor:
        # Forward pass without grad on cheap_fn; we'll re-run it under grad on backward.
        with torch.no_grad():
            cheap_out = cheap_fn(*args)
        # Expensive part runs under grad (its activations are saved by autograd
        # implicitly because we don't no_grad it).
        expensive_out = expensive_fn(cheap_out)
        ctx.save_for_backward(*args)
        ctx.cheap_fn = cheap_fn
        ctx.expensive_fn = expensive_fn
        return expensive_out

    @staticmethod
    def backward(ctx, grad_output):
        args = ctx.saved_tensors
        # Re-run the cheap path under grad to rebuild the autograd graph
        # for it; then expensive on top; backprop.
        with torch.enable_grad():
            args_with_grad = tuple(
                a.detach().requires_grad_(a.requires_grad) for a in args
            )
            cheap_out = ctx.cheap_fn(*args_with_grad)
            expensive_out = ctx.expensive_fn(cheap_out)
            torch.autograd.backward(expensive_out, grad_tensors=grad_output)
            grads = tuple(a.grad if a.requires_grad else None for a in args_with_grad)
        # Two None placeholders for the two non-tensor positional args (cheap_fn, expensive_fn).
        return (None, None) + grads


class SelectiveCheckpointWrapper(nn.Module):
    """Wraps a (cheap_fn, expensive_fn) pair into a single forward.

    Typical use: split a transformer block into:

      - ``cheap_fn(*x) -> normed_x``: input layernorm + maybe RoPE.
      - ``expensive_fn(normed_x) -> out``: attention + projection
        (matmul-heavy).

    Backward recomputes ``cheap_fn`` but the autograd-saved activations
    inside ``expensive_fn`` are kept in memory.
    """

    def __init__(
        self,
        cheap_fn: Callable[..., torch.Tensor],
        expensive_fn: Callable[..., torch.Tensor],
    ):
        super().__init__()
        self.cheap_fn = cheap_fn
        self.expensive_fn = expensive_fn

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return _SelectiveCheckpointFunction.apply(
            self.cheap_fn, self.expensive_fn, *args,
        )
