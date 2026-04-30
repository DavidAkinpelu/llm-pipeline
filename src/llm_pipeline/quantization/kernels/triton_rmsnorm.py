"""Fused RMSNorm Triton kernel.

One-pass mean-square + normalise + scale. Standard layer for Llama /
Qwen-style models.

Forward formula::

    var  = mean(x ** 2, dim=-1)
    out  = x * rsqrt(var + eps) * weight

For Qwen3-style RMSNorm with ``(1 + weight)`` scaling, pass
``add_residual=True`` (defaults to plain weight scaling).

Runtime gating: requires CUDA + ``triton``. CPU/non-Triton hosts get
the torch reference via ``apply_rmsnorm_triton``.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _rmsnorm_kernel(
        x_ptr, w_ptr, out_ptr,
        x_row_stride, out_row_stride,
        n_cols: tl.constexpr, eps: tl.constexpr,
        add_residual_one: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One program per row (over flattened batch dims).

        Computes the per-row mean-square in fp32 even when inputs are
        fp16/bf16 — same trick standard RMSNorm uses to keep numerics
        sane at small std.
        """
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        x = tl.load(x_ptr + row * x_row_stride + col, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        normed = x * rstd
        if add_residual_one:
            scaled = normed * (1.0 + w)
        else:
            scaled = normed * w
        tl.store(out_ptr + row * out_row_stride + col, scaled, mask=mask)


def apply_rmsnorm_triton(
    x: torch.Tensor,                     # [..., n_cols]
    weight: torch.Tensor,                # [n_cols]
    eps: float = 1e-6,
    add_residual_one: bool = False,
) -> torch.Tensor:
    """Fused RMSNorm via Triton. Falls through to a torch reference on
    non-CUDA inputs.
    """
    if not _HAS_TRITON or not x.is_cuda:
        return _torch_reference(x, weight, eps=eps, add_residual_one=add_residual_one)

    orig_shape = x.shape
    n_cols = orig_shape[-1]
    x_flat = x.reshape(-1, n_cols)
    out = torch.empty_like(x_flat)
    n_rows = x_flat.shape[0]

    # Triton needs BLOCK_SIZE to be a power of 2 ≥ n_cols.
    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_cols:
        BLOCK_SIZE *= 2

    _rmsnorm_kernel[(n_rows,)](
        x_flat, weight, out,
        x_flat.stride(0), out.stride(0),
        n_cols=n_cols, eps=eps,
        add_residual_one=int(add_residual_one),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.reshape(orig_shape).to(x.dtype)


def _torch_reference(x, weight, eps, add_residual_one):
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    normed = x.float() * torch.rsqrt(var + eps)
    if add_residual_one:
        out = normed * (1.0 + weight.float())
    else:
        out = normed * weight.float()
    return out.to(x.dtype)
