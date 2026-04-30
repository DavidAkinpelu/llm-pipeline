"""Fused softmax-with-causal-mask Triton kernel.

Numerically-stable online softmax fused with the causal-mask step.
Standard for self-attention's pre-attention-output softmax — fuses
``mask_fill(-inf)`` and ``softmax`` into one pass.

Formula::

    masked = where(j > i, -inf, x[i, j])
    s      = softmax(masked, dim=-1)

Runtime gating: requires CUDA + ``triton``; non-CUDA hosts get the
torch reference automatically.
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
    def _softmax_kernel(
        x_ptr, out_ptr,
        x_row_stride, out_row_stride,
        n_cols: tl.constexpr, row_idx_for_mask,
        causal: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One program per row. ``row_idx_for_mask`` lets the host pass
        the absolute row index (in the original BxHxT logical layout)
        when this kernel is invoked on a flattened view.
        """
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK_SIZE)
        col_mask = col < n_cols

        x = tl.load(x_ptr + row * x_row_stride + col, mask=col_mask, other=-float("inf"))
        x = x.to(tl.float32)

        if causal:
            # Mask future positions (col > row_idx_for_mask).
            i = tl.load(row_idx_for_mask + row).to(tl.int32)
            future = col > i
            x = tl.where(future, -float("inf"), x)

        max_x = tl.max(x, axis=0)
        x_shifted = x - max_x
        exp_x = tl.exp(x_shifted)
        # Mask out the over-fetched columns from the sum.
        exp_x = tl.where(col_mask, exp_x, 0.0)
        denom = tl.sum(exp_x, axis=0)
        out = exp_x / denom

        tl.store(out_ptr + row * out_row_stride + col, out, mask=col_mask)


def fused_softmax_triton(
    x: torch.Tensor,                     # [..., n_cols]
    causal: bool = False,
    row_offset: int = 0,
) -> torch.Tensor:
    """Numerically-stable softmax with optional causal masking.

    With ``causal=True``, the kernel masks positions ``j > i + row_offset``
    in each row to ``-inf``. ``row_offset`` lets you slice into a longer
    sequence (e.g. for the new-token rows of a decode step).
    """
    if not _HAS_TRITON or not x.is_cuda:
        return _torch_reference(x, causal=causal, row_offset=row_offset)

    orig_shape = x.shape
    n_cols = orig_shape[-1]
    x_flat = x.reshape(-1, n_cols)
    out = torch.empty_like(x_flat)
    n_rows = x_flat.shape[0]

    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_cols:
        BLOCK_SIZE *= 2

    # Per-row mask index: row index in the original sequence (rows start
    # at ``row_offset`` and increment by 1, modulo the trailing axes).
    # For 2D inputs (T, T) row r corresponds to index r + row_offset.
    # For higher-rank we just propagate r within the last-dim group.
    row_seq_len = orig_shape[-2] if x.dim() >= 2 else 1
    row_idx = torch.arange(n_rows, device=x.device, dtype=torch.int32) % row_seq_len + row_offset

    _softmax_kernel[(n_rows,)](
        x_flat, out,
        x_flat.stride(0), out.stride(0),
        n_cols=n_cols, row_idx_for_mask=row_idx,
        causal=int(causal), BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.reshape(orig_shape).to(x.dtype)


def _torch_reference(x, causal, row_offset):
    if causal:
        n_rows = x.shape[-2] if x.dim() >= 2 else 1
        n_cols = x.shape[-1]
        row_idx = torch.arange(n_rows, device=x.device).unsqueeze(-1) + row_offset
        col_idx = torch.arange(n_cols, device=x.device).unsqueeze(0)
        mask = col_idx > row_idx                                 # [n_rows, n_cols]
        x = x.masked_fill(mask, float("-inf"))
    return torch.softmax(x.float(), dim=-1).to(x.dtype)
