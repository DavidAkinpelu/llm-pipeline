"""Fused RoPE Triton kernel.

In-place rotary position embedding for Q and K. Reads ``cos`` / ``sin``
once and rotates both tensors in a single launch — saves the two extra
DRAM traversals an unfused implementation pays.

Layout: Q/K shape ``[B, H, T, head_dim]``. The rotation uses the
half-shift convention (matches our existing ``apply_partial_rotary_pos_emb``
in the Qwen3.5 module). Optional partial-rotary support: if ``rotary_dim
< head_dim`` the trailing channels pass through untouched.

Runtime gating: requires CUDA + ``triton``. On non-CUDA hosts the
``apply_rope_triton`` callable raises ``NotImplementedError`` with a
clear message; tests skip cleanly.
"""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _rope_kernel(
        q_ptr, k_ptr, cos_ptr, sin_ptr,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_cb, stride_ct, stride_cd,
        B: tl.constexpr, H_q: tl.constexpr, H_k: tl.constexpr,
        T: tl.constexpr, head_dim: tl.constexpr, rotary_dim: tl.constexpr,
    ):
        """One program per (batch, head, time). Rotates both halves of the
        rotary_dim slice in a single pass; trailing channels untouched.
        """
        bh = tl.program_id(0)
        t = tl.program_id(1)
        b = bh // tl.maximum(H_q, H_k)
        h = bh % tl.maximum(H_q, H_k)

        half = rotary_dim // 2
        col_lo = tl.arange(0, 64)                              # over-fetch; mask below
        mask_lo = col_lo < half
        mask_full = col_lo < rotary_dim

        # Cos/sin shape [B, T, rotary_dim].
        cos_off = b * stride_cb + t * stride_ct + col_lo * stride_cd
        sin_off = b * stride_cb + t * stride_ct + col_lo * stride_cd
        cos_lo = tl.load(cos_ptr + cos_off, mask=mask_lo, other=0.0)
        sin_lo = tl.load(sin_ptr + sin_off, mask=mask_lo, other=0.0)

        # Rotate Q on this (b, h, t).
        if h < H_q:
            base_q = b * stride_qb + h * stride_qh + t * stride_qt
            q_lo = tl.load(q_ptr + base_q + col_lo * stride_qd, mask=mask_lo, other=0.0)
            q_hi = tl.load(q_ptr + base_q + (col_lo + half) * stride_qd, mask=mask_lo, other=0.0)
            new_lo = q_lo * cos_lo - q_hi * sin_lo
            new_hi = q_hi * cos_lo + q_lo * sin_lo
            tl.store(q_ptr + base_q + col_lo * stride_qd, new_lo, mask=mask_lo)
            tl.store(q_ptr + base_q + (col_lo + half) * stride_qd, new_hi, mask=mask_lo)

        # Same for K.
        if h < H_k:
            base_k = b * stride_kb + h * stride_kh + t * stride_kt
            k_lo = tl.load(k_ptr + base_k + col_lo * stride_kd, mask=mask_lo, other=0.0)
            k_hi = tl.load(k_ptr + base_k + (col_lo + half) * stride_kd, mask=mask_lo, other=0.0)
            new_klo = k_lo * cos_lo - k_hi * sin_lo
            new_khi = k_hi * cos_lo + k_lo * sin_lo
            tl.store(k_ptr + base_k + col_lo * stride_kd, new_klo, mask=mask_lo)
            tl.store(k_ptr + base_k + (col_lo + half) * stride_kd, new_khi, mask=mask_lo)


def apply_rope_triton(
    q: torch.Tensor,                     # [B, H_q, T, head_dim]
    k: torch.Tensor,                     # [B, H_k, T, head_dim]
    cos: torch.Tensor,                   # [B, T, rotary_dim]
    sin: torch.Tensor,                   # [B, T, rotary_dim]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """In-place fused RoPE on Q and K. Returns ``(q, k)``.

    Requires Q, K, cos, sin to all be CUDA tensors of the same dtype.
    Falls through to a torch reference on non-CUDA inputs (no exception
    — the caller can call this unconditionally).
    """
    if not _HAS_TRITON or not q.is_cuda:
        return _torch_reference(q, k, cos, sin)

    B, H_q, T, head_dim = q.shape
    H_k = k.shape[1]
    rotary_dim = cos.shape[-1]
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even; got {rotary_dim}")
    if rotary_dim > head_dim:
        raise ValueError(f"rotary_dim ({rotary_dim}) > head_dim ({head_dim})")
    if rotary_dim > 64:
        # The kernel's inline constant assumes ≤64. For larger rotary dims,
        # fall through to the torch reference rather than silently truncate.
        return _torch_reference(q, k, cos, sin)

    grid = (B * max(H_q, H_k), T)
    _rope_kernel[grid](
        q, k, cos, sin,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(0), cos.stride(1), cos.stride(2),
        B=B, H_q=H_q, H_k=H_k, T=T,
        head_dim=head_dim, rotary_dim=rotary_dim,
    )
    return q, k


def _torch_reference(q, k, cos, sin):
    """Reference implementation matching the kernel exactly. Used as the
    CPU/non-Triton fallback and as the equivalence target in tests.
    """
    rotary_dim = cos.shape[-1]
    cos = cos.unsqueeze(1)                                     # [B, 1, T, rotary_dim]
    sin = sin.unsqueeze(1)
    half = rotary_dim // 2

    def _rot(x):
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_lo = x_rot[..., :half]
        x_hi = x_rot[..., half:]
        new_lo = x_lo * cos[..., :half] - x_hi * sin[..., :half]
        new_hi = x_hi * cos[..., :half] + x_lo * sin[..., :half]
        return torch.cat([new_lo, new_hi, x_pass], dim=-1)

    q.copy_(_rot(q))
    k.copy_(_rot(k))
    return q, k
