"""Rotary embeddings for Qwen3.5 / Qwen3.6 — partial RoPE + multimodal RoPE.

Two non-standard wrinkles vs vanilla Llama-style RoPE:

1. **Partial RoPE** (``partial_rotary_factor=0.25``): only the first 25% of
   each head_dim is rotated; the rest passes through unchanged. With
   ``head_dim=256`` this leaves 192 dims un-rotated, which the model uses
   as content channels not tied to position. The same idea was popularized
   by GPT-J and PaLM.

2. **Multimodal RoPE / mRoPE**: when text + image tokens are mixed, we have
   three position grids (text index ``T``, spatial-x ``H``, spatial-y ``W``).
   Each gets its own RoPE band; the bands are interleaved across the
   rotary_dim/2 frequency slots according to ``mrope_section=[11, 11, 10]``:

       slot 0: T  slot 1: H  slot 2: W
       slot 3: T  slot 4: H  slot 5: W
       ...

   For text-only inference the three grids collapse to the same indices,
   so mRoPE is bit-equivalent to vanilla RoPE — but the layout machinery
   has to be there for multimodal to work later.

Reference: ``Qwen3_5TextRotaryEmbedding`` and ``apply_rotary_pos_emb`` in
``transformers/models/qwen3_5/modeling_qwen3_5.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last-dim half-pairs: ``[a, b] → [-b, a]``."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to only the first ``rotary_dim = cos.shape[-1]`` channels of
    each head, leaving the trailing channels untouched.

    ``cos`` / ``sin`` are ``[..., T, rotary_dim]`` (matching the partial dim);
    ``q`` / ``k`` are ``[B, H, T, head_dim]`` with ``head_dim ≥ rotary_dim``.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


def apply_interleaved_mrope(
    freqs: torch.Tensor,
    mrope_section: List[int],
) -> torch.Tensor:
    """Interleave three RoPE bands (T, H, W) into a single per-pair frequency
    layout matching the HF Qwen3.5 reference.

    Input ``freqs`` shape: ``(3, B, T, rotary_dim/2)`` — the three position
    grids' frequency products. Output: ``(B, T, rotary_dim/2)`` with the
    pattern ``T, H, W, T, H, W, ...`` (truncated by the section sizes).

    With ``mrope_section=[11, 11, 10]``:
      - 11 slots get T values  (positions 0, 3, 6, ..., 30)
      - 11 slots get H values  (positions 1, 4, 7, ..., 31)
      - 10 slots get W values  (positions 2, 5, 8, ..., 29)

    Total: 11 + 11 + 10 = 32 = rotary_dim/2 = 64/2.

    For text-only inference, freqs[0] == freqs[1] == freqs[2], so this
    function is a no-op (returns freqs[0]) — but the interleave layout has
    to match the trained weights regardless.
    """
    out = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):           # H (dim=1), W (dim=2)
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        out[..., idx] = freqs[dim, ..., idx]
    return out


@dataclass
class RotaryConfig:
    """Subset of ``Qwen3_5Config`` fields the rotary module needs."""

    head_dim: int
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: Tuple[int, int, int] = (11, 11, 10)
    mrope_interleaved: bool = True
    max_position_embeddings: int = 262_144


class Qwen3_5RotaryEmbedding(nn.Module):
    """Generates ``(cos, sin)`` for partial-rotary, mRoPE-aware RoPE.

    Forward signature accepts either:
      - ``position_ids`` shape ``[B, T]`` — text-only path. Internally
        broadcast to the 3 grids; mRoPE collapses to vanilla RoPE.
      - ``position_ids`` shape ``[3, B, T]`` — multimodal. Each grid carries
        its own indices (text-step / spatial-x / spatial-y).

    Output ``cos`` / ``sin`` shape: ``[B, T, rotary_dim]`` where
    ``rotary_dim = head_dim * partial_rotary_factor`` (rounded to even).
    """

    def __init__(self, config: RotaryConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rotary_dim = (int(config.head_dim * config.partial_rotary_factor) // 2) * 2
        self.theta = config.rope_theta
        self.mrope_section = list(config.mrope_section)
        self.mrope_interleaved = config.mrope_interleaved

        if sum(self.mrope_section) != self.rotary_dim // 2:
            raise ValueError(
                f"mrope_section {self.mrope_section} sums to {sum(self.mrope_section)} "
                f"but expected {self.rotary_dim // 2} (rotary_dim/2)"
            )

        # Inverse-frequency table: rotary_dim/2 frequencies, log-spaced by theta.
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,                         # only used for dtype / device
        position_ids: torch.Tensor,              # [B, T] or [3, B, T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, -1, -1)
        elif position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(
                f"position_ids must be [B, T] or [3, B, T]; got {tuple(position_ids.shape)}"
            )

        # ``[3, 1, rotary_dim/2, 1] @ [3, B, 1, T] → [3, B, rotary_dim/2, T]``
        # then transpose last two for the layout the HF code uses.
        inv_exp = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        pos_exp = position_ids[:, :, None, :].float()
        freqs = (inv_exp @ pos_exp).transpose(2, 3)          # [3, B, T, rotary_dim/2]

        if self.mrope_interleaved:
            freqs = apply_interleaved_mrope(freqs, self.mrope_section)
        else:
            # Concatenated layout: [T-band, H-band, W-band] left-to-right.
            freqs = torch.cat(
                [freqs[i, ..., : self.mrope_section[i]] for i in range(3)],
                dim=-1,
            )

        emb = torch.cat([freqs, freqs], dim=-1)              # [B, T, rotary_dim]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)
