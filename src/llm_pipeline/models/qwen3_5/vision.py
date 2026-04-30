"""Multimodal vision tower for Qwen3.5 / Qwen3.6.

Hand-rolled port of the Qwen3.5 ViT-style image / video encoder + the
mmproj patch merger that lifts vision hidden states into the LLM's
hidden space. Matches the reference implementation in HuggingFace's
``transformers/models/qwen3_5/modeling_qwen3_5.py``.

Architecture
------------

Each image (or video frame group) is processed as:

1. **Patch embedding**: a Conv3d with kernel/stride
   ``(temporal_patch_size, patch_size, patch_size)`` chunks the input
   ``(T, H, W)`` into patches and projects each to ``hidden_size``. For
   pure images, ``temporal_patch_size`` is typically 2 and frames are
   padded/repeated to a multiple — the temporal axis just acts as
   another spatial dimension.

2. **Positional embeddings**: a learned ``num_position_embeddings × hidden``
   table covering an N×N grid (N=√num_position_embeddings); for arbitrary
   image resolutions the patch positions are mapped onto this grid via
   bilinear interpolation (see ``fast_pos_embed_interpolate``).

3. **Vision RoPE**: 2D rotary embeddings applied inside the attention
   heads of every block, computed once from the (T, H, W) grid.

4. **Transformer stack**: ``depth`` (=27 in Qwen3.6) pre-norm blocks
   with non-causal self-attention + GELU MLP. Multiple images per batch
   are concatenated along the patch axis with a ``cu_seqlens`` segmentation
   so attention only mixes within a single image.

5. **Patch merger / mmproj**: groups ``spatial_merge_size × spatial_merge_size``
   adjacent patches, LayerNorms, runs them through a 2-layer MLP that
   projects from ``hidden·spatial_merge_size²`` down to ``out_hidden_size``
   (the LLM's hidden size). The output is the per-vision-token embedding
   sequence consumed by the LLM.

6. **Embedding interleave**: ``replace_placeholder_embeddings`` replaces
   the special-token positions in the LLM's input embedding sequence
   with the vision embeddings produced above.

Status
------

This module is the **pure-PyTorch reference**, suitable for forward-pass
inference, training, and as a teaching artefact. The released Qwen3.6
checkpoints carry the trained vision weights under ``model.visual.*``;
loading them needs the matching name-mapping pass on top of our existing
HF state-dict loader (a small follow-up — most names are direct).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisionConfig:
    """Subset of HF ``Qwen3_5VisionConfig`` fields the tower needs.

    Defaults match the released Qwen3.6 vision tower.
    """

    hidden_size: int = 1152
    out_hidden_size: int = 2048              # projected LLM hidden (varies per release)
    intermediate_size: int = 4304
    depth: int = 27
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16                     # spatial patch
    temporal_patch_size: int = 2             # frame-group patch (1 for true single-image)
    spatial_merge_size: int = 2              # final downsample factor
    num_position_embeddings: int = 2304      # 48×48 absolute pos table
    rope_theta: float = 10_000.0
    layer_norm_eps: float = 1e-6


# --------------------------------------------------------------------------- #
# Patch embedding
# --------------------------------------------------------------------------- #


class Qwen3_5VisionPatchEmbed(nn.Module):
    """Conv3d patch embedder. Input ``(N_patches, C, T_p, P, P)`` → ``(N, hidden)``.

    The caller is responsible for slicing the raw ``(T, H, W)`` video tensor
    into per-patch sub-volumes; this module just runs the Conv3d that maps
    each one to a hidden vector.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim,
            kernel_size=kernel, stride=kernel, bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # ``hidden_states`` is ``[N, C, T_p, P, P]`` (already chunked).
        h = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size,
        )
        return self.proj(h.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


# --------------------------------------------------------------------------- #
# Vision RoPE
# --------------------------------------------------------------------------- #


class Qwen3_5VisionRotaryEmbedding(nn.Module):
    """1-D RoPE base for vision; the 2-D embedding is built by composing
    independent rotaries on H and W indices.
    """

    def __init__(self, dim: int, theta: float = 10_000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)        # [seqlen, dim/2]


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to vision Q/K. ``q`` and ``k`` shape ``[N, H, head_dim]``,
    ``cos`` / ``sin`` shape ``[N, head_dim]``.
    """
    cos = cos.unsqueeze(-2)                            # [N, 1, head_dim]
    sin = sin.unsqueeze(-2)
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    rotate_q = torch.cat([-q2, q1], dim=-1)
    rotate_k = torch.cat([-k2, k1], dim=-1)
    return q * cos + rotate_q * sin, k * cos + rotate_k * sin


# --------------------------------------------------------------------------- #
# MLP, Attention, Block
# --------------------------------------------------------------------------- #


class Qwen3_5VisionMLP(nn.Module):
    """Two-layer GELU MLP. Vision uses ``gelu_pytorch_tanh``."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act = lambda x: F.gelu(x, approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Qwen3_5VisionAttention(nn.Module):
    """Non-causal multi-head attention with optional ``cu_seqlens`` segmentation
    so attention doesn't leak across images concatenated in one batch.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        if self.head_dim * self.num_heads != self.dim:
            raise ValueError(
                f"hidden_size ({self.dim}) must be divisible by num_heads ({self.num_heads})"
            )
        self.scaling = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,                   # [N, hidden]
        cos: torch.Tensor,                              # [N, head_dim]
        sin: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,      # [n_images + 1]
    ) -> torch.Tensor:
        N = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)    # each [N, num_heads, head_dim]
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # SDPA wants ``[B, H, N, D]``.
        if cu_seqlens is None or cu_seqlens.numel() <= 2:
            # Single-image fast path.
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scaling)
            out = out.squeeze(0).transpose(0, 1).reshape(N, self.dim)
        else:
            # Multi-image: split per ``cu_seqlens`` segment, run SDPA per chunk, concat.
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            outs = []
            offset = 0
            for L in lengths:
                qi = q[offset:offset + L].transpose(0, 1).unsqueeze(0)     # [1, H, L, D]
                ki = k[offset:offset + L].transpose(0, 1).unsqueeze(0)
                vi = v[offset:offset + L].transpose(0, 1).unsqueeze(0)
                oi = F.scaled_dot_product_attention(
                    qi, ki, vi, is_causal=False, scale=self.scaling,
                )
                outs.append(oi.squeeze(0).transpose(0, 1).reshape(L, self.dim))
                offset += L
            out = torch.cat(outs, dim=0)

        return self.proj(out)


class Qwen3_5VisionBlock(nn.Module):
    """Pre-norm transformer block: norm → attn → +residual → norm → MLP → +residual."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = Qwen3_5VisionAttention(config)
        self.mlp = Qwen3_5VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = hidden_states + self.attn(self.norm1(hidden_states), cos, sin, cu_seqlens)
        h = h + self.mlp(self.norm2(h))
        return h


# --------------------------------------------------------------------------- #
# Patch merger (a.k.a. mmproj)
# --------------------------------------------------------------------------- #


class Qwen3_5VisionPatchMerger(nn.Module):
    """Spatial 2×2 merger + 2-layer projection to LLM hidden size.

    Halves the spatial resolution by concatenating ``spatial_merge_size²``
    adjacent patches along the channel axis, then runs a small MLP that
    lifts the result from ``hidden·spatial_merge_size²`` to
    ``out_hidden_size`` (the LLM's hidden size).
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.merged_dim = config.hidden_size * (config.spatial_merge_size ** 2)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(self.merged_dim, self.merged_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.merged_dim, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` is ``[N, hidden]`` where the patch order is already
        # spatial-merge-grouped (handled in the model's forward).
        x = self.norm(x).view(-1, self.merged_dim)
        return self.fc2(self.act(self.fc1(x)))


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #


class Qwen3_5VisionModel(nn.Module):
    """End-to-end vision encoder.

    Inputs:
      - ``pixel_values``: ``[N_patches, C, T_p, P, P]`` — raw patch tensor.
        Build this by chunking the raw video/image with stride
        ``(T_p, P, P)`` and flattening the patch grid.
      - ``grid_thw``: ``[n_images, 3]`` long tensor giving each image's
        ``(t, h, w)`` patch grid sizes (before spatial merge). The product
        ``t·h·w`` per row must sum to ``N_patches``.

    Output: ``[N_patches / spatial_merge_size², out_hidden_size]`` — the
    per-vision-token embeddings ready to be inserted into the LLM input
    sequence by ``replace_placeholder_embeddings``.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size

        self.patch_embed = Qwen3_5VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        # √num_position_embeddings = side length of the absolute pos grid.
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        if self.num_grid_per_side ** 2 != config.num_position_embeddings:
            raise ValueError(
                f"num_position_embeddings ({config.num_position_embeddings}) "
                "must be a perfect square"
            )

        head_dim = config.hidden_size // config.num_heads
        # Vision RoPE applies to half the head_dim (the H/W components share).
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2, theta=config.rope_theta)

        self.blocks = nn.ModuleList(
            [Qwen3_5VisionBlock(config) for _ in range(config.depth)]
        )
        self.merger = Qwen3_5VisionPatchMerger(config)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Bilinear-interpolate the learned position table onto each image's
        actual ``(h, w)`` patch grid.

        Output: ``[total_patches_after_merge_permute, hidden]``, ordered to
        match the merger's expected input.
        """
        N = self.num_grid_per_side
        device = self.pos_embed.weight.device

        per_image_emb: List[torch.Tensor] = []
        for t, h, w in grid_thw.tolist():
            # Map each output (h_idx, w_idx) ∈ [0, N-1] back into the grid.
            h_idxs = torch.linspace(0, N - 1, h, device=device)
            w_idxs = torch.linspace(0, N - 1, w, device=device)
            h_floor = h_idxs.floor().long()
            w_floor = w_idxs.floor().long()
            h_ceil = (h_floor + 1).clamp_max(N - 1)
            w_ceil = (w_floor + 1).clamp_max(N - 1)
            dh = (h_idxs - h_floor.float())[:, None]
            dw = (w_idxs - w_floor.float())[None, :]
            base_floor = h_floor[:, None] * N
            base_ceil = h_ceil[:, None] * N

            indices = torch.stack([
                (base_floor + w_floor[None, :]).flatten(),
                (base_floor + w_ceil[None, :]).flatten(),
                (base_ceil + w_floor[None, :]).flatten(),
                (base_ceil + w_ceil[None, :]).flatten(),
            ], dim=0)                                                   # [4, h*w]
            weights = torch.stack([
                ((1 - dh) * (1 - dw)).flatten(),
                ((1 - dh) * dw).flatten(),
                (dh * (1 - dw)).flatten(),
                (dh * dw).flatten(),
            ], dim=0)                                                   # [4, h*w]

            embs = self.pos_embed(indices) * weights[..., None]         # [4, h*w, hidden]
            patch_embs = embs.sum(dim=0)                                 # [h*w, hidden]
            patch_embs = patch_embs.repeat(t, 1)                         # [t*h*w, hidden]

            # Permute into spatial-merge groups so consecutive tokens come
            # from the same merge tile.
            m = self.spatial_merge_size
            patch_embs = (
                patch_embs.view(t, h // m, m, w // m, m, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            per_image_emb.append(patch_embs)

        return torch.cat(per_image_emb, dim=0)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Build per-patch H/W rotary positions, ordered to match
        ``fast_pos_embed_interpolate``.
        """
        m = self.spatial_merge_size
        all_pos: List[torch.Tensor] = []
        for t, h, w in grid_thw.tolist():
            h_pos = torch.arange(h, device=self.pos_embed.weight.device)
            w_pos = torch.arange(w, device=self.pos_embed.weight.device)
            # Per-patch (h, w) indices, post-merge permutation.
            hw = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1)
            hw = hw.view(h // m, m, w // m, m, 2).permute(0, 2, 1, 3, 4).reshape(-1, 2)
            hw = hw.repeat(t, 1)                                        # repeat across temporal patches
            all_pos.append(hw)
        all_pos = torch.cat(all_pos, dim=0)                              # [N_patches, 2]

        # Compose H and W rotary embeddings into a (head_dim/2)-vector each
        # that the attention then interleaves into a head_dim rotation.
        seqlen = all_pos.max().item() + 1
        rope_h = self.rotary_pos_emb(seqlen)                             # [seqlen, head_dim/4]
        rope_w = self.rotary_pos_emb(seqlen)
        return torch.cat([rope_h[all_pos[:, 0]], rope_w[all_pos[:, 1]]], dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,                                       # [N, C, T_p, P, P]
        grid_thw: torch.Tensor,                                          # [n_images, 3]
    ) -> torch.Tensor:
        h = self.patch_embed(pixel_values)                                # [N, hidden]

        # Add interpolated absolute positional embeddings.
        h = h + self.fast_pos_embed_interpolate(grid_thw)

        # Build the rotary cos/sin streams.
        rotary = self.rot_pos_emb(grid_thw)                               # [N, head_dim/2]
        emb = torch.cat([rotary, rotary], dim=-1)                         # [N, head_dim]
        cos, sin = emb.cos(), emb.sin()

        # cu_seqlens for non-leaking cross-image attention.
        per_image_lens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).to(torch.int32)
        cu_seqlens = F.pad(per_image_lens.cumsum(dim=0), (1, 0), value=0)

        for block in self.blocks:
            h = block(h, cos, sin, cu_seqlens=cu_seqlens)

        # Merge spatial 2×2 groups + project to LLM hidden.
        return self.merger(h)


# --------------------------------------------------------------------------- #
# Multimodal embedding interleave
# --------------------------------------------------------------------------- #


def replace_placeholder_embeddings(
    inputs_embeds: torch.Tensor,             # [B, T, H_llm]
    input_ids: torch.LongTensor,              # [B, T]
    vision_embeds: torch.Tensor,              # [N_vision_tokens, H_llm]
    placeholder_token_id: int,
) -> torch.Tensor:
    """Replace every position in ``input_ids`` equal to ``placeholder_token_id``
    with the next available vision embedding from ``vision_embeds``.

    The total count of placeholders across the batch must equal
    ``vision_embeds.shape[0]``; otherwise we raise (catches off-by-one
    errors in the prompt formatter).

    Returns a new ``inputs_embeds`` with the same shape, with the
    placeholder positions overwritten and all other positions untouched.
    """
    if vision_embeds.shape[-1] != inputs_embeds.shape[-1]:
        raise ValueError(
            f"vision_embeds last dim {vision_embeds.shape[-1]} != "
            f"inputs_embeds last dim {inputs_embeds.shape[-1]}"
        )
    mask = input_ids == placeholder_token_id              # [B, T]
    n_slots = mask.sum().item()
    if n_slots != vision_embeds.shape[0]:
        raise ValueError(
            f"placeholder count ({n_slots}) != vision_embeds count "
            f"({vision_embeds.shape[0]})"
        )
    out = inputs_embeds.clone()
    # ``masked_scatter`` would work on contiguous tensors of matching shape;
    # we use the explicit boolean indexing for clarity and to keep the math
    # obvious.
    flat_mask = mask.reshape(-1)
    out_flat = out.reshape(-1, out.shape[-1])
    out_flat[flat_mask] = vision_embeds.to(out_flat.dtype)
    return out_flat.reshape(out.shape)
