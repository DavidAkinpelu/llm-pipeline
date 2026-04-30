"""Output-gated full attention for Qwen3.5 / Qwen3.6.

The "full-attention" (vs Gated DeltaNet) layer in the hybrid stack. Every
4th layer in the released Qwen3.6 models is one of these. Departures from
vanilla Llama-style GQA attention:

1. **Output gate**: ``q_proj`` outputs ``2 × head_dim`` per head; the second
   half is a per-head, per-channel gate that multiplies the attention output
   element-wise (after sigmoid) before ``o_proj``. Same trick as Mixtral /
   DeepSeek-V3 — it lets each head modulate its own contribution.

2. **Q/K RMSNorm before RoPE**: ``q_norm`` and ``k_norm`` (per-head RMSNorm
   over ``head_dim``) are applied to the queries and keys before rotary
   embedding. This stabilises training; it's also what Qwen3 (vanilla)
   does.

3. **Partial RoPE + mRoPE**: rotary embeddings cover only the first 25%
   of ``head_dim``; the rest is left position-agnostic. The rotary slots
   are interleaved across three sub-bands (text / spatial-x / spatial-y)
   for multimodal training.

4. **GQA**: ``num_attention_heads`` queries share ``num_key_value_heads``
   KV heads (24 → 4 on Qwen3.6-27B; 16 → 2 on Qwen3.6-35B-A3B-MoE).

5. **Custom RMSNorm normalisation**: weight is initialised to zeros and
   the forward applies ``output * (1 + weight)`` — so the norm starts at
   identity and the weight learns a residual scale, not a multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotary import apply_partial_rotary_pos_emb


class Qwen3_5RMSNorm(nn.Module):
    """Qwen3.5/3.6 RMSNorm — zero-init weight + ``(1 + weight)`` scaling.

    This is the same RMSNorm Qwen3 uses; the zero-init means the layer
    starts as a pure normalisation (identity-shaped output) and the
    weight learns a residual gain.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.to(x.dtype)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (GQA)."""
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x.unsqueeze(2).expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


@dataclass
class GatedAttentionConfig:
    """Subset of ``Qwen3_5Config`` fields the gated full-attention layer uses."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    attention_bias: bool = False


class Qwen3_5Attention(nn.Module):
    """Output-gated multi-head attention with GQA + partial-RoPE-aware Q/K.

    Forward expects ``cos`` / ``sin`` produced by
    ``Qwen3_5RotaryEmbedding`` (shape ``[B, T, rotary_dim]``). The rotary
    application crops to the first ``rotary_dim`` channels of each head and
    leaves the rest unchanged.

    Causal masking: when ``attention_mask`` is None we apply a strict
    upper-triangular causal mask. Pass an explicit mask to override
    (e.g. for prefix LM, document-aware attention).
    """

    def __init__(self, config: GatedAttentionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attn_dropout = config.attention_dropout
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_q_heads}) must be a multiple of "
                f"num_key_value_heads ({self.num_kv_heads}) for GQA"
            )
        self.kv_repeat = self.num_q_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        # q_proj outputs Q + gate, packed two-per-head along head_dim.
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_q_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_q_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,                      # [B, T, H_in]
        cos: torch.Tensor,                                 # [B, T, rotary_dim]
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = hidden_states.shape

        # --- 1. Q + gate from one fused projection ---
        q_and_gate = self.q_proj(hidden_states).view(B, T, self.num_q_heads, self.head_dim * 2)
        query, gate = torch.chunk(q_and_gate, 2, dim=-1)            # each [B, T, num_q_heads, head_dim]
        gate = gate.reshape(B, T, self.num_q_heads * self.head_dim)  # to match attn_output flat shape

        # --- 2. Q/K/V projections + per-head RMSNorm on Q, K (before RoPE) ---
        query = self.q_norm(query).transpose(1, 2)                  # [B, H_q, T, head_dim]
        key = self.k_norm(
            self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        ).transpose(1, 2)                                           # [B, H_kv, T, head_dim]
        value = (
            self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        )                                                           # [B, H_kv, T, head_dim]

        # --- 3. Partial RoPE on the first rotary_dim channels of each head ---
        query, key = apply_partial_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)

        # --- 4. KV cache append (for autoregressive decode) ---
        if past_kv is not None:
            past_k, past_v = past_kv
            key = torch.cat([past_k, key], dim=2)
            value = torch.cat([past_v, value], dim=2)
        present_kv = (key, value)

        # --- 5. GQA repeat ---
        key = _repeat_kv(key, self.kv_repeat)                       # [B, H_q, T_full, head_dim]
        value = _repeat_kv(value, self.kv_repeat)

        # --- 6. SDPA (causal by default, override via attention_mask) ---
        if attention_mask is None and past_kv is None:
            # Pure prefill: standard causal mask.
            attn = F.scaled_dot_product_attention(
                query, key, value,
                is_causal=True,
                dropout_p=self.attn_dropout if self.training else 0.0,
                scale=self.scale,
            )
        else:
            attn_mask = attention_mask
            if attn_mask is None and past_kv is not None:
                # For cached chunked decode, each query position can only see
                # the prefix plus tokens up to its own offset inside the chunk.
                past_len = past_kv[0].shape[2]
                kv_len = key.shape[2]
                q_pos = past_len + torch.arange(T, device=query.device).view(T, 1)
                k_pos = torch.arange(kv_len, device=query.device).view(1, kv_len)
                attn_mask = torch.zeros((T, kv_len), device=query.device, dtype=query.dtype)
                attn_mask.masked_fill_(k_pos > q_pos, torch.finfo(query.dtype).min)
            # Either explicit mask provided, or we have a past_kv (decode step):
            # SDPA's is_causal flag doesn't generalise to the rectangular case,
            # so build the mask ourselves.
            attn = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                scale=self.scale,
                is_causal=False,
            )

        # --- 7. Output gate + o_proj ---
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.num_q_heads * self.head_dim)
        attn = attn * torch.sigmoid(gate)
        return self.o_proj(attn), present_kv
