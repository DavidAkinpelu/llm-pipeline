"""Qwen3.5 / Qwen3.6 decoder layer — token-mixer + FFN with pre-norm residuals.

Standard pre-norm transformer block. Each layer picks its token-mixer based
on ``config.layer_types[layer_idx]``:

- ``"linear_attention"`` → ``Qwen3_5GatedDeltaNet`` (3 of every 4 layers)
- ``"full_attention"``   → ``Qwen3_5Attention``    (1 of every 4 layers)

The FFN is either dense ``Qwen3_5MLP`` (Qwen3.6-27B) or ``Qwen3_5MoeBlock``
(Qwen3.6-35B-A3B). The block's input/post-attention LayerNorms and residual
structure are the same in both cases.

   residual = h
   h = input_norm(h)
   h = mixer(h)               ← linear or full attention
   h = residual + h
   residual = h
   h = post_norm(h)
   h = mlp(h)                 ← dense or MoE
   h = residual + h
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .attention import GatedAttentionConfig, Qwen3_5Attention, Qwen3_5RMSNorm
from .gated_deltanet import GatedDeltaNetConfig, Qwen3_5GatedDeltaNet
from .mlp import MLPConfig, MoEBlockConfig, Qwen3_5MLP, Qwen3_5MoeBlock


@dataclass
class DecoderLayerConfig:
    """All the bits a decoder layer needs to instantiate its sub-modules."""

    layer_type: str                                  # "linear_attention" | "full_attention"
    hidden_size: int
    rms_norm_eps: float = 1e-6
    # Attention sub-configs (mutually exclusive — only one is used per layer).
    full_attn: Optional[GatedAttentionConfig] = None
    linear_attn: Optional[GatedDeltaNetConfig] = None
    # FFN: dense MLP or MoE block.
    dense_mlp: Optional[MLPConfig] = None
    moe_block: Optional[MoEBlockConfig] = None

    def __post_init__(self) -> None:
        if self.layer_type not in {"linear_attention", "full_attention"}:
            raise ValueError(f"unknown layer_type: {self.layer_type!r}")
        if self.layer_type == "full_attention" and self.full_attn is None:
            raise ValueError("layer_type='full_attention' requires full_attn config")
        if self.layer_type == "linear_attention" and self.linear_attn is None:
            raise ValueError("layer_type='linear_attention' requires linear_attn config")
        if (self.dense_mlp is None) == (self.moe_block is None):
            raise ValueError("exactly one of dense_mlp / moe_block must be set")


class Qwen3_5DecoderLayer(nn.Module):
    """Single hybrid decoder block."""

    def __init__(self, config: DecoderLayerConfig):
        super().__init__()
        self.layer_type = config.layer_type
        self.hidden_size = config.hidden_size

        # Token mixer.
        if config.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config.linear_attn)
            self.self_attn = None
        else:
            self.self_attn = Qwen3_5Attention(config.full_attn)
            self.linear_attn = None

        # FFN: dense or MoE.
        if config.dense_mlp is not None:
            self.mlp = Qwen3_5MLP(config.dense_mlp)
        else:
            self.mlp = Qwen3_5MoeBlock(config.moe_block)

        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        linear_attn_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_caches: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[
            torch.Tensor,
            Optional[Tuple[torch.Tensor, torch.Tensor]],
            Optional[Tuple[torch.Tensor, torch.Tensor]],
        ],
    ]:
        """Forward pass.

        Cache plumbing differs by layer type:
          - **full-attention**: ``past_kv = (k, v)`` is the standard KV cache;
            ``linear_attn_cache`` is ignored.
          - **linear-attention** (Gated DeltaNet): ``linear_attn_cache =
            (recurrent_state, conv_state)`` carries both the recurrent matrix
            S and the trailing ``conv_kernel-1`` frames of the previous
            pre-conv input. ``past_kv`` is ignored.

        With ``return_caches=True`` the forward returns a 3-tuple
        ``(out, present_kv, present_linear_attn_cache)`` where the irrelevant
        slot is ``None``.
        """
        residual = hidden_states
        h = self.input_layernorm(hidden_states)

        present_kv = None
        present_linear = None

        if self.layer_type == "linear_attention":
            initial_rec, conv_state = (
                (None, None) if linear_attn_cache is None else linear_attn_cache
            )
            h, lin_cache = self.linear_attn(
                h,
                initial_state=initial_rec,
                conv_state=conv_state,
                return_final_state=return_caches,
            )
            present_linear = lin_cache
        else:
            if cos is None or sin is None:
                raise ValueError("full_attention layer requires cos and sin")
            h, present_kv = self.self_attn(
                h, cos=cos, sin=sin,
                attention_mask=attention_mask,
                past_kv=past_kv,
            )

        h = residual + h
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        h = residual + h

        if return_caches:
            return h, present_kv, present_linear
        return h
