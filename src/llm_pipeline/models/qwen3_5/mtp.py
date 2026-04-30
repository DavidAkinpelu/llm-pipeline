"""Multi-Token Prediction (MTP) head for Qwen3.5 / Qwen3.6.

Background
----------

The Qwen3.5 config carries ``mtp_num_hidden_layers=1`` and
``mtp_use_dedicated_embeddings=False`` — fields that describe the
DeepSeek-V3-style MTP head used during pretraining: an extra small
transformer layer that predicts the **next-next** token (``x_{t+2}``)
from the main model's hidden state ``h_t`` and the embedding of
``x_{t+1}``.

Used at inference time, MTP supplies a "free" draft prediction one step
ahead of the main model — the foundation for the speculative decoding
helper in ``llm_pipeline/inference/speculative.py``. Used during training,
the MTP loss is summed with the main NTP loss to give the model an
auxiliary signal for longer-horizon prediction.

The released text-only HF Qwen3.6 checkpoints **don't carry** the trained
MTP weights — the public release strips them. So this module's weights
start randomly initialised, and using it for inference acceleration only
makes sense after fine-tuning. For pure architectural completeness +
training-time use, the structure here matches the published spec.

Algorithm (DeepSeek-V3 §2.3, "Multi-Token Prediction")
-------------------------------------------------------

For each token ``t`` and MTP depth ``k`` (we have a single depth k=1
matching ``mtp_num_hidden_layers=1``):

.. code-block:: text

    h_t            = main_model(x_{≤t})[t]               # main hidden state
    e_{t+1}        = embed_tokens(x_{t+1})                # next-token embedding
    h'_t           = M_k @ concat(RMSNorm(h_t), RMSNorm(e_{t+1}))
    h''_t          = MTPDecoderLayer_k(h'_t)              # single transformer layer
    logit^MTP_t    = lm_head(RMSNorm(h''_t))              # shared with main lm_head
    L^MTP          = CrossEntropy(logit^MTP_t, x_{t+2})

The dimensions: ``M_k`` is a ``[hidden, 2*hidden]`` projection. The MTP
layer reuses the same hybrid attention machinery as the main model
(linear or full attention based on the layer index — DeepSeek-V3 and
Qwen3.5 both use full attention for their MTP layer).

Inference / speculative use
---------------------------

Given the main model has just produced token ``x_{t+1}`` and its hidden
state ``h_t``, the MTP head produces a *draft* prediction for
``x_{t+2}``. If the main model accepts (verifies) it, you've decoded
two tokens in one forward step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .attention import GatedAttentionConfig, Qwen3_5Attention, Qwen3_5RMSNorm
from .rotary import Qwen3_5RotaryEmbedding, RotaryConfig


@dataclass
class MTPConfig:
    """Subset of ``Qwen3_5Config`` fields the MTP head uses."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple = (11, 11, 10)
    mrope_interleaved: bool = True
    max_position_embeddings: int = 262_144


class Qwen3_5MTPHead(nn.Module):
    """One DeepSeek-V3-style MTP layer.

    Shares ``embed_tokens`` and ``lm_head`` with the main model; both must
    be passed in at construction time so the parameter sharing is explicit
    (no implicit global lookup).

    Forward returns the MTP logits at every position; downstream code
    handles the offset (the t-th MTP logit predicts ``x_{t+2}``).
    """

    def __init__(
        self,
        config: MTPConfig,
        embed_tokens: nn.Embedding,
        lm_head: nn.Linear,
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens                # shared
        self.lm_head = lm_head                          # shared

        # Two RMSNorms applied separately to the main hidden and the next-token
        # embedding before concatenation — DeepSeek-V3 §2.3.
        self.norm_h = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_e = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Projection from concatenated 2H → H.
        self.proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        # Single full-attention transformer layer. Qwen3.5/DeepSeek-V3 both
        # use full attention here (not the linear-attention variant) — the
        # short MTP horizon doesn't need the linear-attention compression.
        attn_cfg = GatedAttentionConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3_5Attention(attn_cfg)

        # Output norm applied before the (shared) lm_head.
        self.norm_out = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MTP runs the rotary inside its single transformer layer, sharing
        # the config with the main model so positions align.
        self.rotary = Qwen3_5RotaryEmbedding(RotaryConfig(
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            partial_rotary_factor=config.partial_rotary_factor,
            mrope_section=config.mrope_section,
            mrope_interleaved=config.mrope_interleaved,
            max_position_embeddings=config.max_position_embeddings,
        ))

    def forward(
        self,
        main_hidden: torch.Tensor,                      # [B, T, H] from main model
        next_token_ids: torch.LongTensor,                # [B, T] — input_ids shifted by 1
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict logits for the **next-next** token at every position.

        ``main_hidden[:, t]`` is the main model's output hidden at position t.
        ``next_token_ids[:, t]`` is ``x_{t+1}`` (the token that comes after
        position t in the input). The returned logits at position t are the
        MTP prediction for ``x_{t+2}``.
        """
        B, T, _ = main_hidden.shape

        # Concat-project: 2H → H.
        h_norm = self.norm_h(main_hidden)
        e_norm = self.norm_e(self.embed_tokens(next_token_ids))
        h = self.proj(torch.cat([h_norm, e_norm], dim=-1))           # [B, T, H]

        # Single transformer layer with pre-norm residual.
        if position_ids is None:
            position_ids = torch.arange(T, device=h.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary(h, position_ids)

        residual = h
        h = self.input_layernorm(h)
        h, _ = self.self_attn(h, cos=cos, sin=sin, attention_mask=attention_mask)
        h = residual + h

        # No FFN sub-block in DeepSeek-V3's MTP; output norm + shared lm_head.
        h = self.norm_out(h)
        return self.lm_head(h)
