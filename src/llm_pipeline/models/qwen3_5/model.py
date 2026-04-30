"""Full Qwen3.5 / Qwen3.6 model — embedding + decoder stack + LM head.

Assembly note
-------------

This is the ``Qwen3_5TextModel`` + ``Qwen3_5ForCausalLM`` reference, in
pure PyTorch with no HF runtime, autocast wrappers, or generation mixin.
The forward signature is intentionally simple — designed for educational
clarity and direct comparison against the HF source. Use the existing
generic transformers wrapper if you need full HF-style interoperability,
KV cache classes, or beam search.

What the forward does
---------------------

1. Embed token IDs.
2. Compute (cos, sin) once for the whole sequence via ``Qwen3_5RotaryEmbedding``.
3. Walk the layer stack. Full-attention layers consume (cos, sin) and a
   per-layer KV cache; linear-attention layers consume a per-layer recurrent
   state. Caches are passed through as a list-of-tuples, indexed by layer.
4. Final RMSNorm + ``lm_head`` projection to vocab.
5. ``tie_word_embeddings``: if True, ``lm_head.weight`` is bound to
   ``embed_tokens.weight`` (saves ~vocab_size·hidden_size weights, used by
   the smaller dense releases).

Multi-GPU / FSDP, gradient checkpointing, and full-rank loss (vs sliced
``logits_to_keep``) are all left to the trainer — this module is the
math, nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .attention import GatedAttentionConfig, Qwen3_5RMSNorm
from .config import Qwen3_5Config, Qwen3_5_MoE_Config
from .decoder_layer import DecoderLayerConfig, Qwen3_5DecoderLayer
from .gated_deltanet import GatedDeltaNetConfig
from .mlp import MLPConfig, MoEBlockConfig
from .rotary import Qwen3_5RotaryEmbedding, RotaryConfig


def _build_decoder_layer_configs(cfg: Qwen3_5Config) -> List[DecoderLayerConfig]:
    """Materialise per-layer configs from the top-level Qwen3_5Config.

    The two sub-attention configs are shared across all layers of their kind
    (HF builds them inline per layer; we hoist construction here for clarity
    and to make the layer-loop trivially testable).
    """
    full_attn_cfg = GatedAttentionConfig(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rms_norm_eps=cfg.rms_norm_eps,
        attention_dropout=cfg.attention_dropout,
        attention_bias=cfg.attention_bias,
    )
    linear_attn_cfg = GatedDeltaNetConfig(
        hidden_size=cfg.hidden_size,
        linear_num_key_heads=cfg.linear_num_key_heads,
        linear_num_value_heads=cfg.linear_num_value_heads,
        linear_key_head_dim=cfg.linear_key_head_dim,
        linear_value_head_dim=cfg.linear_value_head_dim,
        linear_conv_kernel_dim=cfg.linear_conv_kernel_dim,
        rms_norm_eps=cfg.rms_norm_eps,
        hidden_act=cfg.hidden_act,
    )

    if isinstance(cfg, Qwen3_5_MoE_Config):
        moe_cfg = MoEBlockConfig(
            hidden_size=cfg.hidden_size,
            num_experts=cfg.num_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            moe_intermediate_size=cfg.moe_intermediate_size,
            shared_expert_intermediate_size=cfg.shared_expert_intermediate_size,
            norm_topk_prob=cfg.norm_topk_prob,
            hidden_act=cfg.hidden_act,
        )
        dense_cfg = None
    else:
        dense_cfg = MLPConfig(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            hidden_act=cfg.hidden_act,
        )
        moe_cfg = None

    layers = []
    for lt in cfg.layer_types:
        layers.append(DecoderLayerConfig(
            layer_type=lt,
            hidden_size=cfg.hidden_size,
            rms_norm_eps=cfg.rms_norm_eps,
            full_attn=full_attn_cfg,
            linear_attn=linear_attn_cfg,
            dense_mlp=dense_cfg,
            moe_block=moe_cfg,
        ))
    return layers


def _rotary_config(cfg: Qwen3_5Config) -> RotaryConfig:
    return RotaryConfig(
        head_dim=cfg.head_dim,
        rope_theta=cfg.rope_theta,
        partial_rotary_factor=cfg.partial_rotary_factor,
        mrope_section=cfg.mrope_section,
        mrope_interleaved=cfg.mrope_interleaved,
        max_position_embeddings=cfg.max_position_embeddings,
    )


@dataclass
class Qwen3_5Cache:
    """Per-layer caches for the hybrid stack.

    Each layer's slot is a 2-tuple whose meaning depends on the layer type:
      - ``full_attention``  → ``(past_k, past_v)``, each ``[B, H_kv, T_past, head_dim]``
      - ``linear_attention`` → ``(recurrent_state, conv_state)``,
        ``[B, num_v_heads, D_k, D_v]`` and ``[B, conv_dim, conv_kernel - 1]``

    Both legs of each tuple may individually be ``None`` on the first call.
    """

    layers: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]

    @classmethod
    def empty(cls, num_layers: int) -> "Qwen3_5Cache":
        return cls(layers=[None] * num_layers)

    def get_seq_length(self, full_attn_layer_idx: int) -> int:
        """Length of the cached prefix for a full-attention layer (0 if empty)."""
        slot = self.layers[full_attn_layer_idx]
        if slot is None:
            return 0
        return slot[0].shape[2]


class Qwen3_5Model(nn.Module):
    """Embed → decoder stack → final norm. No LM head."""

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.rotary_emb = Qwen3_5RotaryEmbedding(_rotary_config(config))

        layer_cfgs = _build_decoder_layer_configs(config)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(lc) for lc in layer_cfgs])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Qwen3_5Cache] = None,
        return_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Qwen3_5Cache]]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        B, T, _ = inputs_embeds.shape
        # Default position_ids: arange offset by any cached prefix length.
        if position_ids is None:
            past_len = 0
            if cache is not None:
                # Pick any full-attention layer to read past length from.
                for i, lt in enumerate(self.config.layer_types):
                    if lt == "full_attention" and cache.layers[i] is not None:
                        past_len = cache.get_seq_length(i)
                        break
            position_ids = (
                torch.arange(T, device=inputs_embeds.device) + past_len
            ).unsqueeze(0).expand(B, -1)

        # Rotary computed once for the whole sequence; full-attention layers
        # share the same (cos, sin). Linear-attention layers ignore it.
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)

        present_cache = Qwen3_5Cache.empty(len(self.layers)) if return_cache else None
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            past = cache.layers[i] if cache is not None else None
            if layer.layer_type == "full_attention":
                if return_cache:
                    h, new_kv, _ = layer(
                        h, cos=cos, sin=sin,
                        attention_mask=attention_mask, past_kv=past,
                        return_caches=True,
                    )
                    present_cache.layers[i] = new_kv
                else:
                    h = layer(h, cos=cos, sin=sin, attention_mask=attention_mask, past_kv=past)
            else:
                if return_cache:
                    h, _, new_lin = layer(
                        h, linear_attn_cache=past, return_caches=True,
                    )
                    present_cache.layers[i] = new_lin
                else:
                    h = layer(h, linear_attn_cache=past)

        h = self.norm(h)
        return h, present_cache


class Qwen3_5ForCausalLM(nn.Module):
    """Full causal-LM model — ``Qwen3_5Model`` + ``lm_head``.

    Set ``tie_word_embeddings=True`` on the config to share weights between
    ``embed_tokens`` and ``lm_head`` (sets the ``.weight`` of the second to
    point at the first; saves vocab × hidden parameters at the cost of
    being unable to use a separately initialised LM head).
    """

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Qwen3_5Cache] = None,
        return_cache: bool = False,
        logits_to_keep: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Qwen3_5Cache]]:
        """Forward returns ``(logits, cache)``.

        ``logits_to_keep > 0`` slices the trailing N positions before the LM
        head — saves a vocab-projection matmul during inference where you
        only need the next-token logit.
        """
        h, present_cache = self.model(
            input_ids=input_ids, inputs_embeds=inputs_embeds,
            position_ids=position_ids, attention_mask=attention_mask,
            cache=cache, return_cache=return_cache,
        )
        if logits_to_keep > 0:
            h = h[:, -logits_to_keep:, :]
        logits = self.lm_head(h)
        return logits, present_cache
