"""Typed configs for Qwen3.5 / Qwen3.6 architecture family.

These mirror the fields in the official HF configs verbatim — so you can
roundtrip between ``Qwen3_5Config(...)`` and ``transformers.AutoConfig``
without hand-mapping. See ``__init__.py`` for the architecture overview.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Qwen3_5Config:
    """Dense Qwen3.5 / Qwen3.6 config (e.g. Qwen3.6-27B).

    Field names match the HF ``Qwen3_5Config.text_config`` schema. Defaults
    are taken from the **Qwen3.6-27B** release config.
    """

    # ---- Core dimensions ----
    vocab_size: int = 248320
    hidden_size: int = 5120
    num_hidden_layers: int = 64
    num_attention_heads: int = 24
    num_key_value_heads: int = 4               # GQA: 6 query heads per KV head
    head_dim: int = 256
    intermediate_size: int = 17408              # dense MLP hidden dim

    # ---- Hybrid attention pattern ----
    # Every ``full_attention_interval``-th layer is full attention; the rest
    # are Gated DeltaNet linear-attention. Together with ``layer_types`` this
    # describes the "linear, linear, linear, full" 1-in-4 pattern.
    full_attention_interval: int = 4
    layer_types: Optional[List[str]] = None     # auto-derived in __post_init__

    # ---- Linear-attention (Gated DeltaNet) sub-config ----
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # ---- Activations / norms ----
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6

    # ---- RoPE ----
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25         # only first 25% of head_dim rotated
    mrope_interleaved: bool = True
    mrope_section: Tuple[int, int, int] = (11, 11, 10)

    # ---- Attention extras ----
    attention_dropout: float = 0.0
    attention_bias: bool = False
    attn_output_gate: bool = True               # sigmoid gate on attn output

    # ---- Generation / context ----
    max_position_embeddings: int = 262_144
    tie_word_embeddings: bool = False
    bos_token_id: int = 248044
    eos_token_id: int = 248044
    pad_token_id: Optional[int] = None
    use_cache: bool = True

    # ---- MTP (Multi-Token Prediction) head ----
    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False

    # ---- Numerics ----
    dtype: str = "bfloat16"
    initializer_range: float = 0.02
    mamba_ssm_dtype: str = "float32"

    model_type: str = "qwen3_5"
    architecture_type: str = "decoder_only_hybrid"

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = self._default_layer_types()
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers={self.num_hidden_layers}"
            )
        for lt in self.layer_types:
            if lt not in {"linear_attention", "full_attention"}:
                raise ValueError(f"unknown layer_type: {lt!r}")

    def _default_layer_types(self) -> List[str]:
        return [
            "full_attention" if (i + 1) % self.full_attention_interval == 0
            else "linear_attention"
            for i in range(self.num_hidden_layers)
        ]


@dataclass
class Qwen3_5_MoE_Config(Qwen3_5Config):
    """Sparse-MoE Qwen3.5 / Qwen3.6 config (e.g. Qwen3.6-35B-A3B).

    Inherits everything from ``Qwen3_5Config`` and adds the MoE fields.
    Defaults match the **Qwen3.6-35B-A3B** release config.

    The dense ``intermediate_size`` is ignored on MoE layers; the per-expert
    hidden dim is ``moe_intermediate_size``, and a *shared* expert with
    ``shared_expert_intermediate_size`` runs on every token (DeepSeek-V3 style).
    """

    # Override dense defaults that differ on the MoE release.
    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    intermediate_size: int = 0                  # unused on MoE layers
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    # MoE specifics.
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    norm_topk_prob: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 1e-3

    model_type: str = "qwen3_5_moe"


# --------------------------------------------------------------------------- #
# Architecture presets — convenience constructors for the released models.
# --------------------------------------------------------------------------- #


def qwen3_6_27b() -> Qwen3_5Config:
    """Preset for **Qwen3.6-27B** (dense, 64 layers, 27B params).

    Source: ``Qwen/Qwen3.6-27B`` HF config. Released 2026-04-22.
    """
    return Qwen3_5Config()  # current dataclass defaults match this release.


def qwen3_6_35b_a3b() -> Qwen3_5_MoE_Config:
    """Preset for **Qwen3.6-35B-A3B** (sparse MoE, 35B total / ~3B active).

    Source: ``Qwen/Qwen3.6-35B-A3B`` HF config. Released 2026-04-16.
    Top-8 of 256 routed experts + 1 shared expert per layer; 40 layers,
    hidden_size 2048; 256K context (extensible to 1M).
    """
    return Qwen3_5_MoE_Config()  # current dataclass defaults match this release.
