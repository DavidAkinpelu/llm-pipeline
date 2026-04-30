"""Qwen3.5 / Qwen3.6 architecture support.

Two release lines from the same architectural family share this module:

- **Qwen3.6-27B** (``qwen3_5`` in the HF model_type field) — dense, 27B
  params, 64 layers. Released 2026-04-22.
- **Qwen3.6-35B-A3B** (``qwen3_5_moe`` model_type) — sparse MoE, 35B total
  / ~3B activated, 40 layers, 256 experts, top-8 routing, plus a shared
  expert. Released 2026-04-16.

The architecture is a major departure from vanilla Qwen3:

1. **Hybrid attention** — every 4th layer is a standard full-attention
   transformer; the other 3 are **Gated DeltaNet** linear-attention layers
   (causal, recurrent-friendly, with a small depthwise conv1d for local
   context). This is the "linear_attention / full_attention" interleave
   pattern in the HF config.
2. **Partial RoPE** — only the first 25% of each head_dim gets rotary
   position encoding (``partial_rotary_factor=0.25``). The rest is left
   un-rotated, which matters for very long contexts.
3. **Multimodal RoPE (mRoPE)** — RoPE is split into 3 sub-bands for text /
   spatial-x / spatial-y, with section sizes ``[11, 11, 10]``.
4. **Output-gated attention** (``attn_output_gate=true``) — a sigmoid gate
   on the attention output, like in Mixtral / DeepSeek.
5. **Vision tower** + image / video tokens — these are full multimodal
   models. Text-only inference still works by skipping the vision pipeline.
6. **MTP (Multi-Token Prediction) head** — 1 extra layer for parallel
   token prediction, used for speculative-decoding-friendly training.
7. **MoE specifics**: 256 routed experts (``moe_intermediate_size=512``)
   plus a shared expert (``shared_expert_intermediate_size=512``) that
   *every* token goes through. Top-K = 8 of 256.

What this package provides today
--------------------------------

- ``Qwen3_5Config`` — typed config dataclass with the full set of fields.
- ``Qwen3_5_MoE_Config`` — extended config with the MoE-specific fields.
- Architecture **presets** for the released model sizes (``qwen3_6_27b()``,
  ``qwen3_6_35b_a3b()``) so users can construct the configs without
  pasting magic numbers.

What's *not* here yet
---------------------

A hand-rolled inference engine for these models. Implementing
Gated DeltaNet + mRoPE + output-gated attention + MTP + multimodal vision
is comparable in scope to implementing Mamba + Mixtral + LLaVA in one go,
and is gated on the work tracked in ROADMAP.md ("Hand-rolled Qwen3.5/3.6
engine"). For now, **inference and fine-tuning of these models work via
the generic transformers fallback** (``llm_pipeline.models.generic``):

```python
from llm_pipeline.models.generic.transformers_wrapper import load_hf_model
model, tok = load_hf_model("Qwen/Qwen3.6-27B")        # text-only
model, tok = load_hf_model("Qwen/Qwen3.6-35B-A3B")    # MoE, also text-only
```

The LoRA target modules and DPO/SFT/GRPO trainers all work against this
HF-loaded model through the generic adapter path (the registry entry
``qwen3_5`` / ``qwen3_5_moe`` lists the right projection names for the
hybrid attention layers).

For the MoE auxiliary loss during fine-tuning, the HF Mixtral-style
``router_aux_loss_coef`` field (default 1e-3 in the Qwen3.5 MoE config)
is honored automatically by the transformers `forward` — see the HF
`Qwen3_5MoeForConditionalGeneration` source.
"""

from .config import (
    Qwen3_5Config,
    Qwen3_5_MoE_Config,
    qwen3_6_27b,
    qwen3_6_35b_a3b,
)
from .gated_deltanet import (
    GatedDeltaNetConfig,
    Qwen3_5GatedDeltaNet,
    RMSNormGated,
    chunk_gated_delta_rule,
    forward_chunk_gated_delta_rule,
    l2norm,
    recurrent_gated_delta_rule,
)
from .rotary import (
    Qwen3_5RotaryEmbedding,
    RotaryConfig,
    apply_interleaved_mrope,
    apply_partial_rotary_pos_emb,
    rotate_half,
)
from .attention import (
    GatedAttentionConfig,
    Qwen3_5Attention,
    Qwen3_5RMSNorm,
)
from .mlp import (
    MLPConfig,
    MoEBlockConfig,
    Qwen3_5MLP,
    Qwen3_5MoeBlock,
    Qwen3_5MoeRouter,
)
from .decoder_layer import (
    DecoderLayerConfig,
    Qwen3_5DecoderLayer,
)
from .model import (
    Qwen3_5Cache,
    Qwen3_5ForCausalLM,
    Qwen3_5Model,
)
from .loader import LoadReport, load_qwen3_5_state_dict
from .mtp import MTPConfig, Qwen3_5MTPHead
from .vision import (
    Qwen3_5VisionAttention,
    Qwen3_5VisionBlock,
    Qwen3_5VisionMLP,
    Qwen3_5VisionModel,
    Qwen3_5VisionPatchEmbed,
    Qwen3_5VisionPatchMerger,
    Qwen3_5VisionRotaryEmbedding,
    VisionConfig,
    apply_rotary_pos_emb_vision,
    replace_placeholder_embeddings,
)

__all__ = [
    "Qwen3_5Config",
    "Qwen3_5_MoE_Config",
    "qwen3_6_27b",
    "qwen3_6_35b_a3b",
    "GatedDeltaNetConfig",
    "Qwen3_5GatedDeltaNet",
    "RMSNormGated",
    "chunk_gated_delta_rule",
    "forward_chunk_gated_delta_rule",
    "l2norm",
    "recurrent_gated_delta_rule",
    "Qwen3_5RotaryEmbedding",
    "RotaryConfig",
    "apply_interleaved_mrope",
    "apply_partial_rotary_pos_emb",
    "rotate_half",
    "GatedAttentionConfig",
    "Qwen3_5Attention",
    "Qwen3_5RMSNorm",
    "MLPConfig",
    "MoEBlockConfig",
    "Qwen3_5MLP",
    "Qwen3_5MoeBlock",
    "Qwen3_5MoeRouter",
    "DecoderLayerConfig",
    "Qwen3_5DecoderLayer",
    "Qwen3_5Cache",
    "Qwen3_5Model",
    "Qwen3_5ForCausalLM",
]
