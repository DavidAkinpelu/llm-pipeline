"""Tests for Qwen3.5/3.6 typed configs and architecture detection."""

import pytest

from llm_pipeline.core.registry import ModelRegistry
from llm_pipeline.models.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5_MoE_Config,
    qwen3_6_27b,
    qwen3_6_35b_a3b,
)


# --------------------------------------------------------------------------- #
# Preset values match the released HF configs
# --------------------------------------------------------------------------- #


def test_qwen3_6_27b_preset_matches_hf_config():
    """Source of truth: Qwen/Qwen3.6-27B/config.json (text_config block)."""
    cfg = qwen3_6_27b()
    assert cfg.hidden_size == 5120
    assert cfg.num_hidden_layers == 64
    assert cfg.num_attention_heads == 24
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 256
    assert cfg.intermediate_size == 17408
    assert cfg.vocab_size == 248320
    assert cfg.rope_theta == 10_000_000.0
    assert cfg.partial_rotary_factor == 0.25
    assert cfg.max_position_embeddings == 262_144
    assert cfg.full_attention_interval == 4
    assert cfg.attn_output_gate is True


def test_qwen3_6_35b_a3b_preset_matches_hf_config():
    """Source of truth: Qwen/Qwen3.6-35B-A3B/config.json (text_config block)."""
    cfg = qwen3_6_35b_a3b()
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 40
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 256
    assert cfg.num_experts == 256
    assert cfg.num_experts_per_tok == 8
    assert cfg.moe_intermediate_size == 512
    assert cfg.shared_expert_intermediate_size == 512
    assert cfg.router_aux_loss_coef == 1e-3
    assert cfg.full_attention_interval == 4
    assert cfg.linear_num_value_heads == 32  # MoE-specific override


# --------------------------------------------------------------------------- #
# Layer pattern is correct
# --------------------------------------------------------------------------- #


def test_layer_types_default_pattern():
    """Hybrid pattern: every 4th layer is full_attention, others linear."""
    cfg = qwen3_6_27b()
    # Indices 3, 7, 11, ..., 63 should be full_attention.
    full_idx = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    assert full_idx == list(range(3, 64, 4))
    assert len(cfg.layer_types) == 64


def test_layer_types_for_moe_release():
    cfg = qwen3_6_35b_a3b()
    full_idx = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    assert full_idx == list(range(3, 40, 4))
    assert len(cfg.layer_types) == 40


def test_layer_types_length_must_match_num_layers():
    with pytest.raises(ValueError, match="layer_types"):
        Qwen3_5Config(num_hidden_layers=8, layer_types=["full_attention"] * 4)


def test_layer_types_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown layer_type"):
        Qwen3_5Config(num_hidden_layers=4, layer_types=["full_attention", "linear_attention", "mamba", "full_attention"])


# --------------------------------------------------------------------------- #
# Registry detection for Qwen3.5 / 3.6 names
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name,expected", [
    ("Qwen/Qwen3.6-27B", "qwen3_5"),
    ("Qwen/Qwen3.6-27B-FP8", "qwen3_5"),
    ("Qwen/Qwen3.5-27B", "qwen3_5"),
    ("Qwen/Qwen3.6-35B-A3B", "qwen3_5_moe"),
    ("Qwen/Qwen3.5-35B-A3B", "qwen3_5_moe"),
    ("unsloth/Qwen3.6-35B-A3B-GGUF", "qwen3_5_moe"),
    # Plain qwen3 still routes to qwen3 (the original Qwen3 family).
    ("Qwen/Qwen3-0.6B", "qwen3"),
    ("Qwen/Qwen3-7B-Chat", "qwen3"),
])
def test_registry_detects_qwen3_5_family(name, expected):
    assert ModelRegistry.detect_model_type(name) == expected


def test_qwen3_5_target_modules_include_linear_attention_projections():
    """Hybrid layers add the Gated-DeltaNet projections that LoRA needs to
    know about — these are extra targets compared to vanilla Qwen3 / Llama.
    """
    targets = ModelRegistry.get_target_modules("qwen3_5")
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert name in targets, f"missing full-attention proj: {name}"
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        assert name in targets, f"missing linear-attention proj: {name}"


def test_qwen3_5_architecture_type_is_hybrid():
    assert ModelRegistry.get_architecture_type("qwen3_5") == "decoder_only_hybrid"
    assert ModelRegistry.get_architecture_type("qwen3_5_moe") == "decoder_only_hybrid"


# --------------------------------------------------------------------------- #
# MoE config inherits dense fields cleanly
# --------------------------------------------------------------------------- #


def test_moe_config_inherits_hybrid_layer_types():
    """MoE release has 40 layers but the same 1-in-4 full-attention pattern."""
    cfg = qwen3_6_35b_a3b()
    assert isinstance(cfg, Qwen3_5Config)         # subclass relationship
    assert cfg.full_attention_interval == 4
    n_full = sum(1 for t in cfg.layer_types if t == "full_attention")
    assert n_full == cfg.num_hidden_layers // cfg.full_attention_interval
