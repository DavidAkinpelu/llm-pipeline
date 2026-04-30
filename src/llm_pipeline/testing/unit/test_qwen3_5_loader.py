"""Tests for the HF → hand-rolled state_dict loader."""

import pytest
import torch
import torch.nn as nn

from llm_pipeline.models.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5_MoE_Config,
    Qwen3_5ForCausalLM,
    Qwen3_5MoeBlock,
    load_qwen3_5_state_dict,
)


def _tiny_dense_cfg() -> Qwen3_5Config:
    return Qwen3_5Config(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=4,                 # 3 linear + 1 full
        full_attention_interval=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        rope_theta=10_000_000.0,
        partial_rotary_factor=0.5,
        mrope_section=(1, 0, 0),
        max_position_embeddings=64,
        bos_token_id=0, eos_token_id=0, pad_token_id=0,
    )


def _tiny_moe_cfg() -> Qwen3_5_MoE_Config:
    return Qwen3_5_MoE_Config(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=4,
        full_attention_interval=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        rope_theta=10_000_000.0,
        partial_rotary_factor=0.5,
        mrope_section=(1, 0, 0),
        max_position_embeddings=64,
        bos_token_id=0, eos_token_id=0, pad_token_id=0,
    )


def _build_hf_style_state_dict(model: Qwen3_5ForCausalLM) -> dict:
    """Take the canonical state_dict of our model and translate it back into
    HF's wire format: per-layer MoE experts get re-stacked into 3D tensors,
    everything else is copy-through.

    This is the inverse of ``load_qwen3_5_state_dict``'s expert-expansion
    step — letting us round-trip.
    """
    sd = dict(model.state_dict())
    out: dict = {}

    # Walk per-layer MoE blocks; if a layer's MLP is MoE, collapse the
    # ModuleList experts into 3D tensors with HF naming.
    moe_layer_keys: dict[int, dict[str, list]] = {}
    consumed: set[str] = set()
    for i, layer in enumerate(model.model.layers):
        if not isinstance(layer.mlp, Qwen3_5MoeBlock):
            continue
        E = layer.mlp.num_experts
        gate_ups = []
        downs = []
        for e in range(E):
            gp = sd[f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"]
            up = sd[f"model.layers.{i}.mlp.experts.{e}.up_proj.weight"]
            dn = sd[f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"]
            gate_ups.append(torch.cat([gp, up], dim=0))
            downs.append(dn)
            consumed.add(f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight")
            consumed.add(f"model.layers.{i}.mlp.experts.{e}.up_proj.weight")
            consumed.add(f"model.layers.{i}.mlp.experts.{e}.down_proj.weight")
        out[f"model.layers.{i}.mlp.experts.gate_up_proj"] = torch.stack(gate_ups, dim=0)
        out[f"model.layers.{i}.mlp.experts.down_proj"] = torch.stack(downs, dim=0)

    for k, v in sd.items():
        if k in consumed:
            continue
        out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Dense round-trip
# --------------------------------------------------------------------------- #


def test_dense_roundtrip_loads_every_param():
    """Build two identical-shape models with different random init, dump
    state from #1, load into #2, confirm parameters match exactly.
    """
    cfg = _tiny_dense_cfg()
    src = Qwen3_5ForCausalLM(cfg)
    dst = Qwen3_5ForCausalLM(cfg)

    # Sanity — at construction the two have different weights.
    assert not torch.equal(
        src.model.embed_tokens.weight, dst.model.embed_tokens.weight,
    )

    sd = _build_hf_style_state_dict(src)
    report = load_qwen3_5_state_dict(dst, sd, strict=True)

    assert report.unexpected_hf == []
    assert report.missing_ours == []

    # Every parameter now matches.
    for (n_src, p_src), (n_dst, p_dst) in zip(
        src.named_parameters(), dst.named_parameters()
    ):
        assert n_src == n_dst
        assert torch.equal(p_src, p_dst), f"mismatch on {n_src}"


def test_dense_loaded_model_produces_identical_logits():
    """End-to-end equivalence: after a load, the two models compute the
    same logits on the same input.
    """
    cfg = _tiny_dense_cfg()
    src = Qwen3_5ForCausalLM(cfg).eval()
    dst = Qwen3_5ForCausalLM(cfg).eval()

    sd = _build_hf_style_state_dict(src)
    load_qwen3_5_state_dict(dst, sd, strict=True)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    logits_src, _ = src(input_ids=input_ids)
    logits_dst, _ = dst(input_ids=input_ids)
    torch.testing.assert_close(logits_src, logits_dst, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# MoE round-trip — the interesting case (3D expert tensor expansion)
# --------------------------------------------------------------------------- #


def test_moe_roundtrip_loads_every_param():
    cfg = _tiny_moe_cfg()
    src = Qwen3_5ForCausalLM(cfg)
    dst = Qwen3_5ForCausalLM(cfg)

    sd = _build_hf_style_state_dict(src)
    # Verify the synthetic state_dict really is in HF format (3D experts).
    found_3d = [k for k in sd if "experts.gate_up_proj" in k or k.endswith("experts.down_proj")]
    assert len(found_3d) > 0

    report = load_qwen3_5_state_dict(dst, sd, strict=True)
    assert report.unexpected_hf == []
    assert report.missing_ours == []

    for (n, p_src), (_, p_dst) in zip(src.named_parameters(), dst.named_parameters()):
        assert torch.equal(p_src, p_dst), f"mismatch on {n}"


def test_moe_expert_split_correctly_distributes_gate_up():
    """The ``gate_up_proj[e]`` tensor is split row-wise: first
    ``moe_intermediate_size`` rows go to ``gate_proj``, the rest to ``up_proj``.
    Verify the slicing is byte-identical, not just shape-correct.
    """
    cfg = _tiny_moe_cfg()
    src = Qwen3_5ForCausalLM(cfg)
    dst = Qwen3_5ForCausalLM(cfg)

    sd = _build_hf_style_state_dict(src)
    load_qwen3_5_state_dict(dst, sd)

    # Pick layer 0 (MoE), expert 2 — verify exact rows.
    layer = 0
    e = 2
    src_gate = src.model.layers[layer].mlp.experts[e].gate_proj.weight
    src_up = src.model.layers[layer].mlp.experts[e].up_proj.weight
    dst_gate = dst.model.layers[layer].mlp.experts[e].gate_proj.weight
    dst_up = dst.model.layers[layer].mlp.experts[e].up_proj.weight
    assert torch.equal(src_gate, dst_gate)
    assert torch.equal(src_up, dst_up)


# --------------------------------------------------------------------------- #
# Skip policy — mtp.* and model.visual.* are silently ignored
# --------------------------------------------------------------------------- #


def test_loader_skips_mtp_and_visual_keys():
    cfg = _tiny_dense_cfg()
    model = Qwen3_5ForCausalLM(cfg)
    sd = _build_hf_style_state_dict(model)
    # Inject some skippable keys.
    sd["mtp.layers.0.weight"] = torch.zeros(4)
    sd["model.mtp.norm.weight"] = torch.zeros(4)
    sd["model.visual.patch_embed.weight"] = torch.zeros(4, 4)

    report = load_qwen3_5_state_dict(model, sd)
    assert "mtp.layers.0.weight" in report.skipped_hf_keys
    assert "model.mtp.norm.weight" in report.skipped_hf_keys
    assert "model.visual.patch_embed.weight" in report.skipped_hf_keys
    assert report.unexpected_hf == []


# --------------------------------------------------------------------------- #
# Strict mode
# --------------------------------------------------------------------------- #


def test_strict_mode_raises_on_unexpected_key():
    cfg = _tiny_dense_cfg()
    model = Qwen3_5ForCausalLM(cfg)
    sd = _build_hf_style_state_dict(model)
    sd["model.layers.0.mlp.fictitious_proj.weight"] = torch.zeros(8, 16)
    with pytest.raises(RuntimeError, match="strict load failed"):
        load_qwen3_5_state_dict(model, sd, strict=True)


def test_strict_mode_raises_on_missing_param():
    cfg = _tiny_dense_cfg()
    model = Qwen3_5ForCausalLM(cfg)
    sd = _build_hf_style_state_dict(model)
    del sd["lm_head.weight"]
    with pytest.raises(RuntimeError, match="strict load failed"):
        load_qwen3_5_state_dict(model, sd, strict=True)


def test_lenient_mode_returns_report():
    cfg = _tiny_dense_cfg()
    model = Qwen3_5ForCausalLM(cfg)
    sd = _build_hf_style_state_dict(model)
    sd["unknown.parameter"] = torch.zeros(4)
    report = load_qwen3_5_state_dict(model, sd, strict=False)
    assert "unknown.parameter" in report.unexpected_hf


def test_load_report_summary_string_is_informative():
    cfg = _tiny_dense_cfg()
    model = Qwen3_5ForCausalLM(cfg)
    sd = _build_hf_style_state_dict(model)
    report = load_qwen3_5_state_dict(model, sd)
    text = report.summary()
    assert "loaded" in text and "skipped" in text


# --------------------------------------------------------------------------- #
# Tied embeddings
# --------------------------------------------------------------------------- #


def test_loader_handles_tied_word_embeddings():
    """When tie_word_embeddings=True, the HF checkpoint typically has only
    embed_tokens.weight; lm_head.weight is implicit. Our loader must not
    flag lm_head.weight as missing in that case.
    """
    cfg = _tiny_dense_cfg()
    cfg.tie_word_embeddings = True
    model = Qwen3_5ForCausalLM(cfg)

    sd = _build_hf_style_state_dict(model)
    if "lm_head.weight" in sd:
        del sd["lm_head.weight"]                           # mimic HF's tied-weight save

    report = load_qwen3_5_state_dict(model, sd, strict=True)   # must not raise
    assert "lm_head.weight" not in report.missing_ours


# --------------------------------------------------------------------------- #
# Shape mismatch detection
# --------------------------------------------------------------------------- #


def test_loader_raises_on_shape_mismatch():
    cfg = _tiny_dense_cfg()
    model = Qwen3_5ForCausalLM(cfg)
    sd = _build_hf_style_state_dict(model)
    sd["model.embed_tokens.weight"] = torch.zeros(99, 99)    # wrong shape
    with pytest.raises(ValueError, match="shape mismatch"):
        load_qwen3_5_state_dict(model, sd)
