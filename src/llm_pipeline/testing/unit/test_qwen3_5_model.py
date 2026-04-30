"""Tests for Qwen3.5/3.6 MLP, MoE block, decoder layer, and full model."""

import pytest
import torch

from llm_pipeline.models.qwen3_5 import (
    DecoderLayerConfig,
    GatedAttentionConfig,
    GatedDeltaNetConfig,
    MLPConfig,
    MoEBlockConfig,
    Qwen3_5Cache,
    Qwen3_5Config,
    Qwen3_5_MoE_Config,
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5MLP,
    Qwen3_5Model,
    Qwen3_5MoeBlock,
    Qwen3_5MoeRouter,
)


# --------------------------------------------------------------------------- #
# Dense MLP
# --------------------------------------------------------------------------- #


def test_dense_mlp_forward_shape():
    cfg = MLPConfig(hidden_size=32, intermediate_size=64)
    mlp = Qwen3_5MLP(cfg)
    x = torch.randn(2, 5, 32)
    y = mlp(x)
    assert y.shape == x.shape


def test_dense_mlp_silu_default():
    """SiLU is the default; gelu opt-in path also works."""
    cfg = MLPConfig(hidden_size=8, intermediate_size=16, hidden_act="gelu")
    mlp = Qwen3_5MLP(cfg)
    y = mlp(torch.randn(1, 1, 8))
    assert y.shape == (1, 1, 8)


# --------------------------------------------------------------------------- #
# MoE block
# --------------------------------------------------------------------------- #


def _moe_cfg(**overrides) -> MoEBlockConfig:
    base = dict(
        hidden_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=16,
    )
    base.update(overrides)
    return MoEBlockConfig(**base)


def test_moe_router_renorm_top_k_to_one():
    cfg = _moe_cfg()
    router = Qwen3_5MoeRouter(cfg)
    x = torch.randn(7, cfg.hidden_size)
    _, gates, idx = router(x)
    torch.testing.assert_close(gates.sum(dim=-1), torch.ones(7), atol=1e-5, rtol=1e-5)
    assert idx.shape == (7, cfg.num_experts_per_tok)
    assert idx.min() >= 0 and idx.max() < cfg.num_experts


def test_moe_block_forward_shape():
    cfg = _moe_cfg()
    moe = Qwen3_5MoeBlock(cfg)
    x = torch.randn(2, 3, cfg.hidden_size)
    y = moe(x)
    assert y.shape == x.shape


def test_moe_block_matches_handwritten_reference():
    """End-to-end: routed dispatch + sigmoid-gated shared expert agrees with
    a manual reference implementation.
    """
    cfg = _moe_cfg()
    moe = Qwen3_5MoeBlock(cfg).eval()
    x = torch.randn(1, 2, cfg.hidden_size)
    y = moe(x)

    h = x.reshape(-1, cfg.hidden_size)
    _, gates, idx = moe.router(h)

    routed = torch.zeros_like(h)
    for e in range(cfg.num_experts):
        sel = (idx == e).nonzero(as_tuple=False)
        if sel.numel() == 0:
            continue
        tok = sel[:, 0]
        slot = sel[:, 1]
        out = moe.experts[e](h[tok])
        routed.index_add_(0, tok, out * gates[tok, slot].unsqueeze(-1))

    shared = moe.shared_expert(h)
    shared_gate = torch.sigmoid(moe.shared_expert_gate(h))
    expected = (routed + shared_gate * shared).reshape_as(x)
    torch.testing.assert_close(y, expected, atol=1e-4, rtol=1e-4)


def test_moe_block_router_logits_stashed():
    cfg = _moe_cfg()
    moe = Qwen3_5MoeBlock(cfg)
    x = torch.randn(2, 3, cfg.hidden_size)
    moe(x)
    assert moe.last_router_logits.shape == (6, cfg.num_experts)
    assert moe.last_expert_mask.shape == (6, cfg.num_experts)


# --------------------------------------------------------------------------- #
# Decoder layer
# --------------------------------------------------------------------------- #


def _full_attn_cfg() -> GatedAttentionConfig:
    return GatedAttentionConfig(
        hidden_size=16, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
    )


def _linear_attn_cfg() -> GatedDeltaNetConfig:
    return GatedDeltaNetConfig(
        hidden_size=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
    )


def _dense_mlp_cfg() -> MLPConfig:
    return MLPConfig(hidden_size=16, intermediate_size=32)


def test_decoder_layer_full_attention_path():
    layer_cfg = DecoderLayerConfig(
        layer_type="full_attention", hidden_size=16,
        full_attn=_full_attn_cfg(), dense_mlp=_dense_mlp_cfg(),
    )
    layer = Qwen3_5DecoderLayer(layer_cfg)
    x = torch.randn(1, 4, 16)
    rotary_dim = 4 // 4 * 2  # head_dim * partial=0.5 = 2
    cos = torch.ones(1, 4, rotary_dim)
    sin = torch.zeros(1, 4, rotary_dim)
    y = layer(x, cos=cos, sin=sin)
    assert y.shape == x.shape
    assert layer.linear_attn is None
    assert layer.self_attn is not None


def test_decoder_layer_linear_attention_path():
    layer_cfg = DecoderLayerConfig(
        layer_type="linear_attention", hidden_size=16,
        linear_attn=_linear_attn_cfg(), dense_mlp=_dense_mlp_cfg(),
    )
    layer = Qwen3_5DecoderLayer(layer_cfg)
    x = torch.randn(1, 4, 16)
    y = layer(x)              # no cos/sin needed for linear-attn layers
    assert y.shape == x.shape
    assert layer.self_attn is None
    assert layer.linear_attn is not None


def test_decoder_layer_validates_required_subconfigs():
    with pytest.raises(ValueError, match="full_attention"):
        DecoderLayerConfig(
            layer_type="full_attention", hidden_size=16,
            dense_mlp=_dense_mlp_cfg(),
        )
    with pytest.raises(ValueError, match="linear_attention"):
        DecoderLayerConfig(
            layer_type="linear_attention", hidden_size=16,
            dense_mlp=_dense_mlp_cfg(),
        )


def test_decoder_layer_validates_one_ffn():
    with pytest.raises(ValueError, match="dense_mlp"):
        DecoderLayerConfig(
            layer_type="full_attention", hidden_size=16,
            full_attn=_full_attn_cfg(),
            dense_mlp=_dense_mlp_cfg(),
            moe_block=_moe_cfg(),
        )


def test_decoder_layer_returns_caches_when_requested():
    layer_cfg = DecoderLayerConfig(
        layer_type="full_attention", hidden_size=16,
        full_attn=_full_attn_cfg(), dense_mlp=_dense_mlp_cfg(),
    )
    layer = Qwen3_5DecoderLayer(layer_cfg)
    x = torch.randn(1, 4, 16)
    rotary_dim = 2
    cos = torch.ones(1, 4, rotary_dim)
    sin = torch.zeros(1, 4, rotary_dim)
    y, present_kv, present_rec = layer(x, cos=cos, sin=sin, return_caches=True)
    assert y.shape == x.shape
    assert present_kv is not None
    assert present_rec is None             # full-attn layers don't carry recurrent state


# --------------------------------------------------------------------------- #
# Full model — small dense config
# --------------------------------------------------------------------------- #


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
        partial_rotary_factor=0.5,           # rotary_dim = 2; matches mrope_section sum=1
        mrope_section=(1, 0, 0),
        max_position_embeddings=64,
        bos_token_id=0, eos_token_id=0, pad_token_id=0,
    )


def test_full_model_forward_shape():
    cfg = _tiny_dense_cfg()
    model = Qwen3_5Model(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    h, _ = model(input_ids=input_ids)
    assert h.shape == (1, 5, cfg.hidden_size)


def test_full_model_layer_count_matches_layer_types():
    cfg = _tiny_dense_cfg()
    model = Qwen3_5Model(cfg)
    assert len(model.layers) == cfg.num_hidden_layers
    for i, lt in enumerate(cfg.layer_types):
        assert model.layers[i].layer_type == lt


def test_causal_lm_forward_shape():
    cfg = _tiny_dense_cfg()
    model = Qwen3_5ForCausalLM(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    logits, _ = model(input_ids=input_ids)
    assert logits.shape == (1, 6, cfg.vocab_size)


def test_causal_lm_logits_to_keep_slices_trailing_positions():
    cfg = _tiny_dense_cfg()
    model = Qwen3_5ForCausalLM(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    logits, _ = model(input_ids=input_ids, logits_to_keep=1)
    assert logits.shape == (1, 1, cfg.vocab_size)


def test_causal_lm_tied_weights_share_storage():
    cfg = _tiny_dense_cfg()
    cfg.tie_word_embeddings = True
    model = Qwen3_5ForCausalLM(cfg)
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_causal_lm_untied_weights_separate_storage():
    cfg = _tiny_dense_cfg()
    cfg.tie_word_embeddings = False
    model = Qwen3_5ForCausalLM(cfg)
    assert model.lm_head.weight is not model.model.embed_tokens.weight


def test_full_attention_only_kv_cache_decode_matches_one_shot_prefill():
    """KV cache equivalence on a model with only full-attention layers."""
    cfg = _tiny_dense_cfg()
    cfg.full_attention_interval = 1
    cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
    model = Qwen3_5ForCausalLM(cfg).eval()
    T = 5
    input_ids = torch.randint(0, cfg.vocab_size, (1, T))

    full_logits, _ = model(input_ids=input_ids)
    _, cache = model(input_ids=input_ids[:, : T - 1], return_cache=True)
    step_logits, _ = model(input_ids=input_ids[:, T - 1 :], cache=cache)

    torch.testing.assert_close(full_logits[:, -1:], step_logits, atol=1e-4, rtol=1e-4)


def test_hybrid_model_cache_decode_matches_one_shot_prefill():
    """End-to-end equivalence on a hybrid stack (linear + full attention)
    with the cache-aware Gated DeltaNet preserving conv state across calls.
    """
    cfg = _tiny_dense_cfg()
    # Default _tiny_dense_cfg already has 1-in-4 full attention (3 linear + 1 full).
    assert cfg.layer_types.count("linear_attention") == 3
    model = Qwen3_5ForCausalLM(cfg).eval()
    T = 5
    input_ids = torch.randint(0, cfg.vocab_size, (1, T))

    full_logits, _ = model(input_ids=input_ids)
    _, cache = model(input_ids=input_ids[:, : T - 1], return_cache=True)
    step_logits, _ = model(input_ids=input_ids[:, T - 1 :], cache=cache)

    torch.testing.assert_close(full_logits[:, -1:], step_logits, atol=1e-4, rtol=1e-4)


def test_hybrid_model_token_by_token_decode_matches_one_shot():
    """Stronger version: T sequential 1-token decode steps from an empty
    cache reproduce the one-shot output for every position.
    """
    cfg = _tiny_dense_cfg()
    model = Qwen3_5ForCausalLM(cfg).eval()
    T = 5
    input_ids = torch.randint(0, cfg.vocab_size, (1, T))

    full_logits, _ = model(input_ids=input_ids)

    cache = None
    decoded = []
    for t in range(T):
        step_logits, cache = model(
            input_ids=input_ids[:, t : t + 1], cache=cache, return_cache=True,
        )
        decoded.append(step_logits)
    decoded = torch.cat(decoded, dim=1)

    torch.testing.assert_close(decoded, full_logits, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Full model — small MoE config
# --------------------------------------------------------------------------- #


def _tiny_moe_cfg() -> Qwen3_5_MoE_Config:
    return Qwen3_5_MoE_Config(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=4,                 # 3 linear + 1 full
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
        moe_intermediate_size=16,
        shared_expert_intermediate_size=8,
        rope_theta=10_000_000.0,
        partial_rotary_factor=0.5,
        mrope_section=(1, 0, 0),
        max_position_embeddings=64,
        bos_token_id=0, eos_token_id=0, pad_token_id=0,
    )


def test_moe_model_forward_shape():
    cfg = _tiny_moe_cfg()
    model = Qwen3_5ForCausalLM(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    logits, _ = model(input_ids=input_ids)
    assert logits.shape == (1, 4, cfg.vocab_size)
    # Every layer's MLP should be a MoE block, not dense.
    for layer in model.model.layers:
        assert isinstance(layer.mlp, Qwen3_5MoeBlock)


def test_full_model_gradient_flows():
    cfg = _tiny_dense_cfg()
    model = Qwen3_5ForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    logits, _ = model(input_ids=input_ids)
    logits.sum().backward()
    # Embedding got gradient.
    assert model.model.embed_tokens.weight.grad is not None
    assert model.model.embed_tokens.weight.grad.abs().sum() > 0
    # LM head got gradient.
    assert model.lm_head.weight.grad is not None
    # First and last layer's projections got gradient.
    last = model.model.layers[-1]
    if last.layer_type == "full_attention":
        assert last.self_attn.q_proj.weight.grad is not None
    else:
        assert last.linear_attn.in_proj_qkv.weight.grad is not None


def test_full_model_position_ids_default_to_arange():
    """Auto-generated position_ids should be ``arange(T)`` plus the cached prefix."""
    cfg = _tiny_dense_cfg()
    model = Qwen3_5Model(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 3))
    explicit = torch.arange(3).unsqueeze(0).expand(2, -1)
    h_auto, _ = model(input_ids=input_ids)
    h_explicit, _ = model(input_ids=input_ids, position_ids=explicit)
    torch.testing.assert_close(h_auto, h_explicit, atol=1e-5, rtol=1e-5)
