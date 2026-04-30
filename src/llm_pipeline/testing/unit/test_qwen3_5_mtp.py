"""Tests for the Qwen3.5/3.6 Multi-Token Prediction head."""

import pytest
import torch
import torch.nn as nn

from llm_pipeline.models.qwen3_5 import (
    MTPConfig,
    Qwen3_5MTPHead,
)


def _cfg() -> MTPConfig:
    return MTPConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        partial_rotary_factor=0.5,
        mrope_section=(1, 0, 0),
    )


def _build_head(cfg: MTPConfig, vocab_size: int = 64) -> Qwen3_5MTPHead:
    embed = nn.Embedding(vocab_size, cfg.hidden_size)
    lm_head = nn.Linear(cfg.hidden_size, vocab_size, bias=False)
    return Qwen3_5MTPHead(cfg, embed, lm_head)


# --------------------------------------------------------------------------- #
# Forward shape and basic properties
# --------------------------------------------------------------------------- #


def test_mtp_forward_shape():
    cfg = _cfg()
    head = _build_head(cfg)
    B, T = 2, 5
    main_hidden = torch.randn(B, T, cfg.hidden_size)
    next_ids = torch.randint(0, 64, (B, T))
    logits = head(main_hidden, next_ids)
    assert logits.shape == (B, T, 64)


def test_mtp_shares_lm_head_with_main_model():
    """The lm_head passed in is the same object referenced by the head."""
    cfg = _cfg()
    embed = nn.Embedding(64, cfg.hidden_size)
    lm_head = nn.Linear(cfg.hidden_size, 64, bias=False)
    head = Qwen3_5MTPHead(cfg, embed, lm_head)
    assert head.lm_head is lm_head
    assert head.embed_tokens is embed


def test_mtp_gradient_flows_through_all_components():
    cfg = _cfg()
    head = _build_head(cfg)
    main_hidden = torch.randn(1, 4, cfg.hidden_size, requires_grad=True)
    next_ids = torch.randint(0, 64, (1, 4))
    logits = head(main_hidden, next_ids)
    logits.sum().backward()
    # Components that should receive gradient.
    assert head.proj.weight.grad is not None and head.proj.weight.grad.abs().sum() > 0
    assert head.norm_h.weight.grad is not None
    assert head.norm_e.weight.grad is not None
    assert head.norm_out.weight.grad is not None
    assert head.self_attn.q_proj.weight.grad is not None
    assert head.lm_head.weight.grad is not None


def test_mtp_is_strictly_causal_within_one_layer():
    """The single transformer layer must respect causality — perturbing the
    input at position t shouldn't affect MTP logits at positions < t.
    """
    cfg = _cfg()
    head = _build_head(cfg).eval()
    main_hidden = torch.randn(1, 6, cfg.hidden_size)
    next_ids = torch.randint(0, 64, (1, 6))
    logits_baseline = head(main_hidden, next_ids)

    perturbed = main_hidden.clone()
    perturbed[:, 4:] = torch.randn_like(perturbed[:, 4:]) * 5.0
    logits_perturbed = head(perturbed, next_ids)

    torch.testing.assert_close(
        logits_baseline[:, :4], logits_perturbed[:, :4], atol=1e-5, rtol=1e-5,
    )


def test_mtp_norm_layers_scale_inputs_independently():
    """The two RMSNorms (on h and on e) are independent params — verify by
    setting them differently and checking the output changes.
    """
    cfg = _cfg()
    head = _build_head(cfg).eval()
    main_hidden = torch.randn(1, 3, cfg.hidden_size)
    next_ids = torch.randint(0, 64, (1, 3))
    logits_a = head(main_hidden, next_ids)

    with torch.no_grad():
        head.norm_h.weight.fill_(0.5)        # nudge only norm_h
    logits_b = head(main_hidden, next_ids)
    assert not torch.allclose(logits_a, logits_b)


def test_mtp_head_uses_partial_rotary_factor_from_config():
    """The MTP rotary embedding inherits the main model's partial RoPE
    factor, so the rotary_dim should be ``head_dim * partial_rotary_factor``.
    """
    cfg = MTPConfig(
        hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, partial_rotary_factor=0.5, mrope_section=(1, 1, 0),
    )
    head = _build_head(cfg)
    assert head.rotary.rotary_dim == 4
