"""Tests for Qwen3.5/3.6 rotary embeddings (partial RoPE + mRoPE)."""

import math

import pytest
import torch

from llm_pipeline.models.qwen3_5.rotary import (
    Qwen3_5RotaryEmbedding,
    RotaryConfig,
    apply_interleaved_mrope,
    apply_partial_rotary_pos_emb,
    rotate_half,
)


# --------------------------------------------------------------------------- #
# rotate_half
# --------------------------------------------------------------------------- #


def test_rotate_half_swaps_halves_with_sign_flip():
    x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    # Halves are [0, 1, 2, 3] and [4, 5, 6, 7]; rotate_half = [-4, -5, -6, -7, 0, 1, 2, 3].
    expected = torch.tensor([-4., -5., -6., -7., 0., 1., 2., 3.]).reshape(1, 1, 8)
    torch.testing.assert_close(rotate_half(x), expected)


def test_rotate_half_is_a_quarter_rotation_when_applied_twice():
    """rotate_half∘rotate_half = -id (180° rotation in pair-of-coordinates space)."""
    x = torch.randn(2, 3, 16)
    torch.testing.assert_close(rotate_half(rotate_half(x)), -x)


# --------------------------------------------------------------------------- #
# apply_partial_rotary_pos_emb
# --------------------------------------------------------------------------- #


def test_partial_rotary_leaves_trailing_dims_untouched():
    """With cos/sin only spanning the first rotary_dim channels, channels
    rotary_dim..head_dim should pass through unchanged.
    """
    B, H, T, head_dim = 1, 2, 5, 16
    rotary_dim = 8
    q = torch.randn(B, H, T, head_dim)
    k = torch.randn(B, H, T, head_dim)
    cos = torch.randn(B, T, rotary_dim)
    sin = torch.randn(B, T, rotary_dim)
    q_out, k_out = apply_partial_rotary_pos_emb(q, k, cos, sin)
    torch.testing.assert_close(q_out[..., rotary_dim:], q[..., rotary_dim:])
    torch.testing.assert_close(k_out[..., rotary_dim:], k[..., rotary_dim:])


def test_partial_rotary_zero_position_is_identity():
    """At position 0 (cos=1, sin=0), the rotation is identity even on the
    rotated channels.
    """
    B, H, T, head_dim = 1, 2, 4, 8
    rotary_dim = 4
    q = torch.randn(B, H, T, head_dim)
    k = torch.randn(B, H, T, head_dim)
    cos = torch.ones(B, T, rotary_dim)
    sin = torch.zeros(B, T, rotary_dim)
    q_out, k_out = apply_partial_rotary_pos_emb(q, k, cos, sin)
    torch.testing.assert_close(q_out, q)
    torch.testing.assert_close(k_out, k)


# --------------------------------------------------------------------------- #
# apply_interleaved_mrope
# --------------------------------------------------------------------------- #


def test_mrope_interleave_text_only_returns_t_band():
    """When all three position grids are identical (text-only), the
    interleave is a no-op: every slot ends up holding the T-band value
    (which equals H and W in that case).
    """
    section = [11, 11, 10]
    rotary_half = sum(section)                        # 32
    B, T = 2, 7
    # All three bands carry the same values.
    base = torch.randn(B, T, rotary_half)
    freqs = torch.stack([base, base, base], dim=0)
    out = apply_interleaved_mrope(freqs, section)
    torch.testing.assert_close(out, base)


def test_mrope_interleave_distributes_per_section():
    """With three distinct band tensors, the output should pick the right
    slots from each per the documented interleave pattern (T at 0/3/6/...,
    H at 1/4/7/..., W at 2/5/8/...).
    """
    section = [3, 3, 2]
    rotary_half = sum(section)                        # 8
    B, T = 1, 1
    t_band = torch.full((B, T, rotary_half), 100.0)
    h_band = torch.full((B, T, rotary_half), 200.0)
    w_band = torch.full((B, T, rotary_half), 300.0)
    freqs = torch.stack([t_band, h_band, w_band], dim=0)
    out = apply_interleaved_mrope(freqs, section).flatten().tolist()
    # H positions: slice(1, 9, 3) → indices 1, 4, 7 (three slots, matches section[1]=3).
    # W positions: slice(2, 6, 3) → indices 2, 5    (two slots, matches section[2]=2).
    # T fills the rest.
    expected = [100, 200, 300, 100, 200, 300, 100, 200]
    assert out == expected


# --------------------------------------------------------------------------- #
# Qwen3_5RotaryEmbedding
# --------------------------------------------------------------------------- #


def _rotary_cfg(head_dim=64, partial=0.25, sections=None) -> RotaryConfig:
    """Build a config whose mrope_section sum matches rotary_dim/2.

    Production Qwen3.6 uses (11, 11, 10) summing to 32 with head_dim=256,
    partial=0.25 → rotary_dim/2=32. For test fixtures we shrink head_dim,
    so default to a proportional section split.
    """
    rotary_dim = (int(head_dim * partial) // 2) * 2
    half = rotary_dim // 2
    if sections is None:
        # Approx [11, 11, 10] proportions: ~34%, ~34%, ~32%.
        a = max(1, half * 11 // 32)
        b = max(1, half * 11 // 32)
        c = half - a - b
        sections = (a, b, c)
    return RotaryConfig(
        head_dim=head_dim,
        rope_theta=10_000_000.0,
        partial_rotary_factor=partial,
        mrope_section=sections,
        mrope_interleaved=True,
    )


def test_rotary_module_output_shape():
    cfg = _rotary_cfg()
    rope = Qwen3_5RotaryEmbedding(cfg)
    x = torch.randn(2, 5, cfg.head_dim)
    pos = torch.arange(5).unsqueeze(0).expand(2, -1)
    cos, sin = rope(x, pos)
    rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    assert cos.shape == (2, 5, rotary_dim)
    assert sin.shape == (2, 5, rotary_dim)


def test_rotary_module_position_zero_cos_sin():
    cfg = _rotary_cfg()
    rope = Qwen3_5RotaryEmbedding(cfg)
    x = torch.randn(1, 1, cfg.head_dim)
    pos = torch.zeros(1, 1, dtype=torch.long)
    cos, sin = rope(x, pos)
    torch.testing.assert_close(cos, torch.ones_like(cos))
    torch.testing.assert_close(sin, torch.zeros_like(sin))


def test_rotary_module_text_only_matches_3d_with_same_grids():
    """[B, T] position_ids should produce the same cos/sin as [3, B, T]
    where all three grids carry the same indices.
    """
    cfg = _rotary_cfg()
    rope = Qwen3_5RotaryEmbedding(cfg)
    x = torch.randn(1, 5, cfg.head_dim)
    pos_2d = torch.arange(5).unsqueeze(0)
    pos_3d = pos_2d[None, ...].expand(3, -1, -1)
    c1, s1 = rope(x, pos_2d)
    c2, s2 = rope(x, pos_3d)
    torch.testing.assert_close(c1, c2)
    torch.testing.assert_close(s1, s2)


def test_rotary_module_rejects_misshaped_positions():
    cfg = _rotary_cfg()
    rope = Qwen3_5RotaryEmbedding(cfg)
    x = torch.randn(1, 4, cfg.head_dim)
    with pytest.raises(ValueError, match="position_ids"):
        rope(x, torch.zeros(2, 1, 4, dtype=torch.long))    # leading dim 2 ≠ 3


def test_rotary_module_section_sum_must_match():
    bad_cfg = RotaryConfig(head_dim=64, partial_rotary_factor=0.25, mrope_section=(5, 5, 5))
    # rotary_dim = 16, rotary_dim/2 = 8, but section sums to 15 → must reject.
    with pytest.raises(ValueError, match="mrope_section"):
        Qwen3_5RotaryEmbedding(bad_cfg)


def test_rotary_module_partial_factor_uses_only_partial_dims():
    """Partial rotary should only generate cos/sin of size ``head_dim * factor``,
    not the full head_dim.
    """
    cfg = _rotary_cfg(head_dim=64, partial=0.5, sections=(6, 6, 4))
    rope = Qwen3_5RotaryEmbedding(cfg)
    x = torch.randn(1, 3, cfg.head_dim)
    pos = torch.arange(3).unsqueeze(0)
    cos, sin = rope(x, pos)
    assert cos.shape[-1] == 32           # head_dim * 0.5
    assert cos.shape[-1] != cfg.head_dim


# --------------------------------------------------------------------------- #
# End-to-end: vanilla RoPE behaviour via partial rotation on text-only input
# --------------------------------------------------------------------------- #


def test_partial_rotary_phase_rotation_on_canonical_query():
    """RoPE rotates a 2D pair (a, b) by angle θ to (a cosθ − b sinθ, a sinθ + b cosθ).
    Construct that scenario directly via apply_partial_rotary_pos_emb and
    verify the rotation math.
    """
    B, H, T = 1, 1, 1
    head_dim = 4
    rotary_dim = 4
    # q = [1, 0, 0, 0] in pair (a, b) layout: that's (a=1, b=0) on the first
    # half-pair, (a=0, b=0) on the second. After rotate_half: (-0, -0, 1, 0) →
    # but RoPE views the head as [a0, a1, b0, b1] paired (a0, b0) and (a1, b1).
    q = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    k = torch.zeros_like(q)
    theta = math.pi / 4
    cos = torch.full((B, T, rotary_dim), math.cos(theta))
    sin = torch.full((B, T, rotary_dim), math.sin(theta))
    q_out, _ = apply_partial_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    # The pair (q[0]=1, q[2]=0) rotates to (cosθ, sinθ); the pair (q[1]=0, q[3]=0) stays at (0, 0).
    expected = torch.tensor([[[[math.cos(theta), 0.0, math.sin(theta), 0.0]]]])
    torch.testing.assert_close(q_out, expected, atol=1e-6, rtol=1e-6)
