"""Tests for the Qwen3.5/3.6 vision tower."""

import pytest
import torch

from llm_pipeline.models.qwen3_5 import (
    Qwen3_5VisionModel,
    Qwen3_5VisionPatchEmbed,
    Qwen3_5VisionPatchMerger,
    VisionConfig,
    apply_rotary_pos_emb_vision,
    replace_placeholder_embeddings,
)
from llm_pipeline.models.qwen3_5.vision import (
    Qwen3_5VisionAttention,
    Qwen3_5VisionBlock,
    Qwen3_5VisionRotaryEmbedding,
)


def _tiny_cfg(**overrides) -> VisionConfig:
    """Small but structurally complete config for fast tests."""
    base = dict(
        hidden_size=32,
        out_hidden_size=64,
        intermediate_size=64,
        depth=2,
        num_heads=4,
        in_channels=3,
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=64,            # 8×8 grid
    )
    base.update(overrides)
    return VisionConfig(**base)


# --------------------------------------------------------------------------- #
# Patch embedding
# --------------------------------------------------------------------------- #


def test_patch_embed_output_shape():
    cfg = _tiny_cfg()
    pe = Qwen3_5VisionPatchEmbed(cfg)
    n_patches = 12
    x = torch.randn(n_patches, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
    y = pe(x)
    assert y.shape == (n_patches, cfg.hidden_size)


# --------------------------------------------------------------------------- #
# Rotary
# --------------------------------------------------------------------------- #


def test_rotary_embedding_shape():
    rope = Qwen3_5VisionRotaryEmbedding(dim=8)
    out = rope(seqlen=16)
    assert out.shape == (16, 4)              # dim/2 frequency channels


def test_apply_rotary_pos_emb_vision_preserves_shape_and_norm():
    """Rotation is unitary, so ||q_rot|| == ||q||. Requires the canonical
    duplicate-halves cos/sin layout: ``cos[..., i] == cos[..., half+i]``,
    same for sin — that's how the half-and-rotate trick yields a 2D rotation
    on each (x_i, x_{half+i}) pair.
    """
    N, H, D = 8, 4, 16
    q = torch.randn(N, H, D)
    k = torch.randn(N, H, D)
    half = D // 2
    angles = torch.rand(N, half)
    cos_half = angles.cos()
    sin_half = angles.sin()
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    q2, k2 = apply_rotary_pos_emb_vision(q, k, cos, sin)
    assert q2.shape == q.shape and k2.shape == k.shape
    torch.testing.assert_close(q2.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #


def test_attention_is_non_causal():
    """Vision attention must NOT be causal — perturbing a later patch should
    affect earlier patches' outputs.
    """
    cfg = _tiny_cfg()
    attn = Qwen3_5VisionAttention(cfg).eval()
    N = 8
    h = torch.randn(N, cfg.hidden_size)
    cos = torch.ones(N, cfg.hidden_size // cfg.num_heads)
    sin = torch.zeros_like(cos)

    out1 = attn(h, cos, sin)
    h2 = h.clone()
    h2[-1] = torch.randn_like(h2[-1]) * 5.0
    out2 = attn(h2, cos, sin)
    # Earlier outputs should change (non-causal).
    assert not torch.allclose(out1[0], out2[0], atol=1e-4)


def test_attention_cu_seqlens_segments_isolate_images():
    """Two images concatenated with a cu_seqlens boundary: perturbing image B
    must NOT affect image A's outputs.
    """
    cfg = _tiny_cfg()
    attn = Qwen3_5VisionAttention(cfg).eval()
    L_a, L_b = 4, 6
    h = torch.randn(L_a + L_b, cfg.hidden_size)
    cos = torch.ones(L_a + L_b, cfg.hidden_size // cfg.num_heads)
    sin = torch.zeros_like(cos)
    cu = torch.tensor([0, L_a, L_a + L_b], dtype=torch.int32)

    out1 = attn(h, cos, sin, cu_seqlens=cu)

    h2 = h.clone()
    h2[L_a:] = torch.randn_like(h2[L_a:]) * 10.0
    out2 = attn(h2, cos, sin, cu_seqlens=cu)
    # Image A (indices 0..L_a) must be untouched.
    torch.testing.assert_close(out1[:L_a], out2[:L_a], atol=1e-5, rtol=1e-5)


def test_attention_rejects_indivisible_head_count():
    cfg = _tiny_cfg(hidden_size=33, num_heads=4)
    with pytest.raises(ValueError, match="divisible"):
        Qwen3_5VisionAttention(cfg)


# --------------------------------------------------------------------------- #
# Block
# --------------------------------------------------------------------------- #


def test_vision_block_forward_shape():
    cfg = _tiny_cfg()
    block = Qwen3_5VisionBlock(cfg)
    N = 16
    h = torch.randn(N, cfg.hidden_size)
    cos = torch.ones(N, cfg.hidden_size // cfg.num_heads)
    sin = torch.zeros_like(cos)
    y = block(h, cos, sin)
    assert y.shape == (N, cfg.hidden_size)


# --------------------------------------------------------------------------- #
# Patch merger
# --------------------------------------------------------------------------- #


def test_patch_merger_output_dim_is_out_hidden_size():
    cfg = _tiny_cfg()
    pm = Qwen3_5VisionPatchMerger(cfg)
    # merger expects [N_groups, hidden] where N_groups already accounts for
    # the spatial_merge concatenation; the layer reshapes internally.
    n_groups = 4
    x = torch.randn(n_groups * (cfg.spatial_merge_size ** 2), cfg.hidden_size)
    y = pm(x)
    assert y.shape == (n_groups, cfg.out_hidden_size)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #


def test_vision_model_forward_shape_single_image():
    """Single 16×16 image at patch_size=4 → 4×4 patches → 4 merged tokens."""
    cfg = _tiny_cfg()
    model = Qwen3_5VisionModel(cfg).eval()
    t, h, w = 1, 4, 4
    n_patches = t * h * w
    pixels = torch.randn(n_patches, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.long)
    out = model(pixels, grid_thw)
    # After spatial_merge_size=2, we have (h/2)*(w/2) = 2*2 = 4 vision tokens.
    expected_n = (h // cfg.spatial_merge_size) * (w // cfg.spatial_merge_size) * t
    assert out.shape == (expected_n, cfg.out_hidden_size)


def test_vision_model_forward_shape_two_images():
    cfg = _tiny_cfg()
    model = Qwen3_5VisionModel(cfg).eval()
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    n_patches = int(grid.prod(dim=-1).sum().item())
    pixels = torch.randn(n_patches, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
    out = model(pixels, grid)
    # Two images, each producing 4 merged tokens.
    assert out.shape == (2 * 4, cfg.out_hidden_size)


def test_vision_model_rejects_non_square_position_embeddings():
    bad = _tiny_cfg(num_position_embeddings=63)              # √63 not integer
    with pytest.raises(ValueError, match="perfect square"):
        Qwen3_5VisionModel(bad)


def test_vision_model_gradient_flows():
    cfg = _tiny_cfg()
    model = Qwen3_5VisionModel(cfg)
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    pixels = torch.randn(
        16, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size,
        requires_grad=True,
    )
    out = model(pixels, grid)
    out.sum().backward()
    assert pixels.grad is not None
    assert pixels.grad.abs().sum() > 0
    assert model.patch_embed.proj.weight.grad is not None
    assert model.merger.fc2.weight.grad is not None


def test_vision_model_two_images_dont_leak_attention():
    """End-to-end version of the cu_seqlens isolation test: perturbing
    pixels in image B must not change image A's vision tokens.
    """
    cfg = _tiny_cfg()
    model = Qwen3_5VisionModel(cfg).eval()
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    n_per_image = 16
    pixels_a = torch.randn(n_per_image, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
    pixels_b = torch.randn(n_per_image, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
    pixels = torch.cat([pixels_a, pixels_b], dim=0)

    out1 = model(pixels, grid)
    pixels_b_alt = torch.randn_like(pixels_b) * 10.0
    pixels_alt = torch.cat([pixels_a, pixels_b_alt], dim=0)
    out2 = model(pixels_alt, grid)

    # First 4 vision tokens belong to image A — must be unchanged.
    torch.testing.assert_close(out1[:4], out2[:4], atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Embedding interleave
# --------------------------------------------------------------------------- #


def test_replace_placeholder_embeddings_swaps_only_placeholder_positions():
    B, T, H = 1, 6, 4
    inputs_embeds = torch.zeros(B, T, H)
    inputs_embeds.fill_(7.0)
    input_ids = torch.tensor([[10, 999, 999, 11, 999, 12]])           # placeholder = 999
    vision_embeds = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                                  [5.0, 6.0, 7.0, 8.0],
                                  [9.0, 10.0, 11.0, 12.0]])
    out = replace_placeholder_embeddings(
        inputs_embeds, input_ids, vision_embeds, placeholder_token_id=999,
    )
    # Non-placeholder positions stay at 7.0.
    assert out[0, 0].tolist() == [7.0, 7.0, 7.0, 7.0]
    assert out[0, 3].tolist() == [7.0, 7.0, 7.0, 7.0]
    assert out[0, 5].tolist() == [7.0, 7.0, 7.0, 7.0]
    # Placeholder positions filled in order.
    assert out[0, 1].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert out[0, 2].tolist() == [5.0, 6.0, 7.0, 8.0]
    assert out[0, 4].tolist() == [9.0, 10.0, 11.0, 12.0]


def test_replace_placeholder_rejects_count_mismatch():
    inputs_embeds = torch.zeros(1, 4, 2)
    input_ids = torch.tensor([[1, 99, 2, 99]])               # 2 placeholders
    vision_embeds = torch.zeros(3, 2)                         # 3 vision tokens
    with pytest.raises(ValueError, match="count"):
        replace_placeholder_embeddings(
            inputs_embeds, input_ids, vision_embeds, placeholder_token_id=99,
        )


def test_replace_placeholder_rejects_dim_mismatch():
    with pytest.raises(ValueError, match="last dim"):
        replace_placeholder_embeddings(
            torch.zeros(1, 2, 4), torch.tensor([[99, 99]]),
            torch.zeros(2, 8), placeholder_token_id=99,
        )
