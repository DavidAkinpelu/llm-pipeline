"""Unit tests for outlier-aware Q4_K (``Q4_K_OUT``)."""

import pytest
import torch

from llm_pipeline.quantization import (
    QuantMethod,
    Quantizer,
    decode_q4_k,
    decode_q4_k_out,
    encode_q4_k,
    encode_q4_k_out,
)


def _gaussian(shape, scale=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * scale


def _rel(a, b):
    return ((a - b).pow(2).sum().sqrt() / a.pow(2).sum().sqrt()).item()


def test_q4_k_out_round_trip_one_super_block():
    w = _gaussian(256)
    blob, shape = encode_q4_k_out(w, outlier_k=3)
    out = decode_q4_k_out(blob, shape, outlier_k=3)
    assert out.shape == w.shape
    # Should be at least as good as Q4_K (typically a hair better even on Gaussian).
    e_out = _rel(w, out)
    e_q4 = _rel(w, decode_q4_k(*encode_q4_k(w)))
    assert e_out <= e_q4 + 1e-5


def test_q4_k_out_block_size_matches_outlier_k():
    """Block size = 144 + 3*K bytes per 256 weights."""
    w = _gaussian(256)
    for k, expected_bytes in [(0, 144), (3, 153), (8, 168), (16, 192)]:
        blob, _ = encode_q4_k_out(w, outlier_k=k)
        assert len(blob) == expected_bytes, f"K={k}: got {len(blob)}, expected {expected_bytes}"


def test_q4_k_out_recovers_outliers_exactly():
    """Outlier positions should round-trip bit-exactly to FP16 precision."""
    torch.manual_seed(0)
    w = torch.randn(256) * 0.05
    # Hand-pick outliers far from the bulk so they get selected.
    w[10] = 10.0
    w[100] = -8.0
    w[200] = 5.5
    out = decode_q4_k_out(*encode_q4_k_out(w, outlier_k=3))
    assert (out[10] - w[10]).abs() < 1e-3
    assert (out[100] - w[100]).abs() < 1e-3
    assert (out[200] - w[200]).abs() < 1e-3


def test_q4_k_out_dramatically_better_on_heavy_tailed():
    """Outlier-aware should be much better than vanilla Q4_K on weights
    where ~1% of values are 20σ outliers."""
    torch.manual_seed(0)
    w = torch.randn(2048) * 0.05
    outlier_idx = torch.randperm(2048)[: 2048 // 100]
    w[outlier_idx] += 1.0   # 20σ deviation

    e_q4k = _rel(w, decode_q4_k(*encode_q4_k(w)))
    e_q4k_out = _rel(w, decode_q4_k_out(*encode_q4_k_out(w, outlier_k=3)))
    # The improvement on heavy-tailed should be at least 30%.
    assert (e_q4k - e_q4k_out) / e_q4k > 0.3


def test_q4_k_out_handles_partial_super_block():
    w = _gaussian(300)
    blob, shape = encode_q4_k_out(w, outlier_k=3)
    out = decode_q4_k_out(blob, shape, outlier_k=3)
    assert out.shape == w.shape


def test_q4_k_out_outlier_k_zero_matches_vanilla_q4_k():
    """K=0 must encode/decode identically to vanilla Q4_K."""
    w = _gaussian(2048)
    out_zero = decode_q4_k_out(*encode_q4_k_out(w, outlier_k=0), outlier_k=0)
    out_q4k = decode_q4_k(*encode_q4_k(w))
    assert torch.equal(out_zero, out_q4k)


def test_q4_k_out_validates_outlier_k_range():
    w = _gaussian(256)
    with pytest.raises(ValueError, match="outlier_k must be in"):
        encode_q4_k_out(w, outlier_k=-1)
    with pytest.raises(ValueError, match="outlier_k must be in"):
        encode_q4_k_out(w, outlier_k=256)


def test_q4_k_out_via_quantizer_facade():
    """Q4_K_OUT must be wired into the unified Quantizer dispatch."""
    q = Quantizer(method=QuantMethod.Q4_K_OUT)
    w = _gaussian(2048)
    blob, shape = q.encode(w)
    out = q.decode(blob, shape, method=QuantMethod.Q4_K_OUT)
    assert _rel(w, out) < 0.1
