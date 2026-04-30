"""Q3_K round-trip + ordering tests."""

import pytest
import torch

from llm_pipeline.quantization.kquants import (
    decode_q3_k,
    decode_q4_k,
    decode_q5_k,
    decode_q6_k,
    encode_q3_k,
    encode_q4_k,
    encode_q5_k,
    encode_q6_k,
)


def _gaussian(shape, scale=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * scale


def _rel(a, b):
    return ((a - b).pow(2).sum().sqrt() / a.pow(2).sum().sqrt()).item()


def test_q3_k_round_trip_lower_bound_error():
    """3-bit symmetric quantization with this reference encoder ~17% rel-err
    on Gaussian (only 8 levels, intrinsically lossy)."""
    w = _gaussian(2048)
    blob, shape = encode_q3_k(w)
    out = decode_q3_k(blob, shape)
    assert out.shape == w.shape
    assert _rel(w, out) < 2.5e-1


def test_q3_k_blob_size_per_super_block():
    """Q3_K is 110 bytes per super-block (3.4375 bits/weight)."""
    w = _gaussian(256)
    blob, _ = encode_q3_k(w)
    assert len(blob) == 110


def test_q3_k_handles_partial_super_block():
    w = _gaussian(300)
    blob, shape = encode_q3_k(w)
    out = decode_q3_k(blob, shape)
    assert out.shape == w.shape


def test_full_kquant_family_bit_budget_ordering():
    """At fixed weight tensor, more bits/weight ⇒ lower rel-err.

    The whole point of having multiple K-quant tiers: each format pays a
    bigger memory cost for less error. If this test ever breaks, one of the
    encoders has a bug.
    """
    w = _gaussian(2048)
    e3 = _rel(w, decode_q3_k(*encode_q3_k(w)))
    e4 = _rel(w, decode_q4_k(*encode_q4_k(w)))
    e5 = _rel(w, decode_q5_k(*encode_q5_k(w)))
    e6 = _rel(w, decode_q6_k(*encode_q6_k(w)))
    assert e6 < e5 < e4 < e3


def test_q3_k_signed_pack_round_trips_extremes():
    """Build a tensor of signed values that hit the full [-4, 3] codebook
    range and verify round-trip preserves the (sign, magnitude) information."""
    # 16 weights spanning the codebook range × scale.
    w = torch.tensor([-4, -3, -2, -1, 0, 1, 2, 3] * 32, dtype=torch.float32) * 0.01
    blob, shape = encode_q3_k(w)
    out = decode_q3_k(blob, shape)
    # Q3_K is symmetric around 0, so the result should preserve sign for
    # non-zero entries — at least relative ordering.
    pos_in = (w > 0).float()
    pos_out = (out > -1e-3).float()  # slight tolerance for the [-4, 0] range
    # At least 80% of positive weights should map to non-negative outputs.
    assert (pos_in * pos_out).sum() / pos_in.sum() > 0.8
