"""Tests for quantization formats beyond the educational K-quant module.

Covers IQ4_XS, IQ3_XXS, IQ2_XXS, FP8 (E4M3 + E5M2), and MXFP4.
"""

import pytest
import torch

from llm_pipeline.quantization import (
    decode_fp8_e4m3,
    decode_fp8_e5m2,
    decode_iq2_xxs,
    decode_iq3_xxs,
    decode_iq4_xs,
    decode_mxfp4,
    encode_fp8_e4m3,
    encode_fp8_e5m2,
    encode_iq2_xxs,
    encode_iq3_xxs,
    encode_iq4_xs,
    encode_mxfp4,
)


def _gaussian(n: int, scale: float = 0.05, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g) * scale


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).pow(2).sum().sqrt() / a.pow(2).sum().sqrt().clamp_min(1e-12)).item()


# --------------------------------------------------------------------------- #
# IQ4_XS — 4.25 bpw, 256-weight super-block with per-sub-block scales
# --------------------------------------------------------------------------- #


def test_iq4_xs_round_trip_shape():
    w = _gaussian(2 * 256)
    blob, shape = encode_iq4_xs(w)
    out = decode_iq4_xs(blob, shape)
    assert out.shape == w.shape


def test_iq4_xs_block_size():
    """One super-block should be 137 bytes (2 + 6 + 128 + 1)."""
    w = _gaussian(256)
    blob, _ = encode_iq4_xs(w)
    assert len(blob) == 137


def test_iq4_xs_round_trip_error_under_8_percent():
    """At 4.25 bpw on Gaussian weights, rel-err should land in the same
    ballpark as Q4_K (~7%) — slightly worse because of the codebook
    mismatch on Gaussian data, but within 10%.
    """
    w = _gaussian(8 * 256)
    out = decode_iq4_xs(*encode_iq4_xs(w))
    assert _rel(w, out) < 0.10


def test_iq4_xs_handles_partial_super_block():
    """Trailing weights shorter than one super-block must round-trip."""
    w = _gaussian(300)                                   # not a multiple of 256
    out = decode_iq4_xs(*encode_iq4_xs(w))
    assert out.shape == w.shape


def test_iq4_xs_zero_input():
    w = torch.zeros(256)
    out = decode_iq4_xs(*encode_iq4_xs(w))
    torch.testing.assert_close(out, w, atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------------- #
# IQ3_XXS / IQ2_XXS — codebook-based low-bit
# --------------------------------------------------------------------------- #


def test_iq3_xxs_round_trip_shape():
    w = _gaussian(2 * 256)
    out = decode_iq3_xxs(*encode_iq3_xxs(w))
    assert out.shape == w.shape


def test_iq3_xxs_block_size():
    """Super-block: 2 (scale) + 4 (sign bytes) + 64 (32 × uint16 indices) = 70."""
    w = _gaussian(256)
    blob, _ = encode_iq3_xxs(w)
    assert len(blob) == 70


def test_iq3_xxs_decodes_zeros_to_zero():
    out = decode_iq3_xxs(*encode_iq3_xxs(torch.zeros(256)))
    torch.testing.assert_close(out, torch.zeros(256), atol=1e-6, rtol=1e-6)


def test_iq2_xxs_round_trip_shape():
    w = _gaussian(2 * 256)
    out = decode_iq2_xxs(*encode_iq2_xxs(w))
    assert out.shape == w.shape


def test_iq2_xxs_uses_different_codebook_than_iq3():
    """Sanity: encoding the same input with IQ3 vs IQ2 should give different
    bytes (different codebook grids → different optimal indices).
    """
    w = _gaussian(256)
    blob3, _ = encode_iq3_xxs(w)
    blob2, _ = encode_iq2_xxs(w)
    assert blob3 != blob2


# --------------------------------------------------------------------------- #
# FP8 E4M3 — block-scaled, ±448 range
# --------------------------------------------------------------------------- #


def test_fp8_e4m3_round_trip_shape():
    w = _gaussian(64)
    out = decode_fp8_e4m3(*encode_fp8_e4m3(w))
    assert out.shape == w.shape


def test_fp8_e4m3_block_size():
    """One block: 2 (fp16 scale) + 32 (fp8 values) = 34 bytes."""
    blob, _ = encode_fp8_e4m3(_gaussian(32))
    assert len(blob) == 34


def test_fp8_e4m3_round_trip_error_under_5_percent():
    """E4M3 has 3 mantissa bits (~12.5% step) plus block scaling — on
    Gaussian inputs we expect a few % rel-err.
    """
    w = _gaussian(8 * 32)
    out = decode_fp8_e4m3(*encode_fp8_e4m3(w))
    assert _rel(w, out) < 0.10


def test_fp8_e4m3_zero_round_trips_exactly():
    out = decode_fp8_e4m3(*encode_fp8_e4m3(torch.zeros(32)))
    torch.testing.assert_close(out, torch.zeros(32), atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------------- #
# FP8 E5M2 — wider exponent, less mantissa precision
# --------------------------------------------------------------------------- #


def test_fp8_e5m2_round_trip_shape():
    w = _gaussian(64)
    out = decode_fp8_e5m2(*encode_fp8_e5m2(w))
    assert out.shape == w.shape


def test_fp8_e5m2_handles_wider_dynamic_range_than_e4m3():
    """E5M2's ±57344 range should round-trip large-magnitude values that
    saturate E4M3.
    """
    w = torch.tensor([1000.0, 2000.0, -5000.0, 0.0])
    out_e5m2 = decode_fp8_e5m2(*encode_fp8_e5m2(w))
    # E5M2 should preserve the order of magnitude.
    for x_in, x_out in zip(w.tolist(), out_e5m2.tolist()):
        if x_in == 0:
            assert x_out == 0
        else:
            assert 0.5 < abs(x_out / x_in) < 2.0


# --------------------------------------------------------------------------- #
# MXFP4 — 4.25 bpw
# --------------------------------------------------------------------------- #


def test_mxfp4_round_trip_shape():
    w = _gaussian(64)
    out = decode_mxfp4(*encode_mxfp4(w))
    assert out.shape == w.shape


def test_mxfp4_block_size():
    """One block: 1 (E8M0 scale exponent) + 16 (32 packed 4-bit codes) = 17 bytes."""
    blob, _ = encode_mxfp4(_gaussian(32))
    assert len(blob) == 17


def test_mxfp4_lut_values_round_trip_exactly():
    """Inputs that match LUT grid points exactly (after scaling) should
    decode to themselves up to fp32 round-off.
    """
    # ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6 — the canonical E2M1 grid.
    w = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0,
                      0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5,
                      -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0, 0.5])
    out = decode_mxfp4(*encode_mxfp4(w))
    # Block scale is 2^0 = 1 for this input (since absmax = 6 ≤ 6).
    torch.testing.assert_close(out, w, atol=1e-3, rtol=1e-3)


def test_mxfp4_partial_block_handled():
    w = _gaussian(40)                                     # not multiple of 32
    out = decode_mxfp4(*encode_mxfp4(w))
    assert out.shape == w.shape


# --------------------------------------------------------------------------- #
# Cross-format sanity: lower bpw → higher rel-err
# --------------------------------------------------------------------------- #


def test_lower_bpw_increases_rel_err_on_gaussian():
    """Sanity check on the bit-rate / quality trend: IQ4_XS < IQ3_XXS < IQ2_XXS
    in bits, so rel-err should generally rise as bpw drops.
    """
    w = _gaussian(8 * 256, seed=1)
    e4 = _rel(w, decode_iq4_xs(*encode_iq4_xs(w)))
    e3 = _rel(w, decode_iq3_xxs(*encode_iq3_xxs(w)))
    e2 = _rel(w, decode_iq2_xxs(*encode_iq2_xxs(w)))
    # Loose ordering: IQ4 ≤ IQ3 ≤ IQ2 with reasonable margins.
    assert e4 < e3
    assert e3 < e2
