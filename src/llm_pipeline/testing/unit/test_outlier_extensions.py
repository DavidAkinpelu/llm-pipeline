"""Round-trip tests for the outlier-aware Q3_K/Q5_K/Q6_K formats.

Q4_K_OUT is covered by ``test_q4_k_out.py`` already; this file checks
that the same outlier trick generalises cleanly to the other K-quant
formats — the value gain on heavy-tailed weights should be roughly the
same percentage reduction across the family.
"""

import pytest
import torch

from llm_pipeline.quantization import (
    QuantMethod,
    Quantizer,
    decode_q3_k, decode_q3_k_out,
    decode_q4_k,
    decode_q5_k, decode_q5_k_out,
    decode_q6_k, decode_q6_k_out,
    encode_q3_k, encode_q3_k_out,
    encode_q4_k,
    encode_q5_k, encode_q5_k_out,
    encode_q6_k, encode_q6_k_out,
)


def _gaussian(shape, scale=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * scale


def _heavy_tailed(shape, seed=0):
    """Gaussian + 1% outliers at 20σ."""
    g = torch.Generator().manual_seed(seed)
    n = int(shape if isinstance(shape, int) else shape[0])
    w = torch.randn(n, generator=g) * 0.05
    g2 = torch.Generator().manual_seed(seed + 1)
    n_outliers = max(1, n // 100)
    idx = torch.randperm(n, generator=g2)[:n_outliers]
    w[idx] += 1.0
    return w


def _rel(a, b):
    return ((a - b).pow(2).sum().sqrt() / a.pow(2).sum().sqrt()).item()


# --------------------------------------------------------------------------- #
# Round-trip on Gaussian
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("encode_fn,decode_fn,bound", [
    (encode_q3_k_out, decode_q3_k_out, 0.25),
    (encode_q5_k_out, decode_q5_k_out, 0.07),
    (encode_q6_k_out, decode_q6_k_out, 0.04),
])
def test_outlier_round_trip_gaussian(encode_fn, decode_fn, bound):
    w = _gaussian(2048)
    out = decode_fn(*encode_fn(w, outlier_k=3), outlier_k=3)
    assert _rel(w, out) < bound


# --------------------------------------------------------------------------- #
# Block-size sanity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("encode_fn,base_bytes", [
    (encode_q3_k_out, 110),
    (encode_q5_k_out, 176),
    (encode_q6_k_out, 210),
])
def test_outlier_block_sizes(encode_fn, base_bytes):
    """Block size = base + 3*K bytes per 256 weights."""
    w = _gaussian(256)
    for k, expected in [(0, base_bytes), (3, base_bytes + 9), (8, base_bytes + 24)]:
        blob, _ = encode_fn(w, outlier_k=k)
        assert len(blob) == expected, f"K={k}: got {len(blob)}, expected {expected}"


# --------------------------------------------------------------------------- #
# Heavy-tailed: outlier-aware should be MUCH better than vanilla
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("base_e,base_d,out_e,out_d", [
    (encode_q3_k, decode_q3_k, encode_q3_k_out, decode_q3_k_out),
    (encode_q5_k, decode_q5_k, encode_q5_k_out, decode_q5_k_out),
    (encode_q6_k, decode_q6_k, encode_q6_k_out, decode_q6_k_out),
])
def test_outlier_helps_heavy_tailed(base_e, base_d, out_e, out_d):
    w = _heavy_tailed(2048)
    e_base = _rel(w, base_d(*base_e(w)))
    e_out = _rel(w, out_d(*out_e(w, outlier_k=3), outlier_k=3))
    # Improvement should be at least 30% relative reduction on heavy-tailed.
    assert (e_base - e_out) / e_base > 0.3, (
        f"base={e_base:.4f} out={e_out:.4f} — outlier should reduce error ≥30%"
    )


# --------------------------------------------------------------------------- #
# Outlier positions recovered exactly (FP16 storage rounding only)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("encode_fn,decode_fn", [
    (encode_q3_k_out, decode_q3_k_out),
    (encode_q5_k_out, decode_q5_k_out),
    (encode_q6_k_out, decode_q6_k_out),
])
def test_outlier_positions_recovered_bit_exact_fp16(encode_fn, decode_fn):
    torch.manual_seed(0)
    w = torch.randn(256) * 0.05
    w[10] = 10.0
    w[100] = -8.0
    w[200] = 5.5
    out = decode_fn(*encode_fn(w, outlier_k=3), outlier_k=3)
    # Outlier positions: bit-exact to FP16 storage.
    assert (out[10] - w[10]).abs() < 1e-3
    assert (out[100] - w[100]).abs() < 1e-3
    assert (out[200] - w[200]).abs() < 1e-3


# --------------------------------------------------------------------------- #
# K=0 fallback to vanilla format
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("base_e,base_d,out_e,out_d", [
    (encode_q3_k, decode_q3_k, encode_q3_k_out, decode_q3_k_out),
    (encode_q5_k, decode_q5_k, encode_q5_k_out, decode_q5_k_out),
    (encode_q6_k, decode_q6_k, encode_q6_k_out, decode_q6_k_out),
])
def test_outlier_k_zero_matches_vanilla(base_e, base_d, out_e, out_d):
    """With K=0, the outlier-aware encoder must produce identical output to vanilla."""
    w = _gaussian(2048)
    out_zero = out_d(*out_e(w, outlier_k=0), outlier_k=0)
    out_vanilla = base_d(*base_e(w))
    assert torch.equal(out_zero, out_vanilla)


# --------------------------------------------------------------------------- #
# Quantizer facade dispatch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method,bound", [
    (QuantMethod.Q3_K_OUT, 0.25),
    (QuantMethod.Q5_K_OUT, 0.07),
    (QuantMethod.Q6_K_OUT, 0.04),
])
def test_outlier_via_quantizer_facade(method, bound):
    q = Quantizer(method=method)
    w = _gaussian(2048)
    blob, shape = q.encode(w)
    out = q.decode(blob, shape, method=method)
    assert _rel(w, out) < bound
