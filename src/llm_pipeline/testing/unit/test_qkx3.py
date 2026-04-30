"""Tests for ``make_qkx3_quants`` — iterative refinement on top of qkx2.

qkx3 starts from the qkx2 fit and alternates closed-form (scale, min) re-solves
with re-derived integer levels until levels stop changing. The closed-form
solve in qkx2 is optimal *given* its rounded levels; once the (scale, min)
shifts, the optimal levels shift too, leaving a small amount of error on the
table that qkx3 picks up.

Empirically the gain is small but consistent: ~0.3–0.5% L2 reduction on
Gaussian-ish weights, ~1.5–3% on heavy-tailed weights (where qkx2's outer
``iscale`` candidate search is less optimal).
"""

import numpy as np
import pytest
import torch

from llm_pipeline.quantization.kquants.q4_k import (
    _fit_sub_blocks_qkx2_batched,
    _fit_sub_blocks_qkx3_batched,
    decode_q4_k,
    encode_q4_k,
)
from llm_pipeline.quantization.kquants.q5_k import decode_q5_k, encode_q5_k


def _gaussian(n_blocks, scale=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_blocks * 256, generator=g) * scale


def _heavy_tailed(n_blocks, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = n_blocks * 256
    w = torch.randn(n, generator=g) * 0.05
    g2 = torch.Generator().manual_seed(seed + 1)
    idx = torch.randperm(n, generator=g2)[: n // 100]
    w[idx] += 1.0
    return w


def _rel(a, b):
    return ((a - b).pow(2).sum().sqrt() / a.pow(2).sum().sqrt()).item()


# --------------------------------------------------------------------------- #
# qkx3 ≤ qkx2 by construction (refinement is gated on improvement)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_qkx3_never_worse_than_qkx2_per_subblock(seed):
    """At the per-sub-block level, qkx3 only updates a row if its weighted
    MSE strictly improves on qkx2's seed — so for every row, MSE(qkx3) ≤ MSE(qkx2).
    """
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((64, 32)).astype(np.float32) * 0.05)
    w = np.ones_like(x)

    s2, m2, L2 = _fit_sub_blocks_qkx2_batched(x, w)
    s3, m3, L3 = _fit_sub_blocks_qkx3_batched(x, w)

    # Reconstruct each row's per-weight MSE.
    rec2 = s2[:, None] * L2.astype(np.float32) - m2[:, None]
    rec3 = s3[:, None] * L3.astype(np.float32) - m3[:, None]
    mse2 = ((rec2 - x) ** 2).sum(axis=1)
    mse3 = ((rec3 - x) ** 2).sum(axis=1)

    # Per-row: qkx3 is ≤ qkx2 (small float-noise tolerance).
    assert np.all(mse3 <= mse2 + 1e-7), (mse3 - mse2).max()


# --------------------------------------------------------------------------- #
# End-to-end: qkx3 reduces tensor-level rel-err
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_qkx3_improves_q4_k_gaussian(seed):
    w = _gaussian(64, seed=seed)
    e2 = _rel(w, decode_q4_k(*encode_q4_k(w)))
    e3 = _rel(w, decode_q4_k(*encode_q4_k(w, use_qkx3=True)))
    # Improvement is small but should be strictly positive on Gaussian.
    assert e3 < e2, f"qkx3 ({e3:.4%}) did not improve over qkx2 ({e2:.4%})"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_qkx3_improves_q4_k_heavy_tailed(seed):
    w = _heavy_tailed(64, seed=seed)
    e2 = _rel(w, decode_q4_k(*encode_q4_k(w)))
    e3 = _rel(w, decode_q4_k(*encode_q4_k(w, use_qkx3=True)))
    # Heavy-tailed: qkx2's iscale candidate set is less optimal because the
    # outliers stretch the dynamic range; qkx3 typically claws back ~1-3%.
    assert (e2 - e3) / e2 > 0.005, (
        f"qkx3 ({e3:.4%}) only beat qkx2 ({e2:.4%}) by {(e2-e3)/e2:.2%} — expected ≥ 0.5%"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_qkx3_improves_q5_k_gaussian(seed):
    w = _gaussian(64, seed=seed)
    e2 = _rel(w, decode_q5_k(*encode_q5_k(w)))
    e3 = _rel(w, decode_q5_k(*encode_q5_k(w, use_qkx3=True)))
    assert e3 < e2, f"qkx3 ({e3:.4%}) did not improve over qkx2 ({e2:.4%})"


# --------------------------------------------------------------------------- #
# qkx3 produces wire-compatible bytes (same decoder, only the encoder differs)
# --------------------------------------------------------------------------- #


def test_qkx3_output_decodes_with_standard_decoder():
    """The qkx3 encoder writes the same byte layout as qkx2; the standard
    Q4_K / Q5_K decoders must work on its output without changes.
    """
    w = _gaussian(8, seed=42)

    blob, shape = encode_q4_k(w, use_qkx3=True)
    assert len(blob) == 8 * 144  # standard Q4_K block size
    out = decode_q4_k(blob, shape)
    assert out.shape == w.shape

    blob, shape = encode_q5_k(w, use_qkx3=True)
    assert len(blob) == 8 * 176  # standard Q5_K block size
    out = decode_q5_k(blob, shape)
    assert out.shape == w.shape


# --------------------------------------------------------------------------- #
# Degenerate inputs — must not regress qkx2's behavior
# --------------------------------------------------------------------------- #


def test_qkx3_handles_constant_and_zero_subblocks():
    """Degenerate inputs: all-zero rows (truly degenerate) and constant
    nonzero rows (rng > 0 but every weight is identical). qkx3 must reconstruct
    both cleanly without overflow / NaN.
    """
    x = np.zeros((2, 32), dtype=np.float32)        # truly degenerate
    x = np.concatenate([x, np.full((2, 32), 0.05, dtype=np.float32)], axis=0)
    w = np.ones_like(x)
    s, m, L = _fit_sub_blocks_qkx3_batched(x, w)
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(m))
    rec = s[:, None] * L.astype(np.float32) - m[:, None]
    np.testing.assert_allclose(rec, x, atol=1e-6)


def test_qkx3_n_refine_zero_falls_back_to_qkx2():
    """``n_refine=0`` should bypass the iterative refinement and produce
    bit-identical output to qkx2.
    """
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((16, 32)) * 0.05).astype(np.float32)
    w = np.ones_like(x)
    s2, m2, L2 = _fit_sub_blocks_qkx2_batched(x, w)
    s3, m3, L3 = _fit_sub_blocks_qkx3_batched(x, w, n_refine=0)
    np.testing.assert_array_equal(s2, s3)
    np.testing.assert_array_equal(m2, m3)
    np.testing.assert_array_equal(L2, L3)
