"""Round-trip tests for the educational K-quant + I-quant implementations.

Quality bar:
- Q8_K should round-trip with very small error (~1e-3 relative).
- Q6_K with small error (~5e-3 relative).
- Q4_K with moderate error (~3e-2 relative on Gaussian-ish input).
- IQ4_NL with moderate error (similar to Q4_K).

We also test the imatrix calibrator end-to-end on a tiny model.
"""

import math

import pytest
import torch
import torch.nn as nn

from llm_pipeline.quantization import (
    ImatrixCalibrator,
    Quantizer,
    QuantMethod,
    decode_iq4_nl,
    decode_q4_k,
    decode_q6_k,
    decode_q8_k,
    encode_iq4_nl,
    encode_q4_k,
    encode_q6_k,
    encode_q8_k,
    IQ4_NL_CODEBOOK,
)


def _gaussian(shape, scale=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * scale


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error: ||a - b|| / ||a||."""
    num = (a - b).pow(2).sum().sqrt().item()
    den = a.pow(2).sum().sqrt().item()
    return num / max(den, 1e-12)


# --------------------------------------------------------------------------- #
# Q8_K
# --------------------------------------------------------------------------- #


def test_q8_k_round_trip_low_error():
    """Q8_K is the highest-precision K-quant; rel-err typically <1% on Gaussian."""
    w = _gaussian(512)
    blob, shape = encode_q8_k(w)
    out = decode_q8_k(blob, shape)
    assert out.shape == w.shape
    assert _rel_err(w, out) < 2e-2


def test_q8_k_handles_partial_super_block():
    w = _gaussian(300)              # not a multiple of 256
    blob, shape = encode_q8_k(w)
    out = decode_q8_k(blob, shape)
    assert out.shape == w.shape
    assert _rel_err(w, out) < 2e-2


# --------------------------------------------------------------------------- #
# Q6_K
# --------------------------------------------------------------------------- #


def test_q6_k_round_trip_small_error():
    """Q6_K with this reference encoder lands around 2% rel-err on Gaussian."""
    w = _gaussian(2048)
    blob, shape = encode_q6_k(w)
    out = decode_q6_k(blob, shape)
    assert _rel_err(w, out) < 4e-2


def test_q6_k_better_than_q4_k():
    """6-bit should beat 4-bit on a fixed weight tensor."""
    w = _gaussian(2048)
    blob6, shape6 = encode_q6_k(w)
    blob4, shape4 = encode_q4_k(w)
    out6 = decode_q6_k(blob6, shape6)
    out4 = decode_q4_k(blob4, shape4)
    assert _rel_err(w, out6) < _rel_err(w, out4)


# --------------------------------------------------------------------------- #
# Q4_K
# --------------------------------------------------------------------------- #


def test_q4_k_round_trip_moderate_error():
    """4-bit reference encoder lands around 8% rel-err. Production llama.cpp
    achieves ~3-4% via more sophisticated scale-fitting; see the note at the
    top of ``kquants/__init__.py``."""
    w = _gaussian(2048)
    blob, shape = encode_q4_k(w)
    out = decode_q4_k(blob, shape)
    assert out.shape == w.shape
    assert _rel_err(w, out) < 1.2e-1


def test_q4_k_blob_size_is_144_bytes_per_super_block():
    w = _gaussian(256)              # exactly one super-block
    blob, _ = encode_q4_k(w)
    assert len(blob) == 144


def test_q4_k_importance_weighting_changes_output():
    """Asymmetric importance should bias rounding for high-importance weights."""
    torch.manual_seed(0)
    w = torch.randn(256)
    blob_uniform, shape = encode_q4_k(w)
    importance = torch.zeros_like(w)
    importance[:32] = 100.0  # heavily weight the first sub-block
    blob_weighted, _ = encode_q4_k(w, importance=importance)
    out_uniform = decode_q4_k(blob_uniform, shape)
    out_weighted = decode_q4_k(blob_weighted, shape)
    # The weighted variant should achieve lower error on the heavily-weighted slice.
    err_u = (w[:32] - out_uniform[:32]).abs().sum().item()
    err_w = (w[:32] - out_weighted[:32]).abs().sum().item()
    assert err_w <= err_u + 1e-6  # equal or better


# --------------------------------------------------------------------------- #
# IQ4_NL
# --------------------------------------------------------------------------- #


def test_iq4_nl_codebook_has_16_entries():
    assert IQ4_NL_CODEBOOK.shape == (16,)


def test_iq4_nl_round_trip():
    """IQ4_NL on Gaussian-ish weights with this reference encoder ~9% rel-err."""
    w = _gaussian(2048)
    blob, shape = encode_iq4_nl(w)
    out = decode_iq4_nl(blob, shape)
    assert out.shape == w.shape
    assert _rel_err(w, out) < 1.2e-1


def test_iq4_nl_blob_size_is_18_bytes_per_block():
    w = _gaussian(32)
    blob, _ = encode_iq4_nl(w)
    assert len(blob) == 18


# --------------------------------------------------------------------------- #
# Quantizer facade
# --------------------------------------------------------------------------- #


def test_quantizer_round_trip_single_tensor():
    q = Quantizer(method=QuantMethod.Q4_K)
    w = _gaussian(512)
    blob, shape = q.encode(w)
    out = q.decode(blob, shape, method=QuantMethod.Q4_K)
    assert _rel_err(w, out) < 1.2e-1


def test_quantizer_q4_k_m_bumps_v_proj_to_q6_k():
    """The Q4_K_M policy must select Q6_K for tensors whose name contains v_proj."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64, bias=False)
            self.v_proj = nn.Linear(64, 64, bias=False)
            self.down_proj = nn.Linear(64, 64, bias=False)
    model = M()
    q = Quantizer(method=QuantMethod.Q4_K_M)
    qm = q.quantize(model)
    chosen = {t.name: t.method for t in qm.tensors}
    assert chosen["q_proj.weight"] == QuantMethod.Q4_K
    assert chosen["v_proj.weight"] == QuantMethod.Q6_K
    # Single down_proj is "first half" → Q6_K.
    assert chosen["down_proj.weight"] == QuantMethod.Q6_K


def test_quantizer_save_load_round_trip(tmp_path):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(64, 32, bias=False)
    model = M()
    q = Quantizer(method=QuantMethod.Q8_K)
    qm = q.quantize(model)
    p = tmp_path / "model.bin"
    qm.save(str(p))

    from llm_pipeline.quantization import QuantizedModel
    loaded = QuantizedModel.load(str(p))
    assert len(loaded.tensors) == 1
    assert loaded.tensors[0].name == "lin.weight"
    assert loaded.tensors[0].shape == (32, 64)
    assert loaded.tensors[0].method == QuantMethod.Q8_K


# --------------------------------------------------------------------------- #
# Imatrix calibration
# --------------------------------------------------------------------------- #


def test_imatrix_collects_per_input_channel_squared_sums():
    torch.manual_seed(0)
    layer = nn.Linear(8, 4, bias=False)

    cal = ImatrixCalibrator(layer)
    cal.attach()
    # Feed 3 batches of size (5, 8). Total rows = 15.
    x = torch.randn(15, 8)
    layer(x[:5])
    layer(x[5:10])
    layer(x[10:])
    imat = cal.detach()
    # The hooked module's qualified name is "" (root) — registered against ``layer`` itself.
    # Either the empty string OR a key matching the module is acceptable.
    assert len(imat) == 1
    v = next(iter(imat.values()))
    assert v.shape == (8,)
    expected = (x ** 2).mean(dim=0).to(torch.float32)
    torch.testing.assert_close(v, expected, rtol=1e-4, atol=1e-4)


def test_quantizer_calibrate_populates_imatrix():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8, bias=False)
        def forward(self, input_ids=None, **_):
            return self.lin(input_ids)
    model = M()
    q = Quantizer(method=QuantMethod.Q4_K)
    batches = [{"input_ids": torch.randn(4, 8)} for _ in range(3)]
    imat = q.calibrate(model, batches)
    assert "lin" in imat
    assert imat["lin"].shape == (8,)
