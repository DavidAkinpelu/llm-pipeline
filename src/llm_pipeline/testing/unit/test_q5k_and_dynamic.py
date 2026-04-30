"""Q5_K round-trip + UDQuantizer probe + assignment unit tests."""

import math

import pytest
import torch
import torch.nn as nn

from llm_pipeline.quantization import (
    DEFAULT_CANDIDATES,
    QuantMethod,
    SensitivityReport,
    UDQuantizer,
    UDQuantizerConfig,
    decode_q5_k,
    encode_q5_k,
    encode_q4_k, decode_q4_k,
    encode_q6_k, decode_q6_k,
)


def _gaussian(shape, scale=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * scale


def _rel(a, b):
    return ((a - b).pow(2).sum().sqrt() / a.pow(2).sum().sqrt()).item()


# --------------------------------------------------------------------------- #
# Q5_K
# --------------------------------------------------------------------------- #


def test_q5_k_round_trip_low_error():
    """5-bit reference encoder lands around 4% rel-err — between Q4_K and Q6_K."""
    w = _gaussian(2048)
    blob, shape = encode_q5_k(w)
    out = decode_q5_k(blob, shape)
    assert out.shape == w.shape
    assert _rel(w, out) < 6e-2


def test_q5_k_blob_size_per_super_block():
    w = _gaussian(256)
    blob, _ = encode_q5_k(w)
    assert len(blob) == 176


def test_q5_k_between_q4_k_and_q6_k():
    """Bit budget should determine the error ordering: Q6 < Q5 < Q4."""
    w = _gaussian(2048)
    e4 = _rel(w, decode_q4_k(*encode_q4_k(w)))
    e5 = _rel(w, decode_q5_k(*encode_q5_k(w)))
    e6 = _rel(w, decode_q6_k(*encode_q6_k(w)))
    assert e6 < e5 < e4


def test_q5_k_handles_partial_super_block():
    w = _gaussian(300)
    blob, shape = encode_q5_k(w)
    out = decode_q5_k(blob, shape)
    assert out.shape == w.shape
    assert _rel(w, out) < 8e-2


# --------------------------------------------------------------------------- #
# UDQuantizer
# --------------------------------------------------------------------------- #


class _TinyLM(nn.Module):
    """Stand-in for an HF LM that emits ``.logits`` from input_ids."""
    def __init__(self, vocab=32, d=64, layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(layers)])
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids=None, **_):
        x = self.embed_tokens(input_ids)
        for l in self.layers:
            x = torch.nn.functional.relu(l(x))
        logits = self.lm_head(x)
        return type("O", (), {"logits": logits})()


def test_udquantizer_probe_produces_kl_per_tensor_and_format():
    torch.manual_seed(0)
    m = _TinyLM().eval()
    batches = [{"input_ids": torch.randint(0, 32, (2, 8))}]
    cfg = UDQuantizerConfig(
        candidates=(QuantMethod.Q4_K, QuantMethod.Q6_K),
        kl_budget=1e-2,
        xl=True,
    )
    ud = UDQuantizer(cfg)
    report = ud.probe(m, batches)
    # We expect at least the two layers + lm_head probed; embedding is 1D-Embed
    # weight (still 2D). The probe should visit every 2D param ≥ 64 elements.
    assert any("layers.0" in k for k in report.kl)
    for name, kls in report.kl.items():
        for method in (QuantMethod.Q4_K, QuantMethod.Q6_K):
            assert method in kls
            assert kls[method] >= 0
            # Q6_K should be at most as bad as Q4_K (more bits ⇒ less perturbation).
        assert kls[QuantMethod.Q6_K] <= kls[QuantMethod.Q4_K] + 1e-6


def test_udquantizer_xl_pins_embed_and_lm_head_to_q8_k():
    torch.manual_seed(0)
    m = _TinyLM().eval()
    cfg = UDQuantizerConfig(
        candidates=(QuantMethod.Q4_K, QuantMethod.Q6_K),
        kl_budget=1e-2,
        xl=True,
    )
    ud = UDQuantizer(cfg)
    report = ud.probe(m, [{"input_ids": torch.randint(0, 32, (2, 8))}])
    assignments = ud.assign_formats(report, m)
    assert assignments["embed_tokens.weight"] == QuantMethod.Q8_K
    assert assignments["lm_head.weight"] == QuantMethod.Q8_K


def test_udquantizer_picks_cheapest_under_budget():
    torch.manual_seed(0)
    m = _TinyLM().eval()
    # Stub a SensitivityReport directly so we don't depend on float arithmetic.
    report = SensitivityReport()
    candidates = (QuantMethod.Q4_K, QuantMethod.Q5_K, QuantMethod.Q6_K)
    # layers.0 is OK at Q4_K; layers.1 needs Q6_K.
    report.kl["layers.0.weight"] = {QuantMethod.Q4_K: 1e-3, QuantMethod.Q5_K: 5e-4, QuantMethod.Q6_K: 1e-4}
    report.kl["layers.1.weight"] = {QuantMethod.Q4_K: 9e-2, QuantMethod.Q5_K: 5e-2, QuantMethod.Q6_K: 1e-3}

    ud = UDQuantizer(UDQuantizerConfig(candidates=candidates, kl_budget=2e-3, xl=False))
    a = ud.assign_formats(report, m)
    assert a["layers.0.weight"] == QuantMethod.Q4_K  # cheapest under 2e-3
    assert a["layers.1.weight"] == QuantMethod.Q6_K  # only one under 2e-3


def test_udquantizer_falls_back_to_most_precise_when_no_format_is_acceptable():
    """If every probed format exceeds the budget, the choice should be the
    most-precise candidate — bounded degradation rather than catastrophic."""
    torch.manual_seed(0)
    m = _TinyLM().eval()
    report = SensitivityReport()
    candidates = (QuantMethod.Q4_K, QuantMethod.Q6_K)
    report.kl["layers.0.weight"] = {QuantMethod.Q4_K: 0.5, QuantMethod.Q6_K: 0.4}

    ud = UDQuantizer(UDQuantizerConfig(candidates=candidates, kl_budget=1e-3, xl=False))
    a = ud.assign_formats(report, m)
    assert a["layers.0.weight"] == QuantMethod.Q6_K
