"""Tests for speculative decoding (generic two-model + MTP-driven variants)."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_pipeline.inference.speculative import (
    SpecDecStats,
    mtp_speculative_decode,
    speculative_decode,
)


VOCAB = 32


# --------------------------------------------------------------------------- #
# Toy LMs for the tests — simple enough to predict behaviour, real enough to
# exercise every code path of the speculative loop.
# --------------------------------------------------------------------------- #


def _greedy_target(seq: torch.Tensor) -> torch.Tensor:
    """Deterministic LM that always predicts ``(last_token + 1) % VOCAB``.

    Returns ``[1, T, V]`` logits with a +inf at the predicted token at every
    position. With temperature=0 the model is fully deterministic, which is
    what we want for the equivalence tests.
    """
    B, T = seq.shape
    logits = torch.full((B, T, VOCAB), -1e9)
    targets = (seq + 1) % VOCAB
    logits.scatter_(-1, targets.unsqueeze(-1), 1e9)
    return logits


def _greedy_perfect_draft(seq: torch.Tensor) -> torch.Tensor:
    """Same prediction rule as ``_greedy_target`` — a perfect oracle."""
    return _greedy_target(seq)


def _greedy_bad_draft(seq: torch.Tensor) -> torch.Tensor:
    """Always predicts ``last_token + 13`` — different from the target's
    ``last_token + 1`` for every token. Forces every draft to be rejected.
    """
    B, T = seq.shape
    logits = torch.full((B, T, VOCAB), -1e9)
    targets = (seq + 13) % VOCAB
    logits.scatter_(-1, targets.unsqueeze(-1), 1e9)
    return logits


# --------------------------------------------------------------------------- #
# Generic speculative_decode
# --------------------------------------------------------------------------- #


def test_speculative_decode_shape_and_length():
    prompt = torch.tensor([[5, 7]])
    out, stats = speculative_decode(
        _greedy_target, _greedy_perfect_draft,
        prompt_ids=prompt, max_new_tokens=8, k=4, temperature=0.0,
    )
    assert out.dim() == 2
    assert out.shape[0] == 1
    assert out.shape[1] >= 8


def test_speculative_decode_perfect_draft_matches_target_greedy():
    """With a perfect draft and greedy temperature, the output must match
    plain target-only greedy decoding."""
    prompt = torch.tensor([[3]])
    spec_out, _ = speculative_decode(
        _greedy_target, _greedy_perfect_draft,
        prompt_ids=prompt, max_new_tokens=10, k=4, temperature=0.0,
    )

    # Reference: plain greedy decoding from the target alone.
    ref = prompt.clone()
    for _ in range(10):
        nxt = _greedy_target(ref)[:, -1, :].argmax(dim=-1, keepdim=True)
        ref = torch.cat([ref, nxt], dim=1)
    ref_new = ref[:, prompt.shape[1]:]

    torch.testing.assert_close(spec_out[:, :10], ref_new[:, :10])


def test_speculative_decode_perfect_draft_minimizes_target_calls():
    """A perfect draft should let the target verify ~k+1 tokens per call,
    so target_calls << total_tokens for k > 1.
    """
    prompt = torch.tensor([[1]])
    _, stats = speculative_decode(
        _greedy_target, _greedy_perfect_draft,
        prompt_ids=prompt, max_new_tokens=20, k=4, temperature=0.0,
    )
    # 20 new tokens with k=4 → ~4 target calls (each emits 5 = k+1 tokens).
    assert stats.target_calls <= 6
    assert stats.accepted >= 16              # most drafts should be accepted


def test_speculative_decode_bad_draft_still_correct():
    """A bad draft is allowed — the algorithm should reject every draft and
    still produce target-equivalent output (just without speedup).
    """
    prompt = torch.tensor([[2]])
    spec_out, stats = speculative_decode(
        _greedy_target, _greedy_bad_draft,
        prompt_ids=prompt, max_new_tokens=5, k=4, temperature=0.0,
    )

    # Reference: plain greedy from target.
    ref = prompt.clone()
    for _ in range(5):
        nxt = _greedy_target(ref)[:, -1, :].argmax(dim=-1, keepdim=True)
        ref = torch.cat([ref, nxt], dim=1)

    torch.testing.assert_close(spec_out[:, :5], ref[:, prompt.shape[1]:5 + prompt.shape[1]])
    # Bad draft → 0 acceptances (every draft rejected).
    assert stats.accepted == 0


def test_speculative_decode_rejects_invalid_prompt_shape():
    with pytest.raises(ValueError, match="prompt_ids"):
        speculative_decode(
            _greedy_target, _greedy_perfect_draft,
            prompt_ids=torch.tensor([1, 2, 3]),     # not [1, T]
            max_new_tokens=4, k=2,
        )


def test_speculative_decode_eos_stops_early():
    eos = 6                                       # target predicts (5+1)%32 = 6
    prompt = torch.tensor([[5]])
    out, _ = speculative_decode(
        _greedy_target, _greedy_perfect_draft,
        prompt_ids=prompt, max_new_tokens=20, k=4, temperature=0.0,
        eos_token_id=eos,
    )
    # First emitted token should be 6, then stop.
    assert out[0, 0].item() == eos
    assert out.shape[1] <= 4                       # within one round of k=4


def test_specdec_stats_summary():
    stats = SpecDecStats(rounds=10, target_calls=10, accepted=15)
    assert stats.acceptance_rate() == 1.5
    text = repr(stats)
    assert "rounds=10" in text and "accepted=15" in text


# --------------------------------------------------------------------------- #
# MTP-driven speculative decoding
# --------------------------------------------------------------------------- #


class _ToyMainModel:
    """Wraps ``_greedy_target`` to also return a hidden representation —
    the MTP draft uses it.
    """

    HIDDEN = 8

    def __init__(self):
        # A fixed projection from token-id to hidden, just so MTP has signal.
        torch.manual_seed(0)
        self.embed = nn.Embedding(VOCAB, self.HIDDEN)

    def __call__(self, seq: torch.Tensor):
        logits = _greedy_target(seq)              # [B, T, V]
        hidden = self.embed(seq)                  # [B, T, H]
        return logits, hidden


class _ToyMTPHead(nn.Module):
    """Predicts the same ``last + 1`` rule — perfect draft for the toy target."""

    def forward(self, hidden: torch.Tensor, next_ids: torch.Tensor) -> torch.Tensor:
        # Always predict (next_ids + 1) % VOCAB — a perfect MTP draft.
        B, T = next_ids.shape
        logits = torch.full((B, T, VOCAB), -1e9)
        targets = (next_ids + 1) % VOCAB
        logits.scatter_(-1, targets.unsqueeze(-1), 1e9)
        return logits


def test_mtp_speculative_decode_perfect_head_doubles_throughput():
    """With a perfect MTP head and greedy decoding, every round should
    accept the bonus draft, giving 2 tokens per round (one from main, one
    from MTP).
    """
    main = _ToyMainModel()
    mtp = _ToyMTPHead()
    prompt = torch.tensor([[7]])
    out, stats = mtp_speculative_decode(
        main, mtp, prompt_ids=prompt, max_new_tokens=10, temperature=0.0,
    )
    # 10 new tokens; perfect MTP → roughly 2 tokens per round → ~5 rounds.
    assert stats.rounds <= 6
    # Output should match plain greedy.
    ref = prompt.clone()
    for _ in range(10):
        nxt = _greedy_target(ref)[:, -1, :].argmax(dim=-1, keepdim=True)
        ref = torch.cat([ref, nxt], dim=1)
    torch.testing.assert_close(out[:, :10], ref[:, prompt.shape[1]:10 + prompt.shape[1]])


def test_mtp_speculative_decode_eos_stops_early():
    main = _ToyMainModel()
    mtp = _ToyMTPHead()
    eos = 8
    prompt = torch.tensor([[7]])               # main → 8 (=eos) immediately
    out, _ = mtp_speculative_decode(
        main, mtp, prompt_ids=prompt, max_new_tokens=20, temperature=0.0,
        eos_token_id=eos,
    )
    assert out[0, 0].item() == eos
    assert out.shape[1] <= 1
