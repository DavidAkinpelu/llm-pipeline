"""Tests for constrained generation logit masks."""

import math

import pytest
import torch

from llm_pipeline.inference.constraints import (
    JSONSchemaConstraint,
    PrefixConstraint,
    RegexConstraint,
    _is_json_prefix_valid,
)


class _ToyTokenizer:
    """Single-character vocab tokenizer for the legality-mask tests."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


# --------------------------------------------------------------------------- #
# JSON prefix validity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text,valid", [
    ("", True),
    ("{", True),
    ('{"name"', True),
    ('{"name": "ali', True),
    ('{"name": "ali"}', True),
    ("[1, 2, 3]", True),
    ("[1, 2,", True),
    ("}", False),
    ("{]", False),
    ("[}", False),
])
def test_json_prefix_validity(text, valid):
    assert _is_json_prefix_valid(text) is valid


# --------------------------------------------------------------------------- #
# RegexConstraint
# --------------------------------------------------------------------------- #


def test_regex_constraint_masks_letters_when_pattern_is_digits():
    """Empty prefix + ``\\d+`` pattern: letters are illegal first tokens
    (they could never grow into a digit-only match), digits are legal.
    """
    tokenizer = _ToyTokenizer(["a", "b", "c", "1", "2", "3"])
    constraint = RegexConstraint(pattern=r"\d+")
    logits = torch.zeros(6)
    masked = constraint.apply_to_logits(logits, generated_ids=[], tokenizer=tokenizer)
    # Letters (indices 0, 1, 2) → -inf; digits (3, 4, 5) → 0.
    assert math.isinf(masked[0]) and math.isinf(masked[1]) and math.isinf(masked[2])
    assert masked[3].item() == 0 and masked[4].item() == 0 and masked[5].item() == 0


def test_regex_constraint_falls_back_to_unmasked_if_no_token_legal():
    """When the pattern can't be extended by any single token in the
    vocab, the constraint falls back to allowing everything — surfacing
    the issue at the application layer rather than the sampler choking on
    an all-(-inf) distribution.
    """
    tokenizer = _ToyTokenizer(["x", "y"])
    constraint = RegexConstraint(pattern=r"^\d{3}$")        # nothing in vocab matches
    logits = torch.tensor([1.0, 2.0])
    out = constraint.apply_to_logits(logits, generated_ids=[], tokenizer=tokenizer)
    # Falls back: nothing masked.
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# JSONSchemaConstraint
# --------------------------------------------------------------------------- #


def test_json_schema_constraint_masks_invalid_brackets():
    """If the prefix has an open ``[``, the candidate ``}`` (which would
    close it with a mismatched bracket) is rejected; ``]`` is accepted.
    """
    # Vocab: 0 → "[" (used in the prefix), 1 → "}" (mismatched closer),
    # 2 → "]" (correct closer).
    tokenizer = _ToyTokenizer(["[", "}", "]"])
    constraint = JSONSchemaConstraint({"type": "object"})
    logits = torch.zeros(3)
    # generated_ids=[0] → prefix is "[" — one open square bracket.
    out = constraint.apply_to_logits(logits, generated_ids=[0], tokenizer=tokenizer)
    assert math.isinf(out[1])                      # "}" rejected — mismatched closer
    assert out[2].item() == 0                       # "]" allowed — correct closer


def test_json_schema_constraint_allows_valid_prefix():
    tokenizer = _ToyTokenizer(["a", "b", "{"])
    constraint = JSONSchemaConstraint({"type": "object"})
    logits = torch.zeros(3)
    # Empty prefix — every candidate should still be a possibly-valid prefix.
    out = constraint.apply_to_logits(logits, generated_ids=[], tokenizer=tokenizer)
    # "{" is a valid JSON-doc start — must stay finite.
    assert torch.isfinite(out[2])


# --------------------------------------------------------------------------- #
# PrefixConstraint (custom predicate adaptor)
# --------------------------------------------------------------------------- #


def test_prefix_constraint_custom_predicate():
    """User-defined predicate — only allow 'a' and 'b' regardless of prefix."""
    tokenizer = _ToyTokenizer(["a", "b", "c"])
    constraint = PrefixConstraint(predicate=lambda _p, c: c in {"a", "b"})
    logits = torch.zeros(3)
    out = constraint.apply_to_logits(logits, generated_ids=[], tokenizer=tokenizer)
    assert out[0].item() == 0 and out[1].item() == 0
    assert math.isinf(out[2])
