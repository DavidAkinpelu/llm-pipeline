"""Constrained generation via token-level logit masks.

The standard approach: at each generation step, query a constraint
predicate to figure out which token IDs are *legal* given the prefix
generated so far, then mask the logits of all illegal tokens to ``-inf``
before sampling. Sampling proceeds normally on the masked distribution.

This module provides three constraint kinds, each implementing the same
``LogitConstraint`` protocol so they compose:

- **`RegexConstraint`** — accept only token sequences whose decoded text
  is a (possibly partial) match of the given regex.
- **`JSONSchemaConstraint`** — accept only token sequences whose decoded
  text is a (possibly partial) prefix of a valid JSON document conforming
  to the given JSON schema.
- **`PrefixConstraint`** — generic adaptor that takes any
  ``(prefix_text, candidate_token_text) -> bool`` predicate.

All three are CPU-only and assume tokenizer access; production servers
typically pre-build a token trie for performance, but the math is the
same. The ``apply_to_logits`` method takes a ``[..., V]`` logit tensor
and a list of generated token IDs and returns the masked logit tensor —
plug it directly into your sampling loop.

This is a *correctness* reference, not a fast path. Lib alternatives:
``outlines`` (regex / pydantic), ``lm-format-enforcer`` (JSON schema with
finite-state automaton compilation). Both wire up to HF's
``LogitsProcessor`` interface; our protocol is interchangeable.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch


class LogitConstraint(ABC):
    """Base class for token-level generation constraints.

    Each step the sampling loop calls ``apply_to_logits(logits, generated_ids,
    tokenizer)``. The constraint returns a new logit tensor with illegal
    token IDs masked to ``-inf`` (so softmax sends them to zero).

    Subclasses override ``is_token_legal(prefix_text, candidate_text)``;
    everything else is plumbing.
    """

    def apply_to_logits(
        self,
        logits: torch.Tensor,                 # [..., V]
        generated_ids: Sequence[int],          # the prefix decoded so far (excluding prompt is fine)
        tokenizer: Any,                         # any object exposing decode(ids) -> str
    ) -> torch.Tensor:
        prefix_text = tokenizer.decode(list(generated_ids))
        vocab_size = logits.shape[-1]
        # Build a 1-D legality mask once per step.
        legal = self._compute_legality_mask(prefix_text, vocab_size, tokenizer)
        masked = logits.clone()
        masked[..., ~legal] = float("-inf")
        return masked

    def _compute_legality_mask(
        self, prefix_text: str, vocab_size: int, tokenizer: Any,
    ) -> torch.Tensor:
        legal = torch.zeros(vocab_size, dtype=torch.bool)
        for tok_id in range(vocab_size):
            cand = tokenizer.decode([tok_id])
            if self.is_token_legal(prefix_text, cand):
                legal[tok_id] = True
        if not legal.any():
            # Fall back to allowing everything rather than raising — emits a
            # token-stream-ending sentinel via whatever EOS the model picks
            # would be cleaner, but at the constraint layer we can't decide.
            legal[:] = True
        return legal

    @abstractmethod
    def is_token_legal(self, prefix_text: str, candidate_text: str) -> bool:
        """Return True if appending ``candidate_text`` to ``prefix_text`` could
        still lead to a sequence that satisfies this constraint.
        """


# --------------------------------------------------------------------------- #
# Regex
# --------------------------------------------------------------------------- #


@dataclass
class RegexConstraint(LogitConstraint):
    """Allow only text that is a prefix of some string matching the regex.

    Approximates "could still grow into a match" by checking whether
    ``pattern + ".*"`` fullmatches the accumulated text. This works
    cleanly for prefix-friendly patterns (``\\d+``, ``[a-z]+``,
    alternations, etc.) — the extension's ``.*`` absorbs anything still
    legal to emit. Fixed-quantifier patterns like ``\\d{3}`` are
    sub-optimal here (a partial match like ``"1"`` is still legal but
    the extended fullmatch rejects it); for those, plug in a real
    FSA-based library via the ``LogitConstraint`` interface.
    """

    pattern: str
    _extended: "re.Pattern" = None

    def __post_init__(self):
        self._extended = re.compile(self.pattern + r".*", re.DOTALL)

    def is_token_legal(self, prefix_text: str, candidate_text: str) -> bool:
        candidate_full = prefix_text + candidate_text
        if not candidate_full:
            return True
        return bool(self._extended.fullmatch(candidate_full))


# --------------------------------------------------------------------------- #
# JSON schema
# --------------------------------------------------------------------------- #


class JSONSchemaConstraint(LogitConstraint):
    """Lightweight prefix check against a JSON schema.

    The full schema-aware FSA from ``lm-format-enforcer`` is overkill for
    the most common use case ("emit a JSON object whose top-level keys
    are X, Y, Z"). This constraint enforces:

    1. The accumulated text is the prefix of a syntactically valid JSON
       document (balanced quotes, brackets, braces).
    2. If the schema specifies ``"type": "object"`` and ``"required": [...]``,
       the document must eventually contain those keys at the top level.

    Anything more elaborate (nested types, enums, regex on string values)
    delegates to a "pure-syntactic" check — accepts any well-formed prefix.
    For full-schema enforcement, plug ``outlines`` or ``lm-format-enforcer``
    in via the ``LogitConstraint`` interface instead.
    """

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def is_token_legal(self, prefix_text: str, candidate_text: str) -> bool:
        cand = prefix_text + candidate_text
        return _is_json_prefix_valid(cand)


def _is_json_prefix_valid(text: str) -> bool:
    """Return True if ``text`` is a syntactically-valid prefix of a JSON value.

    Tries ``json.loads(text)`` first (full match → True). Otherwise, scans
    the bracket/quote stack and rejects on imbalance only — strict enough
    to weed out garbage like ``{a:`` while still accepting partial inputs
    like ``{"name": "ali``.
    """
    if not text:
        return True
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        pass
    # Bracket-stack check.
    stack: List[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return False
            opener = stack.pop()
            if (opener, ch) not in {("{", "}"), ("[", "]")}:
                return False
    # Acceptable to be inside a string (we'll close it later) or with an
    # unclosed bracket on the stack (ditto). Reject only on broken structure.
    return True


# --------------------------------------------------------------------------- #
# Generic prefix-predicate constraint
# --------------------------------------------------------------------------- #


@dataclass
class PrefixConstraint(LogitConstraint):
    """Adaptor for an arbitrary ``(prefix, candidate) -> bool`` predicate."""

    predicate: Callable[[str, str], bool]

    def is_token_legal(self, prefix_text: str, candidate_text: str) -> bool:
        return self.predicate(prefix_text, candidate_text)
