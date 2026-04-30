"""Diversity / repetition metrics for generated text.

Useful for instruction-tuned models — catches degenerate "As an AI
assistant..." patterns and mode collapse. Three metrics:

- **distinct_n**: fraction of unique N-grams in the text.
  Higher = more diverse vocabulary.
- **repetition_rate**: fraction of N-grams that appear more than once.
  Inverse perspective; complements distinct-N.
- **self_bleu**: average BLEU between every pair of generations from
  the same prompt. Lower = more diverse outputs.

Tokenisation is whitespace-split here for simplicity. For
language-specific tokenisation pass pre-tokenised lists in.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from typing import List, Sequence, Union


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text_or_tokens: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(text_or_tokens, str):
        return _WORD_RE.findall(text_or_tokens.lower())
    return list(text_or_tokens)


def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    if n < 1:
        raise ValueError(f"n must be ≥ 1; got {n}")
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(text: Union[str, Sequence[str]], n: int = 2) -> float:
    """Fraction of unique N-grams.

    distinct_n = |unique N-grams| / |all N-grams|.
    Returns 0.0 if the text has fewer than n tokens.
    """
    tokens = _tokenize(text)
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def repetition_rate(text: Union[str, Sequence[str]], n: int = 3) -> float:
    """Fraction of N-grams that appear more than once.

    1.0 means every N-gram is repeated; 0.0 means all unique.
    """
    tokens = _tokenize(text)
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(grams)


def _bleu_pair(reference: List[str], hypothesis: List[str], n: int = 4) -> float:
    """Simplified BLEU-N: geometric mean of clipped n-gram precisions
    times a brevity penalty.

    BLEU = BP · exp(Σ_k (1/n) · log p_k) = BP · (Π_k p_k)^(1/n)

    where p_k is the clipped n-gram precision at order k (the weights
    cancel out as 1/n in the geometric mean).
    """
    if not hypothesis:
        return 0.0
    precisions = []
    for k in range(1, n + 1):
        ref_grams = Counter(_ngrams(reference, k))
        hyp_grams = Counter(_ngrams(hypothesis, k))
        if not hyp_grams:
            return 0.0
        clipped = {g: min(c, ref_grams.get(g, 0)) for g, c in hyp_grams.items()}
        num = sum(clipped.values())
        denom = sum(hyp_grams.values())
        if num == 0:
            return 0.0
        precisions.append(num / denom)
    geo = 1.0
    for p in precisions:
        geo *= p ** (1.0 / n)
    bp = 1.0 if len(hypothesis) > len(reference) else (
        2.71828 ** (1 - len(reference) / max(len(hypothesis), 1))
    )
    return geo * bp


def self_bleu(generations: Sequence[Union[str, Sequence[str]]], n: int = 4) -> float:
    """Average BLEU between every pair of generations.

    Lower = more diverse. Identical generations → 1.0; entirely disjoint
    word sets → 0.0.
    """
    if len(generations) < 2:
        return 0.0
    token_lists = [_tokenize(g) for g in generations]
    scores: List[float] = []
    for a, b in combinations(token_lists, 2):
        # Symmetrised: average of bleu(a→b) and bleu(b→a).
        scores.append(0.5 * (_bleu_pair(a, b, n=n) + _bleu_pair(b, a, n=n)))
    return sum(scores) / len(scores)
