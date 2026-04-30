"""Needle-in-a-haystack long-context retrieval eval.

Inserts a "needle" sentence at a chosen depth in a long filler context,
then queries the model. Standard Anthropic / Greg Kamradt style. Sweep
over (context_length, needle_depth) → heatmap of retrieval accuracy.

For our hand-rolled Qwen3.5 / Qwen3.6 with 256K native context, this is
the canonical long-context regression test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


GenerationFn = Callable[[str], str]   # (prompt) -> generated text


# Deterministic filler — repeating Lorem-Ipsum-ish sentences. Avoids the
# need for a corpus dep and keeps tests reproducible.
_FILLER_SENTENCE = (
    "The morning fog drifted through the old stone arches, settling in "
    "the courtyards where ravens preened. "
)


@dataclass
class NIHResult:
    context_length: int                 # in characters
    depth_fraction: float               # 0.0 (start) to 1.0 (end)
    needle: str
    found: bool
    response: str


@dataclass
class NIHEvaluator:
    """Run the needle-in-a-haystack sweep.

    Attributes
    ----------
    generation_fn : (prompt) -> str
        Wraps the model. Should return a single string completion.
    needle : str
        The information to retrieve. Can include a unique-looking token
        ("fluorescent_kitten") so exact-match scoring is unambiguous.
    question : str
        The query appended after the haystack. Default asks for the needle.
    """

    generation_fn: GenerationFn
    needle: str = "The secret password is fluorescent_kitten."
    question: str = "What is the secret password?"
    answer_substring: str = "fluorescent_kitten"
    filler_sentence: str = _FILLER_SENTENCE

    def build_haystack(self, context_chars: int, depth_fraction: float) -> str:
        """Construct a context of approximately ``context_chars`` characters
        with the needle inserted at ``depth_fraction`` of the way through.
        """
        if not 0.0 <= depth_fraction <= 1.0:
            raise ValueError(f"depth_fraction must be in [0, 1]; got {depth_fraction}")
        target = max(context_chars, len(self.needle) + len(self.filler_sentence))
        n_repeats = max(1, target // len(self.filler_sentence))
        filler_full = self.filler_sentence * n_repeats
        insert_at = int(len(filler_full) * depth_fraction)
        return filler_full[:insert_at] + self.needle + filler_full[insert_at:]

    def evaluate_one(
        self, context_length: int, depth_fraction: float,
    ) -> NIHResult:
        """Run a single needle test."""
        haystack = self.build_haystack(context_length, depth_fraction)
        prompt = (
            f"{haystack}\n\nQuestion: {self.question}\n\nAnswer:"
        )
        response = self.generation_fn(prompt)
        found = self.answer_substring.lower() in response.lower()
        return NIHResult(
            context_length=context_length,
            depth_fraction=depth_fraction,
            needle=self.needle,
            found=found,
            response=response,
        )

    def sweep(
        self,
        context_lengths: Sequence[int],
        depth_fractions: Sequence[float],
    ) -> List[NIHResult]:
        """Cartesian product of (length × depth) — produces the standard
        NIH heatmap data.
        """
        results = []
        for L in context_lengths:
            for d in depth_fractions:
                results.append(self.evaluate_one(L, d))
        return results

    @staticmethod
    def accuracy_grid(
        results: Sequence[NIHResult],
    ) -> Dict[Tuple[int, float], bool]:
        """Convenience: results as a ``(length, depth) -> found`` dict."""
        return {(r.context_length, r.depth_fraction): r.found for r in results}
