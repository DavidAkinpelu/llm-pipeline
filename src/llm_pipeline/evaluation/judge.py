"""LLM-as-judge win-rate evaluation.

Gives one model a question + two responses (A and B) and asks it which
is better. Run over a question pool, the average gives a win-rate
estimate of model A vs model B.

Two judging templates ship out of the box:

- **MT-Bench style** (pairwise): "Which response is better, A or B?
  Answer with [[A]], [[B]], or [[C]] (tie)."
- **AlpacaEval style** (direct): same idea, slightly different prompt
  conventions.

Anti-position-bias is built in: each pair is judged twice (A/B then
B/A). A "win" only counts when both judgments agree. Mismatches are
counted as ties — empirically removes ~5-10% of position-driven noise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Sequence


JudgeFn = Callable[[str], str]      # (prompt) -> generated text


_TEMPLATES = {
    "mt_bench": (
        "[Instruction]\n"
        "You are an impartial judge. Compare the two assistant responses "
        "below to the user's question. Pick the one that is more helpful, "
        "accurate, and well-written. If they're equally good, say tie.\n\n"
        "[User question]\n{question}\n\n"
        "[Assistant A]\n{response_a}\n\n"
        "[Assistant B]\n{response_b}\n\n"
        "[Verdict]\n"
        "Reply with exactly one of: [[A]] (A is better), [[B]] (B is better), "
        "or [[C]] (tie). Then explain in one sentence."
    ),
    "alpaca_eval": (
        "Question: {question}\n\n"
        "Response A: {response_a}\n\n"
        "Response B: {response_b}\n\n"
        "Which response is better? Answer with [[A]], [[B]], or [[C]] for tie."
    ),
}


@dataclass
class JudgeVerdict:
    winner: Literal["A", "B", "tie"]
    reasoning: str


@dataclass
class WinRateResult:
    win_rate: float                    # fraction of pairs A won
    n_wins: int
    n_losses: int
    n_ties: int
    verdicts: List[JudgeVerdict] = field(default_factory=list)


_VERDICT_RE = re.compile(r"\[\[(A|B|C)\]\]", re.IGNORECASE)


def _parse_verdict(text: str) -> JudgeVerdict:
    """Extract the judge's pick from its response text."""
    match = _VERDICT_RE.search(text)
    if match is None:
        return JudgeVerdict(winner="tie", reasoning=text.strip())
    letter = match.group(1).upper()
    winner = {"A": "A", "B": "B", "C": "tie"}[letter]
    return JudgeVerdict(winner=winner, reasoning=text.strip())


@dataclass
class LLMJudge:
    """Wraps a judge LLM call with built-in templates + position-bias control.

    >>> judge = LLMJudge(judge_fn=lambda p: openai_complete(p),
    ...                  template="mt_bench", swap_for_position_bias=True)
    >>> verdict = judge.compare("What is 2+2?", "4", "5")
    >>> result = judge.win_rate(questions, responses_a, responses_b)
    """

    judge_fn: JudgeFn
    template: str = "mt_bench"
    swap_for_position_bias: bool = True

    def __post_init__(self):
        if self.template not in _TEMPLATES:
            raise ValueError(
                f"unknown template: {self.template}. "
                f"Available: {list(_TEMPLATES)}"
            )

    def compare(
        self, question: str, response_a: str, response_b: str,
    ) -> JudgeVerdict:
        """One pairwise judgment with optional position-swap consensus."""
        tpl = _TEMPLATES[self.template]
        v1 = _parse_verdict(self.judge_fn(
            tpl.format(question=question, response_a=response_a, response_b=response_b)
        ))
        if not self.swap_for_position_bias:
            return v1
        # Re-judge with A/B swapped; only count as a win if both agree.
        v2 = _parse_verdict(self.judge_fn(
            tpl.format(question=question, response_a=response_b, response_b=response_a)
        ))
        # Map v2 (which had B/A swapped) back to original frame.
        v2_remapped = {"A": "B", "B": "A", "tie": "tie"}[v2.winner]
        if v1.winner == v2_remapped:
            return v1
        return JudgeVerdict(winner="tie", reasoning=f"position-disagreement: {v1.winner} vs {v2_remapped}")

    def win_rate(
        self,
        questions: Sequence[str],
        responses_a: Sequence[str],
        responses_b: Sequence[str],
    ) -> WinRateResult:
        if not (len(questions) == len(responses_a) == len(responses_b)):
            raise ValueError(
                f"length mismatch: {len(questions)} questions, "
                f"{len(responses_a)} A, {len(responses_b)} B"
            )
        verdicts = [
            self.compare(q, a, b)
            for q, a, b in zip(questions, responses_a, responses_b)
        ]
        n_wins = sum(1 for v in verdicts if v.winner == "A")
        n_losses = sum(1 for v in verdicts if v.winner == "B")
        n_ties = sum(1 for v in verdicts if v.winner == "tie")
        n = len(verdicts)
        # Win rate counting ties as half-wins (standard MT-Bench convention).
        win_rate = (n_wins + 0.5 * n_ties) / n if n else 0.0
        return WinRateResult(
            win_rate=win_rate, n_wins=n_wins,
            n_losses=n_losses, n_ties=n_ties, verdicts=verdicts,
        )
