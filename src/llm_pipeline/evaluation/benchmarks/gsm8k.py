"""GSM8K math-reasoning evaluator with chain-of-thought prompting.

Cobbe et al. 2021. ~8500 grade-school math word problems with numeric
answers. Standard recipe: 8-shot CoT prompting, extract the final
numeric answer from the response.

The evaluator doesn't bundle the dataset; pass an iterable of
``{"question", "answer"}`` records (numeric ``answer``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


GenerationFn = Callable[[str], str]


# Default CoT prompts (Cobbe et al. paper's example pool).
_DEFAULT_FEW_SHOT = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = 24 clips in May. Altogether she sold 48 + 24 = 72 clips. The answer is 72.",
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "$12 / 60 = $0.20 per minute. 50 × $0.20 = $10. The answer is 10.",
    },
]


_FINAL_ANSWER_RE = re.compile(
    r"(?:the answer is|####)\s*\$?(-?[\d,]+\.?\d*)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"-?[\d,]+\.?\d*")


def extract_numeric_answer(text: str) -> Optional[float]:
    """Extract the final numeric answer from a CoT response.

    Two strategies, tried in order:
      1. Match "The answer is N" or "#### N" — the conventional GSM8K
         and CoT formats.
      2. Fallback: the last number in the response.

    Returns ``None`` if no number is found.
    """
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return _parse_number(m.group(1))
    matches = _NUMBER_RE.findall(text)
    if matches:
        return _parse_number(matches[-1])
    return None


def _parse_number(s: str) -> Optional[float]:
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class GSM8KResult:
    accuracy: float
    n_correct: int
    n_total: int
    per_item: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GSM8KEvaluator:
    """Run GSM8K with CoT prompting + numeric-answer extraction.

    Parameters
    ----------
    generation_fn : (prompt) -> str
    few_shot : list of {"question", "answer"} dicts, default uses bundled examples.
    n_shot : int — how many of the few-shot pool to use.
    tolerance : float — accept the answer if |pred - gold| <= tolerance × max(|gold|, 1).
    """

    generation_fn: GenerationFn
    few_shot: Optional[Sequence[Dict[str, str]]] = None
    n_shot: int = 8
    tolerance: float = 1e-6

    def __post_init__(self):
        pool = list(self.few_shot or _DEFAULT_FEW_SHOT)
        self._few_shot = pool[: self.n_shot]

    def _format_prompt(self, question: str) -> str:
        prompt = ""
        for ex in self._few_shot:
            prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
        prompt += f"Question: {question}\nAnswer:"
        return prompt

    def evaluate(self, dataset: Iterable[Dict[str, Any]]) -> GSM8KResult:
        n_correct = 0
        n_total = 0
        per_item: List[Dict[str, Any]] = []

        for item in dataset:
            prompt = self._format_prompt(item["question"])
            response = self.generation_fn(prompt)
            pred = extract_numeric_answer(response)
            gold = float(item["answer"])
            correct = (
                pred is not None
                and abs(pred - gold) <= self.tolerance * max(abs(gold), 1.0)
            )
            n_total += 1
            n_correct += int(correct)
            per_item.append({
                "question": item["question"], "gold": gold,
                "predicted": pred, "response": response, "correct": correct,
            })

        return GSM8KResult(
            accuracy=n_correct / max(n_total, 1),
            n_correct=n_correct, n_total=n_total, per_item=per_item,
        )
