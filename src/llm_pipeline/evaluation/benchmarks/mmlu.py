"""MMLU evaluator — multiple-choice with letter-token logit scoring.

Massive Multitask Language Understanding (Hendrycks et al. 2021): 57
subjects, 4-choice MCQ. Standard scoring: read the next-token logits at
the answer position, pick the highest-logit choice from {A, B, C, D}.
This avoids any generation-format brittleness (the model just has to
produce one token).

The evaluator doesn't bundle the dataset (~25 MB CSVs across 57
subjects); pass any iterable of ``{"question", "choices", "answer", "subject"}``
records. ``datasets.load_dataset("cais/mmlu", "all")`` is the canonical
source.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


@dataclass
class MMLUResult:
    overall_accuracy: float
    n_correct: int
    n_total: int
    by_subject: Dict[str, float] = field(default_factory=dict)


@dataclass
class MMLUEvaluator:
    """MMLU runner.

    Parameters
    ----------
    model : torch.nn.Module
        Causal LM with the standard ``forward(input_ids) -> output.logits`` shape.
    tokenizer : Any
        Tokenizer with ``encode(text)`` and ``__call__(text, return_tensors="pt")``.
        Needs to encode "A", "B", "C", "D" each as a single token (or we
        use the first sub-token). Most BPE tokenisers do.
    k_shot : int
        Few-shot examples to prepend. Standard MMLU is 5-shot.
    """

    model: Any
    tokenizer: Any
    k_shot: int = 5
    device: Optional[torch.device] = None

    PROMPT_TEMPLATE = (
        "The following is a multiple choice question about {subject}.\n\n"
        "{question}\n"
        "A. {a}\n"
        "B. {b}\n"
        "C. {c}\n"
        "D. {d}\n"
        "Answer:"
    )

    def __post_init__(self):
        self._answer_token_ids = self._compute_answer_token_ids()

    def _compute_answer_token_ids(self) -> List[int]:
        """Token id for each of "A", "B", "C", "D" — first sub-token if multi."""
        ids = []
        for letter in ("A", "B", "C", "D"):
            # Most tokenisers prepend a leading space for non-initial words;
            # we want the bare letter as it follows "Answer:".
            tok_ids = self.tokenizer.encode(f" {letter}")
            if isinstance(tok_ids, dict):
                tok_ids = tok_ids["input_ids"]
            ids.append(int(tok_ids[-1]))
        return ids

    def _format_prompt(self, item: Dict[str, Any], few_shot: Sequence[Dict[str, Any]]) -> str:
        prompt = ""
        for ex in few_shot:
            answer_letter = "ABCD"[ex["answer"]] if isinstance(ex["answer"], int) else ex["answer"]
            prompt += self.PROMPT_TEMPLATE.format(
                subject=ex.get("subject", "general"),
                question=ex["question"],
                a=ex["choices"][0], b=ex["choices"][1],
                c=ex["choices"][2], d=ex["choices"][3],
            ) + f" {answer_letter}\n\n"
        prompt += self.PROMPT_TEMPLATE.format(
            subject=item.get("subject", "general"),
            question=item["question"],
            a=item["choices"][0], b=item["choices"][1],
            c=item["choices"][2], d=item["choices"][3],
        )
        return prompt

    @torch.no_grad()
    def evaluate(
        self,
        dataset: Iterable[Dict[str, Any]],
        few_shot_pool: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> MMLUResult:
        """Run MMLU on the given dataset; return per-subject + overall accuracy."""
        device = self.device or next(self.model.parameters()).device
        self.model.eval()

        few_shot_pool = list(few_shot_pool or [])
        few_shot = few_shot_pool[: self.k_shot]

        n_correct = 0
        n_total = 0
        by_subject_correct: Dict[str, int] = defaultdict(int)
        by_subject_total: Dict[str, int] = defaultdict(int)

        for item in dataset:
            prompt = self._format_prompt(item, few_shot)
            input_ids = torch.tensor(
                self._encode(prompt), dtype=torch.long, device=device,
            ).unsqueeze(0)
            out = self.model(input_ids=input_ids)
            logits = out.logits if hasattr(out, "logits") else out[0]
            last_token_logits = logits[0, -1, :]
            answer_logits = last_token_logits[self._answer_token_ids]
            pred = int(answer_logits.argmax().item())

            gold = item["answer"]
            if isinstance(gold, str):
                gold = "ABCD".index(gold)
            n_total += 1
            subj = item.get("subject", "general")
            by_subject_total[subj] += 1
            if pred == gold:
                n_correct += 1
                by_subject_correct[subj] += 1

        return MMLUResult(
            overall_accuracy=n_correct / max(n_total, 1),
            n_correct=n_correct,
            n_total=n_total,
            by_subject={
                s: by_subject_correct[s] / by_subject_total[s]
                for s in by_subject_total
            },
        )

    def _encode(self, text: str) -> List[int]:
        out = self.tokenizer.encode(text)
        if isinstance(out, dict):
            return list(out["input_ids"])
        return list(out)
