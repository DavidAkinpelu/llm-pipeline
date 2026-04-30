"""IFEval-style instruction-following evaluation.

Predicate-based: each instruction comes with a predicate that scores
whether the response follows it. The original IFEval (Zhou et al. 2023)
ships ~25 instruction types; we ship a curated subset of the 10 most
common, structured so users can register custom predicates.

Predicates take ``(instruction_args, response) -> bool``. Bundled set:

- ``json_format``: response parses as valid JSON.
- ``contains_keyword``: response contains a keyword (case-insensitive).
- ``forbidden_keyword``: response does NOT contain a forbidden word.
- ``num_words_min`` / ``num_words_max``: word count bounds.
- ``num_sentences_max``: sentence count cap.
- ``num_paragraphs_min``: paragraph count floor.
- ``all_caps``: every alphabetic character is uppercase.
- ``ends_with``: response ends with the given suffix.
- ``starts_with``: response starts with the given prefix.
- ``no_commas``: response contains no commas.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


PredicateFn = Callable[[Dict[str, Any], str], bool]
GenerationFn = Callable[[str], str]


# --------------------------------------------------------------------------- #
# Built-in predicates
# --------------------------------------------------------------------------- #


def _json_format(args: Dict[str, Any], response: str) -> bool:
    response = response.strip()
    # Strip common code-fence wrappers.
    if response.startswith("```"):
        response = response.strip("`")
        if response.lower().startswith("json"):
            response = response[4:]
    try:
        json.loads(response)
        return True
    except json.JSONDecodeError:
        return False


def _contains_keyword(args: Dict[str, Any], response: str) -> bool:
    return args["keyword"].lower() in response.lower()


def _forbidden_keyword(args: Dict[str, Any], response: str) -> bool:
    return args["keyword"].lower() not in response.lower()


def _num_words_min(args: Dict[str, Any], response: str) -> bool:
    return len(response.split()) >= args["min"]


def _num_words_max(args: Dict[str, Any], response: str) -> bool:
    return len(response.split()) <= args["max"]


def _num_sentences_max(args: Dict[str, Any], response: str) -> bool:
    sentences = re.findall(r"[^.!?]+[.!?]", response)
    return len(sentences) <= args["max"]


def _num_paragraphs_min(args: Dict[str, Any], response: str) -> bool:
    paragraphs = [p for p in response.split("\n\n") if p.strip()]
    return len(paragraphs) >= args["min"]


def _all_caps(args: Dict[str, Any], response: str) -> bool:
    letters = [c for c in response if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _ends_with(args: Dict[str, Any], response: str) -> bool:
    return response.rstrip().endswith(args["suffix"])


def _starts_with(args: Dict[str, Any], response: str) -> bool:
    return response.lstrip().startswith(args["prefix"])


def _no_commas(args: Dict[str, Any], response: str) -> bool:
    return "," not in response


_BUILTIN_PREDICATES: Dict[str, PredicateFn] = {
    "json_format": _json_format,
    "contains_keyword": _contains_keyword,
    "forbidden_keyword": _forbidden_keyword,
    "num_words_min": _num_words_min,
    "num_words_max": _num_words_max,
    "num_sentences_max": _num_sentences_max,
    "num_paragraphs_min": _num_paragraphs_min,
    "all_caps": _all_caps,
    "ends_with": _ends_with,
    "starts_with": _starts_with,
    "no_commas": _no_commas,
}


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


@dataclass
class IFEvalItem:
    """One instruction-following test case.

    ``instructions``: list of ``(predicate_name, predicate_args)`` pairs.
    Strict scoring requires *all* of them to pass; loose scoring requires
    at least one.
    """

    prompt: str
    instructions: List[Dict[str, Any]]   # each: {"name": "...", "args": {...}}


@dataclass
class IFEvalResult:
    strict_accuracy: float                # fraction of items where every predicate passed
    loose_accuracy: float                 # fraction where at least one predicate passed
    n_items: int
    per_item: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class IFEvalEvaluator:
    """Run IFEval-style instruction-following on a generation function.

    Custom predicates: pass a dict to ``predicates`` to extend or
    override the bundled set. Each entry is ``name → predicate_fn``.
    """

    generation_fn: GenerationFn
    predicates: Optional[Dict[str, PredicateFn]] = None

    def __post_init__(self):
        merged = dict(_BUILTIN_PREDICATES)
        if self.predicates:
            merged.update(self.predicates)
        self._predicates = merged

    def evaluate(self, items: Iterable[IFEvalItem]) -> IFEvalResult:
        per_item: List[Dict[str, Any]] = []
        n_strict = 0
        n_loose = 0
        n = 0
        for item in items:
            response = self.generation_fn(item.prompt)
            results = []
            for instr in item.instructions:
                name = instr["name"]
                args = instr.get("args", {})
                if name not in self._predicates:
                    raise KeyError(
                        f"unknown predicate: {name!r}; "
                        f"available: {sorted(self._predicates)}"
                    )
                results.append(bool(self._predicates[name](args, response)))
            strict_pass = all(results) if results else False
            loose_pass = any(results) if results else False
            n_strict += int(strict_pass)
            n_loose += int(loose_pass)
            n += 1
            per_item.append({
                "prompt": item.prompt, "response": response,
                "results": results, "strict": strict_pass, "loose": loose_pass,
            })

        return IFEvalResult(
            strict_accuracy=n_strict / max(n, 1),
            loose_accuracy=n_loose / max(n, 1),
            n_items=n, per_item=per_item,
        )
