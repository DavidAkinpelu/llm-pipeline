"""Synthetic-data generation: Self-Instruct + Evol-Instruct orchestration.

Both pipelines use a strong "teacher" LLM to expand a small seed set of
prompts / instructions into a much larger synthetic dataset. The
algorithms differ in how they expand:

- **Self-Instruct** (Wang et al. 2022): repeatedly sample a few random
  seeds, prompt the teacher to generate "more instructions like these",
  parse + dedupe the output, append to the pool.

- **Evol-Instruct** (Xu et al. 2023, the WizardLM paper): start from a
  seed instruction, iteratively rewrite it through a fixed set of
  "evolution" operators — *add constraint*, *deepen*, *concretise*,
  *increase reasoning steps*, *change topic*. After K rounds the
  resulting instruction is much harder than the seed.

Both helpers take a ``SynthGenerator``-compatible callable that runs the
teacher LLM. The orchestration here is LLM-agnostic — plug in our own
inference engine, an HF pipeline, an OpenAI client, or anything that
maps a prompt to a string.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence


# Type alias: a teacher LLM = (prompt -> generated_text).
SynthGenerator = Callable[[str], str]


# --------------------------------------------------------------------------- #
# Self-Instruct
# --------------------------------------------------------------------------- #


@dataclass
class SelfInstructConfig:
    """Knobs for the Self-Instruct expansion loop.

    Attributes
    ----------
    target_size : int
        Stop when the pool reaches this many instructions.
    n_seeds_per_round : int
        How many randomly-sampled seeds to feed the teacher each round.
    seed : int
        RNG seed for deterministic sampling.
    dedupe_threshold : float
        Minimum cosine-of-words-set distance for a generated instruction
        to be considered "novel". Cheap proxy for full embedding
        similarity; tune higher for stricter dedup.
    """

    target_size: int = 100
    n_seeds_per_round: int = 4
    seed: int = 0
    dedupe_threshold: float = 0.6


_INSTRUCTION_RE = re.compile(r"^\s*\d+\.\s*(.+?)\s*$", re.MULTILINE)


def _parse_instructions(text: str) -> List[str]:
    """Extract numbered instructions from a teacher response.

    The classic Self-Instruct prompt asks the teacher to emit
    ``1. ...\n2. ...`` numbered output; we split on those markers.
    """
    matches = _INSTRUCTION_RE.findall(text)
    return [m.strip() for m in matches if m.strip()]


def _word_set(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def _is_novel(candidate: str, pool: Sequence[str], threshold: float) -> bool:
    cand_words = _word_set(candidate)
    if not cand_words:
        return False
    for existing in pool:
        existing_words = _word_set(existing)
        if not existing_words:
            continue
        sim = len(cand_words & existing_words) / max(len(cand_words | existing_words), 1)
        if sim >= threshold:
            return False
    return True


def self_instruct(
    seeds: Sequence[str],
    generator: SynthGenerator,
    config: Optional[SelfInstructConfig] = None,
) -> List[str]:
    """Run the Self-Instruct expansion loop.

    Returns the full instruction pool (seeds + accepted generations).
    Stops when ``target_size`` is reached or no novel candidates have
    been produced for 3 rounds in a row.

    The teacher prompt format follows the Wang et al. paper: list the
    seeds as ``1. ...`` items and ask for additional ``N+1, N+2, ...``
    items in the same style.
    """
    cfg = config or SelfInstructConfig()
    rng = random.Random(cfg.seed)
    pool = list(seeds)
    stagnant = 0

    while len(pool) < cfg.target_size and stagnant < 3:
        sampled = rng.sample(pool, min(cfg.n_seeds_per_round, len(pool)))
        prompt_lines = [
            "Generate more instructions in the style of the examples below.",
            "Output them as a numbered list, one per line. No explanations.",
            "",
        ]
        for i, s in enumerate(sampled, start=1):
            prompt_lines.append(f"{i}. {s}")
        prompt_lines.append(f"{len(sampled) + 1}.")
        prompt = "\n".join(prompt_lines)

        response = generator(prompt)
        new_items = _parse_instructions(response)

        added_this_round = 0
        for item in new_items:
            if _is_novel(item, pool, cfg.dedupe_threshold):
                pool.append(item)
                added_this_round += 1
                if len(pool) >= cfg.target_size:
                    break
        stagnant = 0 if added_this_round else stagnant + 1

    return pool


# --------------------------------------------------------------------------- #
# Evol-Instruct
# --------------------------------------------------------------------------- #


_EVOL_OPERATORS = {
    "add_constraint":
        "Rewrite the instruction below to add one more constraint or "
        "requirement, making it harder to satisfy. Output only the rewritten "
        "instruction.",
    "deepen":
        "Rewrite the instruction below to require deeper reasoning or more "
        "domain expertise. Output only the rewritten instruction.",
    "concretise":
        "Rewrite the instruction below by replacing any general concepts "
        "with specific, concrete examples. Output only the rewritten "
        "instruction.",
    "increase_reasoning":
        "Rewrite the instruction below so that solving it requires more "
        "explicit reasoning steps. Output only the rewritten instruction.",
    "change_topic":
        "Rewrite the instruction below by adapting it to a different domain "
        "or topic, while keeping its structure. Output only the rewritten "
        "instruction.",
}


@dataclass
class EvolInstructConfig:
    """Knobs for the Evol-Instruct rewriting loop.

    Attributes
    ----------
    n_evolutions : int
        Number of evolution rounds per seed instruction.
    operators : list[str]
        Subset of ``_EVOL_OPERATORS`` keys to sample from at each step.
        Default: all five.
    seed : int
        RNG seed.
    """

    n_evolutions: int = 3
    operators: List[str] = field(default_factory=lambda: list(_EVOL_OPERATORS.keys()))
    seed: int = 0

    def __post_init__(self):
        unknown = [op for op in self.operators if op not in _EVOL_OPERATORS]
        if unknown:
            raise ValueError(
                f"unknown evolution operators: {unknown}; "
                f"valid: {list(_EVOL_OPERATORS.keys())}"
            )


def evol_instruct(
    seeds: Sequence[str],
    generator: SynthGenerator,
    config: Optional[EvolInstructConfig] = None,
) -> List[List[str]]:
    """Run Evol-Instruct on each seed.

    Returns one list per seed: ``[seed, evolved_1, evolved_2, ...]``.
    The intermediate steps are kept so a curriculum or difficulty-
    progression dataset can use them.
    """
    cfg = config or EvolInstructConfig()
    rng = random.Random(cfg.seed)
    chains: List[List[str]] = []

    for seed_inst in seeds:
        chain = [seed_inst]
        current = seed_inst
        for _ in range(cfg.n_evolutions):
            op = rng.choice(cfg.operators)
            prompt = (
                f"{_EVOL_OPERATORS[op]}\n\n"
                f"Original instruction:\n{current}\n\n"
                "Rewritten instruction:"
            )
            new_inst = generator(prompt).strip()
            chain.append(new_inst)
            current = new_inst
        chains.append(chain)

    return chains
