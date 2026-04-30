"""HumanEval-style code evaluation.

Standard pass@k metric: generate ``n_samples`` completions per problem,
run each through a subprocess sandbox with timeout, score by whether
the test cases pass.

**Sandbox caveat**: ``subprocess.run`` with a timeout is NOT bulletproof
isolation. Production code-eval should use Docker / firejail / nsjail.
This module is for development-time eval where the generated code is
trusted-ish (e.g., your own model's outputs); for evaluating untrusted
or adversarial code, use a real sandbox.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence


GenerationFn = Callable[[str, int], List[str]]   # (prompt, n_samples) -> [completions]


@dataclass
class HumanEvalProblem:
    task_id: str
    prompt: str                          # function signature + docstring
    test: str                             # python code that defines a check function
    entry_point: str                     # function name to call


@dataclass
class HumanEvalResult:
    pass_at_k: Dict[int, float]          # {k: pass@k}
    per_task: Dict[str, List[bool]] = field(default_factory=dict)
    n_problems: int = 0


def _pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k from n samples with c successes
    (Chen et al. 2021).
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1 - k / (n - i) for i in range(c))


@dataclass
class HumanEvalEvaluator:
    """Run a HumanEval-style benchmark.

    Parameters
    ----------
    generation_fn : (prompt, n) -> [str]
        Wraps the model. Returns ``n`` completions for the prompt.
    n_samples : int
        Per-problem sample count.
    k : list[int]
        Which pass@k values to report. ``k <= n_samples`` for each.
    timeout_s : float
        Per-execution timeout. Caps how long a sample can run before
        being marked as failed.
    """

    generation_fn: GenerationFn
    n_samples: int = 1
    k: Sequence[int] = (1,)
    timeout_s: float = 5.0

    def __post_init__(self):
        for k in self.k:
            if k > self.n_samples:
                raise ValueError(f"pass@{k} requested but only {self.n_samples} samples")

    def evaluate(self, problems: Iterable[HumanEvalProblem]) -> HumanEvalResult:
        per_task: Dict[str, List[bool]] = {}
        for problem in problems:
            completions = self.generation_fn(problem.prompt, self.n_samples)
            results = [self._run_one(problem, c) for c in completions]
            per_task[problem.task_id] = results

        pass_at_k = {}
        for k in self.k:
            estimates = [
                _pass_at_k_estimator(len(results), sum(results), k)
                for results in per_task.values()
            ]
            pass_at_k[k] = sum(estimates) / max(len(estimates), 1)

        return HumanEvalResult(
            pass_at_k=pass_at_k, per_task=per_task,
            n_problems=len(per_task),
        )

    def _run_one(self, problem: HumanEvalProblem, completion: str) -> bool:
        """Run ``problem.prompt + completion + problem.test + check(...)`` in
        a subprocess; return True iff exit code 0.
        """
        program = (
            problem.prompt + completion + "\n\n" +
            problem.test + "\n" +
            f"check({problem.entry_point})\n"
        )
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "candidate.py"
            script.write_text(program)
            try:
                result = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True, timeout=self.timeout_s,
                )
                return result.returncode == 0
            except subprocess.TimeoutExpired:
                return False
            except Exception:
                return False
