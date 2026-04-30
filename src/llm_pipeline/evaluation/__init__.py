"""Evaluation utilities — perplexity, benchmarks, judge, calibration, NIH, diversity."""

from .perplexity import PerplexityEvaluator
from .judge import JudgeVerdict, LLMJudge, WinRateResult
from .callbacks import EvaluationCallback, EvaluationHistory, RegressionWarning
from .calibration import expected_calibration_error, reliability_diagram_data
from .needle_in_haystack import NIHEvaluator, NIHResult
from .diversity import distinct_n, repetition_rate, self_bleu
from .benchmarks import (
    GSM8KEvaluator, GSM8KResult, extract_numeric_answer,
    HumanEvalEvaluator, HumanEvalProblem, HumanEvalResult,
    IFEvalEvaluator, IFEvalItem, IFEvalResult,
    MMLUEvaluator, MMLUResult,
)


__all__ = [
    "PerplexityEvaluator",
    "JudgeVerdict", "LLMJudge", "WinRateResult",
    "EvaluationCallback", "EvaluationHistory", "RegressionWarning",
    "expected_calibration_error", "reliability_diagram_data",
    "NIHEvaluator", "NIHResult",
    "distinct_n", "repetition_rate", "self_bleu",
    "GSM8KEvaluator", "GSM8KResult", "extract_numeric_answer",
    "HumanEvalEvaluator", "HumanEvalProblem", "HumanEvalResult",
    "IFEvalEvaluator", "IFEvalItem", "IFEvalResult",
    "MMLUEvaluator", "MMLUResult",
]
