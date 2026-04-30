"""Tests for the evaluation module: perplexity, calibration, diversity,
LLM judge, needle-in-haystack, callbacks, and the four benchmark evaluators.
"""

import math
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

from llm_pipeline.evaluation import (
    EvaluationCallback,
    EvaluationHistory,
    GSM8KEvaluator,
    HumanEvalEvaluator,
    HumanEvalProblem,
    IFEvalEvaluator,
    IFEvalItem,
    LLMJudge,
    MMLUEvaluator,
    NIHEvaluator,
    PerplexityEvaluator,
    RegressionWarning,
    distinct_n,
    expected_calibration_error,
    extract_numeric_answer,
    reliability_diagram_data,
    repetition_rate,
    self_bleu,
)


# --------------------------------------------------------------------------- #
# Stub model + tokenizer for the evaluators that need them
# --------------------------------------------------------------------------- #


class _DummyTokenizer:
    """Whitespace tokenizer; word → stable int id."""

    def __init__(self):
        self._vocab = {}

    def encode(self, text):
        ids = []
        for w in text.split():
            ids.append(self._vocab.setdefault(w, len(self._vocab) + 1))
        return ids


class _DummyOut:
    def __init__(self, loss=None, logits=None):
        if loss is not None:
            self.loss = loss
        if logits is not None:
            self.logits = logits


class _DummyCausalLM(nn.Module):
    """Returns a constant per-token loss; useful for verifying perplexity wiring."""

    def __init__(self, fixed_loss: float = 1.0, vocab_size: int = 100):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))
        self.fixed_loss = fixed_loss
        self.vocab_size = vocab_size

    def forward(self, input_ids, labels=None):
        if labels is not None:
            return _DummyOut(loss=torch.tensor(self.fixed_loss))
        # Constant logits with token 1 always max.
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size)
        logits[..., 1] = 100.0
        return _DummyOut(logits=logits)


# --------------------------------------------------------------------------- #
# Perplexity
# --------------------------------------------------------------------------- #


def test_perplexity_constant_loss_yields_exp_loss():
    """If the model returns loss=L for every window, perplexity = exp(L)."""
    model = _DummyCausalLM(fixed_loss=1.5)
    tok = _DummyTokenizer()
    evalr = PerplexityEvaluator(model, tok, max_length=4, stride=4)
    out = evalr.evaluate([{"text": "a b c d e f g h i j"}])
    assert abs(out["perplexity"] - math.exp(1.5)) < 1e-3
    assert out["tokens_scored"] > 0


def test_perplexity_no_double_counting_with_stride():
    """Sum of tokens_scored across windows equals total tokens (each
    scored exactly once).
    """
    model = _DummyCausalLM(fixed_loss=2.0)
    tok = _DummyTokenizer()
    text = " ".join(str(i) for i in range(20))
    evalr = PerplexityEvaluator(model, tok, max_length=4, stride=2)
    out = evalr.evaluate([{"text": text}])
    assert out["tokens_scored"] == 20


def test_perplexity_empty_dataset_returns_nan():
    model = _DummyCausalLM()
    tok = _DummyTokenizer()
    evalr = PerplexityEvaluator(model, tok)
    out = evalr.evaluate([])
    assert math.isnan(out["perplexity"])
    assert out["tokens_scored"] == 0


def test_perplexity_rejects_invalid_stride():
    with pytest.raises(ValueError, match="stride"):
        PerplexityEvaluator(_DummyCausalLM(), _DummyTokenizer(), stride=0)
    with pytest.raises(ValueError, match="max_length"):
        PerplexityEvaluator(
            _DummyCausalLM(), _DummyTokenizer(), max_length=4, stride=8,
        )


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_ece_perfect_calibration_is_zero():
    """A perfectly-calibrated predictor has ECE = 0."""
    # 100 examples, half with confidence 0.9 → 90% accurate, half with
    # confidence 0.6 → 60% accurate. Pre-shuffled into one dataset.
    probs_high = np.full((100, 2), 0.1)
    probs_high[:, 1] = 0.9
    labels_high = np.array([1] * 90 + [0] * 10)

    probs_low = np.full((100, 2), 0.4)
    probs_low[:, 1] = 0.6
    labels_low = np.array([1] * 60 + [0] * 40)

    probs = np.concatenate([probs_high, probs_low])
    labels = np.concatenate([labels_high, labels_low])

    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece < 0.05                                      # near-zero, allowing for binning


def test_ece_overconfident_predictor_has_positive_ece():
    """Always-very-confident but only 50% accurate → ECE near 0.5."""
    probs = np.full((100, 2), 0.05)
    probs[:, 1] = 0.95
    labels = np.array([1] * 50 + [0] * 50)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece > 0.4


def test_ece_binary_input_works():
    """Single-column probs treated as P(class=1)."""
    probs = np.array([0.9, 0.9, 0.8, 0.2])
    labels = np.array([1, 1, 1, 0])
    ece = expected_calibration_error(probs, labels, n_bins=5)
    assert 0 <= ece <= 1


def test_reliability_diagram_returns_correct_shape():
    probs = np.random.rand(50, 3)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = np.random.randint(0, 3, 50)
    centers, accs, confs = reliability_diagram_data(probs, labels, n_bins=10)
    assert centers.shape == (10,) and accs.shape == (10,) and confs.shape == (10,)


# --------------------------------------------------------------------------- #
# Diversity
# --------------------------------------------------------------------------- #


def test_distinct_n_all_unique_returns_one():
    assert distinct_n("a b c d e", n=1) == 1.0


def test_distinct_n_all_repeated_returns_low():
    """Same word 10 times → 1 unique 1-gram out of 10 → distinct_1 = 0.1."""
    text = "word " * 10
    assert distinct_n(text, n=1) == 0.1


def test_repetition_rate_no_repeats_is_zero():
    assert repetition_rate("a b c d e f g", n=2) == 0.0


def test_repetition_rate_full_repeats_is_high():
    """Same trigram repeated → repetition rate near 1."""
    text = "the quick fox " * 5
    assert repetition_rate(text, n=3) > 0.5


def test_self_bleu_identical_generations_is_one():
    gens = ["the cat sat on the mat", "the cat sat on the mat", "the cat sat on the mat"]
    score = self_bleu(gens, n=2)
    assert score > 0.99


def test_self_bleu_disjoint_generations_low():
    gens = ["alpha beta gamma", "delta epsilon zeta", "iota kappa lambda"]
    score = self_bleu(gens, n=2)
    assert score < 0.05


# --------------------------------------------------------------------------- #
# LLM judge
# --------------------------------------------------------------------------- #


def test_judge_compare_picks_a():
    judge = LLMJudge(judge_fn=lambda p: "[[A]] A is clearly better", swap_for_position_bias=False)
    v = judge.compare("Q?", "good answer", "bad answer")
    assert v.winner == "A"


def test_judge_position_swap_consensus_required():
    """If first call says A but swap-call disagrees, verdict is tie."""
    calls = [0]

    def judge_fn(p):
        calls[0] += 1
        if calls[0] == 1:
            return "[[A]]"
        return "[[A]]"                                       # second call sees A/B swapped → originally B

    judge = LLMJudge(judge_fn=judge_fn, swap_for_position_bias=True)
    v = judge.compare("Q?", "first", "second")
    assert v.winner == "tie"


def test_judge_position_swap_consensus_agrees():
    calls = [0]

    def judge_fn(p):
        calls[0] += 1
        if calls[0] == 1:
            return "[[A]]"
        # Second call has A/B swapped — original A is now in B-slot, so
        # to agree the judge says [[B]] in that frame.
        return "[[B]]"

    judge = LLMJudge(judge_fn=judge_fn, swap_for_position_bias=True)
    v = judge.compare("Q?", "first", "second")
    assert v.winner == "A"


def test_judge_unparseable_response_is_tie():
    judge = LLMJudge(judge_fn=lambda p: "no verdict here", swap_for_position_bias=False)
    v = judge.compare("Q?", "x", "y")
    assert v.winner == "tie"


def test_judge_win_rate_aggregates_correctly():
    judge = LLMJudge(judge_fn=lambda p: "[[A]]", swap_for_position_bias=False)
    result = judge.win_rate(["q1", "q2", "q3"], ["a1", "a2", "a3"], ["b1", "b2", "b3"])
    assert result.win_rate == 1.0
    assert result.n_wins == 3 and result.n_losses == 0 and result.n_ties == 0


def test_judge_unknown_template_raises():
    with pytest.raises(ValueError, match="template"):
        LLMJudge(judge_fn=lambda p: "", template="nonexistent")


def test_judge_win_rate_length_mismatch_raises():
    judge = LLMJudge(judge_fn=lambda p: "[[A]]", swap_for_position_bias=False)
    with pytest.raises(ValueError, match="length"):
        judge.win_rate(["q1", "q2"], ["a1"], ["b1"])


# --------------------------------------------------------------------------- #
# Needle-in-haystack
# --------------------------------------------------------------------------- #


def test_nih_inserts_needle_at_correct_depth():
    """Depth=0 inserts at the start, depth=1 at the end, depth=0.5 at the middle."""
    eval_ = NIHEvaluator(
        generation_fn=lambda p: "fluorescent_kitten",
        needle="The secret is fluorescent_kitten.",
        answer_substring="fluorescent_kitten",
    )
    h_start = eval_.build_haystack(context_chars=400, depth_fraction=0.0)
    h_mid = eval_.build_haystack(context_chars=400, depth_fraction=0.5)
    h_end = eval_.build_haystack(context_chars=400, depth_fraction=1.0)

    idx_start = h_start.find("The secret")
    idx_mid = h_mid.find("The secret")
    idx_end = h_end.find("The secret")
    # Strict ordering: insertion position increases with depth fraction.
    assert idx_start < idx_mid < idx_end
    assert idx_start == 0                                    # depth=0 → at the very start


def test_nih_evaluate_one_correct_response_marked_found():
    eval_ = NIHEvaluator(
        generation_fn=lambda p: "The password is fluorescent_kitten",
        answer_substring="fluorescent_kitten",
    )
    r = eval_.evaluate_one(context_length=200, depth_fraction=0.5)
    assert r.found is True


def test_nih_evaluate_one_wrong_response_marked_not_found():
    eval_ = NIHEvaluator(
        generation_fn=lambda p: "I don't know",
        answer_substring="fluorescent_kitten",
    )
    r = eval_.evaluate_one(context_length=200, depth_fraction=0.5)
    assert r.found is False


def test_nih_sweep_produces_full_grid():
    eval_ = NIHEvaluator(generation_fn=lambda p: "fluorescent_kitten")
    results = eval_.sweep(context_lengths=[100, 200, 400], depth_fractions=[0.0, 0.5, 1.0])
    assert len(results) == 9
    # Accuracy grid is keyed by (length, depth).
    grid = NIHEvaluator.accuracy_grid(results)
    assert (200, 0.5) in grid


def test_nih_rejects_invalid_depth():
    eval_ = NIHEvaluator(generation_fn=lambda p: "")
    with pytest.raises(ValueError, match="depth_fraction"):
        eval_.build_haystack(100, 1.5)


# --------------------------------------------------------------------------- #
# Evaluation callback
# --------------------------------------------------------------------------- #


class _StubEvaluator:
    """Returns a sequence of pre-set metric values, one per evaluate() call."""

    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.n_calls = 0

    def evaluate(self):
        v = self.sequence[self.n_calls % len(self.sequence)]
        self.n_calls += 1
        return {"perplexity": v}


class _FakeTrainer:
    def __init__(self, step):
        self.global_step = step


def test_callback_fires_at_correct_cadence():
    evalr = _StubEvaluator([10.0, 9.0, 8.0])
    cb = EvaluationCallback(evalr, every_n_steps=100, metric_name="perplexity", mode="min")
    for step in (50, 100, 150, 200):
        cb(_FakeTrainer(step))
    # Fires at 100 and 200 only.
    assert evalr.n_calls == 2
    assert cb.history.steps == [100, 200]


def test_callback_records_history():
    evalr = _StubEvaluator([5.0, 4.0])
    cb = EvaluationCallback(evalr, every_n_steps=10, metric_name="perplexity", mode="min")
    cb(_FakeTrainer(10))
    cb(_FakeTrainer(20))
    assert cb.history.metrics["perplexity"] == [5.0, 4.0]


def test_callback_warns_on_regression():
    """metric goes 5 → 4 (improving) → 100 (regression past 1% threshold)."""
    evalr = _StubEvaluator([5.0, 4.0, 100.0])
    cb = EvaluationCallback(evalr, every_n_steps=10, metric_name="perplexity",
                             mode="min", regression_threshold=0.01)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cb(_FakeTrainer(10))
        cb(_FakeTrainer(20))
        cb(_FakeTrainer(30))
    regression_warnings = [w for w in caught if issubclass(w.category, RegressionWarning)]
    assert len(regression_warnings) == 1


def test_callback_max_mode_handles_higher_is_better():
    """Accuracy goes 0.7 → 0.8 → 0.5 (regression in max mode)."""
    evalr = _StubEvaluator([0.7, 0.8, 0.5])
    cb = EvaluationCallback(evalr, every_n_steps=10, metric_name="perplexity",
                             mode="max", regression_threshold=0.05)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cb(_FakeTrainer(10))
        cb(_FakeTrainer(20))
        cb(_FakeTrainer(30))
    assert any(issubclass(w.category, RegressionWarning) for w in caught)


def test_callback_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        EvaluationCallback(_StubEvaluator([1.0]), mode="huh")


def test_evaluation_history_best_min_picks_lowest():
    h = EvaluationHistory()
    h.record(10, {"loss": 5.0})
    h.record(20, {"loss": 3.0})
    h.record(30, {"loss": 4.0})
    assert h.best("loss", mode="min") == 3.0
    assert h.best("loss", mode="max") == 5.0


# --------------------------------------------------------------------------- #
# MMLU
# --------------------------------------------------------------------------- #


def test_mmlu_with_dummy_model_runs_end_to_end():
    """The dummy model always picks token 1, which corresponds to whichever
    of A/B/C/D maps to that id in the toy tokenizer. We just verify the
    pipeline runs without error and returns a valid result.
    """
    tok = _DummyTokenizer()
    # Pre-encode A B C D so the tokenizer has them.
    for letter in (" A", " B", " C", " D"):
        tok.encode(letter)
    model = _DummyCausalLM(vocab_size=100)
    evalr = MMLUEvaluator(model=model, tokenizer=tok, k_shot=0)
    items = [
        {"question": "Q1", "choices": ["x", "y", "z", "w"], "answer": 0, "subject": "math"},
        {"question": "Q2", "choices": ["a", "b", "c", "d"], "answer": 2, "subject": "history"},
    ]
    result = evalr.evaluate(items)
    assert 0.0 <= result.overall_accuracy <= 1.0
    assert result.n_total == 2
    assert "math" in result.by_subject and "history" in result.by_subject


# --------------------------------------------------------------------------- #
# HumanEval
# --------------------------------------------------------------------------- #


def test_humaneval_pass_when_completion_correct():
    problem = HumanEvalProblem(
        task_id="add",
        prompt="def add(a, b):\n    ",
        test="def check(fn):\n    assert fn(2, 3) == 5\n    assert fn(0, 0) == 0",
        entry_point="add",
    )
    evalr = HumanEvalEvaluator(
        generation_fn=lambda p, n: ["return a + b"] * n,
        n_samples=1, k=[1], timeout_s=10.0,
    )
    result = evalr.evaluate([problem])
    assert result.pass_at_k[1] == 1.0


def test_humaneval_fail_when_completion_wrong():
    problem = HumanEvalProblem(
        task_id="bad", prompt="def add(a, b):\n    ",
        test="def check(fn):\n    assert fn(2, 3) == 5",
        entry_point="add",
    )
    evalr = HumanEvalEvaluator(
        generation_fn=lambda p, n: ["return 99"] * n, n_samples=1, k=[1], timeout_s=10.0,
    )
    result = evalr.evaluate([problem])
    assert result.pass_at_k[1] == 0.0


def test_humaneval_timeout_marks_failure():
    problem = HumanEvalProblem(
        task_id="loop", prompt="def f():\n    ",
        test="def check(fn):\n    fn()",
        entry_point="f",
    )
    evalr = HumanEvalEvaluator(
        generation_fn=lambda p, n: ["while True: pass"] * n,
        n_samples=1, k=[1], timeout_s=0.5,
    )
    result = evalr.evaluate([problem])
    assert result.pass_at_k[1] == 0.0


def test_humaneval_pass_at_k_estimator_correct():
    """3 of 4 samples correct, k=2 — pass@2 = 1 - (1*0)/(4*3) = 1 - 0 = 1.0
    (exact formula: 1 - C(n-c, k)/C(n, k) = 1 - C(1, 2)/C(4, 2) = 1 - 0/6 = 1).
    """
    problem = HumanEvalProblem(
        task_id="add", prompt="def add(a, b):\n    ",
        test="def check(fn):\n    assert fn(2, 3) == 5",
        entry_point="add",
    )
    completions = ["return a + b", "return a + b", "return a + b", "return 0"]
    evalr = HumanEvalEvaluator(
        generation_fn=lambda p, n: completions[:n],
        n_samples=4, k=[1, 2], timeout_s=10.0,
    )
    result = evalr.evaluate([problem])
    assert result.pass_at_k[1] == 0.75                       # 3/4 samples passed
    assert result.pass_at_k[2] == 1.0                         # def find one passing pair


def test_humaneval_rejects_k_larger_than_samples():
    with pytest.raises(ValueError, match="pass@"):
        HumanEvalEvaluator(generation_fn=lambda p, n: [], n_samples=2, k=[5])


# --------------------------------------------------------------------------- #
# IFEval
# --------------------------------------------------------------------------- #


def test_ifeval_strict_passes_when_all_predicates_satisfied():
    items = [IFEvalItem(
        prompt="Respond in JSON with key 'name'",
        instructions=[
            {"name": "json_format", "args": {}},
            {"name": "contains_keyword", "args": {"keyword": "name"}},
        ],
    )]
    evalr = IFEvalEvaluator(generation_fn=lambda p: '{"name": "alice"}')
    result = evalr.evaluate(items)
    assert result.strict_accuracy == 1.0
    assert result.loose_accuracy == 1.0


def test_ifeval_strict_fails_loose_passes_when_partial():
    items = [IFEvalItem(
        prompt="...", instructions=[
            {"name": "json_format", "args": {}},
            {"name": "all_caps", "args": {}},
        ],
    )]
    evalr = IFEvalEvaluator(generation_fn=lambda p: '{"name": "alice"}')
    result = evalr.evaluate(items)
    assert result.strict_accuracy == 0.0                      # JSON yes, all-caps no
    assert result.loose_accuracy == 1.0


def test_ifeval_predicate_num_words_max():
    items = [IFEvalItem(
        prompt="...", instructions=[{"name": "num_words_max", "args": {"max": 3}}],
    )]
    evalr = IFEvalEvaluator(generation_fn=lambda p: "one two three")
    assert evalr.evaluate(items).strict_accuracy == 1.0

    evalr_fail = IFEvalEvaluator(generation_fn=lambda p: "one two three four")
    assert evalr_fail.evaluate(items).strict_accuracy == 0.0


def test_ifeval_unknown_predicate_raises():
    items = [IFEvalItem(prompt="x", instructions=[{"name": "imaginary", "args": {}}])]
    evalr = IFEvalEvaluator(generation_fn=lambda p: "x")
    with pytest.raises(KeyError, match="unknown predicate"):
        evalr.evaluate(items)


def test_ifeval_custom_predicate_extends_builtins():
    def starts_with_dot(args, response):
        return response.startswith(".")

    items = [IFEvalItem(prompt="x", instructions=[{"name": "dot_start", "args": {}}])]
    evalr = IFEvalEvaluator(
        generation_fn=lambda p: ".hello",
        predicates={"dot_start": starts_with_dot},
    )
    assert evalr.evaluate(items).strict_accuracy == 1.0


# --------------------------------------------------------------------------- #
# GSM8K
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text,expected", [
    ("The answer is 42.", 42.0),
    ("The answer is $1,234", 1234.0),
    ("After computing, we get #### 17", 17.0),
    ("Let's see... 2 + 2 = 4. So 4 is the answer.", 4.0),
    ("garbage no number here", None),
])
def test_extract_numeric_answer(text, expected):
    assert extract_numeric_answer(text) == expected


def test_gsm8k_correct_answer_scores_one():
    evalr = GSM8KEvaluator(generation_fn=lambda p: "Let me compute. The answer is 42.")
    result = evalr.evaluate([{"question": "Q?", "answer": 42}])
    assert result.accuracy == 1.0


def test_gsm8k_wrong_answer_scores_zero():
    evalr = GSM8KEvaluator(generation_fn=lambda p: "The answer is 5.")
    result = evalr.evaluate([{"question": "Q?", "answer": 42}])
    assert result.accuracy == 0.0


def test_gsm8k_no_extracted_answer_scores_zero():
    evalr = GSM8KEvaluator(generation_fn=lambda p: "I cannot solve this.")
    result = evalr.evaluate([{"question": "Q?", "answer": 42}])
    assert result.accuracy == 0.0
