"""Tests for the curriculum learning module: scorers, pacing functions,
CurriculumDataset, SelfPacedSampler, and the trainer-hook integration.
"""

import math

import pytest
import torch
import torch.nn as nn

from llm_pipeline.data import (
    CompositeScorer,
    CurriculumDataset,
    CurriculumStepHook,
    ExponentialPacing,
    LengthScorer,
    LinearPacing,
    MetadataScorer,
    PerplexityScorer,
    SelfPacedSampler,
    SqrtPacing,
    StepPacing,
)


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #


def test_length_scorer_shorter_is_easier():
    s = LengthScorer(field="input_ids")
    short = {"input_ids": [1, 2, 3]}
    long = {"input_ids": [1] * 100}
    assert s(short) < s(long)


def test_length_scorer_with_field_fn():
    s = LengthScorer(field_fn=lambda ex: ex["text"].split())
    assert s({"text": "a b c"}) == 3.0
    assert s({"text": "one"}) == 1.0


def test_metadata_scorer_reads_named_field():
    s = MetadataScorer(field="hardness")
    assert s({"hardness": 0.7}) == 0.7
    assert s({"hardness": 5}) == 5.0


def test_composite_scorer_weighted_sum():
    s1 = LengthScorer()
    s2 = MetadataScorer(field="d")
    composite = CompositeScorer([s1, s2], weights=[0.5, 2.0])
    ex = {"input_ids": [1, 2, 3, 4], "d": 1.5}                # length=4, d=1.5
    assert composite(ex) == 0.5 * 4.0 + 2.0 * 1.5             # = 5.0


def test_composite_scorer_uniform_weights_default():
    s1 = LengthScorer()
    s2 = MetadataScorer(field="d")
    composite = CompositeScorer([s1, s2])
    ex = {"input_ids": [1, 2], "d": 3.0}
    assert composite(ex) == 2.0 + 3.0


def test_composite_scorer_rejects_weight_count_mismatch():
    with pytest.raises(ValueError, match="weights"):
        CompositeScorer([LengthScorer()], weights=[1.0, 2.0])


def test_perplexity_scorer_uses_model_loss():
    """A tiny dummy model: each call returns a per-example loss tied to
    sequence length so we can verify the scorer pipes loss through correctly.
    """

    class _DummyOut:
        def __init__(self, loss):
            self.loss = loss

    class _DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._p = nn.Parameter(torch.zeros(1))             # so .parameters() works

        def forward(self, input_ids, labels=None):
            # "Loss" = mean of input ids — deterministic per example.
            return _DummyOut(loss=input_ids.float().mean())

    scorer = PerplexityScorer(_DummyModel(), device=torch.device("cpu"))
    ex_a = {"input_ids": [1, 1, 1]}                            # mean = 1
    ex_b = {"input_ids": [5, 5, 5]}                            # mean = 5
    assert scorer(ex_a) < scorer(ex_b)


# --------------------------------------------------------------------------- #
# Pacing functions
# --------------------------------------------------------------------------- #


def test_linear_pacing_endpoints():
    p = LinearPacing(start_frac=0.1, end_frac=1.0, total_steps=100)
    assert p(0) == 0.1
    assert p(100) == 1.0
    assert p(-5) == 0.1                                         # clamps below
    assert p(500) == 1.0                                         # clamps above


def test_linear_pacing_midpoint():
    p = LinearPacing(start_frac=0.0, end_frac=1.0, total_steps=100)
    assert abs(p(50) - 0.5) < 1e-9


def test_sqrt_pacing_grows_faster_than_linear_initially():
    """At step 1/4 of the way, sqrt(0.25) = 0.5 (50% of the way) vs
    linear's 25%. Sqrt expansion should be ahead.
    """
    sqrt_p = SqrtPacing(start_frac=0.0, end_frac=1.0, total_steps=100)
    lin_p = LinearPacing(start_frac=0.0, end_frac=1.0, total_steps=100)
    assert sqrt_p(25) > lin_p(25)
    assert abs(sqrt_p(25) - 0.5) < 1e-6                         # exactly 0.5 at quarter step


def test_exponential_pacing_grows_slower_than_linear_initially():
    """At step 1/4 of the way, exponential lags linear (slow start)."""
    exp_p = ExponentialPacing(start_frac=0.1, end_frac=1.0, total_steps=100)
    lin_p = LinearPacing(start_frac=0.1, end_frac=1.0, total_steps=100)
    assert exp_p(25) < lin_p(25)


def test_exponential_pacing_rejects_zero_start():
    with pytest.raises(ValueError, match="start_frac"):
        ExponentialPacing(start_frac=0.0, end_frac=1.0, total_steps=100)


def test_step_pacing_holds_at_each_checkpoint():
    p = StepPacing(checkpoints=[(0, 0.1), (1000, 0.5), (5000, 1.0)])
    assert p(0) == 0.1
    assert p(500) == 0.1                                         # still in first plateau
    assert p(1000) == 0.5
    assert p(2000) == 0.5
    assert p(5000) == 1.0
    assert p(99999) == 1.0


def test_step_pacing_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="fraction"):
        StepPacing(checkpoints=[(0, 1.5)])


def test_pacing_validates_fraction_ordering():
    with pytest.raises(ValueError, match="start_frac"):
        LinearPacing(start_frac=0.8, end_frac=0.5, total_steps=100)


# --------------------------------------------------------------------------- #
# CurriculumDataset
# --------------------------------------------------------------------------- #


def _toy_dataset() -> list:
    """20 examples with monotonically increasing length (1 to 20 tokens)."""
    return [{"input_ids": list(range(i + 1))} for i in range(20)]


def test_curriculum_low_pacing_exposes_only_easy_examples():
    base = _toy_dataset()
    ds = CurriculumDataset(
        base, scorer=LengthScorer(),
        pacing=LinearPacing(start_frac=0.1, end_frac=1.0, total_steps=100),
    )
    ds.set_step(0)                                              # 10% → 2 examples
    assert len(ds) == 2
    # The two exposed examples should be the shortest (lengths 1 and 2).
    exposed_lengths = sorted(len(ex["input_ids"]) for ex in ds)
    assert exposed_lengths == [1, 2]


def test_curriculum_full_pacing_exposes_everything():
    base = _toy_dataset()
    ds = CurriculumDataset(
        base, scorer=LengthScorer(),
        pacing=LinearPacing(start_frac=0.1, end_frac=1.0, total_steps=100),
    )
    ds.set_step(100)
    assert len(ds) == len(base)


def test_curriculum_set_step_grows_exposed_set_monotonically():
    base = _toy_dataset()
    ds = CurriculumDataset(
        base, scorer=LengthScorer(),
        pacing=LinearPacing(start_frac=0.1, end_frac=1.0, total_steps=100),
    )
    sizes = []
    for step in (0, 25, 50, 75, 100):
        ds.set_step(step)
        sizes.append(len(ds))
    assert sizes == sorted(sizes)                                # non-decreasing


def test_curriculum_difficulty_scores_indexable():
    base = _toy_dataset()
    ds = CurriculumDataset(
        base, scorer=LengthScorer(),
        pacing=LinearPacing(start_frac=1.0, end_frac=1.0, total_steps=100),
    )
    scores = ds.difficulty_scores
    assert len(scores) == len(base)
    # Scores match LengthScorer semantics: index i has length i+1.
    for i, s in enumerate(scores):
        assert s == float(i + 1)


def test_curriculum_step_hook_calls_set_step():
    base = _toy_dataset()
    ds = CurriculumDataset(
        base, scorer=LengthScorer(),
        pacing=LinearPacing(start_frac=0.1, end_frac=1.0, total_steps=100),
    )
    hook = CurriculumStepHook(ds)

    class _FakeTrainer:
        global_step = 50

    hook(_FakeTrainer())
    assert ds.current_step == 50
    assert ds.current_fraction == LinearPacing(0.1, 1.0, 100)(50)


def test_curriculum_step_hook_rejects_non_curriculum_object():
    with pytest.raises(TypeError, match="set_step"):
        CurriculumStepHook(object())


# --------------------------------------------------------------------------- #
# SelfPacedSampler
# --------------------------------------------------------------------------- #


def test_self_paced_drops_examples_above_threshold():
    base = list(range(10))                                       # 0..9
    # Loss(model, ex) = ex value (0 → easy, 9 → hard).
    sampler = SelfPacedSampler(
        base=base,
        example_loss_fn=lambda model, ex: float(ex),
        threshold_fn=lambda step: 4.5,                           # admit losses ≤ 4.5
    )
    sampler.evaluate(model=None)
    sampler.set_step(0)
    included = list(sampler)
    assert included == [0, 1, 2, 3, 4]


def test_self_paced_threshold_relaxes_with_steps():
    base = list(range(10))

    def threshold_fn(step):
        return float(step) / 100 * 9                              # 0 → 0, 100 → 9

    sampler = SelfPacedSampler(
        base=base,
        example_loss_fn=lambda model, ex: float(ex),
        threshold_fn=threshold_fn,
    )
    sampler.evaluate(model=None)

    sampler.set_step(0)
    early = len(sampler)
    sampler.set_step(50)
    middle = len(sampler)
    sampler.set_step(100)
    late = len(sampler)
    assert early <= middle <= late
    assert late == len(base)                                      # threshold ≥ max loss


def test_self_paced_should_reevaluate_respects_cadence():
    sampler = SelfPacedSampler(
        base=[1, 2, 3],
        example_loss_fn=lambda m, ex: 0.0,
        threshold_fn=lambda s: 1.0,
        re_evaluate_every_n_steps=100,
    )
    assert sampler.should_re_evaluate()                            # never evaluated → True
    sampler.evaluate(model=None)
    sampler.set_step(50)
    assert not sampler.should_re_evaluate()                        # within cadence
    sampler.set_step(150)
    assert sampler.should_re_evaluate()                            # past cadence


def test_self_paced_indexable_after_evaluate():
    base = [10, 20, 30, 40]
    sampler = SelfPacedSampler(
        base=base,
        example_loss_fn=lambda m, ex: float(ex) / 10,
        threshold_fn=lambda s: 2.5,
    )
    sampler.evaluate(model=None)
    sampler.set_step(0)
    assert len(sampler) == 2
    assert sampler[0] == 10                                        # first kept by index order
    assert sampler[1] == 20
