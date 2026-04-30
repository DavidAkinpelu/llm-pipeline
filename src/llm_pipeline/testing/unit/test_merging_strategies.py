"""Unit tests for model-level merging strategies."""

import pytest
import torch

from llm_pipeline.merging import dare_merge, linear_merge, task_arithmetic, ties_merge
from llm_pipeline.merging.strategies.ties import _trim_top_k


def test_linear_merge_weighted_average():
    a = {"w": torch.tensor([1.0, 3.0]), "i": torch.tensor([1, 2])}
    b = {"w": torch.tensor([3.0, 5.0]), "i": torch.tensor([9, 9])}

    out = linear_merge([a, b], weights=[0.25, 0.75])

    torch.testing.assert_close(out["w"], torch.tensor([2.5, 4.5]))
    torch.testing.assert_close(out["i"], a["i"])


def test_task_arithmetic_rejects_alpha_length_mismatch():
    base = {"w": torch.tensor([0.0, 0.0])}
    t1 = {"w": torch.tensor([1.0, 0.0])}
    t2 = {"w": torch.tensor([0.0, 2.0])}

    with pytest.raises(ValueError, match="len\\(alphas\\)=1 != len\\(task_states\\)=2"):
        task_arithmetic(base, [t1, t2], alphas=[2.0])


def test_ties_trim_zero_density_drops_everything():
    t = torch.tensor([1.0, -2.0, 3.0, -4.0])

    out = _trim_top_k(t, 0.0)

    torch.testing.assert_close(out, torch.zeros_like(t))


def test_ties_merge_rejects_shape_mismatch():
    base = {"w": torch.tensor([1.0, 2.0])}
    bad = {"w": torch.tensor([[3.0, 4.0]])}

    with pytest.raises(ValueError, match="Shape mismatch for key 'w'"):
        ties_merge(base, [bad])


def test_dare_merge_rejects_alpha_length_mismatch():
    base = {"w": torch.tensor([0.0, 0.0])}
    t1 = {"w": torch.tensor([1.0, 0.0])}
    t2 = {"w": torch.tensor([0.0, 2.0])}

    with pytest.raises(ValueError, match="len\\(alphas\\)=1 != len\\(task_states\\)=2"):
        dare_merge(base, [t1, t2], alphas=[2.0], drop_p=0.0)


def test_dare_merge_rejects_shape_mismatch():
    base = {"w": torch.tensor([1.0, 2.0])}
    bad = {"w": torch.tensor([[3.0, 4.0]])}

    with pytest.raises(ValueError, match="Shape mismatch for key 'w'"):
        dare_merge(base, [bad], drop_p=0.0)
