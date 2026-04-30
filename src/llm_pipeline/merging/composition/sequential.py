"""Sequential composition of merges.

Useful when you want to merge `A + B`, then merge the result with `C` using
a different strategy.
"""

from typing import Callable, Dict, List

import torch


MergeFn = Callable[..., Dict[str, torch.Tensor]]


def sequential_merge(
    state_dicts: List[Dict[str, torch.Tensor]],
    steps: List[Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """Apply each step in order, threading the result forward.

    `steps[i]` is called with `(accumulator, state_dicts[i+1])` and must
    return the new accumulator. The first state_dict seeds the accumulator.
    """
    if len(state_dicts) < 2:
        raise ValueError("Need at least two state_dicts to compose.")
    if len(steps) != len(state_dicts) - 1:
        raise ValueError("Need one merge step per pair of state_dicts.")

    acc = state_dicts[0]
    for i, step in enumerate(steps):
        acc = step(acc, state_dicts[i + 1])
    return acc
