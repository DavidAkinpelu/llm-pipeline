"""Linear (weighted average) merge of model state_dicts."""

from typing import Dict, List, Optional, Sequence

import torch


def linear_merge(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Weighted average of N state_dicts. Weights default to uniform.

    All state_dicts must share keys and tensor shapes; integer/long buffers
    (e.g. position-id buffers) are taken from the first model.
    """
    if not state_dicts:
        raise ValueError("linear_merge requires at least one state_dict.")
    n = len(state_dicts)
    weights = list(weights) if weights is not None else [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"len(weights)={len(weights)} != len(state_dicts)={n}")

    keys = state_dicts[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = [sd[k] for sd in state_dicts]
        if not tensors[0].is_floating_point():
            out[k] = tensors[0].clone()
            continue
        acc = torch.zeros_like(tensors[0], dtype=torch.float32)
        for w, t in zip(weights, tensors):
            acc.add_(t.to(torch.float32), alpha=float(w))
        out[k] = acc.to(tensors[0].dtype)
    return out
