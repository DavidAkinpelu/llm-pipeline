"""TIES-Merging.

Reference: Yadav et al., "TIES-Merging: Resolving Interference When Merging
Models" (NeurIPS 2023).

Steps:
  1. **Trim**: keep top-k% magnitude entries of each task vector (delta vs base).
  2. **Elect sign**: per-element sign chosen by sum (or count) of trimmed deltas.
  3. **Disjoint merge**: average only deltas matching the elected sign.
  4. Apply the merged delta to the base.
"""

from typing import Dict, Optional, Sequence

import torch


def _trim_top_k(t: torch.Tensor, density: float) -> torch.Tensor:
    """Zero out all but the top-`density` fraction of |t| entries."""
    if density <= 0.0:
        return torch.zeros_like(t)
    if density >= 1.0:
        return t
    n = t.numel()
    k = max(1, int(n * density))
    flat = t.view(-1).abs()
    threshold = torch.kthvalue(flat, n - k + 1).values
    mask = (t.abs() >= threshold).to(t.dtype)
    return t * mask


def ties_merge(
    base_state: Dict[str, torch.Tensor],
    task_states: Sequence[Dict[str, torch.Tensor]],
    density: float = 0.2,
    sign_method: str = "sum",  # "sum" or "count"
    alphas: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    if not task_states:
        raise ValueError("ties_merge requires at least one task state_dict.")
    alphas = list(alphas) if alphas is not None else [1.0] * len(task_states)
    if len(alphas) != len(task_states):
        raise ValueError(f"len(alphas)={len(alphas)} != len(task_states)={len(task_states)}")

    out: Dict[str, torch.Tensor] = {}
    for k, base_v in base_state.items():
        if not base_v.is_floating_point():
            out[k] = base_v.clone()
            continue
        deltas = []
        for alpha, ts in zip(alphas, task_states):
            if k not in ts:
                continue
            if ts[k].shape != base_v.shape:
                raise ValueError(
                    f"Shape mismatch for key '{k}': base {tuple(base_v.shape)} "
                    f"vs task {tuple(ts[k].shape)}"
                )
            d = (ts[k].to(torch.float32) - base_v.to(torch.float32)) * float(alpha)
            deltas.append(_trim_top_k(d, density))
        if not deltas:
            out[k] = base_v.clone()
            continue
        stacked = torch.stack(deltas, dim=0)  # [n, *shape]

        if sign_method == "sum":
            elected_sign = torch.sign(stacked.sum(dim=0))
        elif sign_method == "count":
            elected_sign = torch.sign(torch.sign(stacked).sum(dim=0))
        else:
            raise ValueError(f"Unknown sign_method: {sign_method}")

        agree = torch.sign(stacked) == elected_sign.unsqueeze(0)
        agree = agree & (stacked != 0)
        masked = torch.where(agree, stacked, torch.zeros_like(stacked))
        denom = agree.sum(dim=0).clamp(min=1).to(torch.float32)
        merged_delta = masked.sum(dim=0) / denom
        out[k] = (base_v.to(torch.float32) + merged_delta).to(base_v.dtype)
    return out
