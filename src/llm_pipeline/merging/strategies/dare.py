"""DARE: Drop And REscale merge.

Reference: Yu et al., "Language Models are Super Mario" (DARE), 2023.

For each task delta `Δ = task - base`, randomly drop a fraction `p` of entries
and rescale survivors by `1/(1-p)` so the expectation is preserved. The
rescaled deltas are summed (or fed into TIES on top — `dare_ties`).
"""

from typing import Dict, Optional, Sequence

import torch

from .ties import ties_merge


def _dare(t: torch.Tensor, drop_p: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if drop_p <= 0.0:
        return t
    if drop_p >= 1.0:
        return torch.zeros_like(t)
    keep_mask = (torch.rand(t.shape, generator=generator, device=t.device) >= drop_p).to(t.dtype)
    return t * keep_mask / (1.0 - drop_p)


def dare_merge(
    base_state: Dict[str, torch.Tensor],
    task_states: Sequence[Dict[str, torch.Tensor]],
    drop_p: float = 0.5,
    alphas: Optional[Sequence[float]] = None,
    use_ties: bool = False,
    ties_density: float = 0.2,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """DARE merge with optional TIES on top (DARE-TIES)."""
    if not task_states:
        raise ValueError("dare_merge requires at least one task state_dict.")
    alphas = list(alphas) if alphas is not None else [1.0] * len(task_states)
    if len(alphas) != len(task_states):
        raise ValueError(f"len(alphas)={len(alphas)} != len(task_states)={len(task_states)}")

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    rescaled_states = []
    for ts in task_states:
        rescaled = {}
        for k, base_v in base_state.items():
            if k not in ts or not base_v.is_floating_point():
                rescaled[k] = ts.get(k, base_v).clone()
                continue
            if ts[k].shape != base_v.shape:
                raise ValueError(
                    f"Shape mismatch for key '{k}': base {tuple(base_v.shape)} "
                    f"vs task {tuple(ts[k].shape)}"
                )
            delta = ts[k].to(torch.float32) - base_v.to(torch.float32)
            kept = _dare(delta, drop_p, g)
            rescaled[k] = (base_v.to(torch.float32) + kept).to(base_v.dtype)
        rescaled_states.append(rescaled)

    if use_ties:
        return ties_merge(base_state, rescaled_states, density=ties_density, alphas=alphas)

    # Otherwise sum the rescaled deltas back onto the base.
    out: Dict[str, torch.Tensor] = {}
    for k, base_v in base_state.items():
        if not base_v.is_floating_point():
            out[k] = base_v.clone()
            continue
        acc = base_v.to(torch.float32).clone()
        for alpha, rs in zip(alphas, rescaled_states):
            if k not in rs:
                continue
            acc.add_(rs[k].to(torch.float32) - base_v.to(torch.float32), alpha=float(alpha))
        out[k] = acc.to(base_v.dtype)
    return out
