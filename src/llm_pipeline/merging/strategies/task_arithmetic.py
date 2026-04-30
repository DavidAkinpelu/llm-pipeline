"""Task Arithmetic merge.

`merged = base + sum_i alpha_i * (task_i - base)`

Reference: Ilharco et al., "Editing Models with Task Arithmetic" (2023).
"""

from typing import Dict, Optional, Sequence

import torch


def task_arithmetic(
    base_state: Dict[str, torch.Tensor],
    task_states: Sequence[Dict[str, torch.Tensor]],
    alphas: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    if not task_states:
        raise ValueError("task_arithmetic requires at least one task state_dict.")
    alphas = list(alphas) if alphas is not None else [1.0] * len(task_states)
    if len(alphas) != len(task_states):
        raise ValueError(f"len(alphas)={len(alphas)} != len(task_states)={len(task_states)}")

    out: Dict[str, torch.Tensor] = {}
    for k, base_v in base_state.items():
        if not base_v.is_floating_point():
            out[k] = base_v.clone()
            continue
        acc = base_v.to(torch.float32).clone()
        for alpha, ts in zip(alphas, task_states):
            if k not in ts:
                continue
            delta = ts[k].to(torch.float32) - base_v.to(torch.float32)
            acc.add_(delta, alpha=float(alpha))
        out[k] = acc.to(base_v.dtype)
    return out
