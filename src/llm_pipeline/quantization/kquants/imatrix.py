"""Imatrix calibration: per-input-channel importance for nn.Linear weights.

For a linear layer ``y = W·x + b`` with ``W ∈ R^{out × in}``, the squared
output-error contribution of a quantization perturbation ``ΔW`` is roughly

    E ≈ Σ_i (W_i - W̃_i)² · E[x_i²]

where ``E[x_i²]`` is the empirical second moment of input channel ``i``
over a calibration corpus. The importance vector ``imat ∈ R^{in}`` is
exactly that quantity. K-quant encoders accept it as ``importance`` and
prefer rounding directions that minimize ``imat`` -weighted error.

We hook every ``nn.Linear`` (or ``ColumnParallelLinear`` / etc.; we just
look for ``in_features``) and accumulate the per-input-channel sum of
squared activations.

Usage::

    cal = ImatrixCalibrator(model)
    cal.attach()
    for batch in calibration_loader:
        with torch.no_grad():
            model(**batch)
    imatrix = cal.detach()       # detaches hooks AND returns the result
    # imatrix is a dict: { module_qualified_name: torch.Tensor[in_features] }
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


Imatrix = Dict[str, torch.Tensor]


@dataclass
class ImatrixCalibrator:
    """Hooks all linear-like modules and records their input second moments.

    A module is "linear-like" if it has integer ``in_features`` and a
    ``weight`` parameter whose last dim equals ``in_features``. This covers
    ``nn.Linear``, ``ColumnParallelLinear``, ``RowParallelLinear``, and
    HuggingFace's various Linear-derived classes.

    The result for module ``m`` after calibration is
    ``Σ_b Σ_t x_b_t · x_b_t / N`` where ``N`` is the total number of
    "rows" of ``x`` seen (so it's a mean second moment, not a sum — easier
    to interpret across calibration sizes).
    """

    model: nn.Module
    sums: Dict[str, torch.Tensor] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    _handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def attach(self) -> None:
        if self._handles:
            return  # already attached

        for name, module in self.model.named_modules():
            if not hasattr(module, "in_features") or not hasattr(module, "weight"):
                continue
            if not isinstance(module.in_features, int):
                continue

            def _hook(mod, inputs, _output, _name=name, _in=module.in_features):
                if not inputs:
                    return
                x = inputs[0]
                if not isinstance(x, torch.Tensor):
                    return
                if x.shape[-1] != _in:
                    return
                xf = x.detach().to(torch.float32).reshape(-1, _in)
                running = self.sums.get(_name)
                sq = (xf ** 2).sum(dim=0)
                if running is None:
                    self.sums[_name] = sq.cpu()
                else:
                    self.sums[_name] = running + sq.cpu()
                self.counts[_name] = self.counts.get(_name, 0) + xf.shape[0]

            self._handles.append(module.register_forward_hook(_hook))

    def detach(self) -> Imatrix:
        for h in self._handles:
            h.remove()
        self._handles = []
        out: Imatrix = {}
        for name, sq in self.sums.items():
            n = max(self.counts.get(name, 1), 1)
            out[name] = sq / n
        return out

    def __enter__(self) -> "ImatrixCalibrator":
        self.attach()
        return self

    def __exit__(self, *exc) -> None:
        # Caller is expected to use ``detach()`` to recover the result; we
        # only ensure hooks are removed on context exit.
        if self._handles:
            for h in self._handles:
                h.remove()
            self._handles = []
