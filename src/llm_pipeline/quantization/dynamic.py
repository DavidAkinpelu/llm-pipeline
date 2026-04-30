"""Dynamic per-tensor quantization — Unsloth's "UD" idea.

Standard K-quants apply one bit-width to the whole model. UD-quants don't.
Transformer layers are not equally sensitive to precision loss:

  * Embedding & ``lm_head`` carry structural information the rest of the
    model depends on. Crushing them tanks quality.
  * Attention's ``v_proj`` and the FFN ``down_proj`` are also sensitive.
  * The *bulk* of FFN tensors are redundant and tolerate aggressive
    quantization without measurable degradation.

This module implements that policy via a **per-tensor sensitivity probe**:

  1. Run the FP reference model on a calibration batch — save the
     output logits.
  2. For each candidate (tensor, format) pair, replace just that tensor
     with its dequantized version and re-run forward; the KL divergence
     between the new logits and the reference logits is the *sensitivity
     cost* of that quantization choice.
  3. Pick the smallest format (lowest bits/weight) whose KL stays below a
     budget — typical recipe is "as many bits as you can afford while the
     average size targets a Q4_K_M-equivalent budget".

The ``UDQuantizer.quantize`` method runs the probe and emits a per-tensor
format map that the existing ``Quantizer`` pipeline can consume.

For the **XL suffix** (e.g. ``UD-Q4_K_XL``), embed_tokens and lm_head are
forced to Q8_K regardless of probe outcome — the canonical Unsloth recipe.

This is an *educational* implementation: the probe is N forward passes
per (tensor, format) pair, which is O(layers × candidate_formats). For a
production system you'd batch the probes or use a single-pass Hessian
proxy. The result here is the same per-tensor allocation map.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kquants import QuantMethod, get_block_shape
from .quantizer import (
    QuantizedModel,
    QuantizedTensor,
    Quantizer,
    _BASE_DECODERS,
    _BASE_ENCODERS,
)


# Default candidate formats, ordered cheapest-first.
DEFAULT_CANDIDATES: Tuple[QuantMethod, ...] = (
    QuantMethod.Q4_K,
    QuantMethod.Q5_K,
    QuantMethod.Q6_K,
    QuantMethod.Q8_K,
)


@dataclass
class SensitivityReport:
    """The per-tensor sensitivity matrix the probe produces."""
    # ``kl[name][method]`` = KL(quantized_only_this_tensor || fp_reference).
    kl: Dict[str, Dict[QuantMethod, float]] = field(default_factory=dict)
    # ``size_bytes[name][method]`` = number of bytes the encoded tensor uses.
    size_bytes: Dict[str, Dict[QuantMethod, int]] = field(default_factory=dict)
    fp_reference_norm: float = 0.0


def _kl_logits(
    p_logits: torch.Tensor, q_logits: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> float:
    """KL(softmax(p) || softmax(q)) averaged over positions."""
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    # KL(P || Q) = Σ P · (log P - log Q)
    diff = (p.exp() * (p - q)).sum(dim=-1)
    if mask is not None:
        diff = diff * mask.to(diff.dtype)
        return (diff.sum() / mask.sum().clamp(min=1)).item()
    return diff.mean().item()


@torch.no_grad()
def _logits_for_calibration(model: nn.Module, batches) -> List[torch.Tensor]:
    """Run model over the calibration batches; return one logits tensor per batch."""
    out = []
    for b in batches:
        bb = {k: v for k, v in b.items() if k != "labels"}
        outputs = model(**bb)
        logits = getattr(outputs, "logits", outputs)
        out.append(logits.detach().clone())
    return out


def _patch_with_quantized(
    model: nn.Module,
    name: str,
    encoded_tensor: torch.Tensor,
):
    """Temporarily replace ``model.<name>`` with ``encoded_tensor``.

    Returns a callable that restores the original weight when invoked.
    The replacement is in-place at the parameter level (no copies of the
    rest of the state-dict).
    """
    state = dict(model.named_parameters())
    if name not in state:
        raise KeyError(name)
    param = state[name]
    original = param.data.clone()
    param.data.copy_(encoded_tensor.to(param.dtype).to(param.device))

    def _restore():
        param.data.copy_(original)

    return _restore


def _encode_then_decode(
    tensor: torch.Tensor, method: QuantMethod, importance: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, int]:
    """Round-trip ``tensor`` through ``method`` and return (recovered, bytes)."""
    encoder = _BASE_ENCODERS[method]
    decoder = _BASE_DECODERS[method]
    blob, shape = encoder(tensor, importance=importance)
    recovered = decoder(blob, shape)
    return recovered, len(blob)


@dataclass
class UDQuantizerConfig:
    """Knobs for the dynamic quantizer.

    - ``candidates``: which formats the probe is allowed to consider.
    - ``kl_budget``: per-tensor KL ceiling. The cheapest format with KL ≤
      budget wins. If no format satisfies, falls back to the most precise.
    - ``xl``: when ``True``, ``embed_tokens`` and ``lm_head`` are pinned
      to Q8_K (the standard Unsloth ``_XL`` policy).
    - ``protected_keywords``: names containing any of these keywords
      always get the most-precise candidate, regardless of probe result.
    """
    candidates: Tuple[QuantMethod, ...] = DEFAULT_CANDIDATES
    kl_budget: float = 5e-3
    xl: bool = True
    protected_keywords: Tuple[str, ...] = ()


class UDQuantizer:
    """Dynamic per-tensor quantizer.

    Workflow::

        ud = UDQuantizer(UDQuantizerConfig(kl_budget=5e-3, xl=True))
        report = ud.probe(model, calibration_batches)
        quantized_model = ud.quantize(model, report=report,
                                      imatrix=optional_imatrix_dict)
    """

    def __init__(self, config: Optional[UDQuantizerConfig] = None):
        self.config = config or UDQuantizerConfig()

    # ------------------------------------------------------------------ #
    # Probe
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def probe(self, model: nn.Module, batches, top_n: Optional[int] = None) -> SensitivityReport:
        """Measure each (tensor, format) sensitivity by single-tensor patching.

        ``top_n`` (if given) limits the probe to the ``top_n`` largest
        parameters by element count — cheap heuristic to skip layer norms
        and other small tensors.
        """
        cfg = self.config
        report = SensitivityReport()

        # Reference logits with the *original* model.
        ref_logits = _logits_for_calibration(model, batches)
        report.fp_reference_norm = float(sum(l.float().norm() for l in ref_logits))

        # Decide which tensors to probe.
        params = [(name, p) for name, p in model.named_parameters()
                  if p.dim() == 2 and p.numel() >= 64]
        if top_n is not None:
            params.sort(key=lambda kv: -kv[1].numel())
            params = params[:top_n]

        for name, p in params:
            report.kl[name] = {}
            report.size_bytes[name] = {}
            for method in cfg.candidates:
                # Round-trip to get the dequantized tensor.
                recovered, n_bytes = _encode_then_decode(p.data, method, importance=None)
                report.size_bytes[name][method] = n_bytes
                # Patch, run, restore.
                restore = _patch_with_quantized(model, name, recovered)
                try:
                    new_logits = _logits_for_calibration(model, batches)
                    kl_total = 0.0
                    n_pos = 0
                    for ref, neu in zip(ref_logits, new_logits):
                        # Average KL over the batch.
                        n_b = ref.shape[0] * ref.shape[1]
                        kl_total += _kl_logits(ref, neu) * n_b
                        n_pos += n_b
                    kl = kl_total / max(n_pos, 1)
                finally:
                    restore()
                report.kl[name][method] = kl
        return report

    # ------------------------------------------------------------------ #
    # Quantize
    # ------------------------------------------------------------------ #

    def assign_formats(self, report: SensitivityReport, model: nn.Module) -> Dict[str, QuantMethod]:
        """Pick the cheapest format per tensor that satisfies ``kl_budget``.

        Returns a ``{tensor_name: QuantMethod}`` map. Tensors not in the
        probe report (norms, biases, embeddings) get assigned per the XL /
        protected-keywords policies.
        """
        cfg = self.config
        # Cheapest-first ordering by bits/weight.
        ordered = sorted(cfg.candidates, key=lambda m: get_block_shape(m).bits_per_weight)
        most_precise = ordered[-1]

        assignments: Dict[str, QuantMethod] = {}
        for name, p in model.named_parameters():
            # Always pin embed_tokens / lm_head when XL is on.
            if cfg.xl and ("embed_tokens" in name or "lm_head" in name):
                assignments[name] = QuantMethod.Q8_K
                continue
            if any(kw in name for kw in cfg.protected_keywords):
                assignments[name] = most_precise
                continue
            kls = report.kl.get(name)
            if not kls:
                continue
            # Cheapest format with KL ≤ budget.
            chosen = most_precise
            for m in ordered:
                if kls.get(m, float("inf")) <= cfg.kl_budget:
                    chosen = m
                    break
            assignments[name] = chosen
        return assignments

    def quantize(
        self,
        model: nn.Module,
        report: SensitivityReport,
        imatrix: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[QuantizedModel, Dict[str, QuantMethod]]:
        """Apply the probe-derived per-tensor allocation. Reuses ``Quantizer``
        for the actual encoding so imatrix integration / skip patterns / etc.
        come along for the ride.
        """
        assignments = self.assign_formats(report, model)
        # Use the standard Quantizer as a thin per-tensor encoder.
        base = Quantizer(method=QuantMethod.Q4_K)
        if imatrix is not None:
            base.set_imatrix(imatrix)

        result = QuantizedModel(method=QuantMethod.Q4_K)  # heterogeneous; see ``method`` per tensor
        for name, p in model.named_parameters():
            if any(pat in name for pat in base.skip_pattern):
                continue
            if p.numel() < 32:
                continue
            method = assignments.get(name, QuantMethod.Q4_K)
            encoder = _BASE_ENCODERS[method]
            importance = base._importance_for(name, p.shape)
            blob, shape = encoder(p.data, importance=importance)
            result.tensors.append(QuantizedTensor(name=name, method=method, shape=shape, blob=blob))
        return result, assignments
