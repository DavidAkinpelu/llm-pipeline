"""Load HuggingFace ``Qwen/Qwen3.6-*`` checkpoints into our hand-rolled engine.

Most of our parameter names already match the HF state_dict (we deliberately
followed HF naming when porting `Qwen3_5Attention`, `Qwen3_5GatedDeltaNet`,
`Qwen3_5MLP`, etc.) so the bulk of the loader is direct ``param.copy_()``.

The two real translations:

1. **MoE routed experts**. HF stores all experts of one layer as 3D tensors:

       experts.gate_up_proj  shape [E, 2 * intermediate, hidden]
       experts.down_proj     shape [E, hidden, intermediate]

   Our ``Qwen3_5MoeBlock`` uses an ``nn.ModuleList`` of ``Qwen3_5MLP``
   experts, so we split per expert and further split ``gate_up_proj`` into
   ``gate_proj`` (first ``intermediate`` rows) and ``up_proj`` (last
   ``intermediate`` rows).

2. **Skipped components**. The released checkpoints carry ``mtp.*``
   (Multi-Token Prediction head) and ``model.visual.*`` (vision tower)
   weights that our text-only engine doesn't yet implement. The loader
   ignores them by default; pass ``strict=True`` and they will surface
   as unexpected keys.

Usage
-----

```python
from transformers import AutoModelForCausalLM
from llm_pipeline.models.qwen3_5 import Qwen3_5ForCausalLM, qwen3_6_27b
from llm_pipeline.models.qwen3_5.loader import load_qwen3_5_state_dict

hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B")
ours = Qwen3_5ForCausalLM(qwen3_6_27b())
load_qwen3_5_state_dict(ours, hf.state_dict())
```

The HF model can be discarded after the copy. For very large checkpoints
(35B-A3B), prefer ``safetensors`` shard streaming so you don't have to
hold both copies in memory at once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn

from .mlp import Qwen3_5MoeBlock


# Patterns we deliberately skip — they belong to features (MTP head,
# multimodal vision tower) we don't have on the text-only path.
_SKIP_PATTERNS = (
    re.compile(r"^mtp\."),
    re.compile(r"^model\.mtp\."),
    re.compile(r"^model\.visual\."),
)


@dataclass
class LoadReport:
    """Diagnostic info from a state-dict load."""

    loaded: List[str]                          # our-side names that were updated
    skipped_hf_keys: List[str]                 # HF keys ignored by policy (mtp, visual)
    missing_ours: List[str]                    # our-side names with no HF counterpart
    unexpected_hf: List[str]                   # HF keys with no our-side destination

    def summary(self) -> str:
        return (
            f"loaded {len(self.loaded)} params, "
            f"skipped {len(self.skipped_hf_keys)} HF keys (mtp/visual), "
            f"{len(self.missing_ours)} missing, {len(self.unexpected_hf)} unexpected"
        )


def _is_skipped(hf_key: str) -> bool:
    return any(p.search(hf_key) for p in _SKIP_PATTERNS)


def _expert_layer_index(hf_key: str) -> int | None:
    """Return the layer index ``i`` if ``hf_key`` matches
    ``model.layers.{i}.mlp.experts.{gate_up_proj,down_proj}``, else None.
    """
    m = re.match(r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$", hf_key)
    return int(m.group(1)) if m else None


def _expand_moe_experts(
    hf_key: str, hf_tensor: torch.Tensor, model: nn.Module
) -> Dict[str, torch.Tensor]:
    """Translate HF's 3D expert tensors into our per-expert ModuleList layout.

    ``experts.gate_up_proj`` ``[E, 2*int, hidden]`` →
        ``experts.{e}.gate_proj.weight``  (first ``int`` rows)
        ``experts.{e}.up_proj.weight``    (last  ``int`` rows)

    ``experts.down_proj`` ``[E, hidden, int]`` →
        ``experts.{e}.down_proj.weight``  (sliced per expert)
    """
    out: Dict[str, torch.Tensor] = {}
    layer_idx = _expert_layer_index(hf_key)
    if layer_idx is None:
        return out
    layer_prefix = f"model.layers.{layer_idx}.mlp"
    if hf_key.endswith("gate_up_proj"):
        E, two_int, _hidden = hf_tensor.shape
        intermediate = two_int // 2
        for e in range(E):
            slab = hf_tensor[e]                          # [2*int, hidden]
            out[f"{layer_prefix}.experts.{e}.gate_proj.weight"] = slab[:intermediate].contiguous()
            out[f"{layer_prefix}.experts.{e}.up_proj.weight"] = slab[intermediate:].contiguous()
    elif hf_key.endswith("down_proj"):
        E = hf_tensor.shape[0]
        for e in range(E):
            out[f"{layer_prefix}.experts.{e}.down_proj.weight"] = hf_tensor[e].contiguous()
    return out


def _is_moe_layer(model: nn.Module, layer_idx: int) -> bool:
    return isinstance(model.model.layers[layer_idx].mlp, Qwen3_5MoeBlock)


def load_qwen3_5_state_dict(
    model: nn.Module,
    hf_state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool = False,
    dtype: torch.dtype | None = None,
) -> LoadReport:
    """Copy values from a HuggingFace Qwen3_5 / Qwen3_5_MoE state_dict into
    a hand-rolled :class:`Qwen3_5ForCausalLM`.

    Parameters
    ----------
    model : Qwen3_5ForCausalLM
        Target model. Must be constructed with a config matching the
        checkpoint's architecture (layers, head dims, expert count).
    hf_state_dict : dict
        ``{name: tensor}`` from ``transformers.AutoModelForCausalLM.state_dict()``
        or a safetensors shard.
    strict : bool, default False
        If True, raise on missing or unexpected keys (after applying the
        skip policy for MTP / vision).
    dtype : torch.dtype, optional
        Cast every loaded tensor to this dtype before copying. By default
        we keep the source dtype (typically ``bfloat16``).

    Returns
    -------
    LoadReport
        Includes ``loaded`` / ``skipped_hf_keys`` / ``missing_ours`` /
        ``unexpected_hf``.
    """
    # Build name → param map for our model.
    target_params: Dict[str, torch.Tensor] = dict(model.named_parameters())

    loaded: List[str] = []
    skipped: List[str] = []
    unexpected: List[str] = []
    matched_targets: Set[str] = set()

    def _copy(name: str, value: torch.Tensor) -> None:
        if name not in target_params:
            unexpected.append(name)
            return
        dest = target_params[name]
        if dest.shape != value.shape:
            raise ValueError(
                f"shape mismatch for {name}: ours={tuple(dest.shape)} hf={tuple(value.shape)}"
            )
        with torch.no_grad():
            v = value if dtype is None else value.to(dtype)
            dest.copy_(v.to(dest.dtype))
        loaded.append(name)
        matched_targets.add(name)

    for hf_key, hf_tensor in hf_state_dict.items():
        if _is_skipped(hf_key):
            skipped.append(hf_key)
            continue

        if _expert_layer_index(hf_key) is not None:
            for our_name, sub_tensor in _expand_moe_experts(hf_key, hf_tensor, model).items():
                _copy(our_name, sub_tensor)
            continue

        # Default: HF name maps directly to our name.
        _copy(hf_key, hf_tensor)

    missing = [n for n in target_params if n not in matched_targets]

    # ``inv_freq`` is a non-persistent buffer we recompute from config — it's
    # legitimately absent from HF state_dicts. Same for any tied lm_head when
    # ``tie_word_embeddings=True`` (lm_head.weight is just embed_tokens).
    cfg = model.config
    if getattr(cfg, "tie_word_embeddings", False):
        missing = [n for n in missing if n != "lm_head.weight"]

    report = LoadReport(
        loaded=loaded, skipped_hf_keys=skipped,
        missing_ours=missing, unexpected_hf=unexpected,
    )

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"strict load failed:\n  missing: {missing}\n  unexpected: {unexpected}"
        )
    return report
