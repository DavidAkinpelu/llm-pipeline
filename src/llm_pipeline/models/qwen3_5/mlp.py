"""Feed-forward blocks for Qwen3.5 / Qwen3.6 — dense SwiGLU + MoE.

Two variants, picked per-model by ``Qwen3_5DecoderLayer``:

- ``Qwen3_5MLP``: standard SwiGLU FFN (gate · silu · up → down). Used by
  the dense Qwen3.6-27B release at ``intermediate_size=17408``.
- ``Qwen3_5MoeBlock``: Mixture-of-Experts feed-forward used by the MoE
  release (Qwen3.6-35B-A3B). Pairs a softmax-top-K router over 256 routed
  experts with a sigmoid-gated **shared expert** that runs on every token
  (DeepSeek-V3 style). The shared-expert gate is a scalar per token,
  multiplied into the shared output before adding to the routed sum.

The MoE block is a specialisation of the generic ``MoEFeedForward`` we
shipped earlier — same router + dispatch math, just with the shared-expert
add-on. We keep it separate (rather than parameterising the generic block)
because the shared-expert layout matches HF's ``Qwen3_5MoeSparseMoeBlock``
weight loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MLPConfig:
    """Subset of ``Qwen3_5Config`` fields the dense MLP uses."""

    hidden_size: int
    intermediate_size: int
    hidden_act: str = "silu"


@dataclass
class MoEBlockConfig:
    """Subset of ``Qwen3_5_MoE_Config`` fields the MoE block uses."""

    hidden_size: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    norm_topk_prob: bool = True
    hidden_act: str = "silu"


def _act(name: str):
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"unsupported activation: {name!r}")


# --------------------------------------------------------------------------- #
# Dense SwiGLU MLP
# --------------------------------------------------------------------------- #


class Qwen3_5MLP(nn.Module):
    """Standard SwiGLU feed-forward block."""

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = _act(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# MoE block (routed experts + shared expert with sigmoid gate)
# --------------------------------------------------------------------------- #


class Qwen3_5MoeRouter(nn.Module):
    """Top-K softmax router with renormalised gate weights.

    Mirrors ``Qwen3_5MoeTopKRouter`` from HF: the gate weight is stored as a
    raw ``nn.Parameter`` (no Linear wrapper), softmax is taken in fp32, and
    the top-K probabilities are renormalised so they sum to 1 per token.
    """

    def __init__(self, config: MoEBlockConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk = config.norm_topk_prob
        self.weight = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size))

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Flatten leading dims; expect [N, hidden].
        h_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = F.linear(h_flat, self.weight)                              # [N, E]
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_vals, top_idx = probs.topk(self.top_k, dim=-1)
        if self.norm_topk and self.top_k > 1:
            top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return logits, top_vals.to(hidden_states.dtype), top_idx


class Qwen3_5MoeBlock(nn.Module):
    """Sparse MoE feed-forward: routed experts + sigmoid-gated shared expert.

    Per token:

      out = Σ_{e ∈ topK(t)} gate_e(t) · expert_e(h_t)
            + sigmoid(shared_gate(h_t)) · shared_expert(h_t)

    The router runs softmax → top-K → renormalise. Routed expert output is
    weighted by the renormalised gate; the shared expert always runs and
    is gated by a scalar per token.

    Routed experts use a ModuleList of ``Qwen3_5MLP`` for clarity (HF stores
    them as a 3D tensor for batched dispatch — same math, different layout
    than the educational reference here).
    """

    def __init__(self, config: MoEBlockConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.router = Qwen3_5MoeRouter(config)
        routed_cfg = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.experts = nn.ModuleList([Qwen3_5MLP(routed_cfg) for _ in range(config.num_experts)])

        shared_cfg = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.shared_expert = Qwen3_5MLP(shared_cfg)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

        # Stashed for aux losses across MoE layers (reuses
        # ``llm_pipeline.models.moe.collect_moe_aux_loss`` if you concatenate
        # ``last_router_logits`` / ``last_expert_mask`` from every layer).
        self.last_router_logits: Optional[torch.Tensor] = None
        self.last_expert_mask: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        h = hidden_states.reshape(-1, orig_shape[-1])
        N = h.shape[0]

        # Routing.
        logits, top_gates, top_idx = self.router(h)
        self.last_router_logits = logits
        mask = torch.zeros(N, self.num_experts, dtype=torch.int32, device=h.device)
        mask.scatter_(1, top_idx, 1)
        self.last_expert_mask = mask

        # Routed expert dispatch — gather, run, scatter.
        routed_out = torch.zeros_like(h)
        for e in range(self.num_experts):
            sel = (top_idx == e).nonzero(as_tuple=False)            # [N_e, 2]
            if sel.numel() == 0:
                continue
            tok_ids = sel[:, 0]
            slots = sel[:, 1]
            x_e = h[tok_ids]
            y_e = self.experts[e](x_e)
            gates = top_gates[tok_ids, slots]
            routed_out.index_add_(0, tok_ids, y_e * gates.unsqueeze(-1))

        # Shared expert with sigmoid scalar gate (DeepSeek-V3 style).
        shared = self.shared_expert(h)
        shared_gate = torch.sigmoid(self.shared_expert_gate(h))     # [N, 1]
        shared = shared_gate * shared

        return (routed_out + shared).reshape(orig_shape)
