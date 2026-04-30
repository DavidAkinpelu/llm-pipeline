"""Mixture-of-Experts building blocks.

This module provides a self-contained MoE feed-forward layer (router + N
expert MLPs + dispatch) that can drop into any Llama-style decoder block in
place of a dense SwiGLU MLP. It targets the **naive scatter-gather**
execution path: for each expert, gather the tokens routed to it, run that
expert, scatter the results back. This is what HuggingFace's reference
Mixtral / Qwen3-MoE / DeepSeek implementations use; production stacks
(Megablocks, Tutel) replace it with grouped-GEMM kernels but the math is
the same.

**What this isn't:**

- No tensor-parallel-on-experts. Sharding experts across ranks is the next
  step (``XL`` on the roadmap) and needs an all-to-all dispatch primitive.
- No grouped-GEMM kernels. The for-loop-over-experts here is fine for
  research-scale models (8 experts, top-2) on a single GPU, but won't keep
  up with vLLM-class latency targets at large expert counts.
- No expert-choice routing. We do **token-choice** top-K, which is what
  Mixtral / Qwen3-MoE / DeepSeek-V2 use. Switch-Transformer's expert-choice
  variant (each expert picks its top-K tokens) is a separate algorithm.

**API surface:**

- ``MoEConfig`` — hyperparameter dataclass.
- ``TopKRouter`` — token → (top_k expert IDs, gates).
- ``MoEFeedForward`` — full layer: router + experts + dispatch + combine.
- ``compute_load_balancing_loss`` — Switch-Transformer / Mixtral auxiliary
  loss penalising imbalanced expert utilisation.
- ``compute_router_z_loss`` — DeepSeek-style logsumexp regulariser keeping
  router logits well-conditioned.

**How to integrate into training:**

Each ``MoEFeedForward`` module exposes ``last_router_logits`` after every
forward pass. The ``Trainer`` (or your training loop) collects them, runs
the aux losses, and adds them to the main loss with a small coefficient
(``aux_loss_coef ≈ 0.01`` is the Mixtral default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    """Hyperparameters for an MoE feed-forward layer.

    Attributes
    ----------
    hidden_size : int
        Input/output dimension. Same as the surrounding decoder block.
    intermediate_size : int
        Per-expert SwiGLU hidden dim. Mixtral uses 14336 with hidden=4096.
    num_experts : int
        Total number of experts E. Mixtral=8, Qwen3-30B-MoE=128, DeepSeek-V3=256.
    num_experts_per_token : int
        Top-K. Mixtral=2, DeepSeek-V3=8 (with shared expert).
    norm_topk_prob : bool
        If True, renormalise the top-K gate probabilities so they sum to 1
        (Mixtral / Qwen3-MoE behaviour). DeepSeek-V2 keeps the raw softmax.
    router_jitter : float
        Multiplicative noise added to routing logits during training. Helps
        explore expert assignments in the early steps. 0.0 disables it.
    activation : str
        Expert activation. "silu" matches Mixtral / Qwen3-MoE; "gelu" is
        what GPT / GLaM use.
    """

    hidden_size: int
    intermediate_size: int
    num_experts: int = 8
    num_experts_per_token: int = 2
    norm_topk_prob: bool = True
    router_jitter: float = 0.0
    activation: str = "silu"

    def __post_init__(self) -> None:
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be ≥ 1, got {self.num_experts}")
        if not (1 <= self.num_experts_per_token <= self.num_experts):
            raise ValueError(
                f"num_experts_per_token must be in [1, {self.num_experts}], "
                f"got {self.num_experts_per_token}"
            )
        if self.activation not in {"silu", "gelu"}:
            raise ValueError(f"unsupported activation: {self.activation!r}")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


class TopKRouter(nn.Module):
    """Token-choice top-K softmax router.

    Maps each token's hidden state to a softmax distribution over the E
    experts, picks the top-K, and (optionally) renormalises those K
    gate values so they sum to 1.

    Returns a tuple ``(top_k_indices, top_k_gates, raw_logits)`` where:
      - ``top_k_indices`` is ``[..., K]`` int64 — which experts to route to.
      - ``top_k_gates``   is ``[..., K]`` float — the (possibly normalised)
        weight on each chosen expert.
      - ``raw_logits``    is ``[..., E]`` float — the pre-softmax router
        logits, kept around for the auxiliary load-balance loss.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.norm_topk = config.norm_topk_prob
        self.jitter = config.router_jitter
        # Linear, no bias — Mixtral / Qwen3-MoE convention.
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ``hidden_states`` is [..., H]; flatten leading dims for routing.
        logits = self.gate(hidden_states.to(self.gate.weight.dtype))
        if self.training and self.jitter > 0.0:
            # Multiplicative noise in [1 - j, 1 + j], like Switch-Transformer.
            noise = torch.empty_like(logits).uniform_(1.0 - self.jitter, 1.0 + self.jitter)
            logits = logits * noise
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        topk_gates, topk_idx = probs.topk(self.top_k, dim=-1)
        if self.norm_topk and self.top_k > 1:
            topk_gates = topk_gates / topk_gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return topk_idx, topk_gates.to(hidden_states.dtype), logits


# --------------------------------------------------------------------------- #
# Expert MLP (SwiGLU / GELU)
# --------------------------------------------------------------------------- #


class _ExpertMLP(nn.Module):
    """Single SwiGLU (or GELU-MLP) expert, identical to the dense FFN."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = F.silu if config.activation == "silu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# MoE feed-forward layer
# --------------------------------------------------------------------------- #


class MoEFeedForward(nn.Module):
    """Drop-in replacement for a dense SwiGLU MLP.

    Forward pass:

    1. Route every token through ``TopKRouter`` → top-K (expert_id, gate).
    2. For each expert e, gather the rows of ``hidden_states`` that have e
       in their top-K, run them through that expert's MLP, scale by the
       gate value, and scatter the result back to its original position.
    3. Sum contributions when a token is routed to multiple experts.

    The naive loop over experts is intentional: it's the clearest reference
    implementation and ports cleanly to multi-GPU later (one all-to-all
    around the loop body is enough to shard experts). For research-scale
    models (E ≤ 16) on a single GPU it's also competitive with grouped-GEMM
    once expert utilisation is balanced.

    After every forward pass the module stashes:

    - ``last_router_logits`` — ``[B*T, E]`` float, raw router output.
    - ``last_expert_mask``   — ``[B*T, E]`` int, 1 if token i used expert j.

    Both are used by ``compute_load_balancing_loss`` and ``compute_router_z_loss``.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.router = TopKRouter(config)
        self.experts = nn.ModuleList([_ExpertMLP(config) for _ in range(config.num_experts)])
        # Buffers populated on every forward; not persisted.
        self.last_router_logits: Optional[torch.Tensor] = None
        self.last_expert_mask: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Flatten [B, T, H] → [BT, H]; remember the shape for restoration.
        orig_shape = hidden_states.shape
        h = hidden_states.reshape(-1, orig_shape[-1])
        BT = h.shape[0]

        topk_idx, topk_gates, logits = self.router(h)        # [BT, K], [BT, K], [BT, E]
        self.last_router_logits = logits

        # Build a one-hot expert mask for aux losses ([BT, E] int).
        expert_mask = torch.zeros(BT, self.num_experts, dtype=torch.int32, device=h.device)
        expert_mask.scatter_(1, topk_idx, 1)
        self.last_expert_mask = expert_mask

        out = torch.zeros_like(h)

        # For each expert, find the rows that picked it, run them, scatter back.
        # ``topk_idx`` is [BT, K]; we need (token_idx, k_slot) pairs where
        # topk_idx[token_idx, k_slot] == e.
        for e in range(self.num_experts):
            # Which (token, k-slot) pairs route to this expert?
            sel = (topk_idx == e).nonzero(as_tuple=False)    # [N_e, 2]
            if sel.numel() == 0:
                continue
            token_ids = sel[:, 0]
            k_slots = sel[:, 1]
            x_e = h[token_ids]                                # [N_e, H]
            y_e = self.experts[e](x_e)                        # [N_e, H]
            gates = topk_gates[token_ids, k_slots]            # [N_e]
            out.index_add_(0, token_ids, y_e * gates.unsqueeze(-1))

        return out.reshape(orig_shape)


# --------------------------------------------------------------------------- #
# Auxiliary losses
# --------------------------------------------------------------------------- #


def compute_load_balancing_loss(
    router_logits: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Switch-Transformer / Mixtral load-balancing loss.

    For ``T`` tokens routed to ``E`` experts:

        f_i = (fraction of tokens that picked expert i)
        P_i = (mean router probability on expert i)
        loss = E · Σᵢ (f_i · P_i)

    The dot product is minimised when both vectors are uniform (1/E each),
    in which case ``loss = E · E · (1/E)² = 1.0``. So the loss is in
    ``[1.0, E]`` — the constant offset doesn't matter for gradients.

    The trick: ``f_i`` is computed from a hard top-K mask (no gradient),
    while ``P_i`` is the soft router output (full gradient). So the loss
    pushes the soft probs to *anti-correlate* with the hard utilisation,
    which over time evens both out.

    Parameters
    ----------
    router_logits : [T, E] float
        Pre-softmax router output. Typically the concatenation of
        ``last_router_logits`` from every MoE layer in the model.
    expert_mask : [T, E] int / bool
        ``expert_mask[t, e] = 1`` iff expert ``e`` was in token ``t``'s top-K.
    num_experts : int
        E. Used as the multiplicative constant.
    """
    if router_logits.numel() == 0:
        return router_logits.new_zeros(())
    probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    P = probs.mean(dim=0)                                    # [E]
    f = expert_mask.to(probs.dtype).mean(dim=0)              # [E]
    return num_experts * (f * P).sum()


def compute_router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """DeepSeek / ST-MoE router z-loss.

    ``z_loss = mean_t (logsumexp_e logits[t, e])²``

    Penalises blow-up in the partition function of the router softmax.
    Without it, the router can drift to extreme logits where small noise
    triggers wild expert reassignments. Coefficient ~1e-3 in DeepSeek.
    """
    if router_logits.numel() == 0:
        return router_logits.new_zeros(())
    return torch.logsumexp(router_logits, dim=-1).pow(2).mean()


def collect_moe_aux_loss(
    model: nn.Module,
    load_balance_coef: float = 0.01,
    z_loss_coef: float = 0.0,
) -> torch.Tensor:
    """Walk a model, collect every ``MoEFeedForward``'s last router stats,
    and return the weighted sum of aux losses.

    Returns a zero scalar if the model has no MoE layers (so it's safe to
    add unconditionally to the main loss).
    """
    layers: List[MoEFeedForward] = [m for m in model.modules() if isinstance(m, MoEFeedForward)]
    if not layers:
        return torch.zeros((), device=next(model.parameters()).device)

    # Concatenate the router stats from every MoE layer into one big tensor
    # and run the loss on that — equivalent to averaging the per-layer
    # losses but cheaper, and mathematically identical when every layer
    # has the same ``num_experts``.
    logits_all = torch.cat([m.last_router_logits for m in layers], dim=0)
    mask_all = torch.cat([m.last_expert_mask for m in layers], dim=0)

    total = logits_all.new_zeros(())
    if load_balance_coef > 0.0:
        total = total + load_balance_coef * compute_load_balancing_loss(
            logits_all, mask_all, layers[0].num_experts
        )
    if z_loss_coef > 0.0:
        total = total + z_loss_coef * compute_router_z_loss(logits_all)
    return total
