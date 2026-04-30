"""Tensor-parallel-on-experts MoE: shard experts across N ranks.

The naive ``MoEFeedForward`` runs every expert on the rank that holds it
(typically rank 0); for large MoE models with hundreds of experts this is
both a memory and compute bottleneck. Expert parallelism partitions the
experts across ranks: rank ``r`` owns experts ``[r·E/N, (r+1)·E/N)`` and
runs only those, with the cross-rank routing handled by an all-to-all.

Forward pass per token batch
----------------------------

1. **Local routing** — every rank runs the (replicated) router and gets
   ``(top_k_indices, top_k_gates)`` for its slice of the global token batch.

2. **Permute** — for each chosen expert, figure out which rank owns it and
   build a permutation that groups tokens by destination rank.

3. **All-to-all dispatch** — each rank sends the tokens that need to go to
   experts on other ranks; receives tokens that other ranks routed to its
   own experts. After this every rank has its full local workload.

4. **Local expert compute** — run each local expert on the tokens it received.

5. **All-to-all combine** — reverse the dispatch: each rank sends back the
   processed tokens; receives the per-token contributions for its slice.

6. **Un-permute + gate-weighted sum** — per token, sum the K (gate · output)
   contributions from across all chosen experts.

Status
------

This module is **implemented but not validated end-to-end** — running a
real all-to-all needs at least 2 GPUs and an NCCL backend, neither of
which we have in this dev environment. The single-rank path (world_size=1)
is exercised by the test suite and produces results bit-close to the
reference ``MoEFeedForward``. The multi-rank path is committed for a
cloud-validation pass per the project's hardware policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..models.moe import MoEConfig, TopKRouter, _ExpertMLP


@dataclass
class ExpertParallelConfig:
    """Sharding plan for expert parallelism.

    ``world_size`` is the number of ranks across which experts are split;
    must divide ``MoEConfig.num_experts``. ``rank`` identifies the local
    rank in the group. With ``world_size=1`` the layer behaves as a
    single-rank ``MoEFeedForward`` — useful for unit testing the dispatch
    bookkeeping without needing an NCCL host.
    """

    world_size: int
    rank: int
    process_group: Optional["dist.ProcessGroup"] = None

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError(f"world_size must be ≥ 1; got {self.world_size}")
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"rank must be in [0, {self.world_size}); got {self.rank}")


class ExpertParallelMoE(nn.Module):
    """Drop-in replacement for ``MoEFeedForward`` with experts sharded across ranks.

    Construction is per-rank: each rank instantiates the layer with its own
    ``ep_config.rank``, and only the local-expert MLPs (count = E / world_size)
    are materialised. The router is replicated (every rank has identical
    weights and runs identical math).

    On ``world_size=1`` this falls back to a local-only execution path that
    matches the reference ``MoEFeedForward`` bit-for-bit.

    **Not validated locally — needs ≥2 GPU host with NCCL.**
    """

    def __init__(self, config: MoEConfig, ep_config: ExpertParallelConfig):
        super().__init__()
        if config.num_experts % ep_config.world_size != 0:
            raise ValueError(
                f"num_experts ({config.num_experts}) must be divisible by "
                f"world_size ({ep_config.world_size})"
            )
        self.config = config
        self.ep = ep_config
        self.local_n = config.num_experts // ep_config.world_size
        self.local_start = ep_config.rank * self.local_n           # global expert id offset
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token

        self.router = TopKRouter(config)
        # Each rank owns ``local_n`` consecutive experts.
        self.local_experts = nn.ModuleList([_ExpertMLP(config) for _ in range(self.local_n)])

        # Stash for aux losses, mirroring MoEFeedForward.
        self.last_router_logits: Optional[torch.Tensor] = None
        self.last_expert_mask: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        h = hidden_states.reshape(-1, orig_shape[-1])

        topk_idx, topk_gates, logits = self.router(h)
        self.last_router_logits = logits
        BT = h.shape[0]
        mask = torch.zeros(BT, self.num_experts, dtype=torch.int32, device=h.device)
        mask.scatter_(1, topk_idx, 1)
        self.last_expert_mask = mask

        # Single-rank fast path: no comm needed.
        if self.ep.world_size == 1:
            return self._local_forward(h, topk_idx, topk_gates).reshape(orig_shape)

        # Multi-rank path with all-to-all. Hardware-blocked; the math
        # below mirrors the single-rank path with a dispatch + combine
        # collective inserted around the local compute.
        return self._distributed_forward(h, topk_idx, topk_gates).reshape(orig_shape)

    # ------------------------------------------------------------------ #
    # Single-rank reference (also runs in unit tests).
    # ------------------------------------------------------------------ #

    def _local_forward(
        self,
        h: torch.Tensor,                               # [BT, H]
        topk_idx: torch.Tensor,                        # [BT, K]
        topk_gates: torch.Tensor,                      # [BT, K]
    ) -> torch.Tensor:
        out = torch.zeros_like(h)
        # Iterate over local experts, find tokens that picked them, run, scatter.
        for local_e in range(self.local_n):
            global_e = self.local_start + local_e
            sel = (topk_idx == global_e).nonzero(as_tuple=False)
            if sel.numel() == 0:
                continue
            tok_ids = sel[:, 0]
            slots = sel[:, 1]
            x_e = h[tok_ids]
            y_e = self.local_experts[local_e](x_e)
            gates = topk_gates[tok_ids, slots]
            out.index_add_(0, tok_ids, y_e * gates.unsqueeze(-1))
        return out

    # ------------------------------------------------------------------ #
    # Multi-rank dispatch / combine (NCCL all-to-all).
    # ------------------------------------------------------------------ #

    def _distributed_forward(
        self,
        h: torch.Tensor,                               # [BT, H]
        topk_idx: torch.Tensor,                        # [BT, K]
        topk_gates: torch.Tensor,                      # [BT, K]
    ) -> torch.Tensor:
        ep = self.ep
        world = ep.world_size

        # Determine the destination rank for each (token, slot) pair.
        # Rank r owns experts [r·local_n, (r+1)·local_n).
        dest_rank = topk_idx // self.local_n                     # [BT, K]

        # Build send buffers grouped by destination rank.
        flat_dest = dest_rank.reshape(-1)                        # [BT*K]
        flat_tok = torch.arange(h.shape[0], device=h.device).repeat_interleave(self.top_k)
        flat_slot = torch.arange(self.top_k, device=h.device).repeat(h.shape[0])
        flat_expert = topk_idx.reshape(-1) - dest_rank.reshape(-1) * self.local_n
        flat_gate = topk_gates.reshape(-1)

        send_counts = torch.zeros(world, dtype=torch.long, device=h.device)
        for r in range(world):
            send_counts[r] = (flat_dest == r).sum()
        send_counts_list = send_counts.tolist()

        # Sort the per-(token, slot) records by destination rank.
        order = torch.argsort(flat_dest, stable=True)
        send_h = h[flat_tok[order]]                              # [N_send, H]
        send_local_e = flat_expert[order]                        # [N_send] — which local expert at the dest

        # Exchange counts so each rank knows how much it'll receive.
        recv_counts = torch.zeros(world, dtype=torch.long, device=h.device)
        dist.all_to_all_single(
            recv_counts, send_counts, group=ep.process_group,
        )
        recv_counts_list = recv_counts.tolist()

        # All-to-all the hidden vectors and per-record metadata.
        recv_h = torch.zeros(
            (sum(recv_counts_list), h.shape[-1]), dtype=h.dtype, device=h.device,
        )
        dist.all_to_all_single(
            recv_h, send_h, output_split_sizes=recv_counts_list,
            input_split_sizes=send_counts_list, group=ep.process_group,
        )
        recv_local_e = torch.zeros(sum(recv_counts_list), dtype=torch.long, device=h.device)
        dist.all_to_all_single(
            recv_local_e, send_local_e, output_split_sizes=recv_counts_list,
            input_split_sizes=send_counts_list, group=ep.process_group,
        )

        # Run local experts on received tokens.
        proc = torch.zeros_like(recv_h)
        for local_e in range(self.local_n):
            picks = (recv_local_e == local_e).nonzero(as_tuple=False).squeeze(-1)
            if picks.numel() == 0:
                continue
            proc[picks] = self.local_experts[local_e](recv_h[picks])

        # All-to-all combine: send processed tokens back.
        combine = torch.zeros_like(send_h)
        dist.all_to_all_single(
            combine, proc, output_split_sizes=send_counts_list,
            input_split_sizes=recv_counts_list, group=ep.process_group,
        )

        # Apply gates and scatter into the per-token output buffer.
        out = torch.zeros_like(h)
        for i in range(combine.shape[0]):
            tok = flat_tok[order[i]].item()
            gate = flat_gate[order[i]]
            out[tok] += combine[i] * gate
        return out
