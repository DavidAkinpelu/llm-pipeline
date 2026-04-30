"""Muon optimizer (MomentUm Orthogonalized by Newton-schulz).

Faithful port of the reference implementation by Keller Jordan
(https://github.com/KellerJordan/Muon), with two additions for use inside
this pipeline:

- ``split_muon_param_groups`` heuristically separates a model's parameters
  into the Muon-eligible matmul tensors and the AdamW-eligible remainder
  (embeddings, LM head, biases, norms, scalars). This split is critical:
  Muon is only defined for 2D+ hidden weights, and the reference shows
  empirically that input/output projections must stay on AdamW.
- ``build_muon_optimizer`` constructs the hybrid ``MuonWithAuxAdam`` /
  ``SingleDeviceMuonWithAuxAdam`` optimizer with a single forward step,
  auto-selecting the distributed variant when a process group is live.

The reference Muon classes are kept verbatim so behavior matches the
public benchmarks; only the builder/split are project-local.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz iteration for matrix orthogonalization.

    Quintic iteration with coefficients tuned to maximize slope at zero. Runs
    in bfloat16 on tensor cores. Produces ``US'V^T`` where ``S'_{ii}`` is
    roughly Uniform(0.5, 1.5) — empirically as good as exact ``UV^T``.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(
    grad: torch.Tensor,
    buf1: torch.Tensor,
    buf2: torch.Tensor,
    step: int,
    betas: Tuple[float, float],
    eps: float,
) -> torch.Tensor:
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class Muon(torch.optim.Optimizer):
    """Distributed Muon for 2D hidden weights only.

    Pair with AdamW for embeddings, the LM head, biases, and norms. See
    ``MuonWithAuxAdam`` for a single-optimizer hybrid.
    """

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0, momentum: float = 0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(len(params))[::world_size]:
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """Muon variant for non-distributed settings."""

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0, momentum: float = 0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Distributed hybrid: Muon for matmul groups, AdamW for the rest.

    Param groups must each set ``use_muon=True/False``. Muon groups accept
    ``lr, momentum, weight_decay``; Adam groups accept ``lr, betas, eps,
    weight_decay``.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
                for base_i in range(len(params))[::world_size]:
                    if base_i + rank < len(params):
                        p = params[base_i + rank]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Non-distributed hybrid Muon + AdamW."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


_EMBED_KEYS = ("embed", "wte", "wpe", "tok_emb", "pos_emb")
_HEAD_KEYS = ("lm_head", "output_proj", "out_proj.weight", "head.weight")


def split_muon_param_groups(
    model: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Split trainable params into (muon_eligible, adam_eligible).

    Muon-eligible: 2D+ tensors that are not embeddings or the LM head.
    Adam-eligible: everything else (embeddings, head, biases, norms, scalars).
    """
    muon_params: List[nn.Parameter] = []
    adam_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        is_embed = any(k in lname for k in _EMBED_KEYS)
        is_head = any(k in lname for k in _HEAD_KEYS)
        if p.ndim >= 2 and not is_embed and not is_head:
            muon_params.append(p)
        else:
            adam_params.append(p)
    return muon_params, adam_params


def _distributed_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def build_muon_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Construct a hybrid Muon+AdamW optimizer for ``model``.

    Reads from ``OptimizerConfig``: ``lr`` (Muon matmul lr), ``muon_momentum``,
    ``aux_adam_lr``, ``betas``, ``eps``, ``weight_decay``. Auto-selects the
    distributed variant when a multi-rank process group is live.
    """
    muon_params, adam_params = split_muon_param_groups(model)
    if not muon_params and not adam_params:
        raise ValueError("Model has no trainable parameters.")

    groups = []
    if muon_params:
        groups.append(dict(
            params=muon_params,
            lr=config.lr,
            momentum=config.muon_momentum,
            weight_decay=config.weight_decay,
            use_muon=True,
        ))
    if adam_params:
        groups.append(dict(
            params=adam_params,
            lr=config.aux_adam_lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            use_muon=False,
        ))

    cls = MuonWithAuxAdam if _distributed_active() else SingleDeviceMuonWithAuxAdam
    return cls(groups)
