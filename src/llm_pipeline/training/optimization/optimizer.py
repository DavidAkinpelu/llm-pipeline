"""Optimizer factory.

Wraps PyTorch optimizers and (when available) bitsandbytes paged optimizers
for QLoRA-style training where 8-bit/paged AdamW saves significant memory.
"""

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    # Apply weight decay only to non-bias / non-norm params (common HF default).
    apply_decay_to_norm_and_bias: bool = False
    # Use bitsandbytes 8-bit optimizer if available.
    use_8bit: bool = False
    # Use bitsandbytes paged optimizer if available (best for QLoRA).
    use_paged: bool = False
    # Muon-specific. When name="muon", `lr` is the Muon matmul lr (default
    # 0.02 in the reference; override here). `aux_adam_lr` drives the AdamW
    # groups for embeddings / LM head / biases / norms.
    muon_momentum: float = 0.95
    aux_adam_lr: float = 3e-4


def _split_decay_params(
    params: Iterable[Tuple[str, nn.Parameter]],
    weight_decay: float,
    apply_to_norm_and_bias: bool,
) -> List[dict]:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    no_decay_keys = ("bias", "LayerNorm.weight", "layernorm.weight", "norm.weight")
    for name, p in params:
        if not p.requires_grad:
            continue
        if apply_to_norm_and_bias or not any(k in name for k in no_decay_keys):
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, config: Optional[OptimizerConfig] = None) -> torch.optim.Optimizer:
    """Construct an optimizer matching the requested config.

    Falls back to torch.optim.AdamW if bitsandbytes is unavailable but requested.
    """
    config = config or OptimizerConfig()
    param_groups = _split_decay_params(
        model.named_parameters(),
        config.weight_decay,
        config.apply_decay_to_norm_and_bias,
    )

    name = config.name.lower()
    if name in ("adamw", "adam"):
        if config.use_paged or config.use_8bit:
            try:
                import bitsandbytes as bnb  # type: ignore
                cls = bnb.optim.PagedAdamW8bit if config.use_paged else bnb.optim.AdamW8bit
                return cls(param_groups, lr=config.lr, betas=config.betas, eps=config.eps)
            except ImportError:
                pass
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return cls(param_groups, lr=config.lr, betas=config.betas, eps=config.eps)
    if name == "sgd":
        return torch.optim.SGD(param_groups, lr=config.lr, momentum=config.betas[0])
    if name == "rmsprop":
        return torch.optim.RMSprop(param_groups, lr=config.lr, eps=config.eps)
    if name == "muon":
        from .optimizers import build_muon_optimizer
        return build_muon_optimizer(model, config)
    raise ValueError(f"Unknown optimizer: {config.name}")
