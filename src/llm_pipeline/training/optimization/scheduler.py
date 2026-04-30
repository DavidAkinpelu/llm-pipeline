"""Learning-rate scheduler factory."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class SchedulerConfig:
    name: str = "cosine"  # one of: constant, linear, cosine
    num_warmup_steps: int = 0
    num_training_steps: int = 1000
    min_lr_ratio: float = 0.0  # cosine floor as fraction of peak LR


def _constant_lambda(num_warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        return 1.0
    return lr_lambda


def _linear_lambda(num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 1.0 - progress)
    return lr_lambda


def _cosine_lambda(num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return lr_lambda


def build_scheduler(optimizer: Optimizer, config: Optional[SchedulerConfig] = None) -> torch.optim.lr_scheduler.LRScheduler:
    config = config or SchedulerConfig()
    name = config.name.lower()
    if name == "constant":
        fn = _constant_lambda(config.num_warmup_steps)
    elif name == "linear":
        fn = _linear_lambda(config.num_warmup_steps, config.num_training_steps)
    elif name == "cosine":
        fn = _cosine_lambda(config.num_warmup_steps, config.num_training_steps, config.min_lr_ratio)
    else:
        raise ValueError(f"Unknown scheduler: {config.name}")
    return LambdaLR(optimizer, fn)
