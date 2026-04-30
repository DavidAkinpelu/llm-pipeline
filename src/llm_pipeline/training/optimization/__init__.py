"""Optimizer and scheduler factories for training."""

from .optimizer import build_optimizer, OptimizerConfig
from .scheduler import build_scheduler, SchedulerConfig
from .optimizers import (
    Muon,
    MuonWithAuxAdam,
    SingleDeviceMuon,
    SingleDeviceMuonWithAuxAdam,
    build_muon_optimizer,
    split_muon_param_groups,
)

__all__ = [
    "build_optimizer",
    "OptimizerConfig",
    "build_scheduler",
    "SchedulerConfig",
    "Muon",
    "MuonWithAuxAdam",
    "SingleDeviceMuon",
    "SingleDeviceMuonWithAuxAdam",
    "build_muon_optimizer",
    "split_muon_param_groups",
]
