"""Recent / non-stdlib optimizers.

Each module in this package implements a single optimizer family that is too
self-contained or too idiosyncratic to live alongside the AdamW/SGD/RMSprop
paths in ``optimizer.py``. Add new ones by dropping a module here and adding
a branch to ``build_optimizer`` in the parent module.
"""

from .muon import (
    Muon,
    SingleDeviceMuon,
    MuonWithAuxAdam,
    SingleDeviceMuonWithAuxAdam,
    build_muon_optimizer,
    split_muon_param_groups,
)

__all__ = [
    "Muon",
    "SingleDeviceMuon",
    "MuonWithAuxAdam",
    "SingleDeviceMuonWithAuxAdam",
    "build_muon_optimizer",
    "split_muon_param_groups",
]
