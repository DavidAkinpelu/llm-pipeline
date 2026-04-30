"""Column- and row-parallel ``Linear`` layers.

These are drop-in replacements for ``nn.Linear`` that hold a sliced weight
tensor and insert the appropriate collective on the forward path. Together
they implement the standard tensor-parallel pattern (Megatron / vLLM):

    Y = X · W^T  with W partitioned across ranks.

There are two complementary partition schemes:

- **Column parallel**: split ``W`` along ``out_features`` (rows of the stored
  weight, since PyTorch stores ``W ∈ R^{out × in}``). Input is replicated;
  each rank computes a slice of the output. If the next layer is
  *row*-parallel and accepts a sharded input, we leave the output sharded
  (``gather_output=False``); otherwise we ``all_gather`` so callers see the
  full output.

- **Row parallel**: split ``W`` along ``in_features`` (columns). The input
  must already be sharded along its last dim — typically because it came
  from an upstream column-parallel layer with ``gather_output=False``. Each
  rank computes a partial sum; we ``all_reduce`` to get the full result.

Bias on a row-parallel layer is added *after* the all-reduce so it is not
double-counted.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _is_dist(process_group=None) -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_gather_last_dim(x: torch.Tensor, world_size: int, process_group=None) -> torch.Tensor:
    """Concatenate per-rank tensors along the last dimension."""
    if world_size == 1 or not _is_dist(process_group):
        return x
    x = x.contiguous()
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x, group=process_group)
    return torch.cat(gathered, dim=-1)


class ColumnParallelLinear(nn.Module):
    """Linear layer with output dimension sharded across ranks.

    Stored weight shape: ``(out_features // world_size, in_features)``.

    Args:
        gather_output: When ``True``, all-gather the output so callers see
            the full ``out_features``. Set ``False`` when the next layer is
            row-parallel — that layer will consume the sharded output
            directly.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = True,
        gather_output: bool = True,
        process_group=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if out_features % world_size != 0:
            raise ValueError(
                f"out_features={out_features} not divisible by world_size={world_size}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_local = out_features // world_size
        self.world_size = world_size
        self.rank = rank
        self.gather_output = gather_output
        self.process_group = process_group

        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(self.out_features_local, in_features, **factory))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_local, **factory))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            out = _all_gather_last_dim(out, self.world_size, self.process_group)
        return out

    def load_full_weight(self, full_weight: torch.Tensor, full_bias: Optional[torch.Tensor] = None) -> None:
        """Copy this rank's slice from a full-sized weight (and optional bias)."""
        chunks = full_weight.chunk(self.world_size, dim=0)
        self.weight.data.copy_(chunks[self.rank])
        if full_bias is not None and self.bias is not None:
            self.bias.data.copy_(full_bias.chunk(self.world_size, dim=0)[self.rank])


class RowParallelLinear(nn.Module):
    """Linear layer with input dimension sharded across ranks.

    Stored weight shape: ``(out_features, in_features // world_size)``.

    Args:
        input_is_parallel: ``True`` when the input is already sharded along
            its last dim (e.g. it came from a ``ColumnParallelLinear`` with
            ``gather_output=False``). ``False`` triggers a scatter — rarely
            what you want; use a column-parallel layer upstream instead.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        process_group=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if in_features % world_size != 0:
            raise ValueError(
                f"in_features={in_features} not divisible by world_size={world_size}"
            )
        self.in_features = in_features
        self.in_features_local = in_features // world_size
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.input_is_parallel = input_is_parallel
        self.process_group = process_group

        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_local, **factory))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            # Scatter along last dim — equivalent to taking this rank's slice.
            chunks = x.chunk(self.world_size, dim=-1)
            x = chunks[self.rank].contiguous()
        out = F.linear(x, self.weight, None)  # bias added after reduce
        if self.world_size > 1 and _is_dist(self.process_group):
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.process_group)
        if self.bias is not None:
            out = out + self.bias
        return out

    def load_full_weight(self, full_weight: torch.Tensor, full_bias: Optional[torch.Tensor] = None) -> None:
        """Copy this rank's slice from a full-sized weight."""
        chunks = full_weight.chunk(self.world_size, dim=1)
        self.weight.data.copy_(chunks[self.rank])
        # Bias is replicated (not sharded) since it's added after all-reduce.
        if full_bias is not None and self.bias is not None:
            self.bias.data.copy_(full_bias)
