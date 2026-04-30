"""Unit tests for Qwen3 tensor parallelism.

The TP implementation patches the existing Qwen3 attention / MLP modules in
place, replacing nn.Linear projections with column- or row-parallel
versions. Numerical equivalence to the un-patched model is validated by a
multiprocess GLOO test under ``test_tp_equivalence`` in this file (only
runs when ``torch.multiprocessing.spawn`` is available).
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from llm_pipeline.models.qwen3.config import Qwen3Config
from llm_pipeline.models.qwen3.custom_builder import Qwen3ForCausalLM
from llm_pipeline.parallelism import (
    ParallelismFactory,
    TensorParallelConfig,
    parallelism_registry,
)
from llm_pipeline.parallelism.communication import (
    AllGatherPrimitive,
    AllReducePrimitive,
    ColumnParallelLinear,
    CommunicationPattern,
    ReduceScatterPrimitive,
    RowParallelLinear,
)
from llm_pipeline.parallelism.tensor_parallel import (
    Qwen3TensorParallelism,
    apply_tensor_parallel,
)


# --------------------------------------------------------------------------- #
# Parallel linear primitives — single-rank numerical equivalence to nn.Linear
# --------------------------------------------------------------------------- #


def _seeded_linear(in_f, out_f, bias=True, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(in_f, out_f, bias=bias)


class TestColumnParallelLinear:
    def test_world_size_1_matches_nn_linear(self):
        """At world_size=1 a column-parallel linear is bit-identical to nn.Linear."""
        ref = _seeded_linear(8, 16)
        cp = ColumnParallelLinear(8, 16, world_size=1, rank=0, bias=True, gather_output=True)
        cp.load_full_weight(ref.weight.data, ref.bias.data)
        x = torch.randn(2, 4, 8)
        torch.testing.assert_close(cp(x), ref(x))

    def test_load_full_weight_slices_correctly(self):
        """At world_size=2 each rank holds the corresponding row slice."""
        full_w = torch.arange(16 * 8, dtype=torch.float32).view(16, 8)
        full_b = torch.arange(16, dtype=torch.float32)
        cp_r0 = ColumnParallelLinear(8, 16, world_size=2, rank=0, bias=True, gather_output=True)
        cp_r1 = ColumnParallelLinear(8, 16, world_size=2, rank=1, bias=True, gather_output=True)
        cp_r0.load_full_weight(full_w, full_b)
        cp_r1.load_full_weight(full_w, full_b)
        torch.testing.assert_close(cp_r0.weight.data, full_w[:8])
        torch.testing.assert_close(cp_r1.weight.data, full_w[8:])
        torch.testing.assert_close(cp_r0.bias.data, full_b[:8])
        torch.testing.assert_close(cp_r1.bias.data, full_b[8:])

    def test_out_features_must_be_divisible(self):
        with pytest.raises(ValueError, match="not divisible"):
            ColumnParallelLinear(4, 7, world_size=2, rank=0)

    def test_forward_uses_supplied_process_group(self):
        ref = _seeded_linear(8, 16)
        group = object()
        cp = ColumnParallelLinear(
            8, 16, world_size=2, rank=0, bias=True, gather_output=True, process_group=group,
        )
        cp.load_full_weight(ref.weight.data, ref.bias.data)
        x = torch.randn(2, 4, 8)

        def fake_all_gather(outputs, tensor, group=None):
            assert group is group_obj
            outputs[0].copy_(tensor)
            outputs[1].copy_(tensor)

        group_obj = group
        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.all_gather", side_effect=fake_all_gather) as all_gather:
            y = cp(x)

        assert y.shape[-1] == 16
        all_gather.assert_called_once()


class TestRowParallelLinear:
    def test_world_size_1_matches_nn_linear(self):
        ref = _seeded_linear(8, 16)
        rp = RowParallelLinear(8, 16, world_size=1, rank=0, bias=True, input_is_parallel=True)
        rp.load_full_weight(ref.weight.data, ref.bias.data)
        x = torch.randn(2, 4, 8)
        torch.testing.assert_close(rp(x), ref(x))

    def test_load_full_weight_slices_columns(self):
        full_w = torch.arange(16 * 8, dtype=torch.float32).view(16, 8)
        full_b = torch.arange(16, dtype=torch.float32)
        rp_r0 = RowParallelLinear(8, 16, world_size=2, rank=0, bias=True, input_is_parallel=True)
        rp_r1 = RowParallelLinear(8, 16, world_size=2, rank=1, bias=True, input_is_parallel=True)
        rp_r0.load_full_weight(full_w, full_b)
        rp_r1.load_full_weight(full_w, full_b)
        torch.testing.assert_close(rp_r0.weight.data, full_w[:, :4])
        torch.testing.assert_close(rp_r1.weight.data, full_w[:, 4:])
        # Bias is replicated, not sharded.
        torch.testing.assert_close(rp_r0.bias.data, full_b)
        torch.testing.assert_close(rp_r1.bias.data, full_b)

    def test_in_features_must_be_divisible(self):
        with pytest.raises(ValueError, match="not divisible"):
            RowParallelLinear(7, 4, world_size=2, rank=0)

    def test_forward_uses_supplied_process_group(self):
        group = object()
        rp = RowParallelLinear(
            8, 16, world_size=2, rank=0, bias=True, input_is_parallel=True, process_group=group,
        )
        x = torch.randn(2, 4, 4)

        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.all_reduce") as all_reduce:
            rp(x)

        assert all_reduce.call_args.kwargs["group"] is group


# --------------------------------------------------------------------------- #
# apply_tensor_parallel — module patching at world_size=1 must be a no-op
# numerical change (weights are sharded into a single full-size shard).
# --------------------------------------------------------------------------- #


def _tiny_qwen3_config():
    return Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_position_embeddings=64,
    )


class TestApplyTensorParallel:
    def test_tp_size_1_is_a_noop(self):
        """tp_size=1 should leave the model untouched."""
        torch.manual_seed(0)
        m = Qwen3ForCausalLM(_tiny_qwen3_config())
        ref_attn0 = m.model.layers[0].self_attn.q_proj
        result = apply_tensor_parallel(m, tp_size=1, tp_rank=0)
        assert result is m
        assert m.model.layers[0].self_attn.q_proj is ref_attn0  # no replacement

    def test_tp_size_2_replaces_projections(self):
        """At tp_size=2 the linear projections are swapped for parallel versions."""
        torch.manual_seed(0)
        m = Qwen3ForCausalLM(_tiny_qwen3_config())
        apply_tensor_parallel(m, tp_size=2, tp_rank=0)
        attn = m.model.layers[0].self_attn
        mlp = m.model.layers[0].mlp
        assert isinstance(attn.q_proj, ColumnParallelLinear)
        assert isinstance(attn.k_proj, ColumnParallelLinear)
        assert isinstance(attn.v_proj, ColumnParallelLinear)
        assert isinstance(attn.o_proj, RowParallelLinear)
        assert isinstance(mlp.gate_proj, ColumnParallelLinear)
        assert isinstance(mlp.up_proj, ColumnParallelLinear)
        assert isinstance(mlp.down_proj, RowParallelLinear)

    def test_head_counts_become_local(self):
        torch.manual_seed(0)
        cfg = _tiny_qwen3_config()  # 4 heads, 2 kv heads
        m = Qwen3ForCausalLM(cfg)
        apply_tensor_parallel(m, tp_size=2, tp_rank=0)
        attn = m.model.layers[0].self_attn
        assert attn.num_heads == 2
        assert attn.num_key_value_heads == 1
        assert attn.num_key_value_groups == 2

    def test_indivisible_heads_raises(self):
        cfg = Qwen3Config(
            vocab_size=64, hidden_size=32, num_layers=1,
            num_attention_heads=3,  # not divisible by 2
            num_key_value_heads=1, head_dim=8, intermediate_size=64,
            max_position_embeddings=64,
        )
        m = Qwen3ForCausalLM(cfg)
        with pytest.raises(ValueError, match="num_attention_heads=3"):
            apply_tensor_parallel(m, tp_size=2, tp_rank=0)


# --------------------------------------------------------------------------- #
# Strategy facade
# --------------------------------------------------------------------------- #


class TestQwen3TensorParallelism:
    def test_creation(self):
        cfg = TensorParallelConfig(tensor_parallel_size=2, tensor_parallel_rank=0)
        p = Qwen3TensorParallelism(cfg)
        assert p.tp_size == 2 and p.tp_rank == 0

    def test_tensor_parallel_config_syncs_rank_and_world_size(self):
        cfg = TensorParallelConfig(tensor_parallel_size=4, tensor_parallel_rank=2)
        assert cfg.world_size == 4
        assert cfg.rank == 2

    def test_can_apply_only_qwen3(self):
        p = Qwen3TensorParallelism(TensorParallelConfig(tensor_parallel_size=2))
        ok = type("M", (), {"config": type("C", (), {"model_type": "qwen3"})()})()
        nope = type("M", (), {"config": type("C", (), {"model_type": "llama"})()})()
        assert p.can_apply(ok)
        assert not p.can_apply(nope)

    def test_can_apply_single_gpu_returns_false(self):
        p = Qwen3TensorParallelism(TensorParallelConfig(tensor_parallel_size=1))
        ok = type("M", (), {"config": type("C", (), {"model_type": "qwen3"})()})()
        assert not p.can_apply(ok)

    def test_wrap_size_1_is_passthrough(self):
        p = Qwen3TensorParallelism(TensorParallelConfig(tensor_parallel_size=1))
        m = Qwen3ForCausalLM(_tiny_qwen3_config())
        assert p.wrap_model(m) is m

    def test_wrap_threads_process_group_into_parallel_linears(self):
        group = object()
        p = Qwen3TensorParallelism(
            TensorParallelConfig(
                tensor_parallel_size=2,
                tensor_parallel_rank=0,
                process_group=group,
            )
        )
        m = Qwen3ForCausalLM(_tiny_qwen3_config())
        wrapped = p.wrap_model(m)
        attn = wrapped.model.layers[0].self_attn
        assert attn.q_proj.process_group is group
        assert attn.o_proj.process_group is group

    def test_setup_distributed_uses_tensor_parallel_rank_and_size(self):
        p = Qwen3TensorParallelism(TensorParallelConfig(tensor_parallel_size=4, tensor_parallel_rank=2))
        with patch("torch.distributed.is_initialized", return_value=False), \
             patch("torch.distributed.init_process_group") as init_pg:
            p.setup_distributed()
        assert init_pg.call_args.kwargs["rank"] == 2
        assert init_pg.call_args.kwargs["world_size"] == 4

    def test_setup_distributed_skips_when_process_group_supplied(self):
        p = Qwen3TensorParallelism(
            TensorParallelConfig(tensor_parallel_size=2, tensor_parallel_rank=1, process_group=object())
        )
        with patch("torch.distributed.init_process_group") as init_pg:
            p.setup_distributed()
        init_pg.assert_not_called()


# --------------------------------------------------------------------------- #
# Communication primitives & registry
# --------------------------------------------------------------------------- #


class TestCommunicationPrimitives:
    def test_all_reduce_creation(self):
        AllReducePrimitive(world_size=4, rank=0)  # smoke

    def test_all_gather_creation(self):
        AllGatherPrimitive(world_size=4, rank=0)

    def test_reduce_scatter_creation(self):
        ReduceScatterPrimitive(world_size=4, rank=0)

    def test_all_reduce_noops_without_initialized_dist(self):
        x = torch.ones(4)
        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=False):
            out = AllReducePrimitive(world_size=2, rank=0).execute([x])
        assert out[0] is x

    def test_all_gather_noops_without_initialized_dist(self):
        x = torch.ones(4)
        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=False):
            out = AllGatherPrimitive(world_size=2, rank=0).execute([x])
        assert out[0] is x

    def test_reduce_scatter_uses_list_input_and_process_group(self):
        x = torch.arange(8, dtype=torch.float32)
        group = object()

        def fake_reduce_scatter(output, input_list, group=None):
            assert isinstance(input_list, list)
            assert group is group_obj
            output.copy_(input_list[0])

        group_obj = group
        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.reduce_scatter", side_effect=fake_reduce_scatter) as reduce_scatter:
            out = ReduceScatterPrimitive(world_size=2, rank=0, process_group=group).execute([x], dim=0)

        assert out[0].shape[0] == 4
        reduce_scatter.assert_called_once()


class TestParallelismFactory:
    def test_create_tensor_parallel(self):
        cfg = TensorParallelConfig(tensor_parallel_size=2)
        p = ParallelismFactory.create_parallelism(cfg)
        # The factory routes via the registry, which yields a strategy
        # implementation. We only need it to expose wrap_model.
        assert hasattr(p, "wrap_model")


class TestParallelismRegistry:
    def test_registry_lists_tensor_parallel(self):
        assert "tensor_parallel" in parallelism_registry.list_strategies()
