"""Tests for the chunkwise-parallel Gated Delta Rule.

The chunk version must produce bit-close output (within fp32 round-off) to
the per-token recurrent reference for every shape and seed — that's the
correctness contract that lets us swap it in as the default fast path.
"""

import pytest
import torch

from llm_pipeline.models.qwen3_5 import (
    chunk_gated_delta_rule,
    forward_chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)


def _rand_inputs(B=1, T=16, H=2, D_k=4, D_v=4, seed=0):
    g_seed = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, D_k, generator=g_seed)
    k = torch.randn(B, T, H, D_k, generator=g_seed)
    v = torch.randn(B, T, H, D_v, generator=g_seed)
    g_log = -torch.rand(B, T, H, generator=g_seed) * 0.1
    beta = torch.rand(B, T, H, generator=g_seed)
    return q, k, v, g_log, beta


# --------------------------------------------------------------------------- #
# Equivalence with the recurrent reference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "T,chunk_size",
    [(16, 4), (32, 8), (24, 8), (12, 6), (64, 16)],
)
def test_chunk_matches_recurrent_aligned(seed, T, chunk_size):
    """When ``T`` is a multiple of ``chunk_size`` the no-padding path runs."""
    q, k, v, g, beta = _rand_inputs(T=T, seed=seed)
    out_rec, st_rec = recurrent_gated_delta_rule(q, k, v, g=g, beta=beta)
    out_chk, st_chk = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, chunk_size=chunk_size,
    )
    torch.testing.assert_close(out_chk, out_rec, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(st_chk, st_rec, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("T", [9, 10, 13, 17, 30])
def test_chunk_matches_recurrent_unaligned(T):
    """Sequence lengths that aren't multiples of chunk_size — exercises the
    pad-and-crop path.
    """
    q, k, v, g, beta = _rand_inputs(T=T, seed=42)
    out_rec, st_rec = recurrent_gated_delta_rule(q, k, v, g=g, beta=beta)
    out_chk, st_chk = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=8)
    torch.testing.assert_close(out_chk, out_rec, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(st_chk, st_rec, atol=1e-5, rtol=1e-5)


def test_chunk_propagates_initial_state():
    """Initial state must be honoured by the chunk path the same way it is
    by the recurrent path.
    """
    B, T, H, D_k, D_v = 1, 12, 2, 4, 4
    q, k, v, g, beta = _rand_inputs(B=B, T=T, H=H, D_k=D_k, D_v=D_v, seed=7)
    init = torch.randn(B, H, D_k, D_v)
    out_rec, _ = recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=init)
    out_chk, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=init, chunk_size=4)
    torch.testing.assert_close(out_chk, out_rec, atol=1e-5, rtol=1e-5)


def test_chunk_state_chaining_matches_one_shot():
    """Splitting the call at a chunk boundary and chaining the state should
    reproduce the one-shot output.
    """
    B, T, H, D_k, D_v = 1, 16, 2, 4, 4
    q, k, v, g, beta = _rand_inputs(B=B, T=T, H=H, D_k=D_k, D_v=D_v, seed=3)
    full, final = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=4)

    s = 8
    a_out, a_state = chunk_gated_delta_rule(
        q[:, :s], k[:, :s], v[:, :s], g=g[:, :s], beta=beta[:, :s], chunk_size=4,
    )
    b_out, b_state = chunk_gated_delta_rule(
        q[:, s:], k[:, s:], v[:, s:], g=g[:, s:], beta=beta[:, s:],
        initial_state=a_state, chunk_size=4,
    )
    torch.testing.assert_close(torch.cat([a_out, b_out], dim=1), full, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(b_state, final, atol=1e-5, rtol=1e-5)


def test_chunk_is_strictly_causal():
    """Perturbing the input at step t must not affect outputs at steps < t."""
    q, k, v, g, beta = _rand_inputs(T=16, seed=11)
    out_baseline, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=4)

    v2 = v.clone()
    v2[:, -1] = torch.randn_like(v2[:, -1]) * 100.0
    out_perturbed, _ = chunk_gated_delta_rule(q, k, v2, g=g, beta=beta, chunk_size=4)

    torch.testing.assert_close(out_baseline[:, :-1], out_perturbed[:, :-1])
    assert not torch.allclose(out_baseline[:, -1], out_perturbed[:, -1])


def test_chunk_output_final_state_off():
    q, k, v, g, beta = _rand_inputs(T=8)
    out, state = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, chunk_size=4, output_final_state=False,
    )
    assert state is None
    assert out.shape == (q.shape[0], 8, q.shape[2], v.shape[-1])


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #


def test_dispatcher_uses_recurrent_for_single_token():
    """T==1 has no parallelism to exploit; the dispatcher falls back to the
    per-token recurrence (cheaper than the chunk version's padding overhead).
    Regardless, the result must match the chunk version's for T==1.
    """
    q, k, v, g, beta = _rand_inputs(T=1)
    out_disp, _ = forward_chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
    out_rec, _ = recurrent_gated_delta_rule(q, k, v, g=g, beta=beta)
    torch.testing.assert_close(out_disp, out_rec)


def test_dispatcher_uses_chunk_for_long_sequences():
    """For T > 1 on CPU (no FlashQLA), the dispatcher should land on the
    chunk implementation. We verify by output equivalence with the explicit
    chunk call.
    """
    q, k, v, g, beta = _rand_inputs(T=16)
    out_disp, _ = forward_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=4)
    out_chk, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=4)
    torch.testing.assert_close(out_disp, out_chk)


# --------------------------------------------------------------------------- #
# l2-norm flag agrees with the recurrent reference
# --------------------------------------------------------------------------- #


def test_chunk_l2norm_flag_off_matches_recurrent():
    q, k, v, g, beta = _rand_inputs(T=12, seed=5)
    out_rec, _ = recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm=False)
    out_chk, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=4, use_qk_l2norm=False)
    torch.testing.assert_close(out_chk, out_rec, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# FlashQLA soft-import path
# --------------------------------------------------------------------------- #


def test_flashqla_soft_import_failure_does_not_break_dispatcher():
    """Even when ``flash_qla`` is broken / partially installed (the import
    raises something other than ImportError), the dispatcher must still
    return a correct result via the chunk reference. Regression test for
    the case where ``flash_qla`` is installed but its submodules aren't.
    """
    import sys
    from llm_pipeline.models.qwen3_5.gated_deltanet import _flash_qla_chunk
    # The soft-import is a private helper — verify it returns None on broken
    # installs without raising.
    real = _flash_qla_chunk()
    assert real is None or callable(real)

    # Inject a deliberately broken flash_qla module to force the failure path.
    class _Broken:
        def __getattr__(self, name):
            raise RuntimeError("intentional failure")
    saved = sys.modules.get("flash_qla")
    sys.modules["flash_qla"] = _Broken()
    try:
        assert _flash_qla_chunk() is None
        # Dispatcher still returns the chunk-reference result.
        q, k, v, g, beta = _rand_inputs(T=12, seed=99)
        out_disp, _ = forward_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=4)
        out_chk, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=4)
        torch.testing.assert_close(out_disp, out_chk, atol=1e-5, rtol=1e-5)
    finally:
        if saved is None:
            sys.modules.pop("flash_qla", None)
        else:
            sys.modules["flash_qla"] = saved
