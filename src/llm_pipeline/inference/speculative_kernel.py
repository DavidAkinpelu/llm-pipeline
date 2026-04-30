"""Triton speculative-decoding verification kernel.

Production speculative decoding bottlenecks on the per-position accept/
reject loop in Python (see ``inference/speculative.py``). This kernel
fuses the K verification steps into a single GPU call: given draft
probs and target probs at K positions, decide accept/reject for each
position in parallel and report the first rejection index.

Falls through to a torch reference on non-CUDA / non-Triton hosts.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def verify_drafts_batched(
    target_probs: torch.Tensor,        # [K, V]
    draft_probs: torch.Tensor,          # [K]    — draft_q[i] = q(t_i)
    draft_tokens: torch.Tensor,         # [K]    int64
    rand_uniform: torch.Tensor,         # [K]    pre-sampled in [0, 1)
) -> torch.Tensor:
    """Return per-position accept/reject decisions and first-rejection index.

    Output: ``[K]`` int8 — 1 if accepted, 0 if rejected. The caller
    uses the first 0 as the truncation point.

    Algorithm (per position i):
        p_i  = target_probs[i, draft_tokens[i]]
        q_i  = draft_probs[i]
        accept_i = rand_uniform[i] < min(1, p_i / q_i)

    On rejection at position j, positions j+1..K are not meaningful (the
    standard specdec algorithm stops the chain there), but this kernel
    computes them anyway in parallel — it's cheap and the host slices off
    the tail.
    """
    if not _HAS_TRITON or not target_probs.is_cuda:
        return _torch_reference(target_probs, draft_probs, draft_tokens, rand_uniform)

    K, V = target_probs.shape
    out = torch.empty(K, dtype=torch.int8, device=target_probs.device)
    BLOCK = 32
    grid = ((K + BLOCK - 1) // BLOCK,)
    _verify_kernel[grid](
        target_probs, draft_probs, draft_tokens, rand_uniform, out,
        K, V,
        target_probs.stride(0), target_probs.stride(1),
        BLOCK=BLOCK,
    )
    return out


if _HAS_TRITON:
    @triton.jit
    def _verify_kernel(
        t_ptr, q_ptr, tok_ptr, r_ptr, out_ptr,
        K, V,
        stride_tk, stride_tv,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < K

        tok = tl.load(tok_ptr + offs, mask=mask, other=0)
        # p = target_probs[i, tok[i]]
        p = tl.load(t_ptr + offs * stride_tk + tok * stride_tv, mask=mask, other=0.0)
        q = tl.load(q_ptr + offs, mask=mask, other=1e-9)
        r = tl.load(r_ptr + offs, mask=mask, other=1.0)

        ratio = tl.minimum(p / tl.maximum(q, 1e-9), 1.0)
        accept = (r < ratio).to(tl.int8)
        tl.store(out_ptr + offs, accept, mask=mask)


def _torch_reference(target_probs, draft_probs, draft_tokens, rand_uniform):
    p = target_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
    ratio = (p / draft_probs.clamp_min(1e-9)).clamp(max=1.0)
    return (rand_uniform < ratio).to(torch.int8)
