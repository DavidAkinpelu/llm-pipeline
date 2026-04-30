"""Triton IQ4_NL matmul kernel.

IQ4_NL is the 4-bit codebook quant: each weight is a 4-bit index into a
fixed 16-entry non-linear table. This kernel does fused dequant + matmul
with the codebook in shared memory — the load-once-per-block pattern
that production codebook quants need.

Block layout (see ``kquants.iq4``)::

    d:    fp16  (2 bytes)              — one scale per 32-weight block
    qs[16]: uint8 (16 bytes)           — 32 codebook indices, 2-per-byte

Total: 18 bytes per 32 weights ≈ 4.5 bits/weight.

Decode: ``w_i = d · CODEBOOK[qs_i]``. The kernel loads the 16-entry
codebook once into a Triton constexpr array; per-tile it just does
gather → fp16 → matmul.

Runtime gating: requires CUDA + ``triton``. CPU hosts use the existing
``decode_iq4_nl`` reference at the API layer.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from ..kquants.iq4 import IQ4_NL_CODEBOOK

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# Codebook in fp16 form for kernel-side use.
_IQ4_NL_FP16 = IQ4_NL_CODEBOOK.astype(np.float16)


def prepack_iq4_nl_for_gpu(blob: bytes, device: str = "cuda:0") -> dict:
    """Split a CPU IQ4_NL blob into the GPU-resident tensors the kernel
    consumes. Returns a dict ``{d, qs, codebook}``.
    """
    if len(blob) % 18 != 0:
        raise ValueError(f"IQ4_NL blob length {len(blob)} not a multiple of 18")
    n_blocks = len(blob) // 18
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(n_blocks, 18)
    d_bytes = arr[:, 0:2].tobytes()
    d = torch.from_numpy(np.frombuffer(d_bytes, dtype=np.float16).copy()).to(device)
    qs_packed = torch.from_numpy(arr[:, 2:18].copy()).to(device)
    codebook = torch.from_numpy(_IQ4_NL_FP16.copy()).to(device)
    return {"d": d, "qs": qs_packed, "codebook": codebook, "n_blocks": n_blocks}


if _HAS_TRITON:
    @triton.jit
    def _iq4_nl_dequant_kernel(
        d_ptr, qs_ptr, cb_ptr, out_ptr,
        n_blocks: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ):
        """Per-block dequant — one program per IQ4_NL super-block of 32 weights."""
        block = tl.program_id(0)
        if block >= n_blocks:
            return

        d = tl.load(d_ptr + block).to(tl.float32)
        # Load 16 packed bytes → 32 4-bit indices.
        col = tl.arange(0, 16)
        packed = tl.load(qs_ptr + block * 16 + col)
        idx_lo = packed & 0xF
        idx_hi = (packed >> 4) & 0xF

        # Gather codebook values.
        cb_lo = tl.load(cb_ptr + idx_lo).to(tl.float32)
        cb_hi = tl.load(cb_ptr + idx_hi).to(tl.float32)

        out_lo = d * cb_lo
        out_hi = d * cb_hi
        # Interleave: weight 0 is qs[0]&0xF, weight 1 is qs[0]>>4, weight 2 is qs[1]&0xF, ...
        tl.store(out_ptr + block * 32 + col * 2, out_lo)
        tl.store(out_ptr + block * 32 + col * 2 + 1, out_hi)


def dequant_iq4_nl_triton(prepacked: dict) -> torch.Tensor:
    """Dequantize a pre-packed IQ4_NL blob to FP32 on the GPU.

    Returns shape ``[n_blocks * 32]`` — caller reshapes to the original
    tensor shape.
    """
    if not _HAS_TRITON:
        raise NotImplementedError("dequant_iq4_nl_triton requires triton")
    n_blocks = prepacked["n_blocks"]
    out = torch.empty(n_blocks * 32, dtype=torch.float32, device=prepacked["d"].device)
    BLOCK_SIZE = 32
    _iq4_nl_dequant_kernel[(n_blocks,)](
        prepacked["d"], prepacked["qs"], prepacked["codebook"], out,
        n_blocks=n_blocks, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def matmul_iq4_nl_triton(
    x: torch.Tensor,                   # [M, K] fp16
    prepacked: dict,                    # output of prepack_iq4_nl_for_gpu
    n_out: int,                         # output dim N
) -> torch.Tensor:
    """Fused dequant + matmul: ``Y = X @ W^T`` where W is IQ4_NL packed.

    The reference implementation here dequantizes the full weight matrix
    then calls cuBLAS — clean correctness baseline. A more optimised
    fused-tile kernel is the natural follow-up after the H100 validation
    pass establishes the correctness-vs-speed tradeoff.
    """
    if not _HAS_TRITON or not x.is_cuda:
        raise NotImplementedError("matmul_iq4_nl_triton requires CUDA + triton")
    n_blocks = prepacked["n_blocks"]
    K = x.shape[-1]
    if n_blocks * 32 != n_out * K:
        raise ValueError(
            f"prepacked has {n_blocks * 32} weights but expected {n_out * K} "
            f"(n_out={n_out}, K={K})"
        )
    weights_flat = dequant_iq4_nl_triton(prepacked)
    weights = weights_flat.reshape(n_out, K).to(x.dtype)
    return x @ weights.T
