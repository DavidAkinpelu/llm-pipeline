"""Gated DeltaNet linear-attention module (Qwen3.5/3.6 variant).

Reference: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta
Rule" (arXiv 2412.06464, ICLR 2025), and the HuggingFace ``Qwen3_5GatedDeltaNet``
class. This module is the **pure-PyTorch reference** matching the HF
``torch_recurrent_gated_delta_rule`` path — no Triton kernels, no fused
RMSNorm, no causal-conv1d CUDA fast path. Slow on long sequences but
numerically faithful, and the right teaching artefact.

For production performance on Hopper (H100/H200), the same forward/backward
math is available as TileLang warp-specialised kernels in QwenLM's
**FlashQLA** library (https://github.com/QwenLM/FlashQLA) — 2–3× over FLA
Triton on long sequences. The recurrence here is the unambiguous reference
all of those implementations agree on; ``forward_chunk_gated_delta_rule``
below is the entry point that swaps in FlashQLA when it's importable.

The math, per-token-per-head
-----------------------------

State is a ``[k_dim, v_dim]`` matrix. At each time step:

.. code-block:: text

    α_t  = exp(g_t)                                    # decay  (scalar per head)
    β_t  = sigmoid(b_t)                                # update strength (scalar per head)
    S_t' = α_t · S_{t-1}                                # apply decay
    pred = k_t @ S_t'                                  # current key's predicted value
    Δ_t  = β_t · (v_t − pred)                           # delta-rule correction
    S_t  = S_t' + outer(k_t, Δ_t)                       # rank-1 update
    y_t  = q_t @ S_t                                   # readout

The "delta rule" framing: associative memory with key-replacement instead
of pure write-add. New (k, v) overwrites whatever value was previously
associated with k, weighted by β.

What this module does
---------------------

1. Project hidden state through ``in_proj_qkv`` (combined Q/K/V),
   ``in_proj_z`` (output gate), ``in_proj_b`` (β source), ``in_proj_a``
   (α/g source).
2. Apply a depthwise causal Conv1d (kernel=4) + SiLU on the QKV stream.
   This adds a small local-context window before the recurrence.
3. L2-normalise Q and K (training-stability trick from FLA / DeltaNet papers).
4. Compute ``g = −exp(A_log) · softplus(a + dt_bias)`` so ``α = exp(g) ∈ (0, 1)``.
5. Run the gated-delta-rule recurrence above.
6. Apply z-gated RMSNorm (multiplicative gate from in_proj_z, RMS-normalised
   over head_v_dim).
7. Project back to hidden via ``out_proj``.

GQA-style: if ``num_v_heads > num_k_heads`` (the common case — Qwen3.6-27B
is 16 → 48; the MoE release is 16 → 32), Q and K are repeat-interleaved
along the head axis to match num_v_heads, so each value head gets its own
private S matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise along ``dim``. Matches the FLA implementation: it adds the
    eps *inside* the rsqrt rather than the more common stable form ``x /
    max(||x||, eps)``. The two agree to within machine precision on healthy
    inputs but the in-rsqrt form has a smoother gradient near zero."""
    inv = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv


class RMSNormGated(nn.Module):
    """Z-gated RMSNorm used by Qwen3.5/3.6 GatedDeltaNet.

    ``forward(x, z)`` = ``rmsnorm(x * silu(z)) * weight``. The multiplicative
    gate fuses the SiLU non-linearity that would normally sit between the
    recurrence output and the linear projection — saving one materialisation
    of the (B, T, num_v_heads, head_v_dim) tensor.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is not None:
            x = x * F.silu(z.to(x.dtype))
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps).to(x.dtype)
        return x_normed * self.weight


# --------------------------------------------------------------------------- #
# The recurrence
# --------------------------------------------------------------------------- #


def _flash_qla_chunk():
    """Try to import FlashQLA's chunkwise kernel. Returns the function or None.

    FlashQLA is Hopper-only (H100/H200), needs TileLang, and isn't a hard
    dep — so we soft-import and fall back to ``recurrent_gated_delta_rule``
    when it isn't installed.
    """
    try:
        from flash_qla import chunk_gated_delta_rule as _fn  # type: ignore[import-not-found]
        return _fn
    except Exception:
        return None


def forward_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Forward dispatch for the gated delta rule.

    Chooses an implementation in this order:

    1. **FlashQLA** (``flash_qla.chunk_gated_delta_rule``) — Hopper-only,
       2–3× FLA Triton on long contexts. Used when the input is on CUDA
       and FlashQLA is importable.
    2. **chunkwise reference** (``chunk_gated_delta_rule``) — pure-PyTorch
       O(T·C + T) port of the HF ``torch_chunk_gated_delta_rule``.
       Default fallback. Solves the within-chunk delta rule via a fixed
       ``C``-step closed-form loop (independent of sequence length) and
       only iterates across the ``T/C`` chunk boundaries sequentially.
    3. **per-token recurrence** (``recurrent_gated_delta_rule``) — only
       used for sequence length 1 (decode step) where the chunk version's
       padding adds overhead with no parallelism benefit.

    All three return identical outputs to within fp32 round-off.
    """
    T = query.shape[1]

    flash_fn = _flash_qla_chunk()
    if flash_fn is not None and query.is_cuda:
        scale = query.shape[-1] ** -0.5
        return flash_fn(
            query, key, value, g=g, beta=beta, scale=scale,
            initial_state=initial_state,
            cu_seqlens=None,
        )

    if T == 1:
        return recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm=use_qk_l2norm,
        )

    return chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        chunk_size=chunk_size,
    )


def chunk_gated_delta_rule(
    query: torch.Tensor,           # [B, T, H, D_k]
    key: torch.Tensor,              # [B, T, H, D_k]
    value: torch.Tensor,            # [B, T, H, D_v]
    g: torch.Tensor,                # [B, T, H]
    beta: torch.Tensor,             # [B, T, H]
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pure-PyTorch chunkwise-parallel Gated Delta Rule.

    Port of HF ``torch_chunk_gated_delta_rule``. The trick: within each
    chunk of size ``C`` the recurrence

        S_t = α_t·S_{t-1} − β_t·k_t·k_t^T·S_{t-1} + β_t·k_t·v_t^T

    can be solved in closed form (a lower-triangular linear system) given
    the boundary state ``S_chunk_start``. We compute that triangular
    structure with a single Python-level loop of length ``C`` (independent
    of ``T``); the only ``T``-scaling cost is the inter-chunk recurrence
    that propagates ``S`` across ``T/C`` boundaries.

    The chunk solver builds a triangular ``attn`` matrix that encodes how
    each within-chunk update affects later positions, then applies it as
    a batched matmul. This converts most of the work from a tight Python
    loop into BLAS-grade kernels — the same speedup you'd get from a
    parallel scan, without needing the scan to be associative.

    Returns ``(output, final_state)``.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # [B, T, H, D] → [B, H, T, D] in fp32 for numerical stability.
    query, key, value = [t.transpose(1, 2).contiguous().to(torch.float32) for t in (query, key, value)]
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)
    g = g.transpose(1, 2).contiguous().to(torch.float32)

    B, H, T, D_k = key.shape
    D_v = value.shape[-1]
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad:
        query = F.pad(query, (0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))
    T_full = T + pad
    n_chunks = T_full // chunk_size

    # Standard sqrt(D_k) scaling on Q.
    query = query * (D_k ** -0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    def _into_chunks(x):
        return x.reshape(x.shape[0], x.shape[1], n_chunks, chunk_size, x.shape[-1])

    query, key, value, k_beta, v_beta = [_into_chunks(t) for t in (query, key, value, k_beta, v_beta)]
    g = g.reshape(B, H, n_chunks, chunk_size)

    # ---- Within-chunk closed-form ----
    # Cumulative log-decay inside each chunk.
    g = g.cumsum(dim=-1)
    # Decay mask M[i, j] = exp(g[i] − g[j]) for i ≥ j else 0. Lower-tri inclusive.
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()

    # Pairwise (k_beta · k^T) within the chunk, weighted by decay.
    upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(upper, 0)

    # The within-chunk delta-rule "kernel": triangular fixed-point iteration.
    # Each step folds row i's contributions into the rows above. C iterations
    # are enough because the dependency chain has depth at most C.
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(dim=-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    # Effective intra-chunk values and key-cumdecay for the inter-chunk update.
    v_intra = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    # ---- Inter-chunk recurrence ----
    state = (
        torch.zeros(B, H, D_k, D_v, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    out = torch.zeros_like(value)
    upper_strict = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for c in range(n_chunks):
        q_c = query[:, :, c]
        k_c = key[:, :, c]
        v_c = v_intra[:, :, c]
        # Within-chunk attention-like piece, restricted to the strict lower
        # triangle by the decay_mask (which is already lower-tri).
        attn_c = (q_c @ k_c.transpose(-1, -2) * decay_mask[:, :, c])
        # Subtract the prediction the carried-over state would make at each k_t.
        v_prime = k_cumdecay[:, :, c] @ state
        v_new = v_c - v_prime
        # The "across-chunk" output piece: q_t @ (decayed previous state).
        attn_inter = (q_c * g[:, :, c, :, None].exp()) @ state
        out[:, :, c] = attn_inter + attn_c @ v_new
        # Update the carried state with the chunk's contributions, applying
        # the residual decay from each within-chunk position to the chunk end.
        state = (
            state * g[:, :, c, -1, None, None].exp()
            + (k_c * (g[:, :, c, -1, None] - g[:, :, c]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    final = state if output_final_state else None
    out = out.reshape(B, H, T_full, D_v)[:, :, :T]
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, final


def recurrent_gated_delta_rule(
    query: torch.Tensor,           # [B, T, H, D_k]
    key: torch.Tensor,              # [B, T, H, D_k]
    value: torch.Tensor,            # [B, T, H, D_v]
    g: torch.Tensor,                # [B, T, H]    log-decay (negative; α = exp(g) ∈ (0,1))
    beta: torch.Tensor,             # [B, T, H]    update strength ∈ (0,1)
    initial_state: Optional[torch.Tensor] = None,   # [B, H, D_k, D_v]
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reference per-token recurrence — slow on long sequences but bit-correct.

    Returns ``(output, final_state)`` with shapes ``[B, T, H, D_v]`` and
    ``[B, H, D_k, D_v]``. ``final_state`` is ``None`` if ``output_final_state=False``.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # Move to [B, H, T, D] in fp32 for numerical stability.
    query = query.transpose(1, 2).contiguous().to(torch.float32)
    key = key.transpose(1, 2).contiguous().to(torch.float32)
    value = value.transpose(1, 2).contiguous().to(torch.float32)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)
    g = g.transpose(1, 2).contiguous().to(torch.float32)

    B, H, T, D_k = key.shape
    D_v = value.shape[-1]

    # Per-paper: scale Q by 1/sqrt(D_k) to keep the recurrent dot product in range.
    query = query * (D_k ** -0.5)

    if initial_state is None:
        state = torch.zeros(B, H, D_k, D_v, dtype=value.dtype, device=value.device)
    else:
        state = initial_state.to(value)

    out = torch.zeros(B, H, T, D_v, dtype=value.dtype, device=value.device)
    for t in range(T):
        q_t = query[:, :, t]                                 # [B, H, D_k]
        k_t = key[:, :, t]                                   # [B, H, D_k]
        v_t = value[:, :, t]                                 # [B, H, D_v]
        # α scalar per head; broadcast over (D_k, D_v).
        alpha_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, t].unsqueeze(-1)                 # [B, H, 1]

        state = state * alpha_t                              # decay
        # Predict v from current key: ``k_t @ state`` → [B, H, D_v].
        kv_pred = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_pred) * beta_t                     # delta-rule correction
        # Rank-1 update: state += outer(k_t, delta). Outer over the last two dims.
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # Output: ``q_t @ state`` → [B, H, D_v].
        out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    final_state = state if output_final_state else None
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, final_state


# --------------------------------------------------------------------------- #
# Module
# --------------------------------------------------------------------------- #


@dataclass
class GatedDeltaNetConfig:
    """Subset of ``Qwen3_5Config`` fields the GatedDeltaNet module needs."""

    hidden_size: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int = 4
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"


class Qwen3_5GatedDeltaNet(nn.Module):
    """Pure-PyTorch port of Qwen3.5 / Qwen3.6's Gated DeltaNet linear-attention layer.

    Forward signature is intentionally simple — no cache plumbing in the
    reference. Inference-with-cache will live in a follow-up module that
    wraps this one and threads ``conv_state`` + ``recurrent_state`` through.
    """

    def __init__(self, config: GatedDeltaNetConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel = config.linear_conv_kernel_dim

        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be a multiple of "
                f"num_k_heads ({self.num_k_heads}) for GQA-style sharing"
            )
        self.kv_repeat = self.num_v_heads // self.num_k_heads

        # Input projections.
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # Depthwise causal conv1d on the (Q ⨁ K ⨁ V) stream. Padding is
        # handled in ``forward`` (either with explicit zeros for the first
        # call or with the cached prefix for streaming decode), so the
        # conv1d itself uses ``padding=0``.
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel,
            groups=self.conv_dim,
            padding=0,
            bias=False,
        )

        # Per-head decay parameters: ``g = -exp(A_log) * softplus(a + dt_bias)``.
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0.0, 16.0)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,                              # [B, T, H_in]
        initial_state: Optional[torch.Tensor] = None,             # [B, num_v_heads, D_k, D_v]
        conv_state: Optional[torch.Tensor] = None,                # [B, conv_dim, conv_kernel - 1]
        return_final_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional streaming caches.

        ``initial_state`` carries the recurrent matrix S between calls.
        ``conv_state`` carries the trailing ``conv_kernel - 1`` frames of the
        previous call's pre-conv input so the depthwise conv1d's receptive
        field is continuous across calls — essential for cache-aware decode.

        When ``return_final_state=True``, the second return is a tuple
        ``(recurrent_state, new_conv_state)``; otherwise it's ``None``.
        """
        B, T, _ = hidden_states.shape

        # --- 1. Input projections ---
        mixed_qkv = self.in_proj_qkv(hidden_states)             # [B, T, key*2 + value]
        z = self.in_proj_z(hidden_states)                       # [B, T, value_dim]
        b = self.in_proj_b(hidden_states)                       # [B, T, num_v_heads]
        a = self.in_proj_a(hidden_states)                       # [B, T, num_v_heads]

        # --- 2. Depthwise causal conv1d + SiLU ---
        # Pad/prepend explicitly so the conv kernel sees the right context.
        # First call → pad k-1 zeros on the left (causal). Subsequent calls →
        # prepend ``conv_state`` (the previous tail) so the kernel spans the
        # boundary. After conv, the output length equals the new ``T``.
        x = mixed_qkv.transpose(1, 2)                            # [B, C, T]
        if conv_state is not None:
            x_padded = torch.cat([conv_state, x], dim=-1)        # [B, C, (k-1)+T]
        else:
            x_padded = F.pad(x, (self.conv_kernel - 1, 0))       # left-pad zeros

        new_conv_state = (
            x_padded[:, :, -(self.conv_kernel - 1):].detach().clone()
            if return_final_state and self.conv_kernel > 1
            else None
        )
        x = self.conv1d(x_padded)                                # [B, C, T]
        x = F.silu(x)
        mixed_qkv = x.transpose(1, 2)                            # [B, T, C]

        # --- 3. Split into Q, K, V and reshape into heads ---
        q, k, v = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        q = q.reshape(B, T, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, T, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, T, self.num_v_heads, self.head_v_dim)

        # GQA repeat for value heads.
        if self.kv_repeat > 1:
            q = q.repeat_interleave(self.kv_repeat, dim=2)
            k = k.repeat_interleave(self.kv_repeat, dim=2)

        # --- 4. Gates ---
        beta = b.sigmoid()                                       # [B, T, num_v_heads]
        # ``g`` is the log of the per-step decay α; α = exp(g) ∈ (0, 1).
        # The negation pins it to the stable side of the exponential.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # --- 5. Recurrence ---
        # ``forward_chunk_gated_delta_rule`` picks FlashQLA → chunk reference
        # → recurrent based on T and runtime availability.
        core_out, final_state = forward_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta,
            initial_state=initial_state,
            output_final_state=return_final_state,
            use_qk_l2norm=True,
        )

        # --- 6. Z-gated RMSNorm + output projection ---
        # Norm operates per (head, head_v_dim); reshape so the last axis is head_v_dim.
        core_out = core_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_out = self.norm(core_out, z_flat)
        core_out = core_out.reshape(B, T, self.value_dim)

        cache = (final_state, new_conv_state) if return_final_state else None
        return self.out_proj(core_out), cache
