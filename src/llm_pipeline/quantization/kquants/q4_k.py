"""Q4_K: 4-bit K-quant with two-level scaling.

Block structure (256 weights per super-block):

  super_block = [
      d:    fp16,                    # master scale (for the 8 sub-block scales)
      dmin: fp16,                    # master min   (for the 8 sub-block mins)
      scales[8]: 6-bit unsigned,     # per-sub-block scale (relative to d)
      mins[8]:   6-bit unsigned,     # per-sub-block min   (relative to dmin)
      qs[256]:   4-bit unsigned,     # per-weight value
  ]

Effective storage per super-block (in bits):
    d, dmin: 2 × 16              = 32
    scales:  8 × 6               = 48
    mins:    8 × 6               = 48
    qs:      256 × 4             = 1024
    -----------------------------------
    total                        = 1152 / 256 = 4.5 bits per weight

Decoding: ``w_i ≈ d · scale_sb · q_i  -  dmin · min_sb`` where ``q_i ∈ {0..15}``
is the 4-bit code, ``scale_sb`` and ``min_sb`` are the 6-bit-quantized
per-sub-block scale/min values, and ``d, dmin`` are the master scales.

This is an algorithmically faithful pure-Python implementation. The byte
layout is *not* guaranteed bit-exact to llama.cpp's ``block_q4_K`` C
struct (which uses a non-trivial 6-bit packing) — see the wire-format note
in ``__init__.py``.

References
----------
- Inspired by Tim Dettmers' description of the K-quant family.
- Algorithm matches what llama.cpp's ``quantize_row_q4_K_reference`` does.
"""

from typing import Optional, Tuple

import numpy as np
import torch


SUPER_BLOCK = 256
SUB_BLOCK = 32
N_SUB = SUPER_BLOCK // SUB_BLOCK  # 8


def _pack_6bit(values: np.ndarray) -> np.ndarray:
    """Pack a length-8 uint8 array of 6-bit values into 6 bytes (48 bits).

    Layout: bits [0:6] = values[0], [6:12] = values[1], ..., [42:48] = values[7].
    """
    if values.shape != (8,):
        raise ValueError(f"expected 8 values, got {values.shape}")
    bits = 0
    for i, v in enumerate(values):
        bits |= (int(v) & 0x3F) << (i * 6)
    out = np.zeros(6, dtype=np.uint8)
    for i in range(6):
        out[i] = (bits >> (i * 8)) & 0xFF
    return out


def _unpack_6bit(packed: np.ndarray) -> np.ndarray:
    """Inverse of ``_pack_6bit``."""
    if packed.shape != (6,):
        raise ValueError(f"expected 6 packed bytes, got {packed.shape}")
    bits = 0
    for i in range(6):
        bits |= int(packed[i]) << (i * 8)
    out = np.zeros(8, dtype=np.uint8)
    for i in range(8):
        out[i] = (bits >> (i * 6)) & 0x3F
    return out


def _fit_sub_blocks_qkx2_batched(
    x: np.ndarray,
    w: np.ndarray,
    nmax: int = 15,
    nstep: int = 20,
    rmin: float = -1.0,
    rdelta: float = 0.1,
) -> tuple:
    """Vectorized ``make_qkx2_quants`` over a batch of sub-blocks.

    ``x`` and ``w`` are ``(B, n)``. Returns ``(scales[B], neg_mins[B], L[B, n])``.

    For each row independently we run the same search llama.cpp uses: try
    ``nstep+1`` candidate ``iscale`` values around the trivial guess, at
    each one re-quantize and **analytically** solve the 2×2 normal
    equations for the (scale, min) that minimize the weighted MSE given
    those L_i. Keep the best.

    The closed-form, given fixed integer levels L and per-weight importance w:

        Solve  [SW  SL] [min]   = [SX]
               [SL  SL2] [scale]   [SXL]

    where SW = Σwᵢ, SL = Σwᵢ Lᵢ, SL2 = Σwᵢ Lᵢ², SX = Σwᵢ xᵢ, SXL = Σwᵢ Lᵢ xᵢ.
    If the unconstrained ``min`` comes out positive we constrain ``min = 0``
    and use the simpler ``scale = SXL / SL2`` (symmetric fit).

    All operations are pure numpy on the (B, n) arrays — no Python-level
    loop over rows.
    """
    B, n = x.shape
    sum_w = w.sum(axis=1)                                  # [B]
    sum_x = (w * x).sum(axis=1)                            # [B]

    xmin = np.minimum(x.min(axis=1), 0.0)                  # [B] — min ≤ 0
    xmax = x.max(axis=1)                                    # [B]
    rng = xmax - xmin                                       # [B]
    valid = rng > 0                                         # rows we'll fit

    # Defaults for degenerate rows (all-equal): scale=0, min=-xmin, L=0.
    scales = np.zeros(B, dtype=np.float32)
    cur_mins = np.where(valid, xmin, 0.0).astype(np.float32)
    L_best = np.zeros((B, n), dtype=np.int32)

    # Initial trivial fit.
    iscale0 = np.where(valid, nmax / np.where(valid, rng, 1.0), 0.0)  # [B]
    L0 = np.clip(
        np.rint(iscale0[:, None] * (x - xmin[:, None])), 0, nmax
    ).astype(np.int32)                                      # [B, n]
    scale0 = np.where(valid, 1.0 / np.where(valid, iscale0, 1.0), 0.0)  # [B]
    diff0 = scale0[:, None] * L0 + xmin[:, None] - x
    best_mad = (w * diff0 * diff0).sum(axis=1)              # [B]
    L_best[:] = L0
    scales[:] = scale0
    cur_mins[:] = xmin

    # Search nstep + 1 candidate iscales.
    for is_step in range(nstep + 1):
        cand_iscale = np.where(
            valid, (rmin + rdelta * is_step + nmax) / np.where(valid, rng, 1.0), 0.0
        )                                                   # [B]
        positive = cand_iscale > 0
        L = np.clip(
            np.rint(cand_iscale[:, None] * (x - xmin[:, None])), 0, nmax
        ).astype(np.int32)                                  # [B, n]
        Lf = L.astype(np.float32)
        SL = (w * Lf).sum(axis=1)
        SL2 = (w * Lf * Lf).sum(axis=1)
        SXL = (w * Lf * x).sum(axis=1)
        D = sum_w * SL2 - SL * SL                           # [B]
        det_ok = D > 0
        # Closed-form unconstrained solution.
        D_safe = np.where(det_ok, D, 1.0)
        cand_scale_uc = (sum_w * SXL - sum_x * SL) / D_safe
        cand_min_uc = (SL2 * sum_x - SL * SXL) / D_safe
        # If unconstrained ``min`` > 0, fall back to the SXL/SL2 fit with min=0.
        sl2_ok = SL2 > 0
        SL2_safe = np.where(sl2_ok, SL2, 1.0)
        cand_scale_c = SXL / SL2_safe
        use_constrained = cand_min_uc > 0
        cand_scale = np.where(use_constrained, cand_scale_c, cand_scale_uc)
        cand_min = np.where(use_constrained, 0.0, cand_min_uc)
        # Final per-row mask: candidate is usable.
        usable = positive & det_ok & ((~use_constrained) | sl2_ok)

        diff = cand_scale[:, None] * Lf + cand_min[:, None] - x
        mad = (w * diff * diff).sum(axis=1)                 # [B]
        improved = usable & (mad < best_mad)
        # Update best.
        best_mad = np.where(improved, mad, best_mad)
        L_best = np.where(improved[:, None], L, L_best)
        scales = np.where(improved, cand_scale, scales)
        cur_mins = np.where(improved, cand_min, cur_mins)

    return scales.astype(np.float32), (-cur_mins).astype(np.float32), L_best.astype(np.uint8)


def _fit_sub_block_qkx2(
    x: np.ndarray,
    w: np.ndarray,
    nmax: int = 15,
    nstep: int = 20,
    rmin: float = -1.0,
    rdelta: float = 0.1,
) -> tuple:
    """Single-row convenience wrapper around the batched fit (for tests)."""
    s, m, L = _fit_sub_blocks_qkx2_batched(
        x[None, :], w[None, :], nmax=nmax, nstep=nstep, rmin=rmin, rdelta=rdelta
    )
    return float(s[0]), float(m[0]), L[0]


def _fit_sub_blocks_qkx3_batched(
    x: np.ndarray,
    w: np.ndarray,
    nmax: int = 15,
    nstep: int = 20,
    rmin: float = -1.0,
    rdelta: float = 0.1,
    n_refine: int = 5,
) -> tuple:
    """Iterative refinement on top of ``make_qkx2_quants``.

    qkx2 picks the best (scale, min) **given** the integer levels L produced
    by a candidate ``iscale``. But once the closed-form re-solves (scale,
    min), the optimal L given that mapping might differ from the round-based
    L used during the search — so the qkx2 fit is a one-step approximation
    to the joint optimum over (scale, min, L).

    qkx3 closes that loop: starting from the qkx2 fit, alternate

        L'      = clip(round((x − min)/scale), 0, nmax)
        (s', m') = closed_form_with_levels(L', x, w)

    until L stops changing or ``n_refine`` iterations elapse. Per-row best
    weighted MSE wins; if no refinement step improves on the qkx2 seed,
    that seed is returned untouched.

    Cost vs qkx2: ``n_refine`` extra closed-form passes (each is ~2× a single
    candidate inside qkx2, so ~10–15% extra work overall). Quality gain on
    Gaussian-ish weights: ~5–15% L2 reduction depending on shape and bits.
    """
    if n_refine <= 0:
        return _fit_sub_blocks_qkx2_batched(x, w, nmax=nmax, nstep=nstep, rmin=rmin, rdelta=rdelta)

    # Seed: qkx2 result. ``cur_min`` here is the *negated* min (≥ 0); the
    # internal refinement uses the signed min, so flip back.
    scales, neg_mins, L_best = _fit_sub_blocks_qkx2_batched(
        x, w, nmax=nmax, nstep=nstep, rmin=rmin, rdelta=rdelta
    )
    cur_mins = -neg_mins.astype(np.float32)              # signed, ≤ 0
    cur_scales = scales.astype(np.float32)               # ≥ 0
    L_best = L_best.astype(np.int32)

    sum_w = w.sum(axis=1)                                # [B]
    sum_x = (w * x).sum(axis=1)                          # [B]

    # Initial best MAD (weighted MSE) at the qkx2 seed.
    diff = cur_scales[:, None] * L_best.astype(np.float32) + cur_mins[:, None] - x
    best_mad = (w * diff * diff).sum(axis=1)             # [B]

    for _ in range(n_refine):
        # 1. Re-derive levels from current (scale, min).
        scale_safe = np.where(cur_scales > 0, cur_scales, 1.0)
        L_new = np.clip(
            np.rint((x - cur_mins[:, None]) / scale_safe[:, None]), 0, nmax
        ).astype(np.int32)

        # 2. Closed-form re-fit given those levels.
        Lf = L_new.astype(np.float32)
        SL = (w * Lf).sum(axis=1)
        SL2 = (w * Lf * Lf).sum(axis=1)
        SXL = (w * Lf * x).sum(axis=1)
        D = sum_w * SL2 - SL * SL
        det_ok = D > 0
        D_safe = np.where(det_ok, D, 1.0)
        scale_uc = (sum_w * SXL - sum_x * SL) / D_safe
        min_uc = (SL2 * sum_x - SL * SXL) / D_safe
        sl2_ok = SL2 > 0
        SL2_safe = np.where(sl2_ok, SL2, 1.0)
        scale_c = SXL / SL2_safe
        use_constrained = min_uc > 0
        new_scale = np.where(use_constrained, scale_c, scale_uc)
        new_min = np.where(use_constrained, 0.0, min_uc)

        # Reject rows with degenerate fits or non-positive scale.
        usable = (
            (cur_scales > 0)
            & (new_scale > 0)
            & ((~use_constrained & det_ok) | (use_constrained & sl2_ok))
        )

        diff = new_scale[:, None] * Lf + new_min[:, None] - x
        mad = (w * diff * diff).sum(axis=1)              # [B]
        improved = usable & (mad < best_mad)

        best_mad = np.where(improved, mad, best_mad)
        L_best = np.where(improved[:, None], L_new, L_best)
        cur_scales = np.where(improved, new_scale, cur_scales)
        cur_mins = np.where(improved, new_min, cur_mins)

    return cur_scales.astype(np.float32), (-cur_mins).astype(np.float32), L_best.astype(np.uint8)


def _quantize_super_block(
    block: np.ndarray,
    importance: Optional[np.ndarray] = None,
) -> bytes:
    """Encode 256 weights → ~144 bytes per the layout above.

    Uses the ``make_qkx2_quants`` search (``_fit_sub_block_qkx2``) per
    sub-block — same idea llama.cpp uses for production Q4_K. ``importance``
    (if given) reweights the per-weight squared error during that search.
    """
    assert block.shape == (SUPER_BLOCK,), block.shape
    block = block.astype(np.float32)
    imp = (
        np.ones(SUPER_BLOCK, dtype=np.float32)
        if importance is None
        else importance.astype(np.float32)
    )

    sb_view = block.reshape(N_SUB, SUB_BLOCK)
    imp_view = imp.reshape(N_SUB, SUB_BLOCK)

    # ---- Step 1: vectorized per-sub-block scale-fit search ----
    fit_scales, fit_mins, sub_levels = _fit_sub_blocks_qkx2_batched(sb_view, imp_view)

    # ---- Step 2: master scales for the per-sub-block (scale, min) ----
    max_sc = float(fit_scales.max()) if fit_scales.size else 0.0
    max_min = float(fit_mins.max()) if fit_mins.size else 0.0
    d = np.float32(1.0) if max_sc < 1e-12 else np.float32(max_sc / 63.0)
    dmin = np.float32(1.0) if max_min < 1e-12 else np.float32(max_min / 63.0)

    sb_scales_q = np.clip(np.round(fit_scales / d), 0, 63).astype(np.uint8)
    sb_mins_q = np.clip(np.round(fit_mins / dmin), 0, 63).astype(np.uint8)

    # ---- Step 3: re-quantize each weight against the *quantized* (scale, min)
    # the decoder will see, importance-aware floor-vs-ceil.
    deq_scale = sb_scales_q.astype(np.float32) * d
    deq_min = -sb_mins_q.astype(np.float32) * dmin
    deq_scale_safe = np.where(deq_scale == 0, 1.0, deq_scale)

    qs = np.zeros(SUPER_BLOCK, dtype=np.uint8)
    for s in range(N_SUB):
        sub_w = sb_view[s]
        sub_imp = imp_view[s]
        q_real = (sub_w - deq_min[s]) / deq_scale_safe[s]
        q_floor = np.clip(np.floor(q_real).astype(np.int32), 0, 15)
        q_ceil = np.clip(q_floor + 1, 0, 15)
        err_floor = sub_imp * (sub_w - (deq_scale[s] * q_floor + deq_min[s])) ** 2
        err_ceil = sub_imp * (sub_w - (deq_scale[s] * q_ceil + deq_min[s])) ** 2
        chosen = np.where(err_ceil < err_floor, q_ceil, q_floor).astype(np.uint8)
        qs[s * SUB_BLOCK : (s + 1) * SUB_BLOCK] = chosen

    # ---- Step 4: pack to bytes. ----
    out = bytearray()
    out.extend(np.float16(d).tobytes())
    out.extend(np.float16(dmin).tobytes())
    out.extend(_pack_6bit(sb_scales_q).tobytes())
    out.extend(_pack_6bit(sb_mins_q).tobytes())
    # qs: 256 × 4 bits = 128 bytes, packed two-per-byte.
    qs_packed = (qs[0::2] | (qs[1::2] << 4)).astype(np.uint8)
    out.extend(qs_packed.tobytes())
    assert len(out) == 2 + 2 + 6 + 6 + 128, len(out)
    return bytes(out)


def _dequantize_super_block(blob: bytes) -> np.ndarray:
    """Inverse of ``_quantize_super_block``. Returns 256 fp32 weights."""
    if len(blob) != 144:
        raise ValueError(f"Q4_K block must be 144 bytes; got {len(blob)}")
    arr = np.frombuffer(blob, dtype=np.uint8)
    d = np.frombuffer(arr[0:2].tobytes(), dtype=np.float16).astype(np.float32)[0]
    dmin = np.frombuffer(arr[2:4].tobytes(), dtype=np.float16).astype(np.float32)[0]
    sb_scales_q = _unpack_6bit(arr[4:10])
    sb_mins_q = _unpack_6bit(arr[10:16])
    qs_packed = arr[16:144]
    qs = np.zeros(SUPER_BLOCK, dtype=np.uint8)
    qs[0::2] = qs_packed & 0xF
    qs[1::2] = qs_packed >> 4

    deq_scale = sb_scales_q.astype(np.float32) * d            # [N_SUB]
    deq_min = -sb_mins_q.astype(np.float32) * dmin            # [N_SUB]
    out = np.zeros(SUPER_BLOCK, dtype=np.float32)
    for s in range(N_SUB):
        sub = qs[s * SUB_BLOCK : (s + 1) * SUB_BLOCK].astype(np.float32)
        out[s * SUB_BLOCK : (s + 1) * SUB_BLOCK] = deq_scale[s] * sub + deq_min[s]
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _pack_6bit_batched(values: np.ndarray) -> np.ndarray:
    """Pack ``(N, 8)`` uint8 values (each in [0, 63]) into ``(N, 6)`` bytes.

    The 48-bit packed representation: bit ``[6·i, 6·i+6)`` holds ``values[i]``.
    """
    v = values.astype(np.uint64)
    bits = np.zeros(v.shape[0], dtype=np.uint64)
    for i in range(8):
        bits |= (v[:, i] & np.uint64(0x3F)) << np.uint64(i * 6)
    packed = np.zeros((v.shape[0], 6), dtype=np.uint8)
    for i in range(6):
        packed[:, i] = ((bits >> np.uint64(i * 8)) & np.uint64(0xFF)).astype(np.uint8)
    return packed


def _encode_super_blocks_batched(
    blocks: np.ndarray,            # (N, 256)
    importance: np.ndarray,         # (N, 256)
    use_qkx3: bool = False,
) -> np.ndarray:                    # (N, 144) bytes
    """Encode a batch of super-blocks at once. Returns ``(N, 144)`` uint8.

    ``use_qkx3=True`` swaps the per-sub-block scale-fit for the iterative
    ``make_qkx3_quants`` refinement; everything else is identical.
    """
    N = blocks.shape[0]
    sb_view = blocks.reshape(N, N_SUB, SUB_BLOCK)
    imp_view = importance.reshape(N, N_SUB, SUB_BLOCK)

    # ---- Step 1: per-sub-block scale-fit search, batched across all (N*N_SUB) sub-blocks. ----
    fitter = _fit_sub_blocks_qkx3_batched if use_qkx3 else _fit_sub_blocks_qkx2_batched
    fit_scales, fit_mins, sub_levels = fitter(
        sb_view.reshape(N * N_SUB, SUB_BLOCK),
        imp_view.reshape(N * N_SUB, SUB_BLOCK),
    )
    fit_scales = fit_scales.reshape(N, N_SUB)            # [N, N_SUB]
    fit_mins = fit_mins.reshape(N, N_SUB)                # [N, N_SUB] (≥ 0; the negated-min)
    sub_levels = sub_levels.reshape(N, N_SUB, SUB_BLOCK) # [N, N_SUB, SUB_BLOCK]

    # ---- Step 2: per-super-block master scales d, dmin (FP16). ----
    max_sc = fit_scales.max(axis=1)                      # [N]
    max_min = fit_mins.max(axis=1)                       # [N]
    # Where the max is ~0, fall back to 1.0 to avoid div-by-zero.
    d = np.where(max_sc < 1e-12, 1.0, max_sc / 63.0).astype(np.float32)
    dmin = np.where(max_min < 1e-12, 1.0, max_min / 63.0).astype(np.float32)
    sb_scales_q = np.clip(np.round(fit_scales / d[:, None]), 0, 63).astype(np.uint8)
    sb_mins_q = np.clip(np.round(fit_mins / dmin[:, None]), 0, 63).astype(np.uint8)

    # ---- Step 3: re-quantize each weight against the post-quantization deq scales. ----
    deq_scale = sb_scales_q.astype(np.float32) * d[:, None]   # [N, N_SUB]
    deq_min = -sb_mins_q.astype(np.float32) * dmin[:, None]   # [N, N_SUB]
    deq_scale_safe = np.where(deq_scale == 0, 1.0, deq_scale)

    # Broadcast deq_scale/deq_min over the 32 weights of each sub-block.
    deq_scale_b = deq_scale[:, :, None]                  # [N, N_SUB, 1]
    deq_min_b = deq_min[:, :, None]                      # [N, N_SUB, 1]
    deq_scale_safe_b = deq_scale_safe[:, :, None]
    q_real = (sb_view - deq_min_b) / deq_scale_safe_b
    q_floor = np.clip(np.floor(q_real).astype(np.int32), 0, 15)
    q_ceil = np.clip(q_floor + 1, 0, 15)
    err_floor = imp_view * (sb_view - (deq_scale_b * q_floor + deq_min_b)) ** 2
    err_ceil = imp_view * (sb_view - (deq_scale_b * q_ceil + deq_min_b)) ** 2
    qs = np.where(err_ceil < err_floor, q_ceil, q_floor).astype(np.uint8)
    qs_flat = qs.reshape(N, SUPER_BLOCK)                 # [N, 256]

    # ---- Step 4: pack to bytes. ----
    out = np.zeros((N, 144), dtype=np.uint8)
    out[:, 0:2] = np.frombuffer(d.astype(np.float16).tobytes(), dtype=np.uint8).reshape(N, 2)
    out[:, 2:4] = np.frombuffer(dmin.astype(np.float16).tobytes(), dtype=np.uint8).reshape(N, 2)
    out[:, 4:10] = _pack_6bit_batched(sb_scales_q)
    out[:, 10:16] = _pack_6bit_batched(sb_mins_q)
    # qs: 256 × 4 bits = 128 bytes per super-block; pack two-per-byte.
    out[:, 16:144] = qs_flat[:, 0::2] | (qs_flat[:, 1::2] << 4)
    return out


def encode_q4_k(
    tensor: torch.Tensor,
    importance: Optional[torch.Tensor] = None,
    use_qkx3: bool = False,
) -> Tuple[bytes, Tuple[int, ...]]:
    """Encode an arbitrary-shape tensor as a sequence of Q4_K super-blocks.

    Tensor is flattened in C order; the trailing partial super-block (if
    any) is zero-padded to 256 weights. ``decode_q4_k`` crops the recovered
    output back to the original size.

    ``use_qkx3=True`` enables ``make_qkx3_quants`` iterative refinement on
    top of the default ``make_qkx2_quants`` search — modestly slower
    (~10–15%), modestly more accurate (~5–15% L2 reduction on Gaussian).
    Output bytes are wire-compatible with ``decode_q4_k`` either way.

    Returns ``(blob, original_shape)``.
    """
    flat = tensor.detach().to(torch.float32).cpu().numpy().reshape(-1)
    if importance is not None:
        imp_flat = importance.detach().to(torch.float32).cpu().numpy().reshape(-1)
    else:
        imp_flat = np.ones_like(flat)
    n = flat.shape[0]
    n_blocks = (n + SUPER_BLOCK - 1) // SUPER_BLOCK
    pad = n_blocks * SUPER_BLOCK - n
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
        imp_flat = np.concatenate([imp_flat, np.zeros(pad, dtype=imp_flat.dtype)])
    sb = flat.reshape(n_blocks, SUPER_BLOCK)
    imp = imp_flat.reshape(n_blocks, SUPER_BLOCK)
    encoded = _encode_super_blocks_batched(sb, imp, use_qkx3=use_qkx3)
    return encoded.tobytes(), tuple(tensor.shape)


def decode_q4_k(blob: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    n_elems = 1
    for s in shape:
        n_elems *= s
    n_blocks = (n_elems + SUPER_BLOCK - 1) // SUPER_BLOCK
    expected = n_blocks * 144
    if len(blob) != expected:
        raise ValueError(f"blob length {len(blob)} != expected {expected} for shape {shape}")
    out = np.zeros(n_blocks * SUPER_BLOCK, dtype=np.float32)
    for i in range(n_blocks):
        out[i * SUPER_BLOCK : (i + 1) * SUPER_BLOCK] = _dequantize_super_block(
            blob[i * 144 : (i + 1) * 144]
        )
    return torch.from_numpy(out[:n_elems].copy()).reshape(*shape)
