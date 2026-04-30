"""Unified ``Quantizer`` facade for K-quants and I-quants.

Public surface:

    >>> from llm_pipeline.quantization import Quantizer, QuantMethod
    >>> q = Quantizer(method=QuantMethod.Q4_K_M)
    >>>
    >>> # Optional: collect imatrix from a calibration loader.
    >>> q.calibrate(model, calibration_loader)
    >>>
    >>> # Encode the model's weights.
    >>> result = q.quantize(model)             # returns a QuantizedModel
    >>> result.save("model.gguf-like.bin")     # custom binary; see docstring
    >>>
    >>> # Round-trip a single tensor (handy for testing):
    >>> blob, shape = q.encode(weight)
    >>> recovered = q.decode(blob, shape)

The ``Q4_K_M`` policy mirrors what llama.cpp does: most tensors at Q4_K, but
attention's value projection (``v_proj``) and a fraction of the FFN
down-projections (``down_proj``) get bumped to Q6_K. ``Q4_K_S`` keeps
everything at Q4_K. Other methods (``Q4_K``, ``Q6_K``, ``Q8_K``,
``IQ4_NL``) apply the named format uniformly.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .kquants import (
    Imatrix,
    ImatrixCalibrator,
    QuantMethod,
    decode_iq4_nl,
    decode_q3_k,
    decode_q3_k_out,
    decode_q4_k,
    decode_q4_k_out,
    decode_q5_k,
    decode_q5_k_out,
    decode_q6_k,
    decode_q6_k_out,
    decode_q8_k,
    encode_iq4_nl,
    encode_q3_k,
    encode_q3_k_out,
    encode_q4_k,
    encode_q4_k_out,
    encode_q5_k,
    encode_q5_k_out,
    encode_q6_k,
    encode_q6_k_out,
    encode_q8_k,
    get_block_shape,
)


# --------------------------------------------------------------------------- #
# Per-method dispatch
# --------------------------------------------------------------------------- #


_BASE_ENCODERS = {
    QuantMethod.Q3_K: encode_q3_k,
    QuantMethod.Q3_K_OUT: encode_q3_k_out,
    QuantMethod.Q4_K: encode_q4_k,
    QuantMethod.Q4_K_OUT: encode_q4_k_out,
    QuantMethod.Q5_K: encode_q5_k,
    QuantMethod.Q5_K_OUT: encode_q5_k_out,
    QuantMethod.Q6_K: encode_q6_k,
    QuantMethod.Q6_K_OUT: encode_q6_k_out,
    QuantMethod.Q8_K: encode_q8_k,
    QuantMethod.IQ4_NL: encode_iq4_nl,
}
_BASE_DECODERS = {
    QuantMethod.Q3_K: decode_q3_k,
    QuantMethod.Q3_K_OUT: decode_q3_k_out,
    QuantMethod.Q4_K: decode_q4_k,
    QuantMethod.Q4_K_OUT: decode_q4_k_out,
    QuantMethod.Q5_K: decode_q5_k,
    QuantMethod.Q5_K_OUT: decode_q5_k_out,
    QuantMethod.Q6_K: decode_q6_k,
    QuantMethod.Q6_K_OUT: decode_q6_k_out,
    QuantMethod.Q8_K: decode_q8_k,
    QuantMethod.IQ4_NL: decode_iq4_nl,
}


def _resolve_base_method(method: QuantMethod) -> QuantMethod:
    """Map a variant (Q4_K_S/M) to its underlying per-tensor base format."""
    if method in (QuantMethod.Q4_K_S, QuantMethod.Q4_K_M):
        return QuantMethod.Q4_K
    return method


# --------------------------------------------------------------------------- #
# Per-tensor policy for the S/M/L variants
# --------------------------------------------------------------------------- #


def _policy_for(method: QuantMethod, tensor_name: str, tensor_index: int, total_of_kind: int) -> QuantMethod:
    """Pick the actual format for one named tensor under the given method.

    Implements the standard llama.cpp policy:

      - ``Q4_K_S``: every tensor at Q4_K.
      - ``Q4_K_M``: most tensors at Q4_K, but
            * attention value projection (``v_proj`` / ``attn_v``) → Q6_K
            * the *first half* of the model's FFN ``down_proj`` tensors → Q6_K
        ``tensor_index`` and ``total_of_kind`` let the caller tell us where in
        the layer stack this particular tensor sits, so we can apply the
        "first half" rule.

      - All other methods apply uniformly.
    """
    if method == QuantMethod.Q4_K_S:
        return QuantMethod.Q4_K
    if method == QuantMethod.Q4_K_M:
        lower = tensor_name.lower()
        is_v_proj = "v_proj" in lower or lower.endswith(".attn_v.weight")
        is_down = "down_proj" in lower
        if is_v_proj:
            return QuantMethod.Q6_K
        # "First half" — round up so that with a single down_proj the lone
        # tensor still counts as belonging to the first half.
        if is_down and tensor_index < (total_of_kind + 1) // 2:
            return QuantMethod.Q6_K
        return QuantMethod.Q4_K
    return method


# --------------------------------------------------------------------------- #
# Encoded artifact
# --------------------------------------------------------------------------- #


@dataclass
class QuantizedTensor:
    name: str
    method: QuantMethod
    shape: Tuple[int, ...]
    blob: bytes


@dataclass
class QuantizedModel:
    """Container for the encoded weights of an entire model.

    The on-disk format is a tiny custom binary — one header per tensor
    followed by its blob — meant for round-tripping with this library, not
    llama.cpp.
    """

    method: QuantMethod
    tensors: List[QuantizedTensor] = field(default_factory=list)

    # Bytes per tensor + total size estimate.
    @property
    def total_bytes(self) -> int:
        return sum(len(t.blob) for t in self.tensors)

    def save(self, path: str) -> None:
        """Serialize to a flat binary.

        Layout per tensor:
            uint32 name_len, bytes name (utf-8)
            uint32 method_len, bytes method (utf-8)
            uint32 ndim, uint32[ndim] dims
            uint32 blob_len, bytes blob
        """
        with open(path, "wb") as f:
            f.write(b"LLMPQ\x00")             # magic
            f.write(struct.pack("<I", len(self.tensors)))
            for t in self.tensors:
                _write_str(f, t.name)
                _write_str(f, t.method.value)
                f.write(struct.pack("<I", len(t.shape)))
                for d in t.shape:
                    f.write(struct.pack("<I", int(d)))
                f.write(struct.pack("<I", len(t.blob)))
                f.write(t.blob)

    @classmethod
    def load(cls, path: str) -> "QuantizedModel":
        with open(path, "rb") as f:
            magic = f.read(6)
            if magic != b"LLMPQ\x00":
                raise ValueError(f"bad magic: {magic!r}")
            n = struct.unpack("<I", f.read(4))[0]
            tensors: List[QuantizedTensor] = []
            method: Optional[QuantMethod] = None
            for _ in range(n):
                name = _read_str(f)
                method_str = _read_str(f)
                ndim = struct.unpack("<I", f.read(4))[0]
                shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))
                blob_len = struct.unpack("<I", f.read(4))[0]
                blob = f.read(blob_len)
                m = QuantMethod(method_str)
                tensors.append(QuantizedTensor(name=name, method=m, shape=shape, blob=blob))
                if method is None:
                    method = m
            return cls(method=method or QuantMethod.Q4_K, tensors=tensors)


def _write_str(f, s: str) -> None:
    b = s.encode("utf-8")
    f.write(struct.pack("<I", len(b)))
    f.write(b)


def _read_str(f) -> str:
    n = struct.unpack("<I", f.read(4))[0]
    return f.read(n).decode("utf-8")


# --------------------------------------------------------------------------- #
# Quantizer facade
# --------------------------------------------------------------------------- #


class Quantizer:
    """Encode a model's weights using one of the K-quant / I-quant methods.

    Parameters
    ----------
    method:
        Which format (or variant policy) to use.
    skip_pattern:
        Glob-like substrings; any parameter whose name contains one will be
        kept in float (typical for layer norms and embeddings).
    """

    DEFAULT_SKIP = ("norm", "ln_", "layernorm", "rmsnorm")

    def __init__(
        self,
        method: QuantMethod | str = QuantMethod.Q4_K_M,
        skip_pattern: Optional[Iterable[str]] = None,
    ):
        self.method = method if isinstance(method, QuantMethod) else QuantMethod(method)
        self.skip_pattern = tuple(skip_pattern) if skip_pattern is not None else self.DEFAULT_SKIP
        self.imatrix: Optional[Imatrix] = None

    # -- imatrix ---------------------------------------------------------- #

    def calibrate(self, model: nn.Module, batches: Iterable[Dict[str, torch.Tensor]]) -> Imatrix:
        cal = ImatrixCalibrator(model)
        cal.attach()
        try:
            with torch.no_grad():
                for batch in batches:
                    model(**{k: v for k, v in batch.items() if k != "labels"})
        finally:
            self.imatrix = cal.detach()
        return self.imatrix

    def set_imatrix(self, imatrix: Imatrix) -> None:
        self.imatrix = imatrix

    # -- single-tensor round trip ---------------------------------------- #

    def encode(self, tensor: torch.Tensor, name: Optional[str] = None) -> Tuple[bytes, Tuple[int, ...]]:
        """Encode one tensor under this quantizer's method.

        ``name`` is consulted only when a Q4_K_S/M variant policy is active
        AND the policy depends on tensor identity (which it does for M).
        """
        method = _policy_for(self.method, name or "", tensor_index=0, total_of_kind=1)
        method = _resolve_base_method(method)
        encoder = _BASE_ENCODERS[method]
        importance = self._importance_for(name, tensor.shape)
        return encoder(tensor, importance=importance)

    def decode(self, blob: bytes, shape: Tuple[int, ...], method: Optional[QuantMethod] = None) -> torch.Tensor:
        m = _resolve_base_method(method or self.method)
        decoder = _BASE_DECODERS[m]
        return decoder(blob, shape)

    # -- whole-model encoding -------------------------------------------- #

    def quantize(self, model: nn.Module) -> QuantizedModel:
        """Walk the model and encode each parameter under the right per-tensor method."""
        # Pre-pass: count how many ``down_proj`` tensors there are, so M-policy
        # can apply the "first half → Q6_K" rule.
        down_indices: Dict[str, int] = {}
        n_down = 0
        for name, _p in model.named_parameters():
            if "down_proj" in name:
                down_indices[name] = n_down
                n_down += 1

        result = QuantizedModel(method=self.method)
        for name, p in model.named_parameters():
            if any(pat in name for pat in self.skip_pattern):
                continue
            tensor = p.data.detach()
            if tensor.numel() < 32:           # too small to bother quantizing
                continue
            method = _policy_for(
                self.method,
                tensor_name=name,
                tensor_index=down_indices.get(name, 0),
                total_of_kind=max(n_down, 1),
            )
            base = _resolve_base_method(method)
            encoder = _BASE_ENCODERS[base]
            importance = self._importance_for(name, tensor.shape)
            blob, shape = encoder(tensor, importance=importance)
            result.tensors.append(QuantizedTensor(name=name, method=base, shape=shape, blob=blob))
        return result

    # -- internals ------------------------------------------------------- #

    def _importance_for(self, name: Optional[str], shape: torch.Size) -> Optional[torch.Tensor]:
        """Map an imatrix entry (per-input-channel) onto a full weight shape.

        For a Linear weight with shape ``(out, in)``, broadcast the
        ``in``-sized importance vector across the ``out`` axis.
        """
        if self.imatrix is None or name is None:
            return None
        # imatrix keys are module names; weight names are like ``foo.weight``.
        mod_key = name[:-7] if name.endswith(".weight") else name
        v = self.imatrix.get(mod_key)
        if v is None:
            return None
        if len(shape) == 2 and shape[1] == v.numel():
            return v.expand(shape[0], -1).contiguous()
        return None
