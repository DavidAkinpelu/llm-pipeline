"""GGUF v3 byte-layout writer for llama.cpp / Ollama compatibility.

Format spec
-----------

GGUF (GPT-Generated Unified Format) is the single-file binary container
llama.cpp uses for model weights + metadata. Layout, top-down:

.. code-block:: text

    +--------------------------------------------------------------+
    | header                                                       |
    |   uint32  magic              0x46554747 ("GGUF" little-endian)|
    |   uint32  version            3                                |
    |   uint64  tensor_count                                        |
    |   uint64  metadata_kv_count                                   |
    +--------------------------------------------------------------+
    | metadata key-value pairs (count = metadata_kv_count)         |
    |   each: <utf8-string key, gguf_type value_type, value>       |
    +--------------------------------------------------------------+
    | tensor info table (count = tensor_count)                     |
    |   each: <utf8-string name, uint32 n_dims, uint64[] shape,    |
    |          gguf_type ggml_type, uint64 offset>                 |
    +--------------------------------------------------------------+
    | padding to alignment boundary (default 32 bytes)             |
    +--------------------------------------------------------------+
    | tensor data, in tensor_info order                            |
    | (each tensor padded to alignment after itself)               |
    +--------------------------------------------------------------+

Strings are length-prefixed (uint64 length, then UTF-8 bytes, no NUL).
Arrays in metadata are also length-prefixed and carry an inner type tag.

Source of truth: ``ggml/src/gguf.cpp`` and the GGUF README in the
``ggerganov/ggml`` repository.

What this writer covers
-----------------------

- Header + metadata + tensor info + aligned tensor data (full v3 wire format).
- Metadata value types: bool, uint8/16/32/64, int8/16/32/64, float32/64,
  string, array (homogeneous, including nested arrays).
- Tensor dtypes: F32, F16, BF16, Q8_0 (the simplest K-quant — 1 byte per
  weight + per-block fp16 scale, easy to verify byte-exact).
- Llama-style tensor name mapping (``model.layers.{i}.self_attn.q_proj.weight``
  ⇄ ``blk.{i}.attn_q.weight``) so HF state_dicts round-trip cleanly.

Out of scope (deliberate)
-------------------------

- The full K-quant family at byte-level (Q4_K / Q5_K / Q6_K). The
  educational K-quant module has its own block layout that is NOT
  byte-identical to llama.cpp's; getting bit-exact GGUF blocks needs a
  separate write path that re-packs weights into llama.cpp's exact 6-bit
  packing. That's a clean follow-up — the GGUF infrastructure here makes
  it straightforward to add (just register the new ``ggml_type`` in
  ``GGMLType`` and supply a per-block writer).
- Reading. There's a small ``read_header_for_inspection`` helper for
  test purposes; for full GGUF loading use llama-cpp-python.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from io import BufferedWriter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


GGUF_MAGIC = 0x46554747            # b"GGUF" little-endian
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32             # padding boundary for tensor data


# --------------------------------------------------------------------------- #
# Wire-format type tags
# --------------------------------------------------------------------------- #


class GGUFType(IntEnum):
    """Metadata value type tags. Stable IDs from the GGUF spec."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(IntEnum):
    """Tensor dtype tags. Subset of llama.cpp's ggml types this writer supports."""

    F32 = 0
    F16 = 1
    Q4_0 = 2          # not yet supported by this writer
    Q4_1 = 3          # not yet supported
    Q5_0 = 6          # not yet supported
    Q5_1 = 7          # not yet supported
    Q8_0 = 8          # supported
    Q8_1 = 9          # not yet supported
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    BF16 = 30


_SUPPORTED_TENSOR_TYPES = {GGMLType.F32, GGMLType.F16, GGMLType.BF16, GGMLType.Q8_0}


# --------------------------------------------------------------------------- #
# Q8_0: 32-element block, fp16 scale + int8 per weight (34 bytes / 32 weights)
# --------------------------------------------------------------------------- #


_Q8_0_BLOCK = 32
_Q8_0_BYTES = 2 + _Q8_0_BLOCK            # fp16 scale + 32 int8 values


def _quantize_q8_0(tensor: np.ndarray) -> bytes:
    """Quantize a 1-D fp32 array to GGUF Q8_0 byte layout.

    Per 32-weight block: pick the absmax, scale = absmax / 127, store the
    scale as fp16 followed by 32 signed int8 values (round-to-nearest).
    Length must be a multiple of 32 — pad upstream if not.
    """
    if tensor.dtype != np.float32:
        tensor = tensor.astype(np.float32)
    if tensor.size % _Q8_0_BLOCK != 0:
        raise ValueError(
            f"Q8_0 expects length divisible by {_Q8_0_BLOCK}, got {tensor.size}"
        )
    n_blocks = tensor.size // _Q8_0_BLOCK
    blocks = tensor.reshape(n_blocks, _Q8_0_BLOCK)
    absmax = np.maximum(np.abs(blocks).max(axis=1), 1e-30)
    scales = (absmax / 127.0).astype(np.float16)
    # Decode-side scale is fp16 → cast back for the round.
    decoded_scales = scales.astype(np.float32)[:, None]
    quants = np.clip(np.round(blocks / decoded_scales), -127, 127).astype(np.int8)
    out = bytearray()
    for i in range(n_blocks):
        out.extend(scales[i].tobytes())
        out.extend(quants[i].tobytes())
    return bytes(out)


def _q8_0_block_count(num_elements: int) -> int:
    if num_elements % _Q8_0_BLOCK != 0:
        raise ValueError(f"Q8_0 needs multiple of {_Q8_0_BLOCK}; got {num_elements}")
    return num_elements // _Q8_0_BLOCK


# --------------------------------------------------------------------------- #
# Tensor name translation (HF → GGUF)
# --------------------------------------------------------------------------- #


_HF_TO_GGUF_LLAMA = {
    "model.embed_tokens.weight": "token_embd.weight",
    "lm_head.weight": "output.weight",
    "model.norm.weight": "output_norm.weight",
}

_HF_LAYER_PROJ_TO_GGUF = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


def hf_to_gguf_name(hf_name: str) -> str:
    """Translate a HuggingFace state_dict key to GGUF's canonical name.

    Handles the standard Llama / Qwen-style layout:
    ``model.layers.{i}.self_attn.q_proj.weight`` → ``blk.{i}.attn_q.weight``.
    Unknown names pass through unchanged so a writer can still record them
    (with a warning at the top of the file via metadata).
    """
    if hf_name in _HF_TO_GGUF_LLAMA:
        return _HF_TO_GGUF_LLAMA[hf_name]
    parts = hf_name.split(".", 3)
    if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
        layer_idx = parts[2]
        rest = parts[3]
        if rest in _HF_LAYER_PROJ_TO_GGUF:
            return f"blk.{layer_idx}.{_HF_LAYER_PROJ_TO_GGUF[rest]}"
    return hf_name


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #


@dataclass
class TensorEntry:
    """One tensor scheduled for writing.

    ``data`` is a numpy array in fp32 / fp16 / bf16; quantization happens
    at write time based on ``ggml_type``.
    """

    name: str
    data: np.ndarray
    ggml_type: GGMLType

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    def num_elements(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def encoded_bytes(self) -> bytes:
        flat = self.data.reshape(-1)
        if self.ggml_type == GGMLType.F32:
            return flat.astype(np.float32).tobytes()
        if self.ggml_type == GGMLType.F16:
            return flat.astype(np.float16).tobytes()
        if self.ggml_type == GGMLType.BF16:
            # numpy lacks native bf16; pack via int16 round-half-to-even.
            f32 = flat.astype(np.float32).view(np.uint32)
            bf16 = ((f32 + 0x7FFF + ((f32 >> 16) & 1)) >> 16).astype(np.uint16)
            return bf16.tobytes()
        if self.ggml_type == GGMLType.Q8_0:
            return _quantize_q8_0(flat)
        raise NotImplementedError(
            f"GGMLType.{self.ggml_type.name} not yet supported by this writer "
            f"(supported: F32, F16, BF16, Q8_0)"
        )


class GGUFWriter:
    """Build a GGUF v3 file incrementally, then ``write_to(path)``.

    Typical use:

    >>> w = GGUFWriter(architecture="llama")
    >>> w.add_str("general.name", "tiny-test")
    >>> w.add_uint32("llama.context_length", 2048)
    >>> w.add_tensor("token_embd.weight", embedding_weight, GGMLType.F16)
    >>> w.write_to("model.gguf")
    """

    def __init__(self, architecture: str, alignment: int = DEFAULT_ALIGNMENT):
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise ValueError(f"alignment must be a positive power of 2; got {alignment}")
        self.alignment = alignment
        self._metadata: List[Tuple[str, GGUFType, Any]] = []
        self._tensors: List[TensorEntry] = []
        self.add_str("general.architecture", architecture)
        self.add_uint32("general.alignment", alignment)

    # --- metadata API --- #

    def add_str(self, key: str, value: str) -> None:
        self._metadata.append((key, GGUFType.STRING, value))

    def add_bool(self, key: str, value: bool) -> None:
        self._metadata.append((key, GGUFType.BOOL, bool(value)))

    def add_uint32(self, key: str, value: int) -> None:
        self._metadata.append((key, GGUFType.UINT32, int(value)))

    def add_int32(self, key: str, value: int) -> None:
        self._metadata.append((key, GGUFType.INT32, int(value)))

    def add_uint64(self, key: str, value: int) -> None:
        self._metadata.append((key, GGUFType.UINT64, int(value)))

    def add_float32(self, key: str, value: float) -> None:
        self._metadata.append((key, GGUFType.FLOAT32, float(value)))

    def add_array(self, key: str, inner_type: GGUFType, values: Sequence[Any]) -> None:
        self._metadata.append((key, GGUFType.ARRAY, (inner_type, list(values))))

    # --- tensor API --- #

    def add_tensor(self, name: str, data: torch.Tensor | np.ndarray, ggml_type: GGMLType) -> None:
        if ggml_type not in _SUPPORTED_TENSOR_TYPES:
            raise NotImplementedError(
                f"GGMLType.{ggml_type.name} not supported by this writer; "
                f"supported types: {sorted(t.name for t in _SUPPORTED_TENSOR_TYPES)}"
            )
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().to(torch.float32).numpy()
        self._tensors.append(TensorEntry(name=name, data=data, ggml_type=ggml_type))

    # --- write --- #

    def write_to(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as f:
            self._write_header(f)
            self._write_metadata(f)
            self._write_tensor_info(f)
            self._pad(f, self.alignment)
            self._write_tensor_data(f)

    # --- internal --- #

    def _write_header(self, f: BufferedWriter) -> None:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(self._tensors)))
        f.write(struct.pack("<Q", len(self._metadata)))

    def _write_metadata(self, f: BufferedWriter) -> None:
        for key, vtype, value in self._metadata:
            self._write_string(f, key)
            f.write(struct.pack("<I", int(vtype)))
            self._write_value(f, vtype, value)

    def _write_value(self, f: BufferedWriter, vtype: GGUFType, value: Any) -> None:
        if vtype == GGUFType.STRING:
            self._write_string(f, value)
        elif vtype == GGUFType.BOOL:
            f.write(struct.pack("<?", value))
        elif vtype == GGUFType.UINT8:
            f.write(struct.pack("<B", value))
        elif vtype == GGUFType.INT8:
            f.write(struct.pack("<b", value))
        elif vtype == GGUFType.UINT16:
            f.write(struct.pack("<H", value))
        elif vtype == GGUFType.INT16:
            f.write(struct.pack("<h", value))
        elif vtype == GGUFType.UINT32:
            f.write(struct.pack("<I", value))
        elif vtype == GGUFType.INT32:
            f.write(struct.pack("<i", value))
        elif vtype == GGUFType.UINT64:
            f.write(struct.pack("<Q", value))
        elif vtype == GGUFType.INT64:
            f.write(struct.pack("<q", value))
        elif vtype == GGUFType.FLOAT32:
            f.write(struct.pack("<f", value))
        elif vtype == GGUFType.FLOAT64:
            f.write(struct.pack("<d", value))
        elif vtype == GGUFType.ARRAY:
            inner_type, items = value
            f.write(struct.pack("<I", int(inner_type)))
            f.write(struct.pack("<Q", len(items)))
            for item in items:
                self._write_value(f, inner_type, item)
        else:
            raise ValueError(f"unknown GGUFType: {vtype}")

    def _write_string(self, f: BufferedWriter, s: str) -> None:
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

    def _write_tensor_info(self, f: BufferedWriter) -> None:
        # Tensor offsets are computed from the start of the data section,
        # so we do a first pass to figure out each tensor's offset.
        offsets: List[int] = []
        cur = 0
        for t in self._tensors:
            offsets.append(cur)
            cur += len(t.encoded_bytes())
            cur = _round_up(cur, self.alignment)

        for t, offset in zip(self._tensors, offsets):
            self._write_string(f, t.name)
            f.write(struct.pack("<I", len(t.shape)))
            for dim in t.shape:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", int(t.ggml_type)))
            f.write(struct.pack("<Q", offset))

    def _write_tensor_data(self, f: BufferedWriter) -> None:
        for t in self._tensors:
            data = t.encoded_bytes()
            f.write(data)
            self._pad(f, self.alignment)

    def _pad(self, f: BufferedWriter, alignment: int) -> None:
        pos = f.tell()
        target = _round_up(pos, alignment)
        if target > pos:
            f.write(b"\x00" * (target - pos))


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) & ~(m - 1)


# --------------------------------------------------------------------------- #
# Minimal reader for tests / inspection
# --------------------------------------------------------------------------- #


@dataclass
class GGUFInspection:
    """What ``read_header_for_inspection`` returns. Enough to verify a write
    round-trips structurally; not a full file loader.
    """

    magic: int
    version: int
    tensor_count: int
    metadata_count: int
    metadata: Dict[str, Any]
    tensors: List[Tuple[str, Tuple[int, ...], GGMLType, int]]   # name, shape, type, data_offset
    alignment: int


def read_header_for_inspection(path: str | Path) -> GGUFInspection:
    """Parse the header / metadata / tensor info of a GGUF file.

    Skips the actual tensor data; intended for tests verifying that a
    writer produced a structurally valid file.
    """
    with Path(path).open("rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file (magic={magic:#x})")
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_count = struct.unpack("<Q", f.read(8))[0]

        def read_str() -> str:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8")

        def read_value(vtype: GGUFType) -> Any:
            if vtype == GGUFType.STRING:
                return read_str()
            fmt = {
                GGUFType.BOOL: "<?", GGUFType.UINT8: "<B", GGUFType.INT8: "<b",
                GGUFType.UINT16: "<H", GGUFType.INT16: "<h",
                GGUFType.UINT32: "<I", GGUFType.INT32: "<i",
                GGUFType.UINT64: "<Q", GGUFType.INT64: "<q",
                GGUFType.FLOAT32: "<f", GGUFType.FLOAT64: "<d",
            }
            if vtype in fmt:
                size = struct.calcsize(fmt[vtype])
                return struct.unpack(fmt[vtype], f.read(size))[0]
            if vtype == GGUFType.ARRAY:
                inner = GGUFType(struct.unpack("<I", f.read(4))[0])
                n = struct.unpack("<Q", f.read(8))[0]
                return [read_value(inner) for _ in range(n)]
            raise ValueError(f"unknown vtype {vtype}")

        metadata: Dict[str, Any] = {}
        for _ in range(metadata_count):
            key = read_str()
            vtype = GGUFType(struct.unpack("<I", f.read(4))[0])
            metadata[key] = read_value(vtype)

        tensors: List[Tuple[str, Tuple[int, ...], GGMLType, int]] = []
        for _ in range(tensor_count):
            name = read_str()
            n_dims = struct.unpack("<I", f.read(4))[0]
            shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
            gtype = GGMLType(struct.unpack("<I", f.read(4))[0])
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append((name, shape, gtype, offset))

        alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        return GGUFInspection(
            magic=magic, version=version,
            tensor_count=tensor_count, metadata_count=metadata_count,
            metadata=metadata, tensors=tensors, alignment=alignment,
        )
