"""Educational pure-Python implementations of GGUF-style K-quants and I-quants.

This module is a *learning artifact* — the algorithms here match the standard
formats described in the llama.cpp / GGUF ecosystem (Q4_K, Q6_K, Q8_K,
IQ4_NL, ...) but the byte layout is **not** guaranteed to be bit-exact to
llama.cpp's C structs. To ship a quantized model for use in llama.cpp /
Ollama / llama-cpp-python, run ``llama.cpp/quantize`` on an FP16 GGUF.

What you get here:

  * Numerically faithful K-quant *algorithms* (super-block + sub-block scales).
  * Round-trip ``encode → decode`` you can use to study quantization error.
  * Imatrix calibration that weights quantization error by per-channel input
    activation magnitude — same idea as production GGUF quantizers.
  * A clean ``Quantizer`` facade that picks per-tensor formats based on the
    standard S/M/L variant policy.

Format coverage
---------------
- Legacy: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0   (in ``legacy.py`` — already in gguf.py)
- K-quants: **Q4_K, Q6_K, Q8_K**         (this module)
- I-quants: **IQ4_NL**                    (this module)
- Q3_K, Q5_K, IQ4_XS, IQ2/IQ3 family: not implemented yet — same template.
"""

from .methods import QuantMethod, BlockShape, get_block_shape
from .q3_k import encode_q3_k, decode_q3_k
from .q3_k_out import encode_q3_k_out, decode_q3_k_out
from .q4_k import encode_q4_k, decode_q4_k
from .q4_k_out import encode_q4_k_out, decode_q4_k_out, DEFAULT_OUTLIER_K
from .q5_k import encode_q5_k, decode_q5_k
from .q5_k_out import encode_q5_k_out, decode_q5_k_out
from .q6_k_out import encode_q6_k_out, decode_q6_k_out
from .q6_k import encode_q6_k, decode_q6_k
from .q8_k import encode_q8_k, decode_q8_k
from .iq4 import encode_iq4_nl, decode_iq4_nl, IQ4_NL_CODEBOOK
from .iq4_xs import encode_iq4_xs, decode_iq4_xs
from .iq_low import (
    encode_iq3_xxs, decode_iq3_xxs,
    encode_iq2_xxs, decode_iq2_xxs,
)
from .imatrix import ImatrixCalibrator, Imatrix

__all__ = [
    "QuantMethod",
    "BlockShape",
    "get_block_shape",
    "encode_q3_k",
    "decode_q3_k",
    "encode_q3_k_out",
    "decode_q3_k_out",
    "encode_q4_k",
    "decode_q4_k",
    "encode_q4_k_out",
    "decode_q4_k_out",
    "encode_q5_k",
    "decode_q5_k",
    "encode_q5_k_out",
    "decode_q5_k_out",
    "encode_q6_k_out",
    "decode_q6_k_out",
    "DEFAULT_OUTLIER_K",
    "encode_q6_k",
    "decode_q6_k",
    "encode_q8_k",
    "decode_q8_k",
    "encode_iq4_nl",
    "decode_iq4_nl",
    "IQ4_NL_CODEBOOK",
    "encode_iq4_xs",
    "decode_iq4_xs",
    "encode_iq3_xxs",
    "decode_iq3_xxs",
    "encode_iq2_xxs",
    "decode_iq2_xxs",
    "ImatrixCalibrator",
    "Imatrix",
]
