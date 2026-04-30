"""GPU kernels for K-quant decode + matmul.

Currently provides:

  * ``triton_q4_k`` — pure dequant (Q4_K bytes → FP16 tensor) and a fused
    Q4_K dequant + matmul, both as Triton kernels.

These kernels run on CUDA and require ``triton`` to be importable. CPU
fallbacks are the standard numpy-based encoders/decoders in
``llm_pipeline.quantization.kquants``.
"""

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

if _HAS_TRITON:
    from .triton_q4_k import (
        Q4KGPUWeights,
        dequant_q4_k_triton,
        matmul_q4_k_triton,
        prepack_q4_k_for_gpu,
    )
    from .triton_q5_k import (
        Q5KGPUWeights,
        dequant_q5_k_triton,
        matmul_q5_k_triton,
        prepack_q5_k_for_gpu,
    )
    from .triton_q6_k import (
        Q6KGPUWeights,
        dequant_q6_k_triton,
        matmul_q6_k_triton,
        prepack_q6_k_for_gpu,
    )
    from .triton_q8_k import (
        Q8KGPUWeights,
        dequant_q8_k_triton,
        matmul_q8_k_triton,
        prepack_q8_k_for_gpu,
    )

# These four (RoPE / RMSNorm / softmax / IQ4_NL) and the FP8/MXFP4 kernels
# all gracefully degrade to the torch reference on CPU, so they're safe
# to import unconditionally.
from .triton_rope import apply_rope_triton
from .triton_rmsnorm import apply_rmsnorm_triton
from .triton_softmax import fused_softmax_triton
from .triton_iq4_nl import (
    dequant_iq4_nl_triton,
    matmul_iq4_nl_triton,
    prepack_iq4_nl_for_gpu,
)
from .triton_fp8 import fp8_matmul
from .triton_mxfp4 import mxfp4_matmul

if _HAS_TRITON:
    __all__ = [
        "Q4KGPUWeights",
        "dequant_q4_k_triton",
        "matmul_q4_k_triton",
        "prepack_q4_k_for_gpu",
        "Q5KGPUWeights",
        "dequant_q5_k_triton",
        "matmul_q5_k_triton",
        "prepack_q5_k_for_gpu",
        "Q6KGPUWeights",
        "dequant_q6_k_triton",
        "matmul_q6_k_triton",
        "prepack_q6_k_for_gpu",
        "Q8KGPUWeights",
        "dequant_q8_k_triton",
        "matmul_q8_k_triton",
        "prepack_q8_k_for_gpu",
        "apply_rope_triton",
        "apply_rmsnorm_triton",
        "fused_softmax_triton",
        "dequant_iq4_nl_triton",
        "matmul_iq4_nl_triton",
        "prepack_iq4_nl_for_gpu",
        "fp8_matmul",
        "mxfp4_matmul",
    ]
else:
    __all__ = [
        "apply_rope_triton",
        "apply_rmsnorm_triton",
        "fused_softmax_triton",
        "fp8_matmul",
        "mxfp4_matmul",
    ]
