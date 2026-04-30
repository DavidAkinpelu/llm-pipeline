"""Quantization framework for llm_pipeline."""

from .configs import (
    QuantizationConfig,
    BnBConfig,
    AQLMConfig,
    LoftQConfig
)

# Training quantization
from .training import (
    BnBQuantizedLinear,
    BnBQuantizedLoRA,
    create_bnb_model,
    AQLMQuantizedLinear,
    AQLMQuantizedLoRA,
    create_aqlm_model,
    LoftQInitializer,
    LoftQQuantizedLoRA,
    optimize_quantization_lora_pair
)

# Inference quantization. The inference subpackage pulls in optional backends
# (TensorRT, ONNX, GGUF) that have incomplete fallback stubs upstream, so we
# guard the re-export. Use ``from llm_pipeline.quantization.inference import ...``
# directly if you need a specific backend's class even when the umbrella import
# fails.
try:
    from .inference import (
        PTQQuantizer,
        PTQConfig,
        quantize_for_inference,
        GGUFConverter,
        GGUFConfig,
        convert_to_gguf,
        ONNXConverter,
        ONNXConfig,
        convert_to_onnx,
        TensorRTConverter,
        TensorRTConfig,
        convert_to_tensorrt,
        DeploymentConverter,
        DeploymentConfig,
        deploy_model,
    )
    _INFERENCE_QUANT_AVAILABLE = True
except (ImportError, AttributeError):
    _INFERENCE_QUANT_AVAILABLE = False

# Pure-Python K-quant / I-quant educational module.
from .quantizer import (
    Quantizer,
    QuantizedModel,
    QuantizedTensor,
)
from .dynamic import (
    UDQuantizer,
    UDQuantizerConfig,
    SensitivityReport,
    DEFAULT_CANDIDATES,
)
from .gguf_writer import (
    GGMLType,
    GGUFInspection,
    GGUFType,
    GGUFWriter,
    hf_to_gguf_name,
    read_header_for_inspection,
)
from .fp_low import (
    decode_fp8_e4m3,
    decode_fp8_e5m2,
    decode_mxfp4,
    encode_fp8_e4m3,
    encode_fp8_e5m2,
    encode_mxfp4,
)
from .kquants import (
    QuantMethod,
    BlockShape,
    get_block_shape,
    encode_q3_k, decode_q3_k,
    encode_q3_k_out, decode_q3_k_out,
    encode_q4_k, decode_q4_k,
    encode_q4_k_out, decode_q4_k_out,
    encode_q5_k, decode_q5_k,
    encode_q5_k_out, decode_q5_k_out,
    encode_q6_k, decode_q6_k,
    encode_q6_k_out, decode_q6_k_out,
    encode_q8_k, decode_q8_k,
    encode_iq4_nl, decode_iq4_nl,
    encode_iq4_xs, decode_iq4_xs,
    encode_iq3_xxs, decode_iq3_xxs,
    encode_iq2_xxs, decode_iq2_xxs,
    IQ4_NL_CODEBOOK,
    ImatrixCalibrator,
    Imatrix,
)

from .quant_utils import (
    get_quantization_info,
    estimate_quantized_memory,
    compare_quantization_methods,
    quantization_quality_metrics
)

__all__ = [
    # Configurations
    "QuantizationConfig",
    "BnBConfig", 
    "AQLMConfig",
    "LoftQConfig",
    
    # BitsAndBytes
    "BnBQuantizedLinear",
    "BnBQuantizedLoRA", 
    "create_bnb_model",
    
    # AQLM
    "AQLMQuantizedLinear",
    "AQLMQuantizedLoRA",
    "create_aqlm_model",
    
    # LoftQ
    "LoftQInitializer",
    "LoftQQuantizedLoRA",
    "optimize_quantization_lora_pair",
    
    # Inference quantization
    "PTQQuantizer",
    "PTQConfig",
    "quantize_for_inference",
    "GGUFConverter",
    "GGUFConfig",
    "convert_to_gguf",
    "ONNXConverter",
    "ONNXConfig",
    "convert_to_onnx",
    "TensorRTConverter",
    "TensorRTConfig",
    "convert_to_tensorrt",
    "DeploymentConverter",
    "DeploymentConfig",
    "deploy_model",

    # Utilities
    "get_quantization_info",
    "estimate_quantized_memory",
    "compare_quantization_methods",
    "quantization_quality_metrics",

    # K-quants / I-quants (educational module)
    "Quantizer",
    "QuantizedModel",
    "QuantizedTensor",
    "UDQuantizer",
    "UDQuantizerConfig",
    "SensitivityReport",
    "DEFAULT_CANDIDATES",
    "QuantMethod",
    "BlockShape",
    "get_block_shape",
    "encode_q3_k", "decode_q3_k",
    "encode_q3_k_out", "decode_q3_k_out",
    "encode_q4_k", "decode_q4_k",
    "encode_q4_k_out", "decode_q4_k_out",
    "encode_q5_k", "decode_q5_k",
    "encode_q5_k_out", "decode_q5_k_out",
    "encode_q6_k", "decode_q6_k",
    "encode_q6_k_out", "decode_q6_k_out",
    "encode_q8_k", "decode_q8_k",
    "encode_iq4_nl", "decode_iq4_nl",
    "IQ4_NL_CODEBOOK",
    "ImatrixCalibrator",
    "Imatrix",
]

__version__ = "0.1.0"
