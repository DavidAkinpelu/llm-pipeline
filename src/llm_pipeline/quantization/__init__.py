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

# Inference quantization (to be implemented)
# from .inference import (...)

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
    
    # Utilities
    "get_quantization_info",
    "estimate_quantized_memory", 
    "compare_quantization_methods",
    "quantization_quality_metrics"
]

__version__ = "0.1.0"
