"""Inference quantization configurations."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Literal
from enum import Enum


class QuantizationMethod(Enum):
    """Quantization methods for inference."""
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization-Aware Training


class PrecisionMode(Enum):
    """Precision modes for inference."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


class DeploymentFormat(Enum):
    """Deployment formats."""
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    GGUF = "gguf"
    TORCHSCRIPT = "torchscript"
    COREML = "coreml"


@dataclass
class PTQInferenceConfig:
    """Post-Training Quantization configuration for inference.
    
    Args:
        method: Quantization method - "dynamic", "static", or "qat"
        weight_bits: Bit precision for weights (typically 8 or 4)
        activation_bits: Bit precision for activations (typically 8)
        calibration_samples: Number of samples for calibration
        calibration_batch_size: Batch size for calibration
        weight_scheme: Weight quantization scheme - "symmetric" or "asymmetric"
        activation_scheme: Activation quantization scheme - "symmetric" or "asymmetric"
        quantize_embeddings: Whether to quantize embedding layers
        quantize_attention: Whether to quantize attention layers
        quantize_ffn: Whether to quantize feed-forward network layers
        quantize_layernorm: Whether to quantize layer normalization layers
        per_channel: Whether to use per-channel quantization
        per_tensor: Whether to use per-tensor quantization
        preserve_sparsity: Whether to preserve sparsity during quantization
        enable_logging: Whether to enable logging
        log_level: Log level - "DEBUG", "INFO", "WARNING", or "ERROR"
        log_frequency: Log every N steps
    """
    # Quantization method
    method: QuantizationMethod = QuantizationMethod.DYNAMIC
    
    # Bit precision
    weight_bits: int = 8
    activation_bits: int = 8
    
    # Calibration settings
    calibration_samples: int = 100
    calibration_batch_size: int = 1
    
    # Quantization scheme
    weight_scheme: Literal["symmetric", "asymmetric"] = "symmetric"
    activation_scheme: Literal["symmetric", "asymmetric"] = "symmetric"
    
    # Layer selection
    quantize_embeddings: bool = True
    quantize_attention: bool = True
    quantize_ffn: bool = True
    quantize_layernorm: bool = False
    
    # Advanced settings
    per_channel: bool = True
    per_tensor: bool = False
    preserve_sparsity: bool = False
    
    # Logging
    enable_logging: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_frequency: int = 10


@dataclass
class ONNXInferenceConfig:
    """ONNX inference configuration.
    
    Args:
        export_path: Path to save the exported ONNX model
        opset_version: ONNX opset version (typically 17)
        quantization_type: Quantization type - "dynamic", "static", or "qat"
        weight_type: Weight data type - "QInt8" or "QUInt8"
        activation_type: Activation data type - "QInt8" or "QUInt8"
        quantization_format: Quantization format - "QOperator" or "QDQ"
    """
    # Export settings
    export_path: str = "model.onnx"
    opset_version: int = 17
    
    # Quantization settings
    quantization_type: Literal["dynamic", "static", "qat"] = "dynamic"
    weight_type: Literal["QInt8", "QUInt8"] = "QInt8"
    activation_type: Literal["QInt8", "QUInt8"] = "QUInt8"
    quantization_format: Literal["QOperator", "QDQ"] = "QOperator"
    
    # Static quantization settings
    calibration_samples: int = 100
    calibration_batch_size: int = 1
    
    # Optimization settings
    enable_optimization: bool = True
    optimization_level: Literal["disable", "basic", "extended", "all"] = "all"
    
    # Input/Output settings
    input_names: List[str] = field(default_factory=lambda: ["input_ids", "attention_mask"])
    output_names: List[str] = field(default_factory=lambda: ["logits"])
    dynamic_axes: Dict[str, Dict[int, str]] = field(default_factory=dict)
    
    # Logging
    enable_logging: bool = True
    log_level: str = "INFO"
    suppress_warnings: bool = True


@dataclass
class TensorRTInferenceConfig:
    """TensorRT inference configuration."""
    # Model settings
    model_path: str = "model.trt"
    max_batch_size: int = 1
    max_sequence_length: int = 2048
    max_output_length: int = 512
    
    # Precision settings
    precision_mode: PrecisionMode = PrecisionMode.FP16
    int8_calibration_dataset: Optional[str] = None
    
    # Optimization settings
    optimization_level: int = 5  # 0-5, higher = more optimization
    workspace_size_mb: int = 4096
    max_workspace_size: int = 1 << 30  # 1GB
    
    # Engine settings
    engine_cache_path: str = "engine_cache"
    use_explicit_batch: bool = True
    enable_profiling: bool = False
    
    # Quantization settings
    enable_int8_calibration: bool = False
    calibration_batch_size: int = 1
    calibration_samples: int = 100
    
    # Logging
    enable_logging: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    verbose_logging: bool = False


@dataclass
class GGUFInferenceConfig:
    """GGUF inference configuration."""
    # Model metadata
    model_name: str = "llm_pipeline_model"
    model_type: Literal["llama", "mistral", "qwen", "gpt", "bert", "t5", "other"] = "llama"
    model_version: str = "1.0"
    
    # Quantization settings
    quantization_type: Literal["Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q6_K", "Q8_K"] = "Q8_0"
    block_size: int = 32
    
    # File settings
    use_mmap: bool = True
    tensor_alignment: int = 32
    
    # Metadata
    author: str = "LLM Pipeline"
    description: str = "Quantized model for inference"
    license: str = "MIT"
    
    # Logging
    enable_logging: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@dataclass
class DeploymentInferenceConfig:
    """Deployment inference configuration."""
    # Output settings
    output_dir: str = "deployment"
    model_name: str = "deployed_model"
    
    # Format selection
    export_formats: List[DeploymentFormat] = field(default_factory=lambda: [DeploymentFormat.ONNX, DeploymentFormat.TORCHSCRIPT])
    quantization_enabled: bool = True
    
    # Optimization settings
    optimize_for_mobile: bool = False
    optimize_for_server: bool = True
    enable_fusion: bool = True
    
    # Quantization settings
    quantization_bits: int = 8
    calibration_samples: int = 100
    
    # Metadata
    model_version: str = "1.0"
    model_description: str = "Deployed LLM model"
    model_author: str = "LLM Pipeline"
    
    # Logging
    enable_logging: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@dataclass
class InferenceQuantizationConfig:
    """Unified inference quantization configuration."""
    # Method selection
    primary_method: QuantizationMethod = QuantizationMethod.DYNAMIC
    fallback_methods: List[QuantizationMethod] = field(default_factory=lambda: [QuantizationMethod.STATIC])
    
    # Target deployment
    target_deployment: List[DeploymentFormat] = field(default_factory=lambda: [DeploymentFormat.ONNX])
    target_precision: PrecisionMode = PrecisionMode.FP16
    
    # Performance requirements
    max_latency_ms: Optional[float] = None
    min_throughput_fps: Optional[float] = None
    max_memory_mb: Optional[int] = None
    
    # Quality requirements
    min_quality_threshold: float = 0.95  # 95% of original model quality
    quality_metric: Literal["perplexity", "accuracy", "bleu"] = "perplexity"
    
    # Calibration settings
    calibration_data_path: Optional[str] = None
    calibration_samples: int = 100
    calibration_batch_size: int = 1
    
    # Hardware-specific settings
    target_device: Literal["cpu", "gpu", "mobile", "auto"] = "auto"
    gpu_memory_limit_gb: Optional[float] = None
    
    # Advanced settings
    enable_mixed_precision: bool = True
    enable_fusion: bool = True
    enable_pruning: bool = False
    
    # Logging and debugging
    enable_logging: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    save_intermediate_models: bool = False
    benchmark_models: bool = True


def create_ptq_config(
    method: str = "dynamic",
    weight_bits: int = 8,
    activation_bits: int = 8,
    calibration_samples: int = 100,
    **kwargs
) -> PTQInferenceConfig:
    """Create PTQ configuration with common settings.
    
    Args:
        method: Quantization method
        weight_bits: Weight quantization bits
        activation_bits: Activation quantization bits
        calibration_samples: Number of calibration samples
        **kwargs: Additional configuration parameters
        
    Returns:
        PTQ configuration
    """
    return PTQInferenceConfig(
        method=QuantizationMethod(method),
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        calibration_samples=calibration_samples,
        **kwargs
    )


def create_onnx_config(
    export_path: str = "model.onnx",
    quantization_type: str = "dynamic",
    optimization_level: str = "all",
    **kwargs
) -> ONNXInferenceConfig:
    """Create ONNX configuration with common settings.
    
    Args:
        export_path: Output file path
        quantization_type: Quantization type
        optimization_level: Optimization level
        **kwargs: Additional configuration parameters
        
    Returns:
        ONNX configuration
    """
    return ONNXInferenceConfig(
        export_path=export_path,
        quantization_type=quantization_type,
        optimization_level=optimization_level,
        **kwargs
    )


def create_tensorrt_config(
    model_path: str = "model.trt",
    precision_mode: str = "fp16",
    optimization_level: int = 5,
    **kwargs
) -> TensorRTInferenceConfig:
    """Create TensorRT configuration with common settings.
    
    Args:
        model_path: Output file path
        precision_mode: Precision mode
        optimization_level: Optimization level
        **kwargs: Additional configuration parameters
        
    Returns:
        TensorRT configuration
    """
    return TensorRTInferenceConfig(
        model_path=model_path,
        precision_mode=PrecisionMode(precision_mode),
        optimization_level=optimization_level,
        **kwargs
    )


def create_gguf_config(
    model_name: str = "llm_pipeline_model",
    quantization_type: str = "Q8_0",
    **kwargs
) -> GGUFInferenceConfig:
    """Create GGUF configuration with common settings.
    
    Args:
        model_name: Model name
        quantization_type: Quantization type
        **kwargs: Additional configuration parameters
        
    Returns:
        GGUF configuration
    """
    return GGUFInferenceConfig(
        model_name=model_name,
        quantization_type=quantization_type,
        **kwargs
    )


def create_deployment_config(
    output_dir: str = "deployment",
    export_formats: Optional[List[str]] = None,
    quantization_enabled: bool = True,
    **kwargs
) -> DeploymentInferenceConfig:
    """Create deployment configuration with common settings.
    
    Args:
        output_dir: Output directory
        export_formats: List of export formats
        quantization_enabled: Enable quantization
        **kwargs: Additional configuration parameters
        
    Returns:
        Deployment configuration
    """
    formats = [DeploymentFormat(fmt) for fmt in (export_formats or ["onnx", "torchscript"])]
    
    return DeploymentInferenceConfig(
        output_dir=output_dir,
        export_formats=formats,
        quantization_enabled=quantization_enabled,
        **kwargs
    )


def create_inference_config(
    primary_method: str = "dynamic",
    target_deployment: Optional[List[str]] = None,
    target_precision: str = "fp16",
    **kwargs
) -> InferenceQuantizationConfig:
    """Create unified inference quantization configuration.
    
    Args:
        primary_method: Primary quantization method
        target_deployment: Target deployment formats
        target_precision: Target precision
        **kwargs: Additional configuration parameters
        
    Returns:
        Unified inference configuration
    """
    deployments = [DeploymentFormat(fmt) for fmt in (target_deployment or ["onnx"])]
    
    return InferenceQuantizationConfig(
        primary_method=QuantizationMethod(primary_method),
        target_deployment=deployments,
        target_precision=PrecisionMode(target_precision),
        **kwargs
    )


# Preset configurations for common use cases
PRESET_CONFIGS = {
    "mobile_optimized": InferenceQuantizationConfig(
        primary_method=QuantizationMethod.DYNAMIC,
        target_deployment=[DeploymentFormat.ONNX, DeploymentFormat.TORCHSCRIPT],
        target_precision=PrecisionMode.INT8,
        optimize_for_mobile=True,
        max_memory_mb=512,
        min_quality_threshold=0.90
    ),
    
    "server_optimized": InferenceQuantizationConfig(
        primary_method=QuantizationMethod.STATIC,
        target_deployment=[DeploymentFormat.TENSORRT, DeploymentFormat.ONNX],
        target_precision=PrecisionMode.FP16,
        optimize_for_server=True,
        max_latency_ms=10.0,
        min_throughput_fps=100.0,
        min_quality_threshold=0.98
    ),
    
    "edge_optimized": InferenceQuantizationConfig(
        primary_method=QuantizationMethod.DYNAMIC,
        target_deployment=[DeploymentFormat.GGUF, DeploymentFormat.ONNX],
        target_precision=PrecisionMode.INT8,
        max_memory_mb=1024,
        min_quality_threshold=0.95,
        enable_pruning=True
    ),
    
    "research_optimized": InferenceQuantizationConfig(
        primary_method=QuantizationMethod.QAT,
        target_deployment=[DeploymentFormat.ONNX, DeploymentFormat.TENSORRT],
        target_precision=PrecisionMode.FP16,
        min_quality_threshold=0.99,
        save_intermediate_models=True,
        benchmark_models=True
    )
}


def get_preset_config(preset_name: str) -> InferenceQuantizationConfig:
    """Get preset configuration by name.
    
    Args:
        preset_name: Name of the preset configuration
        
    Returns:
        Preset configuration
        
    Raises:
        ValueError: If preset name is not found
    """
    if preset_name not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {available}")
    
    return PRESET_CONFIGS[preset_name]
