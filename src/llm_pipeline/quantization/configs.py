"""Configuration classes for quantization methods."""

from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
from enum import Enum


class QuantizationMethod(Enum):
    """Supported quantization methods"""
    BNB_4BIT = "bnb_4bit"
    BNB_8BIT = "bnb_8bit"
    AQLM_2BIT = "aqlm_2bit"
    LOFTQ = "loftq"


class QuantizationScheme(Enum):
    """Quantization schemes"""
    FP4 = "fp4"
    NF4 = "nf4"
    INT4 = "int4"
    INT8 = "int8"
    INT2 = "int2"


@dataclass
class QuantizationConfig:
    """Base configuration for quantization"""
    
    method: QuantizationMethod = QuantizationMethod.BNB_4BIT
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    device_map: Optional[str] = "auto"
    max_memory: Optional[Dict[str, str]] = None
    torch_dtype: Optional[str] = "float16"
    
    def __post_init__(self):
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot enable both 4-bit and 8-bit quantization")


@dataclass
class BnBConfig(QuantizationConfig):
    """BitsAndBytes quantization configuration"""
    
    method: QuantizationMethod = field(default=QuantizationMethod.BNB_4BIT, init=False)
    
    # 4-bit specific
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    # 8-bit specific  
    bnb_8bit_compute_dtype: str = "float16"
    bnb_8bit_use_double_quant: bool = False
    
    # Advanced options
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: Optional[List[str]] = None
    llm_int8_enable_fp32_cpu_offload: bool = False
    llm_int8_has_fp16_weight: bool = False
    
    # Memory calculation configuration
    bytes_per_4bit_param: float = 0.5  # 4-bit: 0.5 bytes per parameter
    bytes_per_8bit_param: float = 1.0  # 8-bit: 1 byte per parameter
    memory_unit_factor: int = 1024  # Memory unit conversion factor (1024 for KB/MB/GB)
    
    # Compression ratio configuration
    compression_ratio_4bit: float = 8.0  # 4-bit compression ratio (32 bits / 4 bits)
    compression_ratio_8bit: float = 4.0  # 8-bit compression ratio (32 bits / 8 bits)
    
    # Weight initialization configuration
    weight_init_method: str = "random"  # "random", "zeros", "ones", "xavier", "kaiming"
    weight_init_std: float = 0.02  # Standard deviation for random initialization
    
    # Logging configuration
    enable_logging: bool = True
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    suppress_import_warnings: bool = False  # Suppress import warnings
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.bnb_4bit_quant_type not in ["nf4", "fp4"]:
            raise ValueError(f"Invalid 4-bit quant type: {self.bnb_4bit_quant_type}")
        
        if self.bnb_4bit_compute_dtype not in ["float16", "bfloat16", "float32"]:
            raise ValueError(f"Invalid compute dtype: {self.bnb_4bit_compute_dtype}")
            
        if self.bytes_per_4bit_param <= 0 or self.bytes_per_8bit_param <= 0:
            raise ValueError("Bytes per parameter must be positive")
            
        if self.memory_unit_factor <= 0:
            raise ValueError(f"Memory unit factor must be positive, got {self.memory_unit_factor}")
            
        if self.compression_ratio_4bit <= 0 or self.compression_ratio_8bit <= 0:
            raise ValueError("Compression ratios must be positive")
            
        if self.weight_init_method not in ["random", "zeros", "ones", "xavier", "kaiming"]:
            raise ValueError(f"Invalid weight init method: {self.weight_init_method}")
            
        if self.weight_init_std <= 0:
            raise ValueError(f"Weight init std must be positive, got {self.weight_init_std}")
            
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")


@dataclass
class AQLMConfig(QuantizationConfig):
    """AQLM 2-bit quantization configuration"""
    
    method: QuantizationMethod = field(default=QuantizationMethod.AQLM_2BIT, init=False)
    load_in_4bit: bool = field(default=False, init=False)
    
    # AQLM specific
    num_codebooks: int = 1
    nbits_per_codebook: int = 16
    in_group_size: int = 8
    out_group_size: int = 1
    num_codebooks_per_group: int = 1
    
    # Advanced options
    use_checkpointing: bool = True
    optimize_sequential: bool = True
    
    # Bit width configuration
    baseline_bits: int = 32  # Baseline precision (fp32 = 32 bits)
    compressed_bits: float = 2.0  # AQLM compressed bits per parameter
    
    # Memory calculation configuration
    bytes_per_element: int = 4  # Bytes per element (fp32 = 4, fp16 = 2)
    memory_unit_factor: int = 1024  # Memory unit conversion factor (1024 for KB/MB/GB)
    
    # Model architecture configuration
    default_target_modules: Optional[List[str]] = None  # Default target modules
    
    # Logging configuration
    enable_logging: bool = True
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    suppress_import_warnings: bool = False  # Suppress import warnings
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.nbits_per_codebook not in [8, 16]:
            raise ValueError(f"nbits_per_codebook must be 8 or 16, got {self.nbits_per_codebook}")
        
        if self.in_group_size <= 0 or self.out_group_size <= 0:
            raise ValueError("Group sizes must be positive")
            
        if self.baseline_bits <= 0:
            raise ValueError(f"Baseline bits must be positive, got {self.baseline_bits}")
            
        if self.compressed_bits <= 0:
            raise ValueError(f"Compressed bits must be positive, got {self.compressed_bits}")
            
        if self.bytes_per_element <= 0:
            raise ValueError(f"Bytes per element must be positive, got {self.bytes_per_element}")
            
        if self.memory_unit_factor <= 0:
            raise ValueError(f"Memory unit factor must be positive, got {self.memory_unit_factor}")
            
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")
            
        # Set default target modules if not provided
        if self.default_target_modules is None:
            self.default_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass 
class LoftQConfig(QuantizationConfig):
    """LoftQ quantization-aware LoRA configuration"""
    
    method: QuantizationMethod = field(default=QuantizationMethod.LOFTQ, init=False)
    
    # LoftQ specific
    loftq_bits: int = 4
    loftq_iter: int = 1
    loftq_rank: int = 16
    loftq_alpha: float = 32.0
    
    # Quantization options
    quantization_scheme: QuantizationScheme = QuantizationScheme.NF4
    use_gradient_checkpointing: bool = True
    
    # Optimization options
    optimize_initialization: bool = True
    num_optimization_steps: int = 100
    learning_rate: float = 1e-3
    
    # Initialization configuration
    lora_init_std: float = 0.01  # Standard deviation for LoRA initialization
    lora_init_method: str = "normal"  # "normal", "uniform", "xavier", "kaiming"
    
    # Numerical stability
    epsilon: float = 1e-8  # Small constant for numerical stability
    
    # Quantization configuration
    use_custom_nf4_levels: bool = False  # Use custom NF4 levels
    custom_nf4_levels: Optional[List[float]] = None  # Custom quantization levels
    baseline_bits: int = 8  # Baseline bit width for compression ratio calculation
    
    # Logging configuration
    enable_logging: bool = True
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    log_frequency: int = 10  # Log every N optimization steps
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.loftq_bits not in [2, 4, 8]:
            raise ValueError(f"LoftQ bits must be 2, 4, or 8, got {self.loftq_bits}")
        
        if self.loftq_iter <= 0:
            raise ValueError(f"LoftQ iterations must be positive, got {self.loftq_iter}")
        
        if self.loftq_rank <= 0:
            raise ValueError(f"LoftQ rank must be positive, got {self.loftq_rank}")
            
        if self.lora_init_method not in ["normal", "uniform", "xavier", "kaiming"]:
            raise ValueError(f"Invalid lora_init_method: {self.lora_init_method}")
            
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")
            
        if self.log_frequency <= 0:
            raise ValueError(f"Log frequency must be positive, got {self.log_frequency}")
            
        if self.baseline_bits <= 0:
            raise ValueError(f"Baseline bits must be positive, got {self.baseline_bits}")
            
        if self.epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {self.epsilon}")
