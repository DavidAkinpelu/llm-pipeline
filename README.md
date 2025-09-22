# LLM Pipeline Framework

A comprehensive PyTorch implementation of modern LLM fine-tuning, quantization, and inference techniques including LoRA, DoRA, RSLoRA, and advanced quantization methods. This framework provides maximum control and educational value by implementing everything from scratch with configurable, production-ready components.

## 🚀 Features

- **Large-Scale Training**: Distributed training across multiple GPUs and nodes for billion-parameter models
- **Pure PyTorch Implementation**: No external fine-tuning libraries required
- **Multiple LoRA Variants**: LoRA, DoRA, RSLoRA implementations with configurable parameters
- **Advanced Quantization**: 4-bit (BitsAndBytes), 2-bit (AQLM), and LoftQ quantization support
- **Universal Model Support**: Works with any HuggingFace model architecture
- **Multi-Adapter Management**: Support for multiple adapters with advanced routing capabilities
- **Memory Optimization**: Comprehensive memory management with gradient checkpointing and mixed precision
- **Inference Engine**: Optimized inference pipeline with KV caching and streaming
- **Model Merging**: Advanced merging strategies (TIES, DARE, Linear combinations)
- **Cloud Integration**: Native support for AWS, GCP, and Azure training platforms
- **Configurable Components**: All hardcoded values removed for maximum flexibility
- **Educational Focus**: Built from first principles for deep understanding

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/DavidAkinpelu/llm-pipeline.git
cd llm-pipeline

# Create and activate conda environment
conda create -n llm-pipeline python=3.12
conda activate llm-pipeline

# Install dependencies
pip install torch transformers bitsandbytes aqlm

# Install in development mode
pip install -e .

# Install testing dependencies
pip install pytest pytest-cov
```

## 🏗️ Project Status

This project has completed **Phase 1-3** development with comprehensive core architecture, advanced adapter implementations, and inference engine. **Phase 4** focuses on large-scale training capabilities for production use cases.

### ✅ Implemented Components

#### Core Framework
- **LoRA Configuration System**: Comprehensive configuration classes for LoRA, DoRA, and RSLoRA
- **Model Registry**: Support for 15+ model architectures with automatic detection
- **Model Wrapper**: Automatic LoRA injection into HuggingFace models with memory optimization
- **Base LoRA Module**: Pure PyTorch implementation with configurable initialization
- **Rank=0 Support**: Special handling for pure inference scenarios

#### Adapter Implementations
- **LoRA**: Standard Low-Rank Adaptation with configurable rank and scaling
- **DoRA**: Weight-Decomposed LoRA for better magnitude-direction separation
- **RSLoRA**: Rank-Stabilized LoRA with improved scaling (no rank stabilizer)
- **Multi-Adapter Management**: Advanced routing with configurable attention mechanisms
- **Adapter Merging**: Advanced merging strategies (TIES, DARE_LINEAR, DARE_TIES, WEIGHTED_SUM)

#### Quantization Framework
- **BitsAndBytes Integration**: Fully configurable 4-bit/8-bit quantization with NF4/FP4 support
- **AQLM Integration**: Configurable 2-bit quantization for extreme compression
- **LoftQ Implementation**: Quantization-aware LoRA initialization with configurable parameters
- **Quantization Utilities**: Memory estimation, quality metrics, and comparison tools

#### Inference Engine (Phase 3)
- **Core Inference Engine**: Optimized text generation with KV caching
- **Sampling Strategies**: Greedy, temperature, top-k, top-p, typical-p, Mirostat
- **Streaming Generation**: Real-time token output with configurable constraints
- **Batch Processing**: Efficient handling of multiple inference requests
- **Memory Management**: Dynamic memory estimation and optimization
- **Context Management**: Long sequence handling with configurable windows

#### Model Merging
- **Advanced Strategies**: TIES, DARE_LINEAR, DARE_TIES implementations
- **QLoRA Merging**: Quantized and full-precision merging strategies
- **Merge Caching**: Intelligent caching for merged models
- **Strategy Selection**: Memory-aware automatic strategy selection

#### Utilities
- **Memory Analysis**: Comprehensive memory footprint analysis with optimizer awareness
- **Adapter Benchmarking**: Performance comparison across different adapter types
- **Validation Framework**: Configuration and model compatibility validation
- **Factory Patterns**: Easy model creation and configuration
- **Logging System**: Configurable logging across all components

### 🚧 In Development

#### Large-Scale Training Framework (Phase 4)
- Distributed training across multiple GPUs and nodes
- Multi-GPU training with data parallelism and model parallelism
- Large-scale fine-tuning for billion-parameter models
- Training mode framework (full fine-tuning, LoRA, QLoRA, hybrid modes)
- Advanced optimization strategies and learning rate schedulers
- Memory-efficient training with gradient checkpointing
- Checkpointing and resuming for long-running training jobs
- Support for massive datasets with efficient data loading
- Integration with cloud training platforms (AWS, GCP, Azure)

#### Production Infrastructure (Phase 5)
- API server implementation
- Model serving and deployment tools
- Production monitoring and metrics
- Scalability and load balancing

#### Advanced Features (Phase 6)
- Model compression and pruning
- Advanced quantization techniques
- Multi-modal support
- Performance optimization

## 🎯 Quick Start

### Basic LoRA Training

```python
from llm_pipeline.core import LoRAModelWrapper, LoRAConfig
from transformers import AutoModel

# Load a model
model = AutoModel.from_pretrained("microsoft/DialoGPT-medium")

# Create LoRA configuration
config = LoRAConfig(
    r=16,
    alpha=32.0,
    dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# Wrap model with LoRA
lora_model = LoRAModelWrapper(model, config)

# Print parameter summary
lora_model.print_trainable_parameters()
```

### Advanced Adapter Variants

```python
from llm_pipeline.core import DoRAConfig, RSLoRAConfig
from llm_pipeline.adapters import DoRAModule, RSLoRAModule

# DoRA (Weight-Decomposed LoRA)
dora_config = DoRAConfig(r=16, alpha=32.0, magnitude_init="ones")
dora_model = LoRAModelWrapper(model, dora_config)

# RSLoRA (Rank-Stabilized LoRA)
rslora_config = RSLoRAConfig(r=16, alpha=32.0)
rslora_model = LoRAModelWrapper(model, rslora_config)
```

### Quantized Training

```python
from llm_pipeline.quantization import BnBConfig, AQLMConfig, create_bnb_model, create_aqlm_model

# 4-bit quantization with BitsAndBytes (fully configurable)
bnb_config = BnBConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    enable_logging=True
)
quantized_model = create_bnb_model(model, bnb_config)

# Wrap with LoRA
lora_quantized = LoRAModelWrapper(quantized_model, config)

# 2-bit quantization with AQLM (fully configurable)
aqlm_config = AQLMConfig(
    compressed_bits=2.0,
    baseline_bits=32,
    enable_logging=True
)
aqlm_model = create_aqlm_model(model, aqlm_config)
```

### Multi-Adapter Support

```python
from llm_pipeline.adapters import MultiAdapterLoRALinear

# Add multiple adapters
lora_model.add_adapter("task1", LoRAConfig(r=8, alpha=16))
lora_model.add_adapter("task2", LoRAConfig(r=16, alpha=32))

# Set active adapters
lora_model.set_active_adapters(["task1", "task2"])
```

### Inference Engine

```python
from llm_pipeline.inference import InferenceEngine, SamplingConfig

# Create inference engine with configurable parameters
engine = InferenceEngine(
    model=lora_model,
    tokenizer=tokenizer,
    max_length=512,
    device="cuda"
)

# Generate text with advanced sampling
sampling_config = SamplingConfig(
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1
)

output = engine.generate(
    prompt="Hello, how are you?",
    sampling_config=sampling_config,
    max_new_tokens=100
)
```

### Model Merging

```python
from llm_pipeline.adapters.merging import LoRAAdapterMerger, MergeConfig, MergeStrategy

# Create merger with advanced strategies
merge_config = MergeConfig(
    strategy=MergeStrategy.DARE_TIES,
    density=0.5,
    dare_rescaling_factor=1.0
)

merger = LoRAAdapterMerger(merge_config)

# Merge multiple adapters
merged_adapter = merger.merge_adapters(
    adapters=[adapter1, adapter2, adapter3],
    adapter_names=["task1", "task2", "task3"]
)
```

## 🚀 Large-Scale Training Capabilities

The framework is designed for large-scale training scenarios, supporting everything from single GPU fine-tuning to multi-node distributed training of billion-parameter models.

### Training Modes

#### Full Fine-Tuning
```python
from llm_pipeline.training import FullFineTuningConfig

# Full fine-tuning configuration for large models
config = FullFineTuningConfig(
    learning_rate=1e-5,
    batch_size=4,
    gradient_accumulation_steps=16,
    max_steps=10000,
    warmup_steps=500,
    weight_decay=0.01,
    gradient_checkpointing=True,
    mixed_precision=True
)
```

#### LoRA Fine-Tuning
```python
from llm_pipeline.training import LoRATrainingConfig

# LoRA training for parameter-efficient fine-tuning
config = LoRATrainingConfig(
    lora_config=LoRAConfig(r=16, alpha=32.0),
    learning_rate=2e-4,
    batch_size=8,
    gradient_accumulation_steps=8,
    max_steps=5000
)
```

#### QLoRA Training
```python
from llm_pipeline.training import QLoRATrainingConfig

# QLoRA training with quantization
config = QLoRATrainingConfig(
    lora_config=LoRAConfig(r=16, alpha=32.0),
    quantization_config=BnBConfig(load_in_4bit=True),
    learning_rate=2e-4,
    batch_size=16,
    gradient_accumulation_steps=4
)
```

### Distributed Training

#### Multi-GPU Training
```python
from llm_pipeline.training import DistributedTrainer

# Multi-GPU training with data parallelism
trainer = DistributedTrainer(
    model=model,
    config=training_config,
    strategy="ddp",  # Data Distributed Parallel
    num_gpus=4
)

# Start training
trainer.train()
```

#### Multi-Node Training
```python
# Multi-node training across multiple machines
trainer = DistributedTrainer(
    model=model,
    config=training_config,
    strategy="ddp_multi_node",
    nodes=2,
    gpus_per_node=8
)
```

### Memory Optimization

#### Gradient Checkpointing
```python
# Enable gradient checkpointing for memory efficiency
config = TrainingConfig(
    gradient_checkpointing=True,
    gradient_checkpointing_granularity="layer"  # or "block"
)
```

#### Mixed Precision Training
```python
# Automatic mixed precision for faster training
config = TrainingConfig(
    mixed_precision=True,
    fp16_opt_level="O1"  # or "O2", "O3"
)
```

#### DeepSpeed Integration
```python
# DeepSpeed integration for massive models
config = TrainingConfig(
    deepspeed_config="deepspeed_config.json",
    zero_stage=2,  # ZeRO optimizer state partitioning
    offload_optimizer=True,
    offload_param=True
)
```

### Cloud Training Support

#### AWS SageMaker
```python
from llm_pipeline.cloud.aws import SageMakerTrainer

trainer = SageMakerTrainer(
    model=model,
    config=training_config,
    instance_type="ml.p4d.24xlarge",
    instance_count=4
)
```

#### Google Cloud AI Platform
```python
from llm_pipeline.cloud.gcp import GCPTrainer

trainer = GCPTrainer(
    model=model,
    config=training_config,
    machine_type="n1-standard-96",
    accelerator_type="nvidia-tesla-v100",
    accelerator_count=8
)
```

#### Azure ML
```python
from llm_pipeline.cloud.azure import AzureMLTrainer

trainer = AzureMLTrainer(
    model=model,
    config=training_config,
    compute_target="gpu-cluster",
    node_count=4
)
```

### Large-Scale Data Handling

#### Efficient Data Loading
```python
from llm_pipeline.data import LargeScaleDataLoader

# Efficient data loading for massive datasets
data_loader = LargeScaleDataLoader(
    dataset_path="s3://my-bucket/large-dataset",
    batch_size=32,
    num_workers=16,
    prefetch_factor=4,
    persistent_workers=True
)
```

#### Streaming Data Processing
```python
# Stream processing for datasets that don't fit in memory
data_loader = StreamingDataLoader(
    data_source="https://api.example.com/data-stream",
    batch_size=64,
    buffer_size=10000,
    compression="gzip"
)
```

### Monitoring and Logging

#### Training Monitoring
```python
from llm_pipeline.monitoring import TrainingMonitor

# Comprehensive training monitoring
monitor = TrainingMonitor(
    log_to_wandb=True,
    log_to_tensorboard=True,
    metrics_interval=100,
    checkpoint_interval=1000
)
```

#### Resource Monitoring
```python
# GPU and system resource monitoring
monitor = ResourceMonitor(
    track_gpu_usage=True,
    track_memory=True,
    track_network=True,
    alert_thresholds={"gpu_util": 0.9, "memory_util": 0.8}
)
```

## 🏛️ Architecture

The framework is organized into several modules:

```
src/llm_pipeline/
├── core/                    # Core LoRA implementation and model wrappers
│   ├── config.py           # Configuration classes (LoRA, DoRA, RSLoRA)
│   ├── base_module.py      # Base LoRA module implementation
│   ├── model_wrapper.py    # HuggingFace model wrapper with memory optimization
│   └── registry.py         # Model/adapter registry with 15+ architectures
├── adapters/               # LoRA variants and adapter management
│   ├── lora.py            # Standard LoRA implementation
│   ├── dora.py            # DoRA implementation with magnitude scaling
│   ├── rslora.py          # RSLoRA implementation (no rank stabilizer)
│   ├── adapter_manager.py # Multi-adapter management with routing
│   └── merging.py         # Advanced merging strategies (TIES, DARE)
├── quantization/          # Quantization framework
│   ├── configs.py         # Quantization configuration classes
│   ├── training/          # Training-time quantization
│   │   ├── bnb_integration.py # Fully configurable BitsAndBytes integration
│   │   ├── aqlm_integration.py# Fully configurable AQLM integration
│   │   └── loftq.py       # Configurable LoftQ implementation
│   └── inference/         # Inference-time quantization
├── inference/             # Inference engine (Phase 3 Complete)
│   ├── core/              # Core inference engine with KV caching
│   ├── generation/        # Sampling strategies and constraints
│   ├── merging/           # QLoRA merging strategies
│   └── memory/            # Memory management and optimization
├── training/              # Training framework (Phase 4)
├── merging/               # Model merging strategies (Phase 4)
├── production/            # Production infrastructure (Phase 5)
├── testing/               # Comprehensive test suite
│   ├── unit/              # Unit tests (136 tests)
│   ├── integration/       # Integration tests (13 tests)
│   └── performance/       # Performance benchmarks
└── utils/                 # Utilities and analysis tools
    ├── analysis.py        # Adapter benchmarking and comparison
    ├── memory.py          # Memory optimization with optimizer awareness
    ├── optimizer_memory.py# Dynamic optimizer memory calculation
    ├── validation.py      # Configuration validation
    └── factory.py         # Factory patterns for model creation
```

## 🎓 Educational Value

This framework is designed for educational purposes, providing:

- **Deep Understanding**: Implementation from first principles
- **No Black Boxes**: Every component is transparent and customizable
- **Comprehensive Documentation**: Detailed code comments and examples
- **Performance Insights**: Built-in benchmarking and analysis tools
- **Memory Awareness**: Tools to understand memory usage patterns

## 📊 Supported Models

The framework supports 15+ model architectures through the model registry:

- **Decoder-Only Models**: Llama, Mistral, Qwen, Gemma, Phi, GPT-2, GPT-Neo
- **Encoder-Only Models**: BERT, RoBERTa, DistilBERT, ALBERT
- **Encoder-Decoder Models**: T5, BART, FLAN-T5
- **Custom Models**: Automatic detection and fallback configurations

## 🔧 Configuration & Assumptions

### LoRA Configuration

```python
@dataclass
class LoRAConfig:
    r: int = 8                    # Rank of adaptation
    alpha: float = 16.0           # Scaling parameter
    dropout: float = 0.1          # Dropout rate
    target_modules: List[str]     # Modules to adapt
    bias: str = "none"            # Bias handling
    task_type: str = "CAUSAL_LM"  # Task type
    init_lora_weights: str = "gaussian"  # Initialization method
```

**Default Target Modules**: `["q_proj", "v_proj", "k_proj", "o_proj"]`

**Valid Bias Options**: `["none", "all", "lora_only"]`

**Valid Initialization Methods**: `["gaussian", "kaiming", "xavier"]`

**Valid Task Types**: `["CAUSAL_LM", "MASKED_LM", "SEQUENCE_CLASSIFICATION", "TOKEN_CLASSIFICATION", "QUESTION_ANSWERING", "SEQ2SEQ_LM"]`

### Quantization Configuration

#### BitsAndBytes (4-bit/8-bit)
```python
@dataclass
class BnBConfig:
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_compute_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    bnb_8bit_compute_dtype: str = "float16"
    llm_int8_threshold: float = 6.0
    # Configurable memory calculations
    bytes_per_4bit_param: float = 0.5
    bytes_per_8bit_param: float = 1.0
    memory_unit_factor: int = 1024
    # Configurable compression ratios
    compression_ratio_4bit: float = 8.0
    compression_ratio_8bit: float = 4.0
    # Configurable weight initialization
    weight_init_method: str = "random"  # "random", "zeros", "ones", "xavier", "kaiming"
    weight_init_std: float = 0.02
    # Configurable logging
    enable_logging: bool = True
    log_level: str = "INFO"
    suppress_import_warnings: bool = False
```

**Valid 4-bit Quantization Types**: `["nf4", "fp4"]`

**Valid Compute Dtypes**: `["float16", "bfloat16", "float32"]`

#### AQLM (2-bit)
```python
@dataclass
class AQLMConfig:
    num_codebooks: int = 1
    nbits_per_codebook: int = 16  # Must be 8 or 16
    in_group_size: int = 8
    out_group_size: int = 1
    num_codebooks_per_group: int = 1
    # Configurable bit width calculations
    baseline_bits: int = 32  # Baseline precision (fp32 = 32 bits)
    compressed_bits: float = 2.0  # AQLM compressed bits per parameter
    # Configurable memory calculations
    bytes_per_element: int = 4  # Bytes per element (fp32 = 4, fp16 = 2)
    memory_unit_factor: int = 1024  # Memory unit conversion factor
    # Configurable target modules
    default_target_modules: Optional[List[str]] = None
    # Configurable logging
    enable_logging: bool = True
    log_level: str = "INFO"
    suppress_import_warnings: bool = False
```

**Valid nbits_per_codebook**: `[8, 16]`

#### LoftQ (Quantization-aware LoRA)
```python
@dataclass
class LoftQConfig:
    loftq_bits: int = 4           # Must be 2, 4, or 8
    loftq_iter: int = 1           # Must be positive
    loftq_rank: int = 16          # Must be positive
    loftq_alpha: float = 32.0
    quantization_scheme: QuantizationScheme = QuantizationScheme.NF4
    num_optimization_steps: int = 100
    learning_rate: float = 1e-3
    # Configurable LoRA initialization
    lora_init_std: float = 0.01   # LoRA initialization scaling
    lora_init_method: str = "normal"  # "normal", "uniform", "kaiming", "xavier"
    # Configurable quantization parameters
    epsilon: float = 1e-8         # Numerical stability constant
    use_custom_nf4_levels: bool = False
    custom_nf4_levels: Optional[List[float]] = None
    # Configurable baseline for compression ratio
    baseline_bits: int = 8        # Baseline precision for compression calculation
    # Configurable logging
    enable_logging: bool = True
    log_level: str = "INFO"
    log_frequency: int = 10       # Logging frequency during optimization
```

**Valid LoftQ Bits**: `[2, 4, 8]`

**Valid Quantization Schemes**: `["nf4", "fp4", "int4", "int8", "int2"]`

### Configurable Features

#### Memory Calculation Parameters
- **Bytes per Parameter**: Configurable for different quantization schemes
- **Memory Unit Factor**: Configurable conversion factors (1024 for KB/MB/GB)
- **Compression Ratios**: Configurable compression ratios for different bit widths

#### Weight Initialization Methods
- **LoRA Initialization**: `["normal", "uniform", "kaiming", "xavier"]`
- **DoRA Magnitude**: `["ones", "random", "kaiming"]`
- **Custom Scaling**: Configurable initialization standard deviations

#### Logging and Monitoring
- **Log Levels**: `["DEBUG", "INFO", "WARNING", "ERROR"]`
- **Import Warnings**: Configurable suppression of external library warnings
- **Logging Frequency**: Configurable logging intervals during training/optimization

#### Advanced Configuration
- **Custom NF4 Levels**: User-defined quantization levels for specialized use cases
- **Dynamic Target Modules**: Configurable target modules for different architectures
- **Numerical Stability**: Configurable epsilon values for numerical stability

### NF4 Quantization Levels

The framework uses the standard NF4 (Normal Float 4) quantization levels optimized for normal distributions:

```python
nf4_levels = [
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1.0
]
```

### Model Registry Assumptions

#### Supported Model Architectures

**Decoder-Only Models** (15+ architectures):
- **Llama/Alpaca**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **Mistral/Mixtral**: Same as Llama
- **Qwen**: Same as Llama  
- **Gemma**: Same as Llama
- **Phi**: `["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]`
- **GPT-2**: `["c_attn", "c_proj", "c_fc"]`
- **GPT-Neo**: `["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]`

**Encoder-Only Models**:
- **BERT**: `["query", "key", "value", "dense"]`
- **RoBERTa**: `["query", "key", "value", "dense"]`
- **DistilBERT**: `["q_lin", "k_lin", "v_lin", "out_lin", "fc1", "fc2"]`

**Encoder-Decoder Models**:
- **T5**: `["q", "k", "v", "o", "wi", "wo"]`
- **BART**: `["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]`

#### Default Fallback Configuration
For unknown models:
```python
{
    "target_modules": ["q_proj", "v_proj"],
    "modules_to_save": [],
    "pattern": model_type,
    "architecture_type": "decoder_only"
}
```

### Memory Assumptions

#### Parameter Memory Calculation
- **Full Precision**: 4 bytes per parameter (float32)
- **4-bit Quantization**: 0.5 bytes per parameter
- **8-bit Quantization**: 1 byte per parameter
- **2-bit AQLM**: ~2 bits per parameter (0.25 bytes)

#### Gradient Memory Estimation
- **Gradients**: 4 bytes per trainable parameter
- **Adam Optimizer**: ~8 bytes per trainable parameter
- **Safety Margin**: 0.8 (20% memory buffer)

### Multi-Adapter Configuration

```python
@dataclass
class MultiAdapterConfig:
    max_adapters: int = 8
    enable_routing: bool = False
    adapter_fusion_method: str = "weighted_sum"
```

**Valid Fusion Methods**: `["weighted_sum", "learned_routing", "attention"]`

### Validation Constraints

#### Rank Validation
- **Minimum**: Must be positive
- **Maximum Warning**: > 256 (unusually high)
- **Recommended Range**: 4-64 for most use cases

#### Dropout Validation
- **Range**: [0, 1]
- **Default**: 0.1

#### Alpha Validation
- **Minimum**: Must be positive
- **Typical Range**: 1-64

### Initialization Assumptions

#### LoRA Weight Initialization
- **Gaussian**: `mean=0.0, std=0.02`
- **Kaiming**: `a=sqrt(5)`
- **Xavier**: Uniform distribution

#### DoRA Magnitude Initialization
- **Ones**: Initialize to 1.0
- **Random**: `mean=1.0, std=0.02`
- **Kaiming**: Kaiming uniform

### Hardware Assumptions

#### GPU Memory Requirements
- **Minimum**: 4GB VRAM for 7B models with 4-bit quantization
- **Recommended**: 8GB+ VRAM for optimal performance
- **Multi-GPU**: Automatic device mapping with "auto" strategy

#### CPU Requirements
- **Minimum**: 8GB RAM
- **Recommended**: 16GB+ RAM for large models

## 🚀 Performance

### Memory Efficiency
- **LoRA Training**: 3-10x less memory than full fine-tuning
- **QLoRA Training**: 50-90% memory reduction with quantization
- **Gradient Checkpointing**: 50% additional memory savings
- **Mixed Precision**: 30-50% memory reduction with FP16/BF16

### Training Speed
- **LoRA Training**: 2-3x faster than full fine-tuning
- **Distributed Training**: Linear scaling across multiple GPUs
- **Mixed Precision**: 1.5-2x speedup with automatic mixed precision
- **Data Parallel**: Near-linear scaling with data parallelism

### Scalability
- **Multi-GPU**: Support for 8+ GPUs with data parallelism
- **Multi-Node**: Distributed training across multiple machines
- **Cloud Training**: Native integration with major cloud platforms
- **Large Models**: Tested on models up to 70B parameters

### Quality Preservation
- **LoRA**: Minimal quality degradation (<1% PPL increase)
- **Quantization**: <2% quality loss with proper configuration
- **Merging**: High-quality adapter merging with advanced strategies

## 🤝 Contributing

Contributions are welcome! This project is in active development and we're looking for:

- **Large-Scale Training**: Implement distributed training capabilities and cloud integration
- **Production Infrastructure**: Build API servers and deployment tools
- **Advanced Features**: Multi-modal support, compression, and optimization
- **Documentation**: Improve examples and tutorials
- **Testing**: Add comprehensive test coverage for new features

Please see our contributing guidelines for details.

## 📚 Development Roadmap

See [unified_development_plan.md](unified_development_plan.md) for the complete development roadmap covering comprehensive testing and production deployment.

### Current Focus Areas:
1. **Repository Reorganization** (Phase 1): Testing infrastructure and validation ✅
2. **Inference Quantization** (Phase 2): Post-training quantization and GGUF support ✅
3. **Core Inference Engine** (Phase 3): Optimized inference pipeline ✅
4. **Large-Scale Training** (Phase 4): Distributed training and cloud integration 🚧
5. **Production Infrastructure** (Phase 5): API server and deployment
6. **Advanced Features** (Phase 6): Multi-modal support and optimization

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{llm_pipeline,
  title={LLM Pipeline Framework: Large-Scale Training and Inference for Modern Language Models},
  author={David Akinpelu},
  year={2024},
  url={https://github.com/DavidAkinpelu/llm-pipeline},
  note={A comprehensive PyTorch framework for large-scale LLM fine-tuning, quantization, and inference}
}
```

## 🔗 Links

- **Repository**: [GitHub](https://github.com/DavidAkinpelu/llm-pipeline)
- **Documentation**: [Coming Soon]
- **Issues**: [GitHub Issues](https://github.com/DavidAkinpelu/llm-pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DavidAkinpelu/llm-pipeline/discussions)

---

**Note**: This project is in active development. The API may change between versions. Please check the development roadmap for planned features and timeline.
