"""TensorRT quantization for NVIDIA GPU inference optimization."""

import torch
import torch.nn as nn
import logging
import os
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
import numpy as np

# Optional TensorRT imports with fallback
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    # Create dummy classes for type hints
    class trt:
        class Logger:
            pass
        class Builder:
            pass
        class Network:
            pass
        class IOptimizationProfile:
            pass


@dataclass
class TensorRTConfig:
    """Configuration for TensorRT optimization."""
    # Model settings
    model_path: str = "model.trt"
    max_batch_size: int = 1
    max_sequence_length: int = 2048
    max_output_length: int = 512
    
    # Precision settings
    precision_mode: str = "fp16"  # "fp32", "fp16", "int8"
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
    log_level: str = "INFO"
    verbose_logging: bool = False


class TensorRTLogger(trt.Logger):
    """TensorRT logger with Python logging integration."""
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize TensorRT logger.
        
        Args:
            log_level: Logging level
        """
        super().__init__()
        self.logger = self._setup_logger(log_level)
        
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup Python logger."""
        logger = logging.getLogger(f"{__name__}.TensorRTLogger")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def log(self, severity: int, msg: str):
        """Override TensorRT log method."""
        if severity <= trt.Logger.Severity.VERBOSE:
            self.logger.debug(f"TensorRT: {msg}")
        elif severity <= trt.Logger.Severity.INFO:
            self.logger.info(f"TensorRT: {msg}")
        elif severity <= trt.Logger.Severity.WARNING:
            self.logger.warning(f"TensorRT: {msg}")
        elif severity <= trt.Logger.Severity.ERROR:
            self.logger.error(f"TensorRT: {msg}")
        else:
            self.logger.critical(f"TensorRT: {msg}")


class TensorRTCalibrator(trt.IInt8EntropyCalibrator2):
    """TensorRT INT8 calibration data provider."""
    
    def __init__(self, calibration_data: List[Dict[str, torch.Tensor]], batch_size: int = 1):
        """Initialize TensorRT calibrator.
        
        Args:
            calibration_data: List of calibration samples
            batch_size: Batch size for calibration
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install with: pip install tensorrt")
        
        super().__init__()
        self.calibration_data = calibration_data
        self.batch_size = batch_size
        self.current_index = 0
        self.device_inputs = {}
        
    def get_batch_size(self) -> int:
        """Get batch size for calibration."""
        return self.batch_size
    
    def get_batch(self, names: List[str]) -> List[int]:
        """Get next batch for calibration.
        
        Args:
            names: Input tensor names
            
        Returns:
            List of device memory pointers
        """
        if self.current_index >= len(self.calibration_data):
            return []
        
        # Get batch data
        batch_data = []
        for i in range(self.batch_size):
            if self.current_index + i >= len(self.calibration_data):
                break
            
            sample = self.calibration_data[self.current_index + i]
            batch_data.append(sample)
        
        self.current_index += len(batch_data)
        
        # Allocate device memory and copy data
        device_ptrs = []
        for name in names:
            if name not in self.device_inputs:
                # Allocate device memory
                sample_shape = batch_data[0][name].shape
                device_mem = cuda.mem_alloc(batch_data[0][name].nbytes)
                self.device_inputs[name] = device_mem
            
            # Copy batch data to device
            batch_tensor = torch.stack([sample[name] for sample in batch_data])
            cuda.memcpy_htod(self.device_inputs[name], batch_tensor.cpu().numpy())
            device_ptrs.append(int(self.device_inputs[name]))
        
        return device_ptrs
    
    def read_calibration_cache(self, length: int) -> bytes:
        """Read calibration cache (optional)."""
        return b""
    
    def write_calibration_cache(self, cache: bytes):
        """Write calibration cache (optional)."""
        pass


class TensorRTEngine:
    """TensorRT inference engine."""
    
    def __init__(self, config: TensorRTConfig):
        """Initialize TensorRT engine.
        
        Args:
            config: TensorRT configuration
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install with: pip install tensorrt")
        
        self.config = config
        self.logger = self._setup_logger()
        self.engine = None
        self.context = None
        self.bindings = []
        self.stream = None
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for TensorRT engine."""
        logger = logging.getLogger(f"{__name__}.TensorRTEngine")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def build_engine_from_onnx(self, onnx_path: str, calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None) -> str:
        """Build TensorRT engine from ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            calibration_data: Calibration data for INT8 quantization
            
        Returns:
            Path to TensorRT engine
        """
        self.logger.info(f"Building TensorRT engine from ONNX: {onnx_path}")
        
        # Create TensorRT logger
        trt_logger = TensorRTLogger(self.config.log_level)
        
        # Create builder and network
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        
        # Create ONNX parser
        parser = trt.OnnxParser(network, trt_logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                self.logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    self.logger.error(f"Parser error {error}: {parser.get_error(error)}")
                raise RuntimeError("ONNX parsing failed")
        
        self.logger.info("ONNX model parsed successfully")
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = self.config.max_workspace_size
        
        # Set optimization level
        if hasattr(trt.BuilderFlag, f'LEVEL_{self.config.optimization_level}'):
            optimization_flag = getattr(trt.BuilderFlag, f'LEVEL_{self.config.optimization_level}')
            config.builder_flags = optimization_flag
        
        # Configure precision
        if self.config.precision_mode == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            self.logger.info("Enabled FP16 precision")
        elif self.config.precision_mode == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            self.logger.info("Enabled INT8 precision")
            
            # Setup INT8 calibration if data provided
            if calibration_data and self.config.enable_int8_calibration:
                calibrator = TensorRTCalibrator(
                    calibration_data, 
                    self.config.calibration_batch_size
                )
                config.int8_calibrator = calibrator
                self.logger.info("Setup INT8 calibration")
        
        # Create optimization profile
        profile = builder.create_optimization_profile()
        
        # Configure input shapes (assuming standard language model inputs)
        input_names = ["input_ids", "attention_mask"]
        for name in input_names:
            if network.get_input(name):
                profile.set_shape(
                    name,
                    (1, 1),  # min
                    (self.config.max_batch_size, self.config.max_sequence_length),  # opt
                    (self.config.max_batch_size, self.config.max_sequence_length)   # max
                )
        
        config.add_optimization_profile(profile)
        
        # Build engine
        self.logger.info("Building TensorRT engine...")
        self.engine = builder.build_engine(network, config)
        
        if self.engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        with open(self.config.model_path, 'wb') as f:
            f.write(self.engine.serialize())
        
        self.logger.info(f"TensorRT engine saved: {self.config.model_path}")
        return self.config.model_path
    
    def load_engine(self, engine_path: str):
        """Load TensorRT engine from file.
        
        Args:
            engine_path: Path to TensorRT engine
        """
        self.logger.info(f"Loading TensorRT engine: {engine_path}")
        
        # Create TensorRT logger
        trt_logger = TensorRTLogger(self.config.log_level)
        
        # Create runtime and deserialize engine
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Setup bindings and stream
        self._setup_bindings()
        self.stream = cuda.Stream()
        
        self.logger.info("TensorRT engine loaded successfully")
    
    def _setup_bindings(self):
        """Setup input/output bindings."""
        self.bindings = []
        
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            binding_shape = self.engine.get_binding_shape(i)
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            # Allocate device memory
            size = trt.volume(binding_shape)
            device_mem = cuda.mem_alloc(size * np.dtype(binding_dtype).itemsize)
            self.bindings.append(device_mem)
            
            self.logger.debug(f"Binding {i}: {binding_name}, shape: {binding_shape}, dtype: {binding_dtype}")
    
    def infer(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run inference with TensorRT engine.
        
        Args:
            inputs: Dictionary of input tensors
            
        Returns:
            Dictionary of output tensors
        """
        if self.engine is None or self.context is None:
            raise RuntimeError("TensorRT engine not loaded")
        
        # Copy inputs to device
        for i, (name, tensor) in enumerate(inputs.items()):
            if i < len(self.bindings):
                cuda.memcpy_htod_async(self.bindings[i], tensor.cpu().numpy(), self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy outputs back to host
        outputs = {}
        for i in range(len(inputs), self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            binding_shape = self.engine.get_binding_shape(i)
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            # Allocate host memory
            size = trt.volume(binding_shape)
            host_output = np.empty(binding_shape, dtype=binding_dtype)
            
            # Copy from device
            cuda.memcpy_dtoh_async(host_output, self.bindings[i], self.stream)
            self.stream.synchronize()
            
            outputs[binding_name] = torch.from_numpy(host_output)
        
        return outputs
    
    def benchmark(self, inputs: Dict[str, torch.Tensor], num_runs: int = 100) -> Dict[str, float]:
        """Benchmark TensorRT engine performance.
        
        Args:
            inputs: Input tensors for benchmarking
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        self.logger.info(f"Benchmarking TensorRT engine for {num_runs} runs")
        
        # Warmup
        for _ in range(10):
            self.infer(inputs)
        
        # Benchmark
        import time
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            self.infer(inputs)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        return {
            "average_inference_time_ms": np.mean(times) * 1000,
            "std_inference_time_ms": np.std(times) * 1000,
            "min_inference_time_ms": np.min(times) * 1000,
            "max_inference_time_ms": np.max(times) * 1000,
            "throughput_fps": 1.0 / np.mean(times),
            "p50_latency_ms": np.percentile(times, 50) * 1000,
            "p95_latency_ms": np.percentile(times, 95) * 1000,
            "p99_latency_ms": np.percentile(times, 99) * 1000
        }


class TensorRTConverter:
    """Complete TensorRT conversion pipeline."""
    
    def __init__(self, config: TensorRTConfig):
        """Initialize TensorRT converter.
        
        Args:
            config: TensorRT configuration
        """
        self.config = config
        self.engine = TensorRTEngine(config)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for TensorRT converter."""
        logger = logging.getLogger(f"{__name__}.TensorRTConverter")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def convert_from_onnx(
        self, 
        onnx_path: str,
        calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> str:
        """Convert ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            calibration_data: Calibration data for INT8 quantization
            
        Returns:
            Path to TensorRT engine
        """
        self.logger.info("Starting TensorRT conversion from ONNX")
        
        # Build TensorRT engine
        engine_path = self.engine.build_engine_from_onnx(onnx_path, calibration_data)
        
        # Load engine for immediate use
        self.engine.load_engine(engine_path)
        
        self.logger.info("TensorRT conversion completed")
        return engine_path
    
    def convert_from_pytorch(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor,
        calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None,
        onnx_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Convert PyTorch model to TensorRT engine via ONNX.
        
        Args:
            model: PyTorch model
            dummy_input: Example input tensor
            calibration_data: Calibration data for INT8 quantization
            onnx_config: ONNX export configuration
            
        Returns:
            Path to TensorRT engine
        """
        self.logger.info("Starting TensorRT conversion from PyTorch")
        
        # First convert to ONNX
        from .onnx import convert_to_onnx
        
        onnx_config = onnx_config or {}
        onnx_results = convert_to_onnx(
            model=model,
            dummy_input=dummy_input,
            output_path="temp_model.onnx",
            quantization_type="none",  # Don't quantize ONNX, let TensorRT handle it
            **onnx_config
        )
        
        onnx_path = onnx_results.get('onnx', 'temp_model.onnx')
        
        # Then convert to TensorRT
        engine_path = self.convert_from_onnx(onnx_path, calibration_data)
        
        # Cleanup temporary ONNX file
        if onnx_path == "temp_model.onnx" and os.path.exists(onnx_path):
            os.remove(onnx_path)
        
        self.logger.info("TensorRT conversion from PyTorch completed")
        return engine_path


def convert_to_tensorrt(
    model_or_onnx_path: Union[nn.Module, str],
    dummy_input: Optional[torch.Tensor] = None,
    output_path: str = "model.trt",
    precision_mode: str = "fp16",
    calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None,
    **config_kwargs
) -> str:
    """Convenience function to convert model to TensorRT format.
    
    Args:
        model_or_onnx_path: PyTorch model or path to ONNX model
        dummy_input: Example input tensor (required for PyTorch models)
        output_path: Output TensorRT engine path
        precision_mode: Precision mode (fp32, fp16, int8)
        calibration_data: Calibration data for INT8 quantization
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Path to TensorRT engine
    """
    config = TensorRTConfig(
        model_path=output_path,
        precision_mode=precision_mode,
        **config_kwargs
    )
    
    converter = TensorRTConverter(config)
    
    if isinstance(model_or_onnx_path, str):
        # Convert from ONNX
        return converter.convert_from_onnx(model_or_onnx_path, calibration_data)
    else:
        # Convert from PyTorch
        if dummy_input is None:
            raise ValueError("dummy_input required for PyTorch model conversion")
        
        return converter.convert_from_pytorch(
            model_or_onnx_path, dummy_input, calibration_data
        )
