"""Deployment format converters for production inference."""

import torch
import torch.nn as nn
import logging
import os
import json
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass
import numpy as np

# Optional deployment imports with fallback
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    # Output settings
    output_dir: str = "deployment"
    model_name: str = "deployed_model"
    
    # Format selection
    export_formats: List[str] = None  # ["onnx", "tensorrt", "gguf", "torchscript"]
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
    log_level: str = "INFO"


class DeploymentConverter:
    """Unified deployment converter supporting multiple formats."""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize deployment converter.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.logger = self._setup_logger()
        self.deployed_models = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for deployment converter."""
        logger = logging.getLogger(f"{__name__}.DeploymentConverter")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def deploy_model(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor,
        tokenizer: Optional[Any] = None,
        model_config: Optional[Dict[str, Any]] = None,
        calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, str]:
        """Deploy model to multiple formats.
        
        Args:
            model: PyTorch model
            dummy_input: Example input tensor
            tokenizer: Optional tokenizer
            model_config: Optional model configuration
            calibration_data: Calibration data for quantization
            
        Returns:
            Dictionary mapping format names to file paths
        """
        self.logger.info(f"Deploying model to formats: {self.config.export_formats}")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Default formats if not specified
        formats = self.config.export_formats or ["onnx", "torchscript"]
        
        deployed_paths = {}
        
        for format_name in formats:
            try:
                if format_name == "onnx":
                    path = self._deploy_onnx(model, dummy_input, calibration_data)
                    deployed_paths["onnx"] = path
                elif format_name == "tensorrt":
                    path = self._deploy_tensorrt(model, dummy_input, calibration_data)
                    deployed_paths["tensorrt"] = path
                elif format_name == "gguf":
                    path = self._deploy_gguf(model, model_config)
                    deployed_paths["gguf"] = path
                elif format_name == "torchscript":
                    path = self._deploy_torchscript(model, dummy_input)
                    deployed_paths["torchscript"] = path
                elif format_name == "coreml":
                    path = self._deploy_coreml(model, dummy_input)
                    deployed_paths["coreml"] = path
                else:
                    self.logger.warning(f"Unknown deployment format: {format_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to deploy to {format_name}: {e}")
        
        # Save deployment metadata
        self._save_deployment_metadata(deployed_paths, model_config)
        
        self.logger.info(f"Deployment completed. Models saved to: {self.config.output_dir}")
        return deployed_paths
    
    def _deploy_onnx(
        self, 
        model: nn.Module, 
        dummy_input: torch.Tensor,
        calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> str:
        """Deploy model to ONNX format."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install with: pip install onnx onnxruntime")
        
        self.logger.info("Deploying to ONNX format")
        
        from .onnx import convert_to_onnx
        
        output_path = os.path.join(self.config.output_dir, f"{self.config.model_name}.onnx")
        quantization_type = "dynamic" if self.config.quantization_enabled else "none"
        
        results = convert_to_onnx(
            model=model,
            dummy_input=dummy_input,
            output_path=output_path,
            quantization_type=quantization_type,
            calibration_data=calibration_data,
            enable_optimization=self.config.optimize_for_server
        )
        
        return results.get('quantized', results.get('onnx', output_path))
    
    def _deploy_tensorrt(
        self, 
        model: nn.Module, 
        dummy_input: torch.Tensor,
        calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> str:
        """Deploy model to TensorRT format."""
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install with: pip install tensorrt")
        
        self.logger.info("Deploying to TensorRT format")
        
        from .tensorrt import convert_to_tensorrt
        
        output_path = os.path.join(self.config.output_dir, f"{self.config.model_name}.trt")
        precision_mode = "int8" if self.config.quantization_enabled else "fp16"
        
        return convert_to_tensorrt(
            model=model,
            dummy_input=dummy_input,
            output_path=output_path,
            precision_mode=precision_mode,
            calibration_data=calibration_data,
            optimization_level=5 if self.config.optimize_for_server else 3
        )
    
    def _deploy_gguf(self, model: nn.Module, model_config: Optional[Dict[str, Any]] = None) -> str:
        """Deploy model to GGUF format."""
        self.logger.info("Deploying to GGUF format")
        
        from .gguf import convert_to_gguf
        
        output_path = os.path.join(self.config.output_dir, f"{self.config.model_name}.gguf")
        quantization_type = "Q8_0" if self.config.quantization_enabled else "F16"
        
        return convert_to_gguf(
            model=model,
            output_path=output_path,
            quantization_type=quantization_type,
            model_config=model_config,
            model_name=self.config.model_name,
            description=self.config.model_description,
            author=self.config.model_author
        )
    
    def _deploy_torchscript(self, model: nn.Module, dummy_input: torch.Tensor) -> str:
        """Deploy model to TorchScript format."""
        self.logger.info("Deploying to TorchScript format")
        
        model.eval()
        with torch.no_grad():
            # Trace the model
            traced_model = torch.jit.trace(model, dummy_input)
            
            # Optimize if requested
            if self.config.optimize_for_server:
                traced_model = torch.jit.optimize_for_inference(traced_model)
        
        output_path = os.path.join(self.config.output_dir, f"{self.config.model_name}.pt")
        traced_model.save(output_path)
        
        return output_path
    
    def _deploy_coreml(self, model: nn.Module, dummy_input: torch.Tensor) -> str:
        """Deploy model to CoreML format (iOS/macOS)."""
        self.logger.info("Deploying to CoreML format")
        
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError("CoreML Tools not available. Install with: pip install coremltools")
        
        # Convert to CoreML
        coreml_model = ct.convert(
            model,
            inputs=[ct.TensorType(shape=dummy_input.shape, dtype=np.float32)],
            outputs=None,
            minimum_deployment_target=ct.target.iOS13
        )
        
        # Optimize if requested
        if self.config.optimize_for_mobile:
            coreml_model = ct.models.neural_network.quantization_utils.quantize_weights(
                coreml_model, nbits=8
            )
        
        output_path = os.path.join(self.config.output_dir, f"{self.config.model_name}.mlpackage")
        coreml_model.save(output_path)
        
        return output_path
    
    def _save_deployment_metadata(
        self, 
        deployed_paths: Dict[str, str], 
        model_config: Optional[Dict[str, Any]] = None
    ):
        """Save deployment metadata."""
        metadata = {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "model_description": self.config.model_description,
            "model_author": self.config.model_author,
            "deployment_config": {
                "quantization_enabled": self.config.quantization_enabled,
                "quantization_bits": self.config.quantization_bits,
                "optimize_for_mobile": self.config.optimize_for_mobile,
                "optimize_for_server": self.config.optimize_for_server,
            },
            "deployed_formats": deployed_paths,
            "model_config": model_config or {},
            "deployment_timestamp": torch.utils.data.get_worker_info() or "unknown"
        }
        
        metadata_path = os.path.join(self.config.output_dir, "deployment_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Deployment metadata saved: {metadata_path}")


class ModelBenchmarker:
    """Benchmark deployed models across different formats."""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize model benchmarker.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for model benchmarker."""
        logger = logging.getLogger(f"{__name__}.ModelBenchmarker")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def benchmark_models(
        self,
        model_paths: Dict[str, str],
        test_inputs: Dict[str, torch.Tensor],
        num_runs: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark multiple deployed models.
        
        Args:
            model_paths: Dictionary mapping format names to model paths
            test_inputs: Test input tensors
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary mapping format names to benchmark results
        """
        self.logger.info(f"Benchmarking {len(model_paths)} deployed models")
        
        results = {}
        
        for format_name, model_path in model_paths.items():
            try:
                if format_name == "onnx":
                    benchmark_result = self._benchmark_onnx(model_path, test_inputs, num_runs)
                elif format_name == "tensorrt":
                    benchmark_result = self._benchmark_tensorrt(model_path, test_inputs, num_runs)
                elif format_name == "torchscript":
                    benchmark_result = self._benchmark_torchscript(model_path, test_inputs, num_runs)
                else:
                    self.logger.warning(f"Benchmarking not supported for format: {format_name}")
                    continue
                
                results[format_name] = benchmark_result
                
            except Exception as e:
                self.logger.error(f"Failed to benchmark {format_name}: {e}")
        
        return results
    
    def _benchmark_onnx(
        self, 
        model_path: str, 
        test_inputs: Dict[str, torch.Tensor], 
        num_runs: int
    ) -> Dict[str, float]:
        """Benchmark ONNX model."""
        session = ort.InferenceSession(model_path)
        
        # Prepare inputs
        input_dict = {name: tensor.cpu().numpy() for name, tensor in test_inputs.items()}
        
        # Warmup
        for _ in range(10):
            session.run(None, input_dict)
        
        # Benchmark
        import time
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            session.run(None, input_dict)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return self._calculate_benchmark_stats(times)
    
    def _benchmark_tensorrt(
        self, 
        model_path: str, 
        test_inputs: Dict[str, torch.Tensor], 
        num_runs: int
    ) -> Dict[str, float]:
        """Benchmark TensorRT model."""
        from .tensorrt import TensorRTEngine, TensorRTConfig
        
        config = TensorRTConfig(model_path=model_path)
        engine = TensorRTEngine(config)
        engine.load_engine(model_path)
        
        return engine.benchmark(test_inputs, num_runs)
    
    def _benchmark_torchscript(
        self, 
        model_path: str, 
        test_inputs: Dict[str, torch.Tensor], 
        num_runs: int
    ) -> Dict[str, float]:
        """Benchmark TorchScript model."""
        model = torch.jit.load(model_path)
        model.eval()
        
        # Prepare inputs (assuming single input for simplicity)
        input_tensor = next(iter(test_inputs.values()))
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Benchmark
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                times.append(end_time - start_time)
        
        return self._calculate_benchmark_stats(times)
    
    def _calculate_benchmark_stats(self, times: List[float]) -> Dict[str, float]:
        """Calculate benchmark statistics."""
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


def deploy_model(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_dir: str = "deployment",
    export_formats: Optional[List[str]] = None,
    quantization_enabled: bool = True,
    **config_kwargs
) -> Dict[str, str]:
    """Convenience function to deploy model to multiple formats.
    
    Args:
        model: PyTorch model
        dummy_input: Example input tensor
        output_dir: Output directory for deployed models
        export_formats: List of formats to export
        quantization_enabled: Enable quantization
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Dictionary mapping format names to file paths
    """
    config = DeploymentConfig(
        output_dir=output_dir,
        export_formats=export_formats,
        quantization_enabled=quantization_enabled,
        **config_kwargs
    )
    
    converter = DeploymentConverter(config)
    return converter.deploy_model(model, dummy_input)
