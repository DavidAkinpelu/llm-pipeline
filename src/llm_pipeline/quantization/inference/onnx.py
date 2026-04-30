"""ONNX quantization for inference optimization."""

import torch
import torch.nn as nn
import logging
import os
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass
import numpy as np

# Optional ONNX imports with fallback
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        quantize_dynamic, quantize_static, 
        CalibrationDataReader, QuantType, QuantFormat
    )
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    # Create dummy classes for type hints
    class CalibrationDataReader:
        pass
    class QuantType:
        QInt8 = 0
        QUInt8 = 1
    class QuantFormat:
        QOperator = 0
        QDQ = 1


@dataclass
class ONNXConfig:
    """Configuration for ONNX quantization."""
    # Export settings
    export_path: str = "model.onnx"
    opset_version: int = 17
    
    # Quantization settings
    quantization_type: str = "dynamic"  # "dynamic", "static", "qat"
    weight_type: str = "QInt8"  # "QInt8", "QUInt8"
    activation_type: str = "QUInt8"
    quantization_format: str = "QOperator"  # "QOperator", "QDQ"
    
    # Static quantization settings
    calibration_samples: int = 100
    calibration_batch_size: int = 1
    
    # Optimization settings
    enable_optimization: bool = True
    optimization_level: str = "all"  # "disable", "basic", "extended", "all"
    
    # Input/Output settings
    input_names: List[str] = None
    output_names: List[str] = None
    dynamic_axes: Dict[str, Dict[int, str]] = None
    
    # Logging
    enable_logging: bool = True
    log_level: str = "INFO"
    suppress_warnings: bool = True


class ONNXExporter:
    """ONNX model exporter."""
    
    def __init__(self, config: ONNXConfig):
        """Initialize ONNX exporter.
        
        Args:
            config: ONNX configuration
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX and ONNX Runtime not available. Install with: pip install onnx onnxruntime")
        
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for ONNX exporter."""
        logger = logging.getLogger(f"{__name__}.ONNXExporter")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def export_model(
        self, 
        model: nn.Module, 
        dummy_input: torch.Tensor,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> str:
        """Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model
            dummy_input: Example input tensor
            input_names: Input tensor names
            output_names: Output tensor names
            dynamic_axes: Dynamic axes configuration
            
        Returns:
            Path to exported ONNX model
        """
        self.logger.info(f"Exporting model to ONNX: {self.config.export_path}")
        
        # Use config values or provided values
        input_names = input_names or self.config.input_names or ["input_ids", "attention_mask"]
        output_names = output_names or self.config.output_names or ["logits"]
        dynamic_axes = dynamic_axes or self.config.dynamic_axes or self._get_default_dynamic_axes()
        
        # Set model to evaluation mode
        model.eval()
        
        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                self.config.export_path,
                export_params=True,
                opset_version=self.config.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=self.config.enable_logging
            )
        
        self.logger.info(f"Model exported successfully: {self.config.export_path}")
        return self.config.export_path
    
    def _get_default_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """Get default dynamic axes configuration for language models."""
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
    
    def verify_model(self, model_path: Optional[str] = None) -> bool:
        """Verify exported ONNX model.
        
        Args:
            model_path: Path to ONNX model (uses config path if None)
            
        Returns:
            True if model is valid
        """
        model_path = model_path or self.config.export_path
        
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            
            self.logger.info("ONNX model verification passed")
            return True
            
        except Exception as e:
            self.logger.error(f"ONNX model verification failed: {e}")
            return False
    
    def optimize_model(self, model_path: Optional[str] = None) -> str:
        """Optimize ONNX model.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Path to optimized model
        """
        if not self.config.enable_optimization:
            return model_path or self.config.export_path
        
        model_path = model_path or self.config.export_path
        optimized_path = model_path.replace('.onnx', '_optimized.onnx')
        
        self.logger.info(f"Optimizing ONNX model: {model_path}")
        
        # Optimization levels
        optimization_levels = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        }
        
        optimization_level = optimization_levels.get(
            self.config.optimization_level, 
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        
        # Create optimization session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = optimization_level
        session_options.optimized_model_filepath = optimized_path
        
        # Load model for optimization
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(model_path, session_options, providers=providers)
        
        self.logger.info(f"Optimized model saved: {optimized_path}")
        return optimized_path


class ONNXQuantizer:
    """ONNX model quantizer."""
    
    def __init__(self, config: ONNXConfig):
        """Initialize ONNX quantizer.
        
        Args:
            config: ONNX configuration
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX and ONNX Runtime not available. Install with: pip install onnx onnxruntime")
        
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for ONNX quantizer."""
        logger = logging.getLogger(f"{__name__}.ONNXQuantizer")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def quantize_model(
        self, 
        model_path: str,
        calibration_data_reader: Optional[CalibrationDataReader] = None
    ) -> str:
        """Quantize ONNX model.
        
        Args:
            model_path: Path to ONNX model
            calibration_data_reader: Data reader for static quantization
            
        Returns:
            Path to quantized model
        """
        self.logger.info(f"Quantizing ONNX model: {model_path}")
        
        # Generate output path
        base_name = model_path.replace('.onnx', '')
        quantized_path = f"{base_name}_quantized.onnx"
        
        # Map configuration to ONNX Runtime types
        weight_type = getattr(QuantType, self.config.weight_type, QuantType.QInt8)
        activation_type = getattr(QuantType, self.config.activation_type, QuantType.QUInt8)
        quant_format = getattr(QuantFormat, self.config.quantization_format, QuantFormat.QOperator)
        
        if self.config.quantization_type == "dynamic":
            self._dynamic_quantization(model_path, quantized_path, weight_type)
        elif self.config.quantization_type == "static":
            if calibration_data_reader is None:
                raise ValueError("Calibration data reader required for static quantization")
            self._static_quantization(model_path, quantized_path, calibration_data_reader, weight_type, activation_type, quant_format)
        elif self.config.quantization_type == "qat":
            self._qat_quantization(model_path, quantized_path)
        else:
            raise ValueError(f"Unknown quantization type: {self.config.quantization_type}")
        
        self.logger.info(f"Quantized model saved: {quantized_path}")
        return quantized_path
    
    def _dynamic_quantization(self, model_path: str, output_path: str, weight_type: int):
        """Apply dynamic quantization."""
        self.logger.info("Applying dynamic quantization")
        
        quantize_dynamic(
            model_path,
            output_path,
            weight_type=weight_type,
            per_channel=True,
            reduce_range=True
        )
    
    def _static_quantization(
        self, 
        model_path: str, 
        output_path: str, 
        calibration_data_reader: CalibrationDataReader,
        weight_type: int,
        activation_type: int,
        quant_format: int
    ):
        """Apply static quantization."""
        self.logger.info("Applying static quantization")
        
        quantize_static(
            model_path,
            output_path,
            calibration_data_reader,
            weight_type=weight_type,
            activation_type=activation_type,
            quant_format=quant_format,
            per_channel=True,
            reduce_range=True
        )
    
    def _qat_quantization(self, model_path: str, output_path: str):
        """Apply QAT quantization (simplified)."""
        self.logger.info("Applying QAT quantization")
        # QAT typically requires training-time quantization, so we fall back to dynamic
        self._dynamic_quantization(model_path, output_path, QuantType.QInt8)


class ONNXCalibrationDataReader(CalibrationDataReader):
    """Custom calibration data reader for ONNX quantization."""
    
    def __init__(self, calibration_data: List[Dict[str, torch.Tensor]], input_names: List[str]):
        """Initialize calibration data reader.
        
        Args:
            calibration_data: List of calibration samples
            input_names: Input tensor names
        """
        self.calibration_data = calibration_data
        self.input_names = input_names
        self.current_index = 0
        
    def get_next(self) -> Dict[str, np.ndarray]:
        """Get next calibration sample.
        
        Returns:
            Dictionary of input tensors as numpy arrays
        """
        if self.current_index >= len(self.calibration_data):
            return None
        
        sample = self.calibration_data[self.current_index]
        self.current_index += 1
        
        # Convert to numpy arrays
        return {
            name: sample[name].cpu().numpy() 
            for name in self.input_names 
            if name in sample
        }


class ONNXConverter:
    """Complete ONNX conversion pipeline."""
    
    def __init__(self, config: ONNXConfig):
        """Initialize ONNX converter.
        
        Args:
            config: ONNX configuration
        """
        self.config = config
        self.exporter = ONNXExporter(config)
        self.quantizer = ONNXQuantizer(config)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for ONNX converter."""
        logger = logging.getLogger(f"{__name__}.ONNXConverter")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def convert_and_quantize(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor,
        calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Convert model to ONNX and quantize it.
        
        Args:
            model: PyTorch model
            dummy_input: Example input tensor
            calibration_data: Calibration data for static quantization
            input_names: Input tensor names
            output_names: Output tensor names
            
        Returns:
            Dictionary with paths to exported models
        """
        self.logger.info("Starting ONNX conversion and quantization pipeline")
        
        results = {}
        
        # Step 1: Export to ONNX
        onnx_path = self.exporter.export_model(
            model, dummy_input, input_names, output_names
        )
        results['onnx'] = onnx_path
        
        # Step 2: Verify model
        if not self.exporter.verify_model(onnx_path):
            raise RuntimeError("ONNX model verification failed")
        
        # Step 3: Optimize model
        if self.config.enable_optimization:
            optimized_path = self.exporter.optimize_model(onnx_path)
            results['optimized'] = optimized_path
            onnx_path = optimized_path  # Use optimized model for quantization
        
        # Step 4: Quantize model
        if self.config.quantization_type in ["dynamic", "static"]:
            calibration_data_reader = None
            if self.config.quantization_type == "static" and calibration_data:
                input_names = input_names or self.config.input_names or ["input_ids", "attention_mask"]
                calibration_data_reader = ONNXCalibrationDataReader(calibration_data, input_names)
            
            quantized_path = self.quantizer.quantize_model(onnx_path, calibration_data_reader)
            results['quantized'] = quantized_path
        
        self.logger.info("ONNX conversion and quantization completed")
        return results
    
    def benchmark_model(self, model_path: str, input_shape: Tuple[int, ...], num_runs: int = 100) -> Dict[str, float]:
        """Benchmark ONNX model performance.
        
        Args:
            model_path: Path to ONNX model
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        self.logger.info(f"Benchmarking ONNX model: {model_path}")
        
        # Create inference session
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(model_path, providers=providers)
        
        # Prepare input
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
        
        # Benchmark
        import time
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            session.run(None, {input_name: dummy_input})
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        return {
            "average_inference_time_ms": avg_time * 1000,
            "std_inference_time_ms": std_time * 1000,
            "min_inference_time_ms": min_time * 1000,
            "max_inference_time_ms": max_time * 1000,
            "throughput_fps": 1.0 / avg_time
        }


def convert_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str = "model.onnx",
    quantization_type: str = "dynamic",
    calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None,
    **config_kwargs
) -> Dict[str, str]:
    """Convenience function to convert model to ONNX format.
    
    Args:
        model: PyTorch model
        dummy_input: Example input tensor
        output_path: Output file path
        quantization_type: Quantization type
        calibration_data: Calibration data for static quantization
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Dictionary with paths to exported models
    """
    config = ONNXConfig(
        export_path=output_path,
        quantization_type=quantization_type,
        **config_kwargs
    )
    
    converter = ONNXConverter(config)
    return converter.convert_and_quantize(
        model, dummy_input, calibration_data
    )
