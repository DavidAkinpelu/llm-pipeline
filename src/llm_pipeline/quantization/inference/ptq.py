"""Post-Training Quantization (PTQ) for inference optimization."""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Literal
from dataclasses import dataclass
import numpy as np


@dataclass
class PTQConfig:
    """Configuration for Post-Training Quantization."""
    # Quantization method
    method: Literal["dynamic", "static", "qat"] = "dynamic"
    
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


class PTQQuantizer:
    """Post-Training Quantization implementation."""
    
    def __init__(self, config: PTQConfig):
        """Initialize PTQ quantizer.
        
        Args:
            config: PTQ configuration
        """
        self.config = config
        self.logger = self._setup_logger()
        self.calibration_data = []
        self.quantization_params = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for PTQ."""
        logger = logging.getLogger(f"{__name__}.PTQQuantizer")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def quantize_model(self, model: nn.Module, calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None) -> nn.Module:
        """Quantize a model using PTQ.
        
        Args:
            model: Model to quantize
            calibration_data: Optional calibration data for static quantization
            
        Returns:
            Quantized model
        """
        self.logger.info(f"Starting PTQ quantization with method: {self.config.method}")
        
        if self.config.method == "dynamic":
            return self._dynamic_quantization(model)
        elif self.config.method == "static":
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
            return self._static_quantization(model, calibration_data)
        elif self.config.method == "qat":
            return self._qat_quantization(model)
        else:
            raise ValueError(f"Unknown quantization method: {self.config.method}")
    
    def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to the model."""
        self.logger.info("Applying dynamic quantization")
        
        # Quantize weights only (activations quantized at runtime)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Embedding},
            dtype=torch.qint8
        )
        
        self.logger.info("Dynamic quantization completed")
        return quantized_model
    
    def _static_quantization(self, model: nn.Module, calibration_data: List[Dict[str, torch.Tensor]]) -> nn.Module:
        """Apply static quantization with calibration data."""
        self.logger.info("Applying static quantization with calibration")
        
        # Set quantization config
        quantization_config = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=True
            ),
            weight=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
                reduce_range=True
            )
        )
        
        # Prepare model for quantization
        model.eval()
        model.qconfig = quantization_config
        
        # Prepare for calibration
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with provided data
        self.logger.info(f"Calibrating with {len(calibration_data)} samples")
        with torch.no_grad():
            for i, sample in enumerate(calibration_data):
                if i >= self.config.calibration_samples:
                    break
                    
                # Forward pass for calibration
                if 'input_ids' in sample:
                    _ = model(**sample)
                else:
                    _ = model(sample.get('input', sample))
                
                if i % self.config.log_frequency == 0:
                    self.logger.info(f"Calibrated {i+1}/{len(calibration_data)} samples")
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        self.logger.info("Static quantization completed")
        return quantized_model
    
    def _qat_quantization(self, model: nn.Module) -> nn.Module:
        """Apply Quantization-Aware Training setup."""
        self.logger.info("Setting up Quantization-Aware Training")
        
        # Set QAT config
        qat_config = torch.quantization.QConfig(
            activation=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=True
            ),
            weight=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MovingAveragePerChannelMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
                reduce_range=True
            )
        )
        
        # Apply QAT configuration
        model.qconfig = qat_config
        torch.quantization.prepare_qat(model, inplace=True)
        
        self.logger.info("QAT setup completed - model ready for training")
        return model
    
    def get_quantization_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Get quantization statistics for the model.
        
        Args:
            model: Quantized model
            
        Returns:
            Dictionary with quantization statistics
        """
        stats = {
            "total_parameters": 0,
            "quantized_parameters": 0,
            "quantization_ratio": 0.0,
            "memory_reduction": 0.0,
            "layer_stats": {}
        }
        
        for name, module in model.named_modules():
            if isinstance(module, (torch.quantized.Linear, torch.quantized.Embedding)):
                # Count quantized parameters
                param_count = sum(p.numel() for p in module.parameters())
                stats["quantized_parameters"] += param_count
                stats["layer_stats"][name] = {
                    "type": type(module).__name__,
                    "parameters": param_count,
                    "quantized": True
                }
            elif hasattr(module, 'weight') and module.weight is not None:
                # Count regular parameters
                param_count = module.weight.numel()
                stats["total_parameters"] += param_count
                stats["layer_stats"][name] = {
                    "type": type(module).__name__,
                    "parameters": param_count,
                    "quantized": False
                }
        
        # Calculate ratios
        total_params = stats["total_parameters"] + stats["quantized_parameters"]
        if total_params > 0:
            stats["quantization_ratio"] = stats["quantized_parameters"] / total_params
            stats["memory_reduction"] = 1.0 - (stats["quantized_parameters"] * 1 + stats["total_parameters"] * 4) / (total_params * 4)
        
        return stats
    
    def compare_model_sizes(self, original_model: nn.Module, quantized_model: nn.Module) -> Dict[str, Any]:
        """Compare model sizes before and after quantization.
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            
        Returns:
            Size comparison statistics
        """
        def get_model_size(model):
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            return param_size + buffer_size
        
        original_size = get_model_size(original_model)
        quantized_size = get_model_size(quantized_model)
        
        return {
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "size_reduction_mb": (original_size - quantized_size) / (1024 * 1024),
            "compression_ratio": original_size / quantized_size if quantized_size > 0 else 0,
            "size_reduction_percent": ((original_size - quantized_size) / original_size * 100) if original_size > 0 else 0
        }


class LayerWisePTQ:
    """Layer-wise PTQ for fine-grained control."""
    
    def __init__(self, config: PTQConfig):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.LayerWisePTQ")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def quantize_layer(self, layer: nn.Module, layer_type: str) -> nn.Module:
        """Quantize a specific layer based on its type.
        
        Args:
            layer: Layer to quantize
            layer_type: Type of layer ("embedding", "attention", "ffn", "layernorm")
            
        Returns:
            Quantized layer
        """
        if layer_type == "embedding" and not self.config.quantize_embeddings:
            return layer
        elif layer_type == "attention" and not self.config.quantize_attention:
            return layer
        elif layer_type == "ffn" and not self.config.quantize_ffn:
            return layer
        elif layer_type == "layernorm" and not self.config.quantize_layernorm:
            return layer
        
        # Apply quantization based on layer type
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            return torch.quantization.quantize_dynamic(
                layer, {type(layer)}, dtype=torch.qint8
            )
        
        return layer
    
    def quantize_model_selective(self, model: nn.Module, layer_mapping: Dict[str, str]) -> nn.Module:
        """Quantize model with selective layer quantization.
        
        Args:
            model: Model to quantize
            layer_mapping: Mapping of layer names to types
            
        Returns:
            Selectively quantized model
        """
        self.logger.info("Starting selective layer quantization")
        
        for name, module in model.named_modules():
            if name in layer_mapping:
                layer_type = layer_mapping[name]
                quantized_module = self.quantize_layer(module, layer_type)
                
                if quantized_module != module:
                    # Replace module in parent
                    parent_name = '.'.join(name.split('.')[:-1])
                    if parent_name:
                        parent_module = model
                        for part in parent_name.split('.'):
                            parent_module = getattr(parent_module, part)
                        setattr(parent_module, name.split('.')[-1], quantized_module)
                    else:
                        # Root module
                        for attr_name in dir(model):
                            if getattr(model, attr_name) is module:
                                setattr(model, attr_name, quantized_module)
                    
                    self.logger.info(f"Quantized layer: {name} ({layer_type})")
        
        self.logger.info("Selective layer quantization completed")
        return model


def create_ptq_config(
    method: str = "dynamic",
    weight_bits: int = 8,
    activation_bits: int = 8,
    calibration_samples: int = 100,
    **kwargs
) -> PTQConfig:
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
    return PTQConfig(
        method=method,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        calibration_samples=calibration_samples,
        **kwargs
    )


def quantize_for_inference(
    model: nn.Module,
    method: str = "dynamic",
    calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None,
    **config_kwargs
) -> nn.Module:
    """Convenience function to quantize model for inference.
    
    Args:
        model: Model to quantize
        method: Quantization method
        calibration_data: Calibration data for static quantization
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Quantized model
    """
    config = create_ptq_config(method=method, **config_kwargs)
    quantizer = PTQQuantizer(config)
    return quantizer.quantize_model(model, calibration_data)
