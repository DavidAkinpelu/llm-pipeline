"""LoftQ: Quantization-aware LoRA initialization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from dataclasses import dataclass

from ...core.config import LoRAConfig
from ...core.base_module import BaseLoRAModule
from ..configs import LoftQConfig, QuantizationScheme


def _setup_loftq_logger(config: LoftQConfig) -> logging.Logger:
    """Setup logger based on LoftQ configuration."""
    logger = logging.getLogger("loftq")
    logger.setLevel(getattr(logging, config.log_level))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


@dataclass
class LoftQInitResult:
    """Result from LoftQ initialization"""
    quantized_weight: torch.Tensor
    lora_A: torch.Tensor
    lora_B: torch.Tensor
    reconstruction_error: float
    num_iterations: int


class LoftQInitializer:
    """LoftQ quantization-aware LoRA initialization"""
    
    def __init__(self, config: LoftQConfig):
        self.config = config
        self.logger = _setup_loftq_logger(config) if config.enable_logging else None
        
    def _get_nf4_levels(self) -> torch.Tensor:
        """Get NF4 quantization levels (configurable)."""
        if self.config.use_custom_nf4_levels and self.config.custom_nf4_levels is not None:
            return torch.tensor(self.config.custom_nf4_levels)
        else:
            # Default NF4 levels (optimized for normal distribution)
            return torch.tensor([
                -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1.0
            ])
    
    def _initialize_lora_matrix(self, shape: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialize LoRA matrix based on configuration.
        
        Args:
            shape: Shape of the matrix to initialize
            device: Device for the tensor
            dtype: Data type for the tensor
            
        Returns:
            Initialized LoRA matrix
        """
        if self.config.lora_init_method == "normal":
            return torch.randn(shape, device=device, dtype=dtype) * self.config.lora_init_std
        elif self.config.lora_init_method == "uniform":
            # Uniform in [-std, std]
            return torch.rand(shape, device=device, dtype=dtype) * 2 * self.config.lora_init_std - self.config.lora_init_std
        elif self.config.lora_init_method == "xavier":
            # Xavier uniform initialization
            bound = (6.0 / (shape[0] + shape[1])) ** 0.5
            return torch.rand(shape, device=device, dtype=dtype) * 2 * bound - bound
        elif self.config.lora_init_method == "kaiming":
            # Kaiming normal initialization
            return torch.randn(shape, device=device, dtype=dtype) * (2.0 / shape[1]) ** 0.5
        else:
            raise ValueError(f"Unknown initialization method: {self.config.lora_init_method}")
    
    def _get_quantization_range(self, bits: int) -> Tuple[int, int]:
        """Get quantization range for given bit width.
        
        Args:
            bits: Number of bits for quantization
            
        Returns:
            Tuple of (min_value, max_value)
        """
        if bits == 2:
            return -2, 1  # 4 levels: -2, -1, 0, 1
        elif bits == 4:
            return -8, 7  # 16 levels: -8 to 7
        elif bits == 8:
            return -128, 127  # 256 levels: -128 to 127
        else:
            # General formula for n-bit signed integers
            max_val = (2 ** (bits - 1)) - 1
            min_val = -(2 ** (bits - 1))
            return min_val, max_val
        
    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor according to scheme"""
        if self.config.quantization_scheme == QuantizationScheme.NF4:
            return self._quantize_nf4(tensor)
        elif self.config.quantization_scheme == QuantizationScheme.FP4:
            return self._quantize_fp4(tensor)
        elif self.config.quantization_scheme == QuantizationScheme.INT4:
            return self._quantize_int4(tensor)
        elif self.config.quantization_scheme == QuantizationScheme.INT8:
            return self._quantize_int8(tensor)
        else:
            raise ValueError(f"Unsupported quantization scheme: {self.config.quantization_scheme}")
    
    def _quantize_nf4(self, tensor: torch.Tensor) -> torch.Tensor:
        """NF4 (Normal Float 4) quantization"""
        # Get configurable NF4 levels
        nf4_levels = self._get_nf4_levels().to(device=tensor.device, dtype=tensor.dtype)
        
        # Normalize tensor
        scale = tensor.abs().max()
        normalized = tensor / (scale + self.config.epsilon)
        
        # Find closest quantization level
        distances = torch.abs(normalized.unsqueeze(-1) - nf4_levels.unsqueeze(0).unsqueeze(0))
        indices = torch.argmin(distances, dim=-1)
        quantized_normalized = nf4_levels[indices]
        
        return quantized_normalized * scale
    
    def _quantize_fp4(self, tensor: torch.Tensor) -> torch.Tensor:
        """FP4 quantization"""
        # Simple uniform quantization to 4 bits
        scale = tensor.abs().max()
        normalized = tensor / (scale + self.config.epsilon)
        
        # Quantize to [-1, 1] range with 2^4 = 16 levels
        num_levels = 2 ** self.config.loftq_bits
        levels = torch.linspace(-1, 1, num_levels, device=tensor.device, dtype=tensor.dtype)
        distances = torch.abs(normalized.unsqueeze(-1) - levels.unsqueeze(0).unsqueeze(0))
        indices = torch.argmin(distances, dim=-1)
        quantized_normalized = levels[indices]
        
        return quantized_normalized * scale
    
    def _quantize_int4(self, tensor: torch.Tensor) -> torch.Tensor:
        """INT4 quantization"""
        # Get quantization range for current bit width
        min_val, max_val = self._get_quantization_range(self.config.loftq_bits)
        scale = tensor.abs().max()
        
        # Quantize to n-bit integers
        quantized = torch.clamp(torch.round(tensor / scale * max_val), min_val, max_val)
        return quantized / max_val * scale
    
    def _quantize_int8(self, tensor: torch.Tensor) -> torch.Tensor:
        """INT8 quantization"""
        # Get quantization range for current bit width
        min_val, max_val = self._get_quantization_range(self.config.loftq_bits)
        scale = tensor.abs().max()
        
        # Quantize to n-bit integers
        quantized = torch.clamp(torch.round(tensor / scale * max_val), min_val, max_val)
        return quantized / max_val * scale
    
    def initialize_loftq(
        self, 
        weight: torch.Tensor,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        num_iter: Optional[int] = None
    ) -> LoftQInitResult:
        """Perform LoftQ initialization"""
        
        rank = rank or self.config.loftq_rank
        alpha = alpha or self.config.loftq_alpha
        num_iter = num_iter or self.config.loftq_iter
        
        # Initialize LoRA matrices using configurable initialization
        lora_A = self._initialize_lora_matrix((rank, weight.size(1)), weight.device, weight.dtype)
        lora_B = torch.zeros(weight.size(0), rank, device=weight.device, dtype=weight.dtype)
        
        original_weight = weight.clone()
        best_error = float('inf')
        best_result = None
        
        for iteration in range(num_iter):
            # Step 1: Fix LoRA, optimize quantization
            lora_weight = (lora_B @ lora_A) * (alpha / rank)
            residual = original_weight - lora_weight
            quantized_residual = self.quantize_tensor(residual)
            
            # Step 2: Fix quantization, optimize LoRA via SVD
            reconstruction_target = original_weight - quantized_residual
            
            # SVD decomposition for LoRA update
            U, S, Vt = torch.svd(reconstruction_target)
            
            # Take top-k components
            lora_B = (U[:, :rank] @ torch.diag(S[:rank].sqrt())).contiguous()
            lora_A = (torch.diag(S[:rank].sqrt()) @ Vt[:rank, :]).contiguous()
            
            # Compute reconstruction error
            reconstructed = quantized_residual + (lora_B @ lora_A) * (alpha / rank)
            error = F.mse_loss(reconstructed, original_weight).item()
            
            if error < best_error:
                best_error = error
                best_result = LoftQInitResult(
                    quantized_weight=quantized_residual.clone(),
                    lora_A=lora_A.clone(),
                    lora_B=lora_B.clone(),
                    reconstruction_error=error,
                    num_iterations=iteration + 1
                )
        
        return best_result


class LoftQQuantizedLoRA(nn.Module):
    """LoftQ-initialized quantized LoRA layer"""
    
    def __init__(
        self,
        base_weight: torch.Tensor,
        loftq_config: LoftQConfig,
        lora_config: LoRAConfig,
        bias: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.loftq_config = loftq_config
        self.lora_config = lora_config
        self.in_features = base_weight.size(1)
        self.out_features = base_weight.size(0)
        
        # Initialize with LoftQ
        initializer = LoftQInitializer(loftq_config)
        result = initializer.initialize_loftq(
            base_weight,
            lora_config.r,
            lora_config.alpha,
            loftq_config.loftq_iter
        )
        
        # Store quantized base weight (frozen)
        self.register_buffer('quantized_weight', result.quantized_weight)
        
        # LoRA parameters (trainable)
        self.lora_A = nn.Parameter(result.lora_A)
        self.lora_B = nn.Parameter(result.lora_B)
        
        # Bias
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        
        # Dropout
        self.dropout = nn.Dropout(lora_config.dropout) if lora_config.dropout > 0 else nn.Identity()
        
        # Store initialization info
        self.initialization_error = result.reconstruction_error
        self.scaling = lora_config.alpha / lora_config.r
        
        # Setup logging
        self.logger = _setup_loftq_logger(loftq_config) if loftq_config.enable_logging else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: quantized base + LoRA adaptation"""
        # Base quantized layer
        result = F.linear(x, self.quantized_weight, self.bias)
        
        # LoRA adaptation
        lora_result = x @ self.lora_A.T
        lora_result = self.dropout(lora_result)
        lora_result = lora_result @ self.lora_B.T
        result += lora_result * self.scaling
        
        return result
    
    def get_loftq_stats(self) -> Dict[str, Any]:
        """Get LoftQ initialization statistics"""
        return {
            "initialization_error": self.initialization_error,
            "quantization_scheme": self.loftq_config.quantization_scheme.value,
            "loftq_rank": self.loftq_config.loftq_rank,
            "loftq_alpha": self.loftq_config.loftq_alpha,
            "loftq_iterations": self.loftq_config.loftq_iter,
            "compression_ratio": self.loftq_config.baseline_bits / self.loftq_config.loftq_bits,
        }


def optimize_quantization_lora_pair(
    original_weight: torch.Tensor,
    loftq_config: LoftQConfig,
    lora_config: LoRAConfig,
    num_optimization_steps: Optional[int] = None
) -> LoftQQuantizedLoRA:
    """Optimize quantization-LoRA pair with iterative refinement"""
    
    num_steps = num_optimization_steps or loftq_config.num_optimization_steps
    
    # Create initial LoftQ layer
    loftq_layer = LoftQQuantizedLoRA(original_weight, loftq_config, lora_config)
    
    if not loftq_config.optimize_initialization:
        return loftq_layer
    
    # Iterative optimization
    optimizer = torch.optim.Adam([loftq_layer.lora_A, loftq_layer.lora_B], lr=loftq_config.learning_rate)
    
    best_error = float('inf')
    best_state = None
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Forward pass to reconstruct original weight
        lora_weight = (loftq_layer.lora_B @ loftq_layer.lora_A) * loftq_layer.scaling
        reconstructed = loftq_layer.quantized_weight + lora_weight
        
        # Reconstruction loss
        loss = F.mse_loss(reconstructed, original_weight)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track best result
        if loss.item() < best_error:
            best_error = loss.item()
            best_state = {
                'lora_A': loftq_layer.lora_A.data.clone(),
                'lora_B': loftq_layer.lora_B.data.clone()
            }
        
        if step % loftq_config.log_frequency == 0:
            if loftq_config.enable_logging:
                if hasattr(loftq_layer, 'logger') and loftq_layer.logger:
                    loftq_layer.logger.info(f"Step {step}: Reconstruction error = {loss.item():.6f}")
                else:
                    print(f"Step {step}: Reconstruction error = {loss.item():.6f}")
    
    # Restore best state
    if best_state is not None:
        loftq_layer.lora_A.data = best_state['lora_A']
        loftq_layer.lora_B.data = best_state['lora_B']
        loftq_layer.initialization_error = best_error
    
    if loftq_config.enable_logging:
        if hasattr(loftq_layer, 'logger') and loftq_layer.logger:
            loftq_layer.logger.info(f"LoftQ optimization complete. Final error: {best_error:.6f}")
        else:
            print(f"LoftQ optimization complete. Final error: {best_error:.6f}")
    return loftq_layer
