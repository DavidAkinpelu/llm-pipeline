"""GGUF format support for model deployment and quantization."""

import struct
import json
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, BinaryIO
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np


@dataclass
class GGUFConfig:
    """Configuration for GGUF format conversion."""
    # Model metadata
    model_name: str = "llm_pipeline_model"
    model_type: str = "llama"  # llama, mistral, qwen, etc.
    model_version: str = "1.0"
    
    # Quantization settings
    quantization_type: str = "Q8_0"  # Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q6_K, Q8_K
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
    log_level: str = "INFO"


class GGUFWriter:
    """Writer for GGUF format."""
    
    def __init__(self, config: GGUFConfig):
        """Initialize GGUF writer.
        
        Args:
            config: GGUF configuration
        """
        self.config = config
        self.logger = self._setup_logger()
        self.kv_data = {}
        self.tensors = []
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for GGUF writer."""
        logger = logging.getLogger(f"{__name__}.GGUFWriter")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def add_metadata(self, key: str, value: Any, value_type: str = "string"):
        """Add metadata key-value pair.
        
        Args:
            key: Metadata key
            value: Metadata value
            value_type: Value type (string, int, float, bool, array)
        """
        self.kv_data[key] = {"value": value, "type": value_type}
        self.logger.debug(f"Added metadata: {key} = {value} ({value_type})")
    
    def add_tensor(self, name: str, tensor: torch.Tensor, quantization_type: Optional[str] = None):
        """Add tensor to the GGUF file.
        
        Args:
            name: Tensor name
            tensor: PyTorch tensor
            quantization_type: Quantization type (overrides config)
        """
        quant_type = quantization_type or self.config.quantization_type
        quantized_data = self._quantize_tensor(tensor, quant_type)
        
        self.tensors.append({
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "quantization_type": quant_type,
            "data": quantized_data,
            "original_size": tensor.numel() * tensor.element_size(),
            "quantized_size": len(quantized_data)
        })
        
        self.logger.debug(f"Added tensor: {name}, shape: {tensor.shape}, quantized: {quant_type}")
    
    def _quantize_tensor(self, tensor: torch.Tensor, quant_type: str) -> bytes:
        """Quantize tensor according to GGUF format.
        
        Args:
            tensor: Input tensor
            quant_type: Quantization type
            
        Returns:
            Quantized tensor as bytes
        """
        tensor_np = tensor.detach().cpu().numpy().astype(np.float32)
        
        if quant_type == "Q8_0":
            return self._quantize_q8_0(tensor_np)
        elif quant_type == "Q4_0":
            return self._quantize_q4_0(tensor_np)
        elif quant_type == "Q4_1":
            return self._quantize_q4_1(tensor_np)
        elif quant_type == "Q5_0":
            return self._quantize_q5_0(tensor_np)
        elif quant_type == "Q5_1":
            return self._quantize_q5_1(tensor_np)
        elif quant_type == "Q6_K":
            return self._quantize_q6_k(tensor_np)
        elif quant_type == "Q8_K":
            return self._quantize_q8_k(tensor_np)
        else:
            raise ValueError(f"Unsupported quantization type: {quant_type}")
    
    def _quantize_q8_0(self, tensor: np.ndarray) -> bytes:
        """Quantize to Q8_0 format (8-bit with scale)."""
        # Reshape to blocks
        block_size = self.config.block_size
        tensor_flat = tensor.flatten()
        num_blocks = (tensor_flat.shape[0] + block_size - 1) // block_size
        
        quantized_data = bytearray()
        
        for i in range(num_blocks):
            block_start = i * block_size
            block_end = min(block_start + block_size, tensor_flat.shape[0])
            block = tensor_flat[block_start:block_end]
            
            # Calculate scale
            scale = np.max(np.abs(block)) / 127.0
            
            # Quantize
            quantized_block = np.round(block / scale).astype(np.int8)
            
            # Pack: scale (float32) + quantized data (int8)
            quantized_data.extend(struct.pack('<f', scale))
            quantized_data.extend(quantized_block.tobytes())
        
        return bytes(quantized_data)
    
    def _quantize_q4_0(self, tensor: np.ndarray) -> bytes:
        """Quantize to Q4_0 format (4-bit with scale)."""
        block_size = self.config.block_size
        tensor_flat = tensor.flatten()
        num_blocks = (tensor_flat.shape[0] + block_size - 1) // block_size
        
        quantized_data = bytearray()
        
        for i in range(num_blocks):
            block_start = i * block_size
            block_end = min(block_start + block_size, tensor_flat.shape[0])
            block = tensor_flat[block_start:block_end]
            
            # Pad block to full size
            if len(block) < block_size:
                block = np.pad(block, (0, block_size - len(block)))
            
            # Calculate scale
            scale = np.max(np.abs(block)) / 7.0  # 4-bit: -8 to 7
            
            # Quantize to 4-bit
            quantized_block = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
            
            # Pack 4-bit values into bytes
            packed_block = bytearray()
            packed_block.extend(struct.pack('<f', scale))
            
            for j in range(0, len(quantized_block), 2):
                val1 = quantized_block[j]
                val2 = quantized_block[j + 1] if j + 1 < len(quantized_block) else 0
                packed_byte = (val1 & 0xF) | ((val2 & 0xF) << 4)
                packed_block.append(packed_byte)
            
            quantized_data.extend(packed_block)
        
        return bytes(quantized_data)
    
    def _quantize_q4_1(self, tensor: np.ndarray) -> bytes:
        """Quantize to Q4_1 format (4-bit with scale and bias)."""
        block_size = self.config.block_size
        tensor_flat = tensor.flatten()
        num_blocks = (tensor_flat.shape[0] + block_size - 1) // block_size
        
        quantized_data = bytearray()
        
        for i in range(num_blocks):
            block_start = i * block_size
            block_end = min(block_start + block_size, tensor_flat.shape[0])
            block = tensor_flat[block_start:block_end]
            
            # Pad block to full size
            if len(block) < block_size:
                block = np.pad(block, (0, block_size - len(block)))
            
            # Calculate scale and bias
            min_val = np.min(block)
            max_val = np.max(block)
            scale = (max_val - min_val) / 15.0  # 4-bit: 0 to 15
            bias = min_val
            
            # Quantize to 4-bit
            quantized_block = np.clip(np.round((block - bias) / scale), 0, 15).astype(np.uint8)
            
            # Pack
            packed_block = bytearray()
            packed_block.extend(struct.pack('<ff', scale, bias))
            
            for j in range(0, len(quantized_block), 2):
                val1 = quantized_block[j]
                val2 = quantized_block[j + 1] if j + 1 < len(quantized_block) else 0
                packed_byte = (val1 & 0xF) | ((val2 & 0xF) << 4)
                packed_block.append(packed_byte)
            
            quantized_data.extend(packed_block)
        
        return bytes(quantized_data)
    
    def _quantize_q5_0(self, tensor: np.ndarray) -> bytes:
        """Quantize to Q5_0 format (5-bit with scale)."""
        block_size = self.config.block_size
        tensor_flat = tensor.flatten()
        num_blocks = (tensor_flat.shape[0] + block_size - 1) // block_size
        
        quantized_data = bytearray()
        
        for i in range(num_blocks):
            block_start = i * block_size
            block_end = min(block_start + block_size, tensor_flat.shape[0])
            block = tensor_flat[block_start:block_end]
            
            # Pad block to full size
            if len(block) < block_size:
                block = np.pad(block, (0, block_size - len(block)))
            
            # Calculate scale
            scale = np.max(np.abs(block)) / 15.0  # 5-bit: -16 to 15
            
            # Quantize to 5-bit
            quantized_block = np.clip(np.round(block / scale), -16, 15).astype(np.int16)
            
            # Pack 5-bit values into bytes
            packed_block = bytearray()
            packed_block.extend(struct.pack('<f', scale))
            
            for j in range(0, len(quantized_block), 8):  # 8 * 5-bit = 40 bits = 5 bytes
                chunk = quantized_block[j:j+8]
                if len(chunk) < 8:
                    chunk = np.pad(chunk, (0, 8 - len(chunk)))
                
                # Pack 8 5-bit values into 5 bytes
                packed_bytes = np.zeros(5, dtype=np.uint8)
                for k, val in enumerate(chunk):
                    byte_idx = k * 5 // 8
                    bit_offset = (k * 5) % 8
                    val_uint = (val + 16) & 0x1F  # Convert to unsigned 5-bit
                    
                    if bit_offset <= 3:
                        packed_bytes[byte_idx] |= val_uint << bit_offset
                    else:
                        packed_bytes[byte_idx] |= val_uint << bit_offset
                        packed_bytes[byte_idx + 1] |= val_uint >> (8 - bit_offset)
                
                packed_block.extend(packed_bytes.tobytes())
            
            quantized_data.extend(packed_block)
        
        return bytes(quantized_data)
    
    def _quantize_q5_1(self, tensor: np.ndarray) -> bytes:
        """Quantize to Q5_1 format (5-bit with scale and bias)."""
        # Similar to Q4_1 but with 5-bit precision
        return self._quantize_q5_0(tensor)  # Simplified implementation
    
    def _quantize_q6_k(self, tensor: np.ndarray) -> bytes:
        """Quantize to Q6_K format (6-bit K-quantized)."""
        # Simplified Q6_K implementation
        return self._quantize_q4_0(tensor)  # Fallback to Q4_0
    
    def _quantize_q8_k(self, tensor: np.ndarray) -> bytes:
        """Quantize to Q8_K format (8-bit K-quantized)."""
        # Simplified Q8_K implementation
        return self._quantize_q8_0(tensor)  # Fallback to Q8_0
    
    def write_file(self, filepath: str):
        """Write GGUF file.
        
        Args:
            filepath: Output file path
        """
        self.logger.info(f"Writing GGUF file to: {filepath}")
        
        # Add standard metadata
        self._add_standard_metadata()
        
        with open(filepath, 'wb') as f:
            # Write GGUF magic
            f.write(b'GGUF')
            
            # Write version
            f.write(struct.pack('<I', 3))  # Version 3
            
            # Write tensor count
            f.write(struct.pack('<Q', len(self.tensors)))
            
            # Write metadata count
            f.write(struct.pack('<Q', len(self.kv_data)))
            
            # Write metadata
            self._write_metadata(f)
            
            # Write tensors
            self._write_tensors(f)
        
        self.logger.info(f"GGUF file written successfully: {filepath}")
    
    def _add_standard_metadata(self):
        """Add standard GGUF metadata."""
        self.add_metadata("general.name", self.config.model_name)
        self.add_metadata("general.description", self.config.description)
        self.add_metadata("general.author", self.config.author)
        self.add_metadata("general.license", self.config.license)
        self.add_metadata("general.version", self.config.model_version)
        self.add_metadata("general.quantization_version", "2.0")
        self.add_metadata("general.file_type", "F16" if self.config.quantization_type == "F16" else "MOSTLY_Q4_0")
    
    def _write_metadata(self, f: BinaryIO):
        """Write metadata section."""
        for key, data in self.kv_data.items():
            # Write key
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<Q', len(key_bytes)))
            f.write(key_bytes)
            
            # Write value type
            type_map = {
                'string': 0,
                'int': 1,
                'float': 2,
                'bool': 3,
                'array': 4
            }
            f.write(struct.pack('<I', type_map.get(data['type'], 0)))
            
            # Write value
            value = data['value']
            if data['type'] == 'string':
                value_bytes = str(value).encode('utf-8')
                f.write(struct.pack('<Q', len(value_bytes)))
                f.write(value_bytes)
            elif data['type'] == 'int':
                f.write(struct.pack('<Q', int(value)))
            elif data['type'] == 'float':
                f.write(struct.pack('<f', float(value)))
            elif data['type'] == 'bool':
                f.write(struct.pack('<B', 1 if value else 0))
            elif data['type'] == 'array':
                # Write array length and data
                array_data = json.dumps(value).encode('utf-8')
                f.write(struct.pack('<Q', len(array_data)))
                f.write(array_data)
    
    def _write_tensors(self, f: BinaryIO):
        """Write tensor section."""
        # Calculate tensor data offset
        tensor_data_offset = f.tell() + sum(
            len(tensor['name'].encode('utf-8')) + 8 +  # name length + name
            8 +  # shape length
            len(tensor['shape']) * 8 +  # shape data
            4 +  # type
            8 +  # offset
            8    # size
            for tensor in self.tensors
        )
        
        # Write tensor headers
        current_offset = tensor_data_offset
        for tensor in self.tensors:
            # Write tensor name
            name_bytes = tensor['name'].encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            
            # Write shape
            f.write(struct.pack('<Q', len(tensor['shape'])))
            for dim in tensor['shape']:
                f.write(struct.pack('<Q', dim))
            
            # Write type (simplified)
            type_map = {'Q8_0': 8, 'Q4_0': 2, 'Q4_1': 3, 'Q5_0': 4, 'Q5_1': 5, 'Q6_K': 6, 'Q8_K': 7}
            f.write(struct.pack('<I', type_map.get(tensor['quantization_type'], 8)))
            
            # Write offset and size
            f.write(struct.pack('<Q', current_offset))
            f.write(struct.pack('<Q', len(tensor['data'])))
            
            current_offset += len(tensor['data'])
        
        # Write tensor data
        for tensor in self.tensors:
            f.write(tensor['data'])


class GGUFConverter:
    """Converter from PyTorch models to GGUF format."""
    
    def __init__(self, config: GGUFConfig):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.GGUFConverter")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def convert_model(self, model: nn.Module, output_path: str, model_config: Optional[Dict[str, Any]] = None):
        """Convert PyTorch model to GGUF format.
        
        Args:
            model: PyTorch model
            output_path: Output GGUF file path
            model_config: Optional model configuration
        """
        self.logger.info(f"Converting model to GGUF format: {output_path}")
        
        writer = GGUFWriter(self.config)
        
        # Add model configuration
        if model_config:
            for key, value in model_config.items():
                writer.add_metadata(f"llama.{key}", value)
        
        # Convert model parameters
        self._convert_parameters(model, writer)
        
        # Write file
        writer.write_file(output_path)
        
        self.logger.info("Model conversion completed")
    
    def _convert_parameters(self, model: nn.Module, writer: GGUFWriter):
        """Convert model parameters to GGUF tensors."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Skip gradients, only save weights
                continue
            
            # Map parameter names to GGUF format
            gguf_name = self._map_parameter_name(name)
            writer.add_tensor(gguf_name, param.data)
            
            self.logger.debug(f"Converted parameter: {name} -> {gguf_name}")
    
    def _map_parameter_name(self, name: str) -> str:
        """Map PyTorch parameter name to GGUF format."""
        # Common mappings for different model architectures
        mappings = {
            'transformer.wte.weight': 'token_embd.weight',
            'transformer.wpe.weight': 'position_embd.weight',
            'transformer.ln_f.weight': 'output_norm.weight',
            'transformer.ln_f.bias': 'output_norm.bias',
            'lm_head.weight': 'output.weight',
        }
        
        # Handle transformer blocks
        if 'transformer.h.' in name:
            # Extract layer number and component
            parts = name.split('.')
            layer_idx = parts[2]
            component = '.'.join(parts[3:])
            
            # Map components
            component_map = {
                'ln_1.weight': f'blk.{layer_idx}.attn_norm.weight',
                'ln_1.bias': f'blk.{layer_idx}.attn_norm.bias',
                'attn.c_attn.weight': f'blk.{layer_idx}.attn_qkv.weight',
                'attn.c_attn.bias': f'blk.{layer_idx}.attn_qkv.bias',
                'attn.c_proj.weight': f'blk.{layer_idx}.attn_output.weight',
                'attn.c_proj.bias': f'blk.{layer_idx}.attn_output.bias',
                'ln_2.weight': f'blk.{layer_idx}.ffn_norm.weight',
                'ln_2.bias': f'blk.{layer_idx}.ffn_norm.bias',
                'mlp.c_fc.weight': f'blk.{layer_idx}.ffn_gate.weight',
                'mlp.c_fc.bias': f'blk.{layer_idx}.ffn_gate.bias',
                'mlp.c_proj.weight': f'blk.{layer_idx}.ffn_down.weight',
                'mlp.c_proj.bias': f'blk.{layer_idx}.ffn_down.bias',
            }
            
            return component_map.get(component, name)
        
        return mappings.get(name, name)


def convert_to_gguf(
    model: nn.Module,
    output_path: str,
    quantization_type: str = "Q8_0",
    model_config: Optional[Dict[str, Any]] = None,
    **config_kwargs
) -> str:
    """Convenience function to convert model to GGUF format.
    
    Args:
        model: PyTorch model
        output_path: Output file path
        quantization_type: GGUF quantization type
        model_config: Model configuration
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Output file path
    """
    config = GGUFConfig(quantization_type=quantization_type, **config_kwargs)
    converter = GGUFConverter(config)
    converter.convert_model(model, output_path, model_config)
    return output_path
