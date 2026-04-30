"""Flash Attention with automatic fallback to standard attention."""

import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AttentionConfig


class StandardAttentionLayer(nn.Module):
    """Standard PyTorch attention implementation."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standard attention forward pass.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len, head_dim]
            attention_mask: Attention mask
            past_key_values: Past key-value states
            
        Returns:
            Attention output and updated key-value states
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / (head_dim ** 0.5)
        
        # Apply attention mask if provided (supports 2D or 4D additive masks)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Convert padding mask (1=keep, 0=mask) to additive mask and broadcast
                padding_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2)
                padding_mask = padding_mask.to(attention_scores.dtype) * torch.finfo(attention_scores.dtype).min
                attention_scores = attention_scores + padding_mask
            elif attention_mask.dim() == 4:
                attention_scores = attention_scores + attention_mask.to(attention_scores.dtype)
            else:
                # Fallback: try to broadcast as-is
                attention_scores = attention_scores + attention_mask.to(attention_scores.dtype)
        
        if self.config.scale_attention_softmax_in_fp32:
            attention_scores = attention_scores.float()
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        if self.config.scale_attention_softmax_in_fp32:
            attention_probs = attention_probs.type_as(query)
        
        if self.config.attention_dropout > 0:
            attention_probs = F.dropout(attention_probs, p=self.config.attention_dropout, training=self.training)
        
        attention_output = torch.matmul(attention_probs, value)
        
        new_key_values = (key, value)
        
        return attention_output, new_key_values


class FlashAttentionLayer(nn.Module):
    """
    Flash Attention implementation with automatic fallback.
    
    Automatically detects if Flash Attention is available and falls back
    to standard attention if not.
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.use_flash_attn = self._check_flash_attn_availability()
        self.fallback_layer = StandardAttentionLayer(config)
        
        if not self.use_flash_attn:
            warnings.warn(
                "Flash Attention not available, using standard attention. "
                "Install flash-attn for better performance: pip install flash-attn",
                UserWarning
            )
    
    def _check_flash_attn_availability(self) -> bool:
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Flash Attention forward pass with automatic fallback.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len, head_dim]
            attention_mask: Attention mask
            past_key_values: Past key-value states
            
        Returns:
            Attention output and updated key-value states
        """
        if self.use_flash_attn:
            return self._flash_attention_forward(query, key, value, attention_mask, past_key_values)
        else:
            return self.fallback_layer.forward(query, key, value, attention_mask, past_key_values)
    
    def _flash_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Flash Attention forward pass.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len, head_dim]
            attention_mask: Attention mask
            past_key_values: Past key-value states
            
        Returns:
            Attention output and updated key-value states
        """
        try:
            from flash_attn import flash_attn_func
            
            if not query.is_cuda:
                raise RuntimeError("Flash Attention requires CUDA")
            
            if query.dtype not in [torch.float16, torch.bfloat16]:
                raise RuntimeError(f"Flash Attention only supports fp16 and bf16, got {query.dtype}")
            
            batch_size, num_heads, seq_len, head_dim = query.shape
            
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            
            attention_output = flash_attn_func(
                q, k, v,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                softmax_scale=1.0 / (head_dim ** 0.5),
                causal=True,  # Assuming causal attention
            )
            
            attention_output = attention_output.transpose(1, 2)
            
            new_key_values = (key, value)
            
            return attention_output, new_key_values
            
        except Exception as e:
            # Silent fallback for expected conditions (CPU, unsupported dtype)
            if "CUDA" in str(e) or "data type" in str(e):
                pass  # Expected fallback, no warning needed
            else:
                warnings.warn(f"Flash Attention failed, falling back to standard attention: {e}")
            return self.fallback_layer.forward(query, key, value, attention_mask, past_key_values)


def get_attention_layer(config: AttentionConfig) -> Union[FlashAttentionLayer, StandardAttentionLayer]:
    """
    Get the best available attention layer.
    
    Args:
        config: Attention configuration
        
    Returns:
        Flash Attention layer if available, otherwise standard attention layer
    """
    if config.use_flash_attention:
        return FlashAttentionLayer(config)
    else:
        return StandardAttentionLayer(config)
