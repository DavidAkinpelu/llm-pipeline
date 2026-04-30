"""Qwen3 model builder (no accelerate dependency)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class CausalLMOutput:
    """Output class for causal language model"""
    logits: torch.Tensor
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


def _create_causal_mask(input_shape: tuple, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create additive causal mask"""
    bsz, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask_cond = torch.arange(tgt_len, device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(tgt_len, 1), 0)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


class Qwen3Config:
    """Lightweight Qwen3 config without transformers dependency."""
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, "hidden_act"):
            self.hidden_act = "silu"
        if not hasattr(self, "rms_norm_eps"):
            self.rms_norm_eps = 1e-6
        if not hasattr(self, "attention_bias"):
            self.attention_bias = False
        if not hasattr(self, "rope_theta"):
            self.rope_theta = 1000000


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class Qwen3Attention(nn.Module):
    """Qwen3 attention module."""

    def __init__(self, config: Qwen3Config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        
        # Projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        
        # Q/K normalization layers (Qwen3 specific)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod  
    def _apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # cos/sin unsqueeze at dim=1 for [B, H, T, D] broadcasting
        cos = cos.unsqueeze(1)  # [B, 1, T, D]  
        sin = sin.unsqueeze(1)  # [B, 1, T, D]
        
        q_embed = (q * cos) + (Qwen3Attention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Qwen3Attention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads"""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass"""
        input_shape = hidden_states.shape[:-1]  # [B, T]
        hidden_shape = (*input_shape, -1, self.head_dim)  # [B, T, -1, head_dim]
        
        # Step 1: Project to Q, K, V and reshape - HF order
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B, H, T, D]
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # [B, H_kv, T, D]
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B, H_kv, T, D]
        
        # Step 1b: Apply Q/K normalization per head - HF order
        query_states = self.q_norm(query_states)  # [B, H, T, D]
        key_states = self.k_norm(key_states)      # [B, H_kv, T, D]
        
        # Step 2: Apply rotary position embeddings - HF order
        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Step 3: Handle past key values (cache)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        present_key_value = (key_states, value_states) if use_cache else None
        
        # Step 4: Repeat K/V for GQA - HF method
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Step 5: Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        
        # Step 6: Apply attention mask (additive) and causal mask
        batch_size, num_heads, seq_len, kv_len = attn_weights.shape
        
        # Build causal mask aware of past length when using cache
        past_len = kv_len - seq_len
        q_pos = past_len + torch.arange(seq_len, device=attn_weights.device).view(1, 1, seq_len, 1)
        k_pos = torch.arange(kv_len, device=attn_weights.device).view(1, 1, 1, kv_len)
        causal_mask = (k_pos > q_pos)
        min_val = torch.finfo(attn_weights.dtype).min
        causal_mask = causal_mask.to(attn_weights.dtype) * min_val  # [1,1,seq_len,kv_len]
        attn_weights = attn_weights + causal_mask
        
        # Apply additional padding mask if provided (support 2D and 4D)
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                # Already 4D additive mask. Align key length.
                additional_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + additional_mask
            elif attention_mask.dim() == 2:
                # Convert 2D padding mask (1=attend, 0=pad) to additive mask
                padding_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
                padding_mask = padding_mask * torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights + padding_mask
            else:
                raise ValueError(
                    f"Unsupported attention_mask rank: {attention_mask.dim()}D. Expected 2D or 4D."
                )
        
        # Step 7: Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Step 8: Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Step 9: Reshape and project output
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, present_key_value


class Qwen3MLP(nn.Module):
    """Qwen3 MLP module (Gemma-style)."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 decoder layer."""

    def __init__(self, config: Qwen3Config, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, _, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class Qwen3RotaryEmbedding(nn.Module):
    """Qwen3 Rotary Position Embedding."""
    
    def __init__(self, config, device=None):
        super().__init__()
        self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        
        # Initialize inv_freq
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self.attention_scaling = 1.0
        
    @torch.no_grad()
    def forward(self, x, position_ids):
        """Forward pass"""
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3Model(nn.Module):
    """Qwen3 model."""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        # nn.Embedding asserts padding_idx < vocab_size; fall back to None when
        # the configured pad token is outside the (possibly downscaled-for-tests) vocab.
        embed_pad = self.padding_idx if (self.padding_idx is not None and self.padding_idx < self.vocab_size) else None
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, embed_pad)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        # Input embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Cache position handling like HF
        if cache_position is None:
            if past_key_values is not None and len(past_key_values) > 0:
                # Get the sequence length from the first layer's key cache
                past_seen_tokens = past_key_values[0][0].shape[2]  # K cache shape: [B, H, T, D]
            else:
                past_seen_tokens = 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        # Note: attention_mask handling is done in the attention layer
        
        hidden_states = inputs_embeds
        
        # Create position embeddings to be shared across decoder layers - HF style
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Handle past key values
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        next_decoder_cache = () if use_cache else None
        
        # Pass through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values else None
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                **kwargs,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)

        outputs = {"last_hidden_state": hidden_states}
        if use_cache:
            outputs["past_key_values"] = next_decoder_cache
        return outputs


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 model for causal language modeling."""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutput:
        """Forward pass."""
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = model_outputs["last_hidden_state"]
        next_cache = model_outputs.get("past_key_values")

        logits = self.lm_head(hidden_states)

        return CausalLMOutput(
            logits=logits,
            past_key_values=next_cache,
            hidden_states=hidden_states
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate text using centralized Sampler for token selection."""
        from llm_pipeline.inference.sampling import Sampler, SamplingConfig

        self.eval()
        with torch.no_grad():
            device = input_ids.device
            batch_size = input_ids.size(0)

            # Initialize sequences and masks
            generated_ids = input_ids.clone()
            attention_mask = torch.ones_like(generated_ids, device=device)

            # Initialize sampler
            sampling_config = SamplingConfig(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.0,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            sampler = Sampler(sampling_config)

            past_key_values = None
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_new_tokens):
                if past_key_values is None:
                    outputs = self(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                        past_key_values=None,
                        use_cache=True,
                    )
                else:
                    # Decode only the last token
                    last_token = generated_ids[:, -1:].contiguous()
                    outputs = self(
                        input_ids=last_token,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                logits = outputs.logits
                past_key_values = outputs.past_key_values

                next_token_logits = logits[:, -1, :]

                if do_sample:
                    next_token = sampler.sample_token(next_token_logits)
                    next_token = next_token.view(-1, 1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                if eos_token_id is not None and torch.any(finished):
                    fill_token_id = pad_token_id if pad_token_id is not None else eos_token_id
                    fill_tokens = torch.full_like(next_token, fill_token_id)
                    next_token = torch.where(finished.unsqueeze(-1), fill_tokens, next_token)

                # Append token and extend mask
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
                ], dim=-1)

                # EOS handling (stop only those sequences that reached EOS)
                if eos_token_id is not None:
                    finished = finished | (next_token.squeeze(-1) == eos_token_id)
                    if torch.all(finished):
                        break

            return generated_ids


def build_qwen3_model_from_weights(
    weights: Dict[str, torch.Tensor], 
    config: Qwen3Config
) -> Qwen3ForCausalLM:
    """
    Build Qwen3 model from loaded weights - nano-vLLM style.
    
    Args:
        weights: Dictionary of model weights
        config: Model configuration
        
    Returns:
        Initialized Qwen3 model
    """
    # Determine device from first weight
    device = next(iter(weights.values())).device
    dtype = next(iter(weights.values())).dtype
    
    model = Qwen3ForCausalLM(config)
    
    # Move model to the same device and dtype as weights
    model = model.to(device=device, dtype=dtype)
    
    # Load weights into model
    model_dict = model.state_dict()
    
    for name, param in weights.items():
        if name in model_dict:
            if param.shape == model_dict[name].shape:
                model_dict[name] = param.to(device=device, dtype=dtype)
            else:
                print(f"Warning: Shape mismatch for {name}: {param.shape} vs {model_dict[name].shape}")
        else:
            print(f"Warning: Parameter {name} not found in model")
            
    model.load_state_dict(model_dict, strict=False)
    
    return model
