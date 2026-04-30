"""Factory for building Qwen3 models and tokenizer without accelerate."""

import re
from typing import Tuple, Optional
import torch
from transformers import AutoConfig, AutoTokenizer
import os
from pathlib import Path

from .config import Qwen3Config
from .custom_loader import load_model
from .custom_builder import Qwen3ForCausalLM


# Hugging Face repo id: ``namespace/name`` with restricted character set, no
# leading slash, no path traversal.
_HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _resolve_model_path(model_path: str) -> str:
    """Return a local directory path. Downloads from the Hub if needed."""
    p = Path(model_path)
    if p.exists():
        return str(p)
    # Looks like an HF id and isn't a local path -> snapshot-download it.
    if _HF_ID_RE.match(model_path):
        from huggingface_hub import snapshot_download
        return str(snapshot_download(model_path))
    return str(p)


def _parameter_counts(
    *,
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    intermediate_size: int,
    head_dim: int,
    tie_word_embeddings: bool,
    attention_bias: bool = False,
) -> dict:
    q_proj_out = num_attention_heads * head_dim
    kv_proj_out = num_key_value_heads * head_dim

    embed_params = vocab_size * hidden_size
    attention_params = (
        hidden_size * q_proj_out
        + hidden_size * kv_proj_out * 2
        + q_proj_out * hidden_size
    )
    if attention_bias:
        attention_params += q_proj_out + kv_proj_out * 2 + hidden_size

    qk_norm_params = head_dim * 2
    mlp_params = hidden_size * intermediate_size * 3
    layer_norm_params = hidden_size * 2
    params_per_layer = attention_params + qk_norm_params + mlp_params + layer_norm_params

    final_norm_params = hidden_size
    lm_head_params = 0 if tie_word_embeddings else hidden_size * vocab_size
    total_params = embed_params + (params_per_layer * num_layers) + final_norm_params + lm_head_params

    return {
        "embedding_parameters": embed_params,
        "attention_parameters_per_layer": attention_params,
        "qk_norm_parameters_per_layer": qk_norm_params,
        "mlp_parameters_per_layer": mlp_params,
        "layer_norm_parameters_per_layer": layer_norm_params,
        "parameters_per_layer": params_per_layer,
        "final_norm_parameters": final_norm_params,
        "lm_head_parameters": lm_head_params,
        "total_parameters": total_params,
    }


class Qwen3Factory:
    """Factory for Qwen3 models and tokenizer."""
    
    @staticmethod
    def create_model_and_tokenizer(
        model_path: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Qwen3ForCausalLM, AutoTokenizer]:
        """Create Qwen3 model and tokenizer and load weights from disk."""

        model_path = _resolve_model_path(model_path)

        # Auto-select device/dtype if not provided
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        hf_config = AutoConfig.from_pretrained(model_path)
        config = Qwen3Config.from_huggingface_config(hf_config)
        model = Qwen3ForCausalLM(config)
        load_model(model, model_path)
        model = model.to(device=device, dtype=dtype)
        return model, tokenizer
    
    @staticmethod
    def create_model_only(
        model_path: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Qwen3ForCausalLM:
        """Create only the Qwen3 model and load weights from disk."""

        model_path = _resolve_model_path(model_path)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        hf_config = AutoConfig.from_pretrained(model_path)
        config = Qwen3Config.from_huggingface_config(hf_config)
        model = Qwen3ForCausalLM(config)
        load_model(model, model_path)
        model = model.to(device=device, dtype=dtype)
        return model
    
    @staticmethod
    def create_tokenizer_only(model_path: str) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(model_path)
    
    @staticmethod
    def get_model_info(model_path: str) -> dict:
        model_path = _resolve_model_path(model_path)

        config = AutoConfig.from_pretrained(model_path)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        attention_bias = getattr(config, "attention_bias", False)
        counts = _parameter_counts(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            head_dim=head_dim,
            tie_word_embeddings=tie_word_embeddings,
            attention_bias=attention_bias,
        )
        return {
            "model_type": getattr(config, "model_type", "qwen3"),
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "intermediate_size": config.intermediate_size,
            "max_position_embeddings": config.max_position_embeddings,
            "head_dim": head_dim,
            "torch_dtype": str(config.torch_dtype),
            "use_cache": True,
            "tie_word_embeddings": tie_word_embeddings,
            "attention_bias": attention_bias,
            **counts,
        }
    
    @staticmethod
    def estimate_memory_usage(model_path: str, dtype: torch.dtype = torch.float16) -> dict:
        info = Qwen3Factory.get_model_info(model_path)
        if "total_parameters" not in info:
            counts = _parameter_counts(
                vocab_size=info["vocab_size"],
                hidden_size=info["hidden_size"],
                num_layers=info["num_layers"],
                num_attention_heads=info["num_attention_heads"],
                num_key_value_heads=info["num_key_value_heads"],
                intermediate_size=info["intermediate_size"],
                head_dim=info["head_dim"],
                tie_word_embeddings=info.get("tie_word_embeddings", False),
                attention_bias=info.get("attention_bias", False),
            )
            info = {**info, **counts}
        if dtype == torch.float16 or dtype == torch.bfloat16:
            bytes_per_param = 2
        elif dtype == torch.float32:
            bytes_per_param = 4
        else:
            bytes_per_param = torch.tensor([], dtype=dtype).element_size()
        memory_mb = (info["total_parameters"] * bytes_per_param) / (1024 * 1024)
        memory_gb = memory_mb / 1024
        # Rule of thumb: weights + activations + KV cache + framework overhead.
        # 1.5x model memory is a conservative floor for inference; training is larger.
        recommended_gpu_memory_gb = max(memory_gb * 1.5, memory_gb + 1.0)
        return {
            "total_parameters": info["total_parameters"],
            "embedding_parameters": info["embedding_parameters"],
            "attention_parameters_per_layer": info["attention_parameters_per_layer"],
            "qk_norm_parameters_per_layer": info["qk_norm_parameters_per_layer"],
            "mlp_parameters_per_layer": info["mlp_parameters_per_layer"],
            "layer_norm_parameters_per_layer": info["layer_norm_parameters_per_layer"],
            "parameters_per_layer": info["parameters_per_layer"],
            "final_norm_parameters": info["final_norm_parameters"],
            "lm_head_parameters": info["lm_head_parameters"],
            "memory_mb": memory_mb,
            "memory_gb": memory_gb,
            "model_memory_mb": memory_mb,
            "model_memory_gb": memory_gb,
            "bytes_per_param": bytes_per_param,
            "dtype": str(dtype),
            "recommended_gpu_memory_gb": recommended_gpu_memory_gb,
        }
