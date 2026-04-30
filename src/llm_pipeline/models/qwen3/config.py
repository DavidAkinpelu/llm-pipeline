"""Qwen3 model configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Qwen3Config:
    """Configuration for Qwen3 model."""
    
    # Model architecture parameters
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    intermediate_size: int = 3072
    max_position_embeddings: int = 32768
    head_dim: int = 64
    
    # Model type
    model_type: str = "qwen3"
    architecture_type: str = "decoder_only"
    
    # Activation and normalization
    hidden_act: str = "silu"
    norm_epsilon: float = 1e-6
    
    # RoPE parameters
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    
    # Attention parameters
    attention_dropout: float = 0.0
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window_size: int = 4096
    
    # Generation parameters
    pad_token_id: int = 151643
    eos_token_id: int = 151643
    bos_token_id: int = 151643
    
    # Additional parameters
    tie_word_embeddings: bool = False
    use_cache: bool = True
    torch_dtype: str = "float16"
    
    @classmethod
    def from_huggingface_config(cls, config):
        """Create Qwen3Config from HuggingFace config.

        Tolerant to transformers >=5.0 field renames (e.g. ``rope_theta`` →
        ``default_theta``, ``torch_dtype`` → ``dtype``).
        """
        def _first(*names, default=None):
            for n in names:
                if hasattr(config, n):
                    v = getattr(config, n)
                    if v is not None:
                        return v
            return default

        return cls(
            vocab_size=_first('vocab_size', default=32000),
            hidden_size=_first('hidden_size', default=768),
            num_layers=_first('num_hidden_layers', 'num_layers', default=28),
            num_attention_heads=_first('num_attention_heads', default=16),
            num_key_value_heads=_first('num_key_value_heads', default=8),
            intermediate_size=_first('intermediate_size', default=3072),
            max_position_embeddings=_first('max_position_embeddings', default=32768),
            head_dim=_first('head_dim', default=64),
            model_type=_first('model_type', default='qwen3'),
            hidden_act=_first('hidden_act', default='silu'),
            norm_epsilon=_first('rms_norm_eps', 'norm_epsilon', default=1e-6),
            attention_bias=_first('attention_bias', default=False),
            rope_theta=_first('rope_theta', 'default_theta', default=10000.0),
            rope_scaling=_first('rope_scaling', default=None),
            attention_dropout=_first('attention_dropout', default=0.0),
            use_sliding_window=_first('use_sliding_window', default=False),
            sliding_window_size=_first('sliding_window', 'sliding_window_size', default=4096),
            pad_token_id=_first('pad_token_id', default=151643),
            eos_token_id=_first('eos_token_id', default=151643),
            bos_token_id=_first('bos_token_id', default=151643),
            tie_word_embeddings=_first('tie_word_embeddings', default=False),
            use_cache=_first('use_cache', default=True),
            torch_dtype=str(_first('dtype', 'torch_dtype', default='float16')),
        )

    # Compatibility properties for components expecting HF-style names
    @property
    def rms_norm_eps(self) -> float:
        return self.norm_epsilon

    @property
    def num_hidden_layers(self) -> int:
        return self.num_layers
