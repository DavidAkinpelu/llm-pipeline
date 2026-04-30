"""Model architecture registry.

Maps a model class name (or HF id substring) to its LoRA target-module
fingerprint. New architectures register by adding an entry to ``_REGISTRY``.

Most decoder-only LLMs share the Llama-style fingerprint (q_proj, k_proj,
v_proj, o_proj, gate_proj, up_proj, down_proj); the ``llama_like`` fallback
covers them.
"""

from typing import Any, Dict, List, Optional
import re


_LLAMA_LIKE = {
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "modules_to_save": ["embed_tokens", "lm_head"],
    "architecture_type": "decoder_only",
}

_BERT_LIKE = {
    "target_modules": ["query", "key", "value", "dense"],
    "modules_to_save": ["embeddings", "cls"],
    "architecture_type": "encoder_only",
}

# Qwen3.5/3.6 hybrid stack (Gated DeltaNet linear-attention + full attention)
# uses Llama-style q/k/v/o on the full-attention layers, and the
# in_proj_qkv / in_proj_z / in_proj_b / in_proj_a / out_proj fingerprint on
# the Gated-DeltaNet linear-attention layers. See ``models/qwen3_5/`` for
# the detailed architecture notes.
_QWEN3_5_LIKE = {
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",                  # full-attention layers
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",    # linear-attention input projs
        "out_proj",                                               # linear-attention output proj
        "gate_proj", "up_proj", "down_proj",                     # dense MLP / per-expert MLP
    ],
    "modules_to_save": ["embed_tokens", "lm_head"],
    "architecture_type": "decoder_only_hybrid",
}


# Order matters: ``detect_model_type`` returns the *first* matching pattern,
# so put more-specific patterns (e.g. qwen3_5_moe) before their broader
# counterparts (qwen3_5, qwen3).
_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen3_5_moe":   {**_QWEN3_5_LIKE, "pattern": r"qwen3[_\.\-]?(5|6).*(moe|a\d+b)"},
    "qwen3_5":       {**_QWEN3_5_LIKE, "pattern": r"qwen3[_\.\-]?(5|6)\b"},
    "qwen3":         {**_LLAMA_LIKE,    "pattern": r"qwen3(?!_5|\.5|\.6|-?5|-?6)"},
    "qwen2":         {**_LLAMA_LIKE,    "pattern": r"qwen2"},
    "mixtral":       {**_LLAMA_LIKE,    "pattern": r"mixtral"},
    "mistral":       {**_LLAMA_LIKE,    "pattern": r"mistral"},
    "llama":         {**_LLAMA_LIKE,    "pattern": r"llama"},
    "gemma":         {**_LLAMA_LIKE,    "pattern": r"gemma"},
    "phi":           {**_LLAMA_LIKE,    "pattern": r"\bphi\b"},
    "roberta":       {**_BERT_LIKE,     "pattern": r"roberta"},
    "bert":          {**_BERT_LIKE,     "pattern": r"bert"},
}


class ModelRegistry:
    """Registry of supported model architectures."""

    QWEN3_CONFIG = _REGISTRY["qwen3"]  # backwards compat (used by some tests)

    @classmethod
    def detect_model_type(cls, model_name_or_class: str) -> Optional[str]:
        s = model_name_or_class.lower()
        for name, cfg in _REGISTRY.items():
            if re.search(cfg["pattern"], s):
                return name
        return None

    @classmethod
    def get_target_modules(cls, model_type: str = "qwen3") -> List[str]:
        cfg = _REGISTRY.get(model_type, _LLAMA_LIKE)
        return list(cfg["target_modules"])

    @classmethod
    def get_modules_to_save(cls, model_type: str = "qwen3") -> List[str]:
        cfg = _REGISTRY.get(model_type, _LLAMA_LIKE)
        return list(cfg["modules_to_save"])

    @classmethod
    def get_architecture_type(cls, model_type: str = "qwen3") -> str:
        cfg = _REGISTRY.get(model_type, _LLAMA_LIKE)
        return cfg["architecture_type"]

    @classmethod
    def list_supported_models(cls) -> List[str]:
        return list(_REGISTRY.keys())

    @classmethod
    def get_model_info(cls, model_type: str = "qwen3") -> Dict[str, Any]:
        cfg = _REGISTRY.get(model_type, _LLAMA_LIKE)
        return dict(cfg)

    @classmethod
    def register(cls, name: str, config: Dict[str, Any]) -> None:
        """Register a custom architecture at runtime."""
        required = {"target_modules", "modules_to_save", "pattern", "architecture_type"}
        missing = required - config.keys()
        if missing:
            raise ValueError(f"Missing keys for architecture {name}: {missing}")
        _REGISTRY[name] = dict(config)
