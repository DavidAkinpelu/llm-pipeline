"""Model registry for supported architectures and their configurations."""

from typing import List, Dict, Any, Optional
import re


class ModelRegistry:
    """Registry for model architectures and their target modules"""
    
    SUPPORTED_MODELS = {
        "llama": {
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "modules_to_save": ["embed_tokens", "lm_head"],
            "pattern": r"llama|alpaca",
            "architecture_type": "decoder_only"
        },
        "mistral": {
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "modules_to_save": ["embed_tokens", "lm_head"],
            "pattern": r"mistral|mixtral",
            "architecture_type": "decoder_only"
        },
        "qwen": {
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "modules_to_save": ["embed_tokens", "lm_head"],
            "pattern": r"qwen",
            "architecture_type": "decoder_only"
        },
        "gemma": {
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "modules_to_save": ["embed_tokens", "lm_head"],
            "pattern": r"gemma",
            "architecture_type": "decoder_only"
        },
        "phi": {
            "target_modules": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
            "modules_to_save": ["embed_tokens", "lm_head"],
            "pattern": r"phi",
            "architecture_type": "decoder_only"
        },
        "bert": {
            "target_modules": ["query", "key", "value", "dense"],
            "modules_to_save": ["embeddings", "classifier", "pooler"],
            "pattern": r"bert",
            "architecture_type": "encoder_only"
        },
        "roberta": {
            "target_modules": ["query", "key", "value", "dense"],
            "modules_to_save": ["embeddings", "classifier", "pooler"],
            "pattern": r"roberta",
            "architecture_type": "encoder_only"
        },
        "distilbert": {
            "target_modules": ["q_lin", "k_lin", "v_lin", "out_lin", "ffn.lin1", "ffn.lin2"],
            "modules_to_save": ["embeddings", "classifier"],
            "pattern": r"distilbert",
            "architecture_type": "encoder_only"
        },
        "gpt2": {
            "target_modules": ["c_attn", "c_proj", "c_fc"],
            "modules_to_save": ["wte", "wpe", "lm_head"],
            "pattern": r"gpt2",
            "architecture_type": "decoder_only"
        },
        "gpt_neox": {
            "target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "modules_to_save": ["embed_in", "embed_out"],
            "pattern": r"gpt_neox|pythia",
            "architecture_type": "decoder_only"
        },
        "t5": {
            "target_modules": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
            "modules_to_save": ["shared", "lm_head"],
            "pattern": r"t5|flan",
            "architecture_type": "encoder_decoder"
        },
        "bart": {
            "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            "modules_to_save": ["shared", "lm_head"],
            "pattern": r"bart",
            "architecture_type": "encoder_decoder"
        }
    }

    @classmethod
    def detect_model_type(cls, model_name_or_class: str) -> Optional[str]:
        """Auto-detect model type from model name or class name."""
        model_str = model_name_or_class.lower()
        
        # Check patterns in order of specificity (more specific first)
        # This prevents "roberta" from matching "bert" pattern
        ordered_models = [
            ("roberta", cls.SUPPORTED_MODELS["roberta"]),
            ("bert", cls.SUPPORTED_MODELS["bert"]),
            ("distilbert", cls.SUPPORTED_MODELS["distilbert"]),
            ("mixtral", cls.SUPPORTED_MODELS["mistral"]),  # mixtral is mistral-based
            ("llama", cls.SUPPORTED_MODELS["llama"]),  # llama before alpaca
            ("alpaca", cls.SUPPORTED_MODELS["llama"]),  # alpaca is llama-based
            ("mistral", cls.SUPPORTED_MODELS["mistral"]),
            ("qwen", cls.SUPPORTED_MODELS["qwen"]),
            ("gemma", cls.SUPPORTED_MODELS["gemma"]),
            ("phi", cls.SUPPORTED_MODELS["phi"]),
            ("t5", cls.SUPPORTED_MODELS["t5"]),
            ("bart", cls.SUPPORTED_MODELS["bart"]),
        ]
        
        # Add any dynamically registered models to the end
        for model_type, config in cls.SUPPORTED_MODELS.items():
            if model_type not in [m[0] for m in ordered_models]:
                ordered_models.append((model_type, config))
        
        for model_type, config in ordered_models:
            if model_type in cls.SUPPORTED_MODELS: 
                pattern = config["pattern"]
                if re.search(pattern, model_str):
                    return model_type
        
        return None

    @classmethod
    def get_target_modules(cls, model_type: str) -> List[str]:
        """Get target modules for a model type"""
        model_type = model_type.lower()
        
        if model_type in cls.SUPPORTED_MODELS:
            return cls.SUPPORTED_MODELS[model_type]["target_modules"]
        
        # Pattern matching fallback
        detected_type = cls.detect_model_type(model_type)
        if detected_type:
            return cls.SUPPORTED_MODELS[detected_type]["target_modules"]
        
        return ["q_proj", "v_proj"]
    
    @classmethod
    def get_modules_to_save(cls, model_type: str) -> List[str]:
        """Get modules that should be saved (not adapted)"""
        model_type = model_type.lower()
        
        if model_type in cls.SUPPORTED_MODELS:
            return cls.SUPPORTED_MODELS[model_type]["modules_to_save"]
        
        # Pattern matching fallback
        detected_type = cls.detect_model_type(model_type)
        if detected_type:
            return cls.SUPPORTED_MODELS[detected_type]["modules_to_save"]
        
        return []
    
    @classmethod
    def get_architecture_type(cls, model_type: str) -> str:
        """Get architecture type (encoder_only, decoder_only, encoder_decoder)"""
        model_type = model_type.lower()
        
        if model_type in cls.SUPPORTED_MODELS:
            return cls.SUPPORTED_MODELS[model_type]["architecture_type"]
        
        detected_type = cls.detect_model_type(model_type)
        if detected_type:
            return cls.SUPPORTED_MODELS[detected_type]["architecture_type"]
        
        raise ValueError(f"Unknown model type: {model_type}")
    
    @classmethod
    def register_model(
        cls, 
        model_name: str, 
        target_modules: List[str],
        modules_to_save: Optional[List[str]] = None,
        pattern: Optional[str] = None,
        architecture_type: str = "decoder_only"
    ):
        """Register a new model architecture"""
        cls.SUPPORTED_MODELS[model_name] = {
            "target_modules": target_modules,
            "modules_to_save": modules_to_save or [],
            "pattern": pattern or model_name,
            "architecture_type": architecture_type
        }
    
    @classmethod
    def list_supported_models(cls) -> List[str]:
        """Get list of all supported model types"""
        return list(cls.SUPPORTED_MODELS.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """Get complete configuration for a model type"""
        model_type = model_type.lower()
        
        # Direct match first
        if model_type in cls.SUPPORTED_MODELS:
            return cls.SUPPORTED_MODELS[model_type].copy()
        
        # Pattern matching fallback
        detected_type = cls.detect_model_type(model_type)
        if detected_type:
            return cls.SUPPORTED_MODELS[detected_type].copy()
        
        # Return default config for unknown models
        return {
            "target_modules": ["q_proj", "v_proj"],
            "modules_to_save": [],
            "pattern": model_type,
            "architecture_type": "decoder_only"
        }