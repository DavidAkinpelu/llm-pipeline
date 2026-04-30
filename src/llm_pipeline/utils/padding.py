"""Utilities for determining appropriate padding configuration based on model architecture."""

from typing import Literal


def get_appropriate_padding_side(model_config) -> Literal["left", "right"]:
    """Determine the appropriate padding side based on model architecture.
    
    Args:
        model_config: HuggingFace model configuration object
        
    Returns:
        Appropriate padding side: "left" for decoder-only models, "right" for others
        
    Raises:
        ValueError: If model configuration is invalid
    """
    if not hasattr(model_config, 'model_type'):
        # Default to right padding if model type is not available
        return "right"
    
    model_type = model_config.model_type.lower()
    
    # Decoder-only models (GPT, LLaMA, etc.) typically use left padding
    # This is because they generate text from left to right, and we want
    # the actual prompt to be at the end (right side) for generation
    decoder_only_models = {
        'gpt', 'gpt2', 'gpt_neo', 'gpt_neox', 'gpt_j', 'gpt_codegen',
        'llama', 'llama2', 'llama3', 'mistral', 'mixtral', 'qwen', 
        'qwen2', 'phi', 'phi2', 'phi3', 'gemma', 'gemma2',
        'falcon', 'mpt', 'bloom', 'opt', 'xglm', 'codegen',
        'dolly', 'stablelm', 'redpajama', 'openllama'
    }
    
    # Encoder-decoder models (T5, BART, etc.) typically use right padding
    encoder_decoder_models = {
        't5', 'mt5', 'ul2', 'bart', 'pegasus', 'marian', 'mbart',
        'blenderbot', 'blenderbot_small', 'prophetnet', 'xlm_prophetnet'
    }
    
    # Encoder-only models (BERT, RoBERTa, etc.) typically use right padding
    encoder_only_models = {
        'bert', 'roberta', 'distilbert', 'electra', 'albert',
        'xlm_roberta', 'camembert', 'flaubert', 'longformer',
        'reformer', 'convbert', 'squeezebert', 'mobilebert',
        'deberta', 'deberta_v2', 'funnel', 'mpnet', 'nystromformer'
    }
    
    if model_type in decoder_only_models:
        return "left"
    elif model_type in encoder_decoder_models:
        return "right"
    elif model_type in encoder_only_models:
        return "right"
    else:
        # Default to right padding for unknown model types
        return "right"


def configure_tokenizer_padding(tokenizer, model_config, force_padding_side: str = None):
    """Configure tokenizer padding side based on model architecture.
    
    Args:
        tokenizer: HuggingFace tokenizer
        model_config: HuggingFace model configuration object
        force_padding_side: Override automatic detection with specific padding side
        
    Returns:
        Original padding side (for restoration later)
    """
    if not hasattr(tokenizer, 'padding_side'):
        return None
        
    original_padding_side = tokenizer.padding_side
    
    if force_padding_side:
        tokenizer.padding_side = force_padding_side
    else:
        tokenizer.padding_side = get_appropriate_padding_side(model_config)
        
    return original_padding_side


def get_padding_explanation(model_config) -> str:
    """Get explanation for why a particular padding side is chosen.
    
    Args:
        model_config: HuggingFace model configuration object
        
    Returns:
        Human-readable explanation of padding side choice
    """
    if not hasattr(model_config, 'model_type'):
        return "Unknown model type - using right padding as default"
    
    model_type = model_config.model_type.lower()
    padding_side = get_appropriate_padding_side(model_config)
    
    if padding_side == "left":
        return f"Model '{model_type}' is decoder-only - using left padding for proper text generation"
    else:
        if model_type in ['t5', 'bart', 'pegasus']:
            return f"Model '{model_type}' is encoder-decoder - using right padding"
        elif model_type in ['bert', 'roberta', 'distilbert']:
            return f"Model '{model_type}' is encoder-only - using right padding"
        else:
            return f"Model '{model_type}' - using right padding as default"
