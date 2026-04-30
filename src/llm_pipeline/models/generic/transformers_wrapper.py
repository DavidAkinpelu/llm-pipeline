"""Thin transformers wrapper for any HF causal LM.

Use this when you want to apply LoRA / training / merging to a model that
doesn't have a hand-rolled implementation in ``llm_pipeline.models``.
The Qwen3 path remains the optimized in-house implementation.
"""

from typing import Optional, Tuple

import torch


def load_causal_lm(
    model_path: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = False,
) -> Tuple[torch.nn.Module, "object"]:
    """Load any HF causal LM + tokenizer.

    Returns (model, tokenizer). The model is moved to the target device with
    the requested dtype.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers is required for the generic model loader."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    return model, tokenizer


class GenericFactory:
    """Mirror of ``Qwen3Factory.create_model_and_tokenizer`` for arbitrary HF models."""

    @staticmethod
    def create_model_and_tokenizer(
        model_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = False,
    ) -> Tuple[torch.nn.Module, "object"]:
        return load_causal_lm(model_path, device=device, dtype=dtype, trust_remote_code=trust_remote_code)
