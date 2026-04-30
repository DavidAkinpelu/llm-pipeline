"""Model implementations.

- ``qwen3``: hand-rolled, fast Qwen3 implementation used by the inference engine.
- ``qwen3_5``: typed configs + presets for Qwen3.5 / Qwen3.6 models
  (Qwen3.6-27B dense, Qwen3.6-35B-A3B MoE). Inference is via the generic
  transformers fallback today; hand-rolled engine is in the roadmap.
- ``generic``: transformers-based wrapper for any HF causal LM.
- ``moe``: generic Mixture-of-Experts building blocks (router, experts,
  aux losses) usable as a drop-in for any Llama-style SwiGLU MLP.
"""
