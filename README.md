# llm-pipeline

A PyTorch framework for LLM fine-tuning, quantization, and inference — implemented from scratch for full control and educational transparency, with no dependency on `peft`.

## What's in it

- **Adapters**: LoRA, DoRA (weight-decomposed), RSLoRA (rank-stabilized) — pure PyTorch
- **Quantization**: BitsAndBytes (4 / 8-bit), AQLM (2-bit), LoftQ (quantization-aware LoRA initialization)
- **Inference**: KV-cached generation with greedy / temperature / top-k / top-p / typical-p / Mirostat sampling, streaming output
- **Adapter merging**: TIES, DARE-Linear, DARE-TIES strategies; QLoRA-aware merging
- **Model registry**: 15+ HuggingFace architectures auto-detected (Llama, Mistral, Qwen, Gemma, Phi, GPT-2 / Neo, BERT, RoBERTa, T5, BART)
- **Memory tools**: footprint analysis with optimizer-state awareness, dynamic estimation across training modes

## Quickstart

```bash
git clone https://github.com/DavidAkinpelu/llm-pipeline.git
cd llm-pipeline
pip install -e .
```

```python
from llm_pipeline.core import LoRAModelWrapper, LoRAConfig
from transformers import AutoModel

model  = AutoModel.from_pretrained("microsoft/DialoGPT-medium")
config = LoRAConfig(r=16, alpha=32.0,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
lora_model = LoRAModelWrapper(model, config)
lora_model.print_trainable_parameters()
```

DoRA, RSLoRA, BnB / AQLM quantization, multi-adapter routing, inference, and merging examples follow the same pattern — see the modules under `src/llm_pipeline/`.

## Architecture

```
src/llm_pipeline/
  core/          # config classes, base LoRA module, model wrapper, registry
  adapters/      # LoRA / DoRA / RSLoRA + multi-adapter manager + merging
  quantization/  # BnB / AQLM / LoftQ integrations
  inference/     # KV-cached engine, sampling, streaming
  merging/       # adapter merge strategies (TIES, DARE)
  utils/         # memory analysis, factory patterns, validation
  testing/       # unit + integration test suite
```

## Status

Active. Core, adapters, quantization, inference, and merging are implemented and tested. Distributed training and production serving are work-in-progress.

## License

MIT
