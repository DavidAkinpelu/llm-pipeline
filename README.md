# llm-pipeline

An educational, end-to-end PyTorch toolkit for studying how modern LLM
training, inference, quantization, and serving actually work. The goal is
one repo where you can read — and run — pure-PyTorch / pure-Python
implementations of techniques you'd otherwise piece together from a dozen
libraries and papers.

This is a learning project, not a production library. Where the standard
tools (`transformers`, `peft`, `bitsandbytes`, `llama.cpp`) are
battle-tested, this repo prioritizes **clarity of implementation over
feature completeness or peak performance**. Every algorithm aims to be
readable as a reference, with hooks to compare against the canonical
version.

## What's in here

- **Parameter-efficient fine-tuning** — LoRA, DoRA, RSLoRA, multi-adapter
  routing, adapter merging (TIES, DARE, etc.).
- **Training modes** — SFT, DPO, GRPO, PPO/RLHF, ORPO, KTO, reward
  modelling, distillation.
- **Optimizers** — AdamW (incl. paged 8-bit via bitsandbytes), SGD,
  RMSprop, and **Muon** (matmul-only updates with AdamW auto-routed to
  embeddings, LM head, biases, and norms). Recent optimizers live under
  `training/optimization/optimizers/`.
- **Inference** — paged attention KV cache, continuous batching,
  speculative decoding (two-model + MTP-driven), and an OpenAI-compatible
  serving layer.
- **Quantization** — bitsandbytes / AQLM / LoftQ for training; an
  educational pure-Python K-quant + I-quant family (`Q3_K`–`Q8_K`,
  `IQ4_NL`, outlier-aware variants) with Triton GPU kernels for the hot
  paths; a from-scratch GGUF v3 writer.
- **Hand-rolled Qwen3 / Qwen3.5 / Qwen3.6** — pure-PyTorch implementations
  including Gated DeltaNet linear-attention, hybrid attention layer
  interleave, MoE with shared expert, MTP head, vision tower.
- **Distributed training** — DDP, FSDP, tensor parallelism (Qwen3),
  expert-parallel MoE.
- **Model merging** — full state-dict TIES / DARE / task-arithmetic /
  linear.

The Qwen3 family inference paths use hand-rolled implementations; the
training, serving, and merging paths work against any HuggingFace causal
LM.

## Install

Python 3.9–3.12 + CUDA. The base install pulls torch + transformers; opt
in to extras for quantization or serving.

```bash
git clone https://github.com/DavidAkinpelu/llm-pipeline.git
cd llm-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e ".[serving,quantization,dev]"
```

Extras: `serving` (FastAPI + Prometheus), `quantization` (bitsandbytes +
aqlm + scipy), `dev` (pytest, black, flake8, mypy). Flash-attention is
optional — install separately when you want the speedup; the engine falls
back gracefully when it's missing.

## Quick taste

Inference:

```python
import torch
from llm_pipeline.inference import Qwen3InferenceEngine, Qwen3InferenceConfig

engine = Qwen3InferenceEngine(Qwen3InferenceConfig(
    model_path="Qwen/Qwen3-0.6B", device="cuda:0", dtype=torch.float16,
))
prompt = engine.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Why is the sky blue?"}],
    tokenize=False, add_generation_prompt=True,
)
print(engine.generate(prompt, max_tokens=64, temperature=0.3))
```

SFT training (sketch — see `examples/` for runnable scripts):

```python
from llm_pipeline.training import Trainer, TrainerConfig, OptimizerConfig

trainer = Trainer(
    model, dataloader,
    config=TrainerConfig(
        output_dir="./checkpoints", max_steps=200, precision="bf16",
        optimizer=OptimizerConfig(name="adamw", lr=5e-5, use_paged=True),
    ),
)
trainer.train()
```

Swap `name="adamw"` for `name="muon"` to use Muon. The same pattern
applies to DPO, GRPO, PPO, ORPO, KTO, reward-model, and distillation
trainers — each is a `Trainer` subclass with its own config.

Quantize a model with the educational K-quant family:

```python
from llm_pipeline.quantization import Quantizer, QuantMethod
qm = Quantizer(method=QuantMethod.Q4_K_M).quantize(model)
qm.save("model.llmpq")
```

OpenAI-compatible server:

```bash
python -m llm_pipeline.production.serving.server \
    --model Qwen/Qwen3-0.6B --device cuda:0 --dtype fp16 --port 8000
```

A reference Dockerfile lives at
`src/llm_pipeline/production/deployment/Dockerfile`.

## Project layout

```
src/llm_pipeline/
├── adapters/        LoRA / DoRA / RSLoRA + multi-adapter
├── core/            wrappers, registry, configs
├── inference/       Qwen3 engine, paged attn, speculative decoding
├── merging/         full-state-dict merging
├── models/          hand-rolled Qwen3 / Qwen3.5 / Qwen3.6 + generic HF loader
├── parallelism/     tensor parallelism, expert parallelism
├── production/      FastAPI server, Prometheus, Dockerfile
├── quantization/    BnB / AQLM / LoftQ + educational K-quants + GGUF writer
├── testing/         pytest suites
├── training/        Trainer, optimizers (incl. Muon), schedulers, training modes
└── utils/
```

## Tests

```bash
pytest src/llm_pipeline/testing/             # full suite
pytest src/llm_pipeline/testing/unit/        # unit only (no GPU)
```

Triton-kernel and CUDA-only tests are auto-skipped when the toolchain is
absent.

## Documentation

This README is a deliberately thin overview. Full documentation —
architecture deep-dives, derivations, and tutorials walking through each
implementation — is the next phase of work and will land under `docs/`.

## License

MIT.
