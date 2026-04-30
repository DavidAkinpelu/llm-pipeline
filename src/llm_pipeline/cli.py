"""``llm-pipeline`` console entry point.

Subcommands
-----------
- ``infer``  — one-off chat/completion against a model.
- ``serve``  — run the OpenAI-compatible FastAPI server.
- ``train``  — light wrapper that prints a pointer to ``Trainer`` (full
  pipelines are too configuration-heavy for a CLI; programmatic use only).
- ``merge``  — merge two or more model state_dicts using a chosen strategy.
"""

from __future__ import annotations

import argparse
import sys
from typing import List


def _cmd_infer(args: argparse.Namespace) -> int:
    import torch
    from .inference import Qwen3InferenceEngine, Qwen3InferenceConfig

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    engine = Qwen3InferenceEngine(
        Qwen3InferenceConfig(
            model_path=args.model,
            device=args.device,
            dtype=dtype_map[args.dtype],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    )
    if args.chat:
        msgs = [{"role": "user", "content": args.prompt}]
        text = engine.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    else:
        text = args.prompt
    out = engine.generate(text, max_tokens=args.max_new_tokens, temperature=args.temperature)
    print(out if isinstance(out, str) else "\n---\n".join(out))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .production.serving.server import run as serve
    serve(model_path=args.model, host=args.host, port=args.port, device=args.device, dtype=args.dtype)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    print(
        "Programmatic use is recommended:\n"
        "  from llm_pipeline.training import Trainer, TrainerConfig, SFTDataCollator\n"
        "See README.md for an end-to-end SFT example.",
        file=sys.stderr,
    )
    return 1


def _cmd_merge(args: argparse.Namespace) -> int:
    import torch
    from .merging import linear_merge, task_arithmetic, ties_merge, dare_merge

    state_dicts = [torch.load(p, map_location="cpu") for p in args.inputs]
    weights = [float(w) for w in args.weights] if args.weights else None

    if args.strategy == "linear":
        merged = linear_merge(state_dicts, weights=weights)
    elif args.strategy == "task_arithmetic":
        if not args.base:
            print("--base required for task_arithmetic", file=sys.stderr)
            return 2
        base = torch.load(args.base, map_location="cpu")
        merged = task_arithmetic(base, state_dicts, alphas=weights)
    elif args.strategy == "ties":
        if not args.base:
            print("--base required for ties", file=sys.stderr)
            return 2
        base = torch.load(args.base, map_location="cpu")
        merged = ties_merge(base, state_dicts, density=args.density, alphas=weights)
    elif args.strategy == "dare":
        if not args.base:
            print("--base required for dare", file=sys.stderr)
            return 2
        base = torch.load(args.base, map_location="cpu")
        merged = dare_merge(base, state_dicts, drop_p=args.drop_p, alphas=weights)
    else:
        print(f"Unknown strategy: {args.strategy}", file=sys.stderr)
        return 2

    torch.save(merged, args.output)
    print(f"Saved merged state_dict to {args.output}")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_infer = sub.add_parser("infer", help="One-off generation")
    p_infer.add_argument("--model", required=True)
    p_infer.add_argument("--prompt", required=True)
    p_infer.add_argument("--device", default="cuda:0")
    p_infer.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    p_infer.add_argument("--max-new-tokens", type=int, default=256)
    p_infer.add_argument("--temperature", type=float, default=0.7)
    p_infer.add_argument("--chat", action="store_true", help="Wrap prompt in chat template")
    p_infer.set_defaults(fn=_cmd_infer)

    p_serve = sub.add_parser("serve", help="Run OpenAI-compatible HTTP server")
    p_serve.add_argument("--model", required=True)
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--device", default="cuda:0")
    p_serve.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    p_serve.set_defaults(fn=_cmd_serve)

    p_train = sub.add_parser("train", help="Pointer to programmatic training API")
    p_train.set_defaults(fn=_cmd_train)

    p_merge = sub.add_parser("merge", help="Merge model state_dicts")
    p_merge.add_argument("--strategy", choices=("linear", "task_arithmetic", "ties", "dare"), default="linear")
    p_merge.add_argument("--base", help="Base state_dict (.pt) for task_arithmetic / ties / dare")
    p_merge.add_argument("--inputs", nargs="+", required=True, help="Task state_dicts (.pt)")
    p_merge.add_argument("--weights", nargs="*", help="Per-input weights/alphas")
    p_merge.add_argument("--density", type=float, default=0.2, help="TIES density (top-k fraction)")
    p_merge.add_argument("--drop-p", type=float, default=0.5, help="DARE drop probability")
    p_merge.add_argument("--output", required=True, help="Output .pt path")
    p_merge.set_defaults(fn=_cmd_merge)

    args = parser.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
