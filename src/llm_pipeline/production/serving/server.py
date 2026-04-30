"""OpenAI-compatible HTTP server backed by Qwen3InferenceEngine.

Endpoints:
  GET  /health                  — liveness/readiness probe.
  GET  /metrics                 — Prometheus exposition (if prometheus_client installed).
  GET  /v1/models               — list available models.
  POST /v1/completions          — text completion (OpenAI-compatible).
  POST /v1/chat/completions     — chat completion (OpenAI-compatible), optional SSE streaming.

Run with:
    python -m llm_pipeline.production.serving.server --model Qwen/Qwen3-0.6B --port 8000
or programmatically: ``run(model_path="Qwen/Qwen3-0.6B")``.
"""

import argparse
import json
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

from ..monitoring.metrics import (
    MetricsRegistry,
    metrics_response,
    timed_request,
)

# These imports are required at module import time so FastAPI's signature
# introspection can resolve the request-body schemas. The serving extra
# (``pip install llm-pipeline[serving]``) installs all of them.
try:
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import StreamingResponse, Response
    from .schemas import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatCompletionChoice,
        ChatMessage,
        CompletionRequest,
        CompletionResponse,
        CompletionChoice,
        ModelCard,
        ModelList,
        Usage,
    )
    _SERVING_DEPS_OK = True
except ImportError:
    _SERVING_DEPS_OK = False


def _require_serving_deps() -> None:
    if not _SERVING_DEPS_OK:
        raise ImportError(
            "Serving dependencies missing. Install with `pip install llm-pipeline[serving]`."
        )


def build_app(
    model_path: str,
    device: str = "cuda:0",
    dtype: str = "fp16",
    max_batch_size: int = 8,
    enable_streaming: bool = True,
):
    """Construct a FastAPI app wrapped around Qwen3InferenceEngine."""
    _require_serving_deps()

    import torch
    from ...inference import Qwen3InferenceEngine, Qwen3InferenceConfig

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    engine = Qwen3InferenceEngine(
        Qwen3InferenceConfig(
            model_path=model_path,
            device=device,
            dtype=dtype_map[dtype],
            max_batch_size=max_batch_size,
            enable_streaming=enable_streaming,
        )
    )

    metrics = MetricsRegistry()
    app = FastAPI(title="llm-pipeline", version="0.1.0")
    app.state.engine = engine
    app.state.model_id = model_path
    app.state.metrics = metrics

    # ---------------------------------------------------------------- #
    # Health + metrics
    # ---------------------------------------------------------------- #

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_path}

    @app.get("/metrics")
    def get_metrics():
        body, content_type = metrics_response(metrics)
        return Response(content=body, media_type=content_type)

    # ---------------------------------------------------------------- #
    # Models
    # ---------------------------------------------------------------- #

    @app.get("/v1/models")
    def list_models():
        return ModelList(data=[ModelCard(id=model_path, created=int(time.time()))])

    # ---------------------------------------------------------------- #
    # Completions
    # ---------------------------------------------------------------- #

    def _count_tokens(text: str) -> int:
        return len(engine.tokenizer(text, add_special_tokens=False).input_ids)

    def _validate_model_name(requested_model: str) -> None:
        if requested_model != model_path:
            raise HTTPException(
                status_code=404,
                detail=f"model {requested_model!r} not available on this server",
            )

    def _validate_generation_options(n: int, stop: Optional[Any]) -> None:
        if n != 1:
            raise HTTPException(status_code=400, detail="n > 1 is not supported")
        if stop is not None:
            raise HTTPException(status_code=400, detail="stop sequences are not supported")

    @app.post("/v1/completions")
    def completions(req: CompletionRequest = Body(...)):
        _validate_model_name(req.model)
        _validate_generation_options(req.n, req.stop)
        prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]

        if req.stream:
            if len(prompts) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="streaming completions only support a single prompt",
                )

            def event_stream() -> Iterable[str]:
                rid = f"cmpl-{uuid.uuid4().hex}"
                created = int(time.time())
                chunks: List[str] = []
                with timed_request(metrics, "completions_stream"):
                    for chunk in engine.stream_generate(
                        prompts[0],
                        max_tokens=req.max_tokens or 256,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                    ):
                        chunks.append(chunk)
                        yield _sse({
                            "id": rid,
                            "object": "text_completion",
                            "created": created,
                            "model": model_path,
                            "choices": [{
                                "index": 0,
                                "text": chunk,
                                "finish_reason": None,
                            }],
                        })

                    completion_tokens = _count_tokens("".join(chunks))
                    metrics.completion_tokens.inc(completion_tokens)
                    yield _sse({
                        "id": rid,
                        "object": "text_completion",
                        "created": created,
                        "model": model_path,
                        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                    })
                    yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        with timed_request(metrics, "completions"):
            outs = engine.generate(
                prompts if len(prompts) > 1 else prompts[0],
                max_tokens=req.max_tokens or 256,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            )
            outs = outs if isinstance(outs, list) else [outs]

            prompt_tokens = sum(_count_tokens(p) for p in prompts)
            completion_tokens = sum(_count_tokens(o) for o in outs)
            metrics.completion_tokens.inc(completion_tokens)

            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=model_path,
                choices=[CompletionChoice(index=i, text=o) for i, o in enumerate(outs)],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

    # ---------------------------------------------------------------- #
    # Chat completions
    # ---------------------------------------------------------------- #

    def _apply_chat_template(messages: List[ChatMessage]) -> str:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        if hasattr(engine.tokenizer, "apply_chat_template"):
            return engine.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        # Fallback: simple chat formatting.
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant:"

    def _sse(payload: Dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest = Body(...)):
        _validate_model_name(req.model)
        _validate_generation_options(req.n, req.stop)
        prompt = _apply_chat_template(req.messages)
        prompt_tokens = _count_tokens(prompt)

        if req.stream:
            def event_stream() -> Iterable[str]:
                rid = f"chatcmpl-{uuid.uuid4().hex}"
                created = int(time.time())
                first = True
                completion_chars = 0
                with timed_request(metrics, "chat_completions_stream"):
                    for chunk in engine.stream_generate(
                        prompt,
                        max_tokens=req.max_tokens or 256,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                    ):
                        completion_chars += len(chunk)
                        yield _sse({
                            "id": rid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_path,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": chunk} if first else {"content": chunk},
                                "finish_reason": None,
                            }],
                        })
                        first = False
                    yield _sse({
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_path,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    })
                    yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        with timed_request(metrics, "chat_completions"):
            out = engine.generate(
                prompt,
                max_tokens=req.max_tokens or 256,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            )
            out = out[0] if isinstance(out, list) else out
            completion_tokens = _count_tokens(out)
            metrics.completion_tokens.inc(completion_tokens)
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=model_path,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=out),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

    return app


def run(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "cuda:0",
    dtype: str = "fp16",
) -> None:
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "uvicorn is required to run the serving layer. Install with `pip install llm-pipeline[serving]`."
        ) from e
    app = build_app(model_path=model_path, device=device, dtype=dtype)
    uvicorn.run(app, host=host, port=port)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible server for llm-pipeline")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    args = parser.parse_args()
    run(model_path=args.model, host=args.host, port=args.port, device=args.device, dtype=args.dtype)


if __name__ == "__main__":
    _cli()
