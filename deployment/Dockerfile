# Multi-stage Dockerfile for the llm-pipeline serving stack.
#
# Build:
#   docker build -t llm-pipeline:latest -f deployment/Dockerfile .
#
# Run:
#   docker run --gpus all -p 8000:8000 \
#       -e LLM_PIPELINE_MODEL=Qwen/Qwen3-0.6B \
#       llm-pipeline:latest

# ---- Builder ----
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ src/

RUN python3.11 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir ".[serving,quantization]"

# ---- Runtime ----
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    LLM_PIPELINE_HOST=0.0.0.0 \
    LLM_PIPELINE_PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

ENTRYPOINT ["llm-pipeline"]
CMD ["serve"]
