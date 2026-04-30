# Minimal CUDA-enabled image for serving llm-pipeline.
# Build:  docker build -f src/llm_pipeline/production/deployment/Dockerfile -t llm-pipeline:latest .
# Run:    docker run --gpus all -p 8000:8000 -e MODEL=Qwen/Qwen3-0.6B llm-pipeline:latest

ARG CUDA_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src/ ./src/

RUN pip install --upgrade pip && \
    pip install ".[serving]"

EXPOSE 8000

ENV MODEL=""
ENV DTYPE=fp16
ENV DEVICE=cuda:0

ENTRYPOINT ["python", "-m", "llm_pipeline.production.serving.server"]
CMD ["--model", "${MODEL}", "--port", "8000"]
