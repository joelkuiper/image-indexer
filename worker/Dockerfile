# Use a lightweight python base image. For production serverless GPU loads,
# inherit from a CUDA/PyTorch base, e.g.:
#   FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# The slim base below is fine for CPU smoke-tests of the handler contract.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs (project rule: never pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install deps into the system interpreter (no venv layer inside the container).
ENV UV_SYSTEM_PYTHON=1
COPY pyproject.toml .
RUN touch README.md && \
    uv pip install \
      "runpod>=1.9.1" \
      "torch>=2.2.0" \
      "torchvision>=0.17.0" \
      "transformers>=4.57.0" \
      "accelerate>=0.27.0" \
      "pillow>=10.2.0"

# Bake the model weights into the image so cold starts don't pay the download.
# (Comment out to fetch lazily at first request instead.)
ENV HF_HOME=/app/hf-cache
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('google/siglip2-so400m-patch16-384'); \
    snapshot_download('Qwen/Qwen3-VL-4B-Instruct')" || \
    echo "Skipping weight pre-bake (will download at runtime)."

COPY worker/ /app/worker/

WORKDIR /app/worker
CMD ["python", "-u", "handler.py"]

