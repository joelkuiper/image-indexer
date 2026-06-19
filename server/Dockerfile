# Self-hosted inference server with CLIP + Qwen3-VL
#
# Exposes:
#   POST /api/inference — Submit inference job
#   GET  /api/status/{id} — Poll job status
#
# Environment variables:
#   SERVER_HOST — Host to bind (default: 0.0.0.0)
#   SERVER_PORT — Port to listen on (default: 8080)
#   SERVER_DB_PATH — SQLite database path (default: /app/server.db)
#   SERVER_MAX_WORKERS — Max concurrent jobs (default: 3)
#   HF_HOME — HuggingFace cache directory (default: /app/model-cache)
#   LOG_LEVEL — Logging level (default: INFO)
FROM ghcr.io/astral-sh/uv:latest

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY server/ server/

# Install dependencies
ENV UV_SYSTEM_PYTHON=1 \
    HF_HOME=/app/model-cache \
    TRANSFORMERS_VERBOSITY=error

RUN uv pip install \
      "torch>=2.2.0" \
      "torchvision>=0.17.0" \
      "transformers>=4.57.0" \
      "pillow>=10.2.0" \
      "accelerate>=0.27.0" \
      "fastapi>=0.115.0" \
      "uvicorn[standard]>=0.32.0" \
      "pydantic>=2.0.0" \
      --no-cache-dir

# Pre-bake model weights for fast cold starts
# Uncomment to include models in image (~15GB), comment for smaller image (~2GB)
# RUN python -c "from huggingface_hub import snapshot_download; \\"
#     snapshot_download('openai/clip-vit-base-patch32'); \\"
#     snapshot_download('Qwen/Qwen3-VL-4B-Instruct')"

# Copy server code
RUN mkdir -p /app/model-cache
COPY server/ /app/server/

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "server.inference_server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
