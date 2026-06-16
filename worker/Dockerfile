# Production RunPod worker with SigLIP2 + Qwen3-VL
#
# This image is ~15GB because it bakes model weights for fast cold starts.
# For quick infra validation, use worker/hello/Dockerfile instead (~3-4GB).
#
# RunPod endpoint contract:
#   POST https://api.runpod.ai/v2/{endpoint_id}/run
#   Headers: Authorization: Bearer ***
#   Body: {"input": {"image_b64": "...", "task": "embed|caption|all"}}
#
# Response: {"embedding": [1152 floats], "description": "...", "models": {...}}
#
# Cold start: ~60s (loads 5GB weights into GPU VRAM)
# Warm request: ~200-500ms per image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# uv for fast, reproducible installs (project rule: never pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install deps into the system interpreter (no venv layer inside the container).
ENV UV_SYSTEM_PYTHON=1 \
    HF_HOME=/app/hf-cache \
    TRANSFORMERS_VERBOSITY=error \
    TORCH_CUDA_ARCH_LIST="5.0;5.2;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

COPY pyproject.toml .
RUN touch README.md && \
    uv pip install \
      "runpod>=1.9.1" \
      "torch>=2.2.0" \
      "torchvision>=0.17.0" \
      "transformers>=4.57.0" \
      "accelerate>=0.27.0" \
      "pillow>=10.2.0" \
      "--no-cache-dir"

# Bake both model weights (~5GB) so cold starts don't pay the download.
# On first request, models load from local cache into GPU VRAM (~60s).
# Subsequent requests: ~200-500ms per image.
#
# If you want to test the build without downloading weights (faster), comment this out.
# Weights will then download on first request (slower cold start, but smaller image).
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('google/siglip2-so400m-patch16-384'); \
    snapshot_download('Qwen/Qwen3-VL-4B-Instruct')" || \
    echo "Skipping weight pre-bake (will download at runtime)."

COPY worker/ /app/worker/

WORKDIR /app/worker

# RunPod calls this on container start. Handler registers with /run endpoint.
CMD ["python", "-u", "handler.py"]
