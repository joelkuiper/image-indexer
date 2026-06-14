"""Production RunPod serverless handler — image indexing with SigLIP2 + Qwen3-VL.

This handler powers the real image indexing pipeline. Not for infra validation.

API Contract (RunPod /run endpoint):
  POST https://api.runpod.ai/v2/{endpoint_id}/run
  Headers:
    Authorization: Bearer ***    Content-Type: application/json
  Body:
    {
      "input": {
        "image_b64": "<base64 encoded image>",
        "task": "embed" | "caption" | "all"
      }
    }

Response:
  {
    "embedding": [1152 floats],  # SigLIP2 visual+text embedding (if task=embed|all)
    "description": "...",         # Qwen3-VL natural language caption (if task=caption|all)
    "models": {
      "siglip": "google/siglip2-so400m-patch16-384",
      "qwen": "Qwen/Qwen3-VL-4B-Instruct"
    }
  }

Tasks:
  - "embed":   Returns 1152-d SigLIP2 embedding for semantic search
  - "caption": Returns Qwen3-VL natural language description for lexical search
  - "all":     Returns both (default)

Performance:
  - Cold start: ~60s (loads 5GB weights from baked cache into GPU VRAM)
  - Warm request: ~200-500ms per image
  - Subsequent requests reuse loaded models (no repeated downloads)

Deployment:
  - Image: ~15GB (bakes both model weights for fast cold starts)
  - GPU: RTX 3070 or L4 recommended (~$0.0004/sec on RunPod)
  - Scale: Min 0 workers (scale to zero), Max 1-2
"""
import base64
import io

from worker.handler import embed_image, caption_image, load_models


def handler(job):
    """RunPod serverless handler for image indexing.
    
    Authorization is handled by RunPod infrastructure, not here.
    The handler is called with a job dict containing the input payload.
    """
    job_input = job.get("input", {})
    
    # Load models on first call (cached for subsequent requests)
    load_models()
    
    # Decode base64 image
    image_b64 = job_input.get("image_b64")
    if not image_b64:
        return {"error": "Missing required field: image_b64"}
    
    try:
        img_bytes = base64.b64decode(image_b64)
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return {"error": f"Failed to decode image: {str(e)}"}
    
    # Determine task
    task = job_input.get("task", "all")
    
    result = {}
    
    # SigLIP2 embedding for semantic search
    if task in ("embed", "all"):
        try:
            embedding_siglip = embed_image(img, "siglip")
            result["embedding"] = embedding_siglip
            result["embedding_model"] = "google/siglip2-so400m-patch16-384"
        except Exception as e:
            return {"error": f"SigLIP2 embedding failed: {str(e)}"}
    
    # Qwen3-VL caption for lexical search
    if task in ("caption", "all"):
        try:
            description_qwen = caption_image(img, "qwen")
            result["description"] = description_qwen
            result["caption_model"] = "Qwen/Qwen3-VL-4B-Instruct"
        except Exception as e:
            return {"error": f"Qwen3-VL caption failed: {str(e)}"}
    
    # Include model registry
    result["models"] = {
        "siglip": "google/siglip2-so400m-patch16-384",
        "qwen": "Qwen/Qwen3-VL-4B-Instruct"
    }
    
    return result


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
