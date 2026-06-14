"""Minimal RunPod hello-world handler.

Proves the infrastructure works:
- Container starts ✓
- Handler receives job ✓
- Base64 image decodes ✓
- CUDA is available ✓
- Response returns correctly ✓

No models, no heavy deps. Just torch + Pillow for the CUDA check and decode.
"""
import base64
import io
import time

import runpod
import torch
from PIL import Image


def handler(job):
    start = time.time()
    
    job_input = job.get("input", {})
    image_b64 = job_input.get("image_b64")
    
    # CUDA check
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU only"
    
    # If they sent an image, decode it and report dimensions
    image_info = None
    if image_b64:
        try:
            img_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_bytes))
            image_info = {
                "format": img.format,
                "size": img.size,
                "mode": img.mode,
                "bytes_decoded": len(img_bytes),
            }
        except Exception as e:
            image_info = {"error": f"Failed to decode: {e}"}
    
    elapsed_ms = round((time.time() - start) * 1000, 2)
    
    return {
        "greeting": "Hello from RunPod! 🚀",
        "compute": {
            "cuda_available": cuda_available,
            "gpu": gpu_name,
            "torch_version": torch.__version__,
        },
        "image": image_info,
        "handler_time_ms": elapsed_ms,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
