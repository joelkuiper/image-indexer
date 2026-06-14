"""RunPod serverless handler — infrastructure validation hello world.

Proves the full stack works:
- Container starts on RunPod GPU
- Handler receives jobs via /run endpoint
- Authorization handled by RunPod (not in handler)
- Base64 image decoding works
- CUDA/GPU access available
- Response contract correct

API contract:
  POST https://api.runpod.ai/v2/{endpoint_id}/run
  Headers: Authorization: Bearer YOUR_API_KEY
  Body: { "input": { "image_b64": "...", "task": "embed" } }
"""
import base64
import io

import runpod
import torch
from PIL import Image


def handler(job):
    """RunPod serverless handler.
    
    Input: {"image_b64": "<base64>", "task": "validate"}
    Output: greeting, compute info, decoded image info
    
    Authorization is handled by RunPod infrastructure, not here.
    """
    job_input = job.get("input", {})
    
    # CUDA check
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU only"
    torch_version = torch.__version__
    
    # Decode image if provided
    image_info = None
    image_b64 = job_input.get("image_b64")
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
            image_info = {"error": f"Failed to decode: {str(e)}"}
    
    return {
        "greeting": "Hello from RunPod! 🚀",
        "compute": {
            "cuda_available": cuda_available,
            "gpu": gpu_name,
            "torch_version": torch_version,
        },
        "image": image_info,
    }


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
