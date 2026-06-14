# Production Worker Deployment Guide

## Overview

This deploys the real image indexing handler with SigLIP2 + Qwen3-VL models.
It's the production version — not for infrastructure validation.

**Differences from hello-world:**
- Image size: ~15GB (vs ~3-4GB for hello)
- Models: Bakes SigLIP2 + Qwen3-VL weights into image
- Cold start: ~60s (vs ~30s for hello)
- Warm request: ~200-500ms per image

## API Contract

Same as hello-world, but returns real embeddings and captions:

```bash
POST https://api.runpod.ai/v2/{endpoint_id}/run
Headers:
  Authorization: Bearer ***  Content-Type: application/json
Body:
  {
    "input": {
      "image_b64": "<base64 encoded image>",
      "task": "embed" | "caption" | "all"
    }
  }
```

**Example response:**
```json
{
  "embedding": [0.0123, -0.0456, ...],  // 1152 floats
  "description": "A photo of a mountain landscape at sunset...",
  "models": {
    "siglip": "google/siglip2-so400m-patch16-384",
    "qwen": "Qwen/Qwen3-VL-4B-Instruct"
  }
}
```

## Deploy from RunPod UI

1. **RunPod Console** → Serverless → **New Endpoint**

2. **Deploy source**:
   - Choose "GitHub Repository"
   - Repo: `joelkuiper/image-indexer`
   - Branch: `main`
   - Dockerfile path: `worker/Dockerfile`

3. **GPU & scaling**:
   - GPU: RTX 3070 or L4 (~$0.0004/sec)
   - Min Workers: `0` (scale to zero)
   - Max Workers: `1` (start small, increase if needed)
   - Idle Timeout: `60` seconds

4. Click **Deploy** — RunPod builds the container (~15GB, ~10-15 min)

5. Note your endpoint ID from the URL after deploy.

## Test the Production Worker

```bash
# Set your credentials
export RUNPOD_API_KEY=$(cat ~/.runpod-token)
export ENDPOINT_ID="your_endpoint_id_here"

# Test with embed only
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d '{
    "input": {
      "image_b64": "'"$(base64 -w 0 ./test_image.jpg)"'",
      "task": "embed"
    }
  }'

# Response (async):
# {"id": "req_abc123", "status": "IN_QUEUE", ...}

# Check status
curl "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/req_abc123" \
  -H "Authorization: Bearer ***}
# Response: {"status": "COMPLETED", "output": {"embedding": [...], "models": {...}}}
```

## Performance Expectations

- **First request (cold start)**: ~60s
  - Downloads ~15GB image (if not cached on RunPod)
  - Loads 5GB model weights into GPU VRAM
  
- **Subsequent requests (warm)**: ~200-500ms per image
  - Models already loaded in VRAM
  - No re-download, no re-initialization

- **Idle timeout**: After 60s with no requests, worker scales to zero
  - Next request triggers cold start again (~60s)

## Cost Calculation

Example: Index 1000 images
- Cold start: 60s × $0.0004/s = $0.024
- 1000 images × 0.5s avg × $0.0004/s = $0.20
- **Total**: ~$0.22 for 1000 images

## Troubleshooting

**Build fails:**
- Check RunPod logs (Endpoint → Logs tab)
- Common issue: HuggingFace token needed for gated models (both models are open, so shouldn't happen)

**CUDA out of memory:**
- Models require ~5GB VRAM
- RTX 3070 (8GB) or L4 (8GB) work fine
- Don't use A10 (24GB) unless you have budget — overkill

**Timeout on first request:**
- Expected! Cold start takes ~60s
- RunPod default timeout is 120s, so it should complete
- If it fails, check if image finished building

**Models not loading:**
- Check `HF_HOME=/app/hf-cache` env var is set
- Weights should be baked into image at `/app/hf-cache`
- If missing, Dockerfile `snapshot_download` step failed during build

## Next Steps

Once production worker is running:
1. Build local CLI client that:
   - Reads images from directory
   - Encodes to base64
   - POSTs to RunPod `/run` endpoint
   - Parses response and stores in local DB

2. Implement batching:
   - RunPod supports batch requests
   - Send 10-50 images per request for efficiency

3. Add error handling:
   - Retry logic for failed requests
   - Timeout handling for slow responses
   - Local fallback if RunPod is down

4. Monitor costs:
   - RunPod Console → Billing → Usage
   - Set budget alerts to avoid surprises
