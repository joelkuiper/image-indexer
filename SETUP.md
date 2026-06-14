# RunPod Hello World — Deploy & Test

## Deploy vanuit RunPod UI (geen GHCR nodig)

RunPod kan direct vanuit je GitHub repo bouwen:

1. **RunPod Console** → Serverless → **New Endpoint**
2. Vul in:
   - **Endpoint Name**: `image-indexer-hello`
   - **Template**: Custom
   - **Docker Image**: Laat leeg (we gebruiken Dockerfile deploy)
   - **GitHub Repo**: `joelkuiper/image-indexer`
   - **Branch**: `main`
   - **Dockerfile Path**: `worker/hello/Dockerfile`
3. **GPU & Scaling**:
   - **GPU Types**: RTX 3070 of L4 (goedkoopst voor test)
   - **Min Workers**: 0 (scale to zero)
   - **Max Workers**: 1
   - **Idle Timeout**: 60 seconds
4. **Advanced Settings**:
   - **FlashBoot**: Enabled (snellere cold starts)
5. Klik **Deploy**

⚠️ RunPod buildt de container automatisch (~3-4 GB, ~5 min).

## API Contract

RunPod endpoints gebruiken `/run` (async) of `/runsync` (sync):

```bash
POST https://api.runpod.ai/v2/{endpoint_id}/run
Headers:
  Authorization: Bearer YOUR_API_KEY
  Content-Type: application/json
Body:
  {
    "input": {
      "image_b64": "<base64 encoded image>",
      "task": "validate"
    }
  }
```

**API Key**: RunPod Console → User Settings → API Keys

**Endpoint ID**: Staat in de URL na deploy, bv: `https://www.runpod.io/console/serverless/user/endpoint/80to509hzmd3q4` → ID is `80to509hzmd3q4`

## Local Testing

Test de handler zonder RunPod:
```bash
cd ~/Repositories/image-indexer
uv run python scripts/test_hello_handler.py
```

Verwacht:
```
Testing hello-world handler...
--------------------------------------------------

[Test 1] Basic call (no image):
  ✓ Greeting: Hello from RunPod! 🚀
  ✓ CUDA: False (lokaal, geen GPU)
  ✓ GPU: CPU only
  ✓ PyTorch: 2.1.0+cpu
  ✓ Image: null (as expected)

[Test 2] Call with base64 image:
  ✓ Image decoded: (2, 2)
  ✓ Format: PNG
  ✓ Mode: RGB
  ✓ Bytes: 79

==================================================
✅ All tests passed! Handler is RunPod-ready.
==================================================

RunPod /run endpoint contract:
  POST https://api.runpod.ai/v2/{endpoint_id}/run
  Headers: Authorization: Bearer YOUR_API_KEY
  Body: { 'input': { 'image_b64': '...', 'task': 'validate' } }
```

## Remote Testing

Test het deployed endpoint:
```bash
# Get your API key from RunPod Console → User Settings
export RUNPOD_API_KEY=$(cat ~/.runpod-token)

# Test basic call
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d '{"input":{}}'

# Response (async - returns request ID):
# {"id": "req_abc123", "status": "IN_QUEUE", ...}

# Check status:
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/req_abc123" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

Of gebruik de test script (update `ENDPOINT_ID` in het script):
```bash
./scripts/test_runpod_endpoint.sh YOUR_API_KEY
```

## Expected Response

Wanneer de container draait op GPU:
```json
{
  "greeting": "Hello from RunPod! 🚀",
  "compute": {
    "cuda_available": true,
    "gpu": "NVIDIA RTX 3070",
    "torch_version": "2.1.0+cu118"
  },
  "image": {
    "format": "PNG",
    "size": [2, 2],
    "mode": "RGB",
    "bytes_decoded": 79
  }
}
```

## Troubleshooting

- **Build failed**: Check RunPod logs (Endpoint → Logs tab)
- **CUDA not available**: Ensure GPU is selected in endpoint config
- **Timeout**: First call takes 30-60s (cold start), subsequent calls ~5-10s
- **Auth error**: Verify API key in headers

## Next Steps

✅ Hello world werkt? → Deploy de echte worker met SigLIP2 + Qwen3-VL

Zelfde proces, maar met:
- `worker/worker.py` (echte handler)
- `worker/worker_dockerfile` (15GB image met models)

Kosten: ~$0.0004/sec voor RTX 3070, scale to zero = alleen betalen bij gebruik.
