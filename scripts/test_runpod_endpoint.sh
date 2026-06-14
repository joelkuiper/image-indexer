#!/bin/bash
# Test the deployed RunPod hello-world endpoint
# Usage: ./scripts/test_runpod_endpoint.sh YOUR_API_KEY

if [ -z "$1" ]; then
    echo "Usage: $0 YOUR_RUNPOD_API_KEY"
    echo "Get your API key from: https://www.runpod.io/console/user/settings"
    exit 1
fi

API_KEY="$1"
ENDPOINT_ID="80to509hzmd3q4"  # Replace with your actual endpoint ID
URL="https://api.runpod.ai/v2/${ENDPOINT_ID}/run"

echo "Testing RunPod endpoint: ${URL}"
echo "=========================================="

# Test 1: Basic call (no image)
echo -e "\n[Test 1] Basic call (no image):"
curl -s -X POST "${URL}" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d '{"input":{}}' | jq .

# Create a tiny test image (2x2 red PNG)
echo -e "\n[Test 2] With base64 image:"
IMG_B64=$(python3 -c "
import base64, io
from PIL import Image
img = Image.new('RGB', (2, 2), color=(255, 0, 0))
buf = io.BytesIO()
img.save(buf, format='PNG')
print(base64.b64encode(buf.getvalue()).decode())
")

curl -s -X POST "${URL}" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "{\"input\":{\"image_b64\":\"${IMG_B64}\",\"task\":\"validate\"}}" | jq .

echo -e "\n=========================================="
echo "✅ Tests complete!"
