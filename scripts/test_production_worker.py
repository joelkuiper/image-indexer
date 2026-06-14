# Local test script for the production worker
# Loads API key from ~/.runpod-token (not in code!)

import os
import sys
import base64
import requests
import json

# Read credentials from secure files
def load_runpod_key():
    key_path = os.path.expanduser("~/.runpod-token")
    if not os.path.exists(key_path):
        print(f"Error: {key_path} not found")
        print("Create it with: echo '*** > ~/.runpod-token && chmod 600 ~/.runpod-token")
        sys.exit(1)
    
    with open(key_path) as f:
        return f.read().strip()

def test_production_worker():
    api_key = load_runpod_key()
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    
    if not endpoint_id:
        print("Error: RUNPOD_ENDPOINT_ID not set")
        print("Get it from RunPod Console → Serverless → Your Endpoint")
        sys.exit(1)
    
    # Create a simple test image (red square)
    from PIL import Image
    import io
    
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
    
    # Call the endpoint
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "input": {
            "image_b64": img_b64,
            "task": "all"  # Get both embedding and description
        }
    }
    
    print(f"Testing endpoint: {url}")
    print("Sending test image (100x100 red square)...")
    
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    
    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        print(response.text)
        sys.exit(1)
    
    result = response.json()
    
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    # Parse response
    print("\n✓ Success!")
    print(f"Request ID: {result.get('id')}")
    print(f"Status: {result.get('status')}")
    
    if result.get("status") == "COMPLETED":
        output = result.get("output", {})
        
        if "embedding" in output:
            embedding = output["embedding"]
            print(f"\nEmbedding:")
            print(f"  Dimension: {len(embedding)}")
            print(f"  First 5 values: {embedding[:5]}")
        
        if "description" in output:
            print(f"\nDescription:")
            print(f"  {output['description']}")
        
        if "models" in output:
            print(f"\nModels used:")
            for name, model_id in output["models"].items():
                print(f"  {name}: {model_id}")
    else:
        print(f"\nRequest still processing: {result.get('status')}")
        print("Check status with: GET /v2/{endpoint_id}/status/{request_id}")

if __name__ == "__main__":
    test_production_worker()
