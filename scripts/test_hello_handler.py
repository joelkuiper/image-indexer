"""Test the hello-world handler locally (simulates RunPod /run contract)."""
import sys
import base64
import io

# Add worker/hello to path
sys.path.insert(0, "worker/hello")

from handler import handler
from PIL import Image


def test_hello_handler():
    print("Testing hello-world handler...")
    print("-" * 50)
    
    # Test 1: No image, just CUDA check
    print("\n[Test 1] Basic call (no image):")
    job = {"input": {}}
    result = handler(job)
    
    assert "greeting" in result
    print(f"  ✓ Greeting: {result['greeting']}")
    print(f"  ✓ CUDA: {result['compute']['cuda_available']}")
    print(f"  ✓ GPU: {result['compute']['gpu']}")
    print(f"  ✓ PyTorch: {result['compute']['torch_version']}")
    assert result["image"] is None
    print(f"  ✓ Image: null (as expected)")
    
    # Test 2: With base64 image
    print("\n[Test 2] Call with base64 image:")
    
    # Create a tiny test image
    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    
    job = {"input": {"image_b64": img_b64, "task": "validate"}}
    result = handler(job)
    
    assert result["image"] is not None
    assert result["image"]["size"] == (2, 2)
    assert result["image"]["mode"] == "RGB"
    assert result["image"]["format"] == "PNG"
    
    print(f"  ✓ Image decoded: {result['image']['size']}")
    print(f"  ✓ Format: {result['image']['format']}")
    print(f"  ✓ Mode: {result['image']['mode']}")
    print(f"  ✓ Bytes: {result['image']['bytes_decoded']}")
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! Handler is RunPod-ready.")
    print("=" * 50)
    
    print("\nRunPod /run endpoint contract:")
    print("  POST https://api.runpod.ai/v2/{endpoint_id}/run")
    print("  Headers: Authorization: Bearer YOUR_API_KEY")
    print("  Body: { 'input': { 'image_b64': '...', 'task': 'validate' } }")


if __name__ == "__main__":
    test_hello_handler()
