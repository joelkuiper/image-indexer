#!/usr/bin/env python3
"""Test the hello-world handler locally (no RunPod, no Docker)."""
import sys
from pathlib import Path

# Add worker/hello to path
sys.path.insert(0, str(Path(__file__).parent.parent / "worker" / "hello"))

from handler import handler


def test_hello_world():
    # Test 1: No image, just CUDA check
    print("Test 1: Basic handler call (no image)...")
    job = {"input": {}}
    result = handler(job)
    
    assert "greeting" in result
    assert result["greeting"] == "Hello from RunPod! 🚀"
    assert "compute" in result
    assert "cuda_available" in result["compute"]
    assert "gpu" in result["compute"]
    assert result["image"] is None
    
    print(f"  ✓ Greeting: {result['greeting']}")
    print(f"  ✓ CUDA: {result['compute']['cuda_available']}")
    print(f"  ✓ GPU: {result['compute']['gpu']}")
    print(f"  ✓ Handler time: {result['handler_time_ms']}ms")
    
    # Test 2: With a tiny test image
    print("\nTest 2: Handler with base64 image...")
    from PIL import Image
    import io
    import base64
    
    # Create a tiny 2x2 RGB image
    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    
    job = {"input": {"image_b64": b64}}
    result = handler(job)
    
    assert result["image"] is not None
    assert result["image"]["size"] == (2, 2)
    assert result["image"]["mode"] == "RGB"
    assert result["image"]["format"] == "PNG"
    
    print(f"  ✓ Image decoded: {result['image']['size']}")
    print(f"  ✓ Format: {result['image']['format']}")
    print(f"  ✓ Bytes decoded: {result['image']['bytes_decoded']}")
    
    print("\n✅ All tests passed! Handler is ready for RunPod.")


if __name__ == "__main__":
    test_hello_world()
