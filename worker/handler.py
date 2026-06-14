"""RunPod serverless handler for image-indexer.

Two models, two jobs:
  * SigLIP2 (so400m-patch16-384)  -> 1152-d embedding in a JOINT image/text space,
    so the on-device CLI can do text->image AND image->image semantic search.
  * Qwen3-VL-4B-Instruct          -> rich natural-language caption (subjects,
    setting, mood, OCR) used for the FTS5 lexical index.

Request:
    {"input": {"image_b64": "<base64 raw image bytes>", "task": "all"}}
    task in {"embed", "caption", "all"} (default "all").

Response:
    {"embedding": [float x1152], "embedding_dim": 1152,
     "description": "...", "models": {...}}
"""

import base64
import io

from typing import Any

import runpod
import torch
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoModelForImageTextToText,
    AutoProcessor,
)

EMBED_MODEL_ID = "openai/clip-vit-base-patch32"
CAPTION_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
EMBED_DIM = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float32
)

# Lazily-initialised globals (warm-start reuse across serverless invocations).
embed_model = None
embed_processor = None
caption_model = None
caption_processor = None


def load_models():
    """Load both models once. Subsequent calls are no-ops (warm worker)."""
    global embed_model, embed_processor, caption_model, caption_processor

    if embed_model is None:
        print(f"Loading CLIP embedder ({EMBED_MODEL_ID}) on {device}...")
        embed_processor = CLIPProcessor.from_pretrained(EMBED_MODEL_ID)
        embed_model = CLIPModel.from_pretrained(
            EMBED_MODEL_ID,
            torch_dtype=torch.float32,  # CLIP is stable on float32/fp16, bfloat16 can be weird on some CPU/GPUs
            device_map="auto" if device == "cuda" else None,
        ).eval()
        if device == "cpu":
            embed_model.to(device)

    if caption_model is None:
        print(f"Loading Qwen3-VL captioner ({CAPTION_MODEL_ID}) on {device}...")
        caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID)
        caption_model = AutoModelForImageTextToText.from_pretrained(
            CAPTION_MODEL_ID,
            torch_dtype=_dtype,
            device_map="auto" if device == "cuda" else None,
        ).eval()
        if device == "cpu":
            caption_model.to(device)


def embed_image(image: Image.Image) -> list[float]:
    """CLIP image embedding, L2-normalised, as a plain Python float list."""
    assert embed_processor is not None
    assert embed_model is not None
    inputs = embed_processor(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        output_obj = embed_model.get_image_features(**inputs)
        if hasattr(output_obj, "pooler_output"):
            feats = output_obj.pooler_output
        else:
            feats = output_obj
    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    return feats[0].cpu().to(torch.float32).tolist()


CAPTION_PROMPT = (
    "You are an expert photo curator building a searchable archive. "
    "Describe this image in 2-4 sentences. Cover the main subjects, the setting, "
    "the mood and lighting, dominant colours, and transcribe any visible text. "
    "Write plain descriptive prose with no preamble."
)


def caption_image(image: Image.Image) -> str:
    """Qwen3-VL caption via the chat-template API (handles vision tokens for us)."""
    assert caption_processor is not None
    assert caption_model is not None
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]
    inputs = caption_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated = caption_model.generate(
            **inputs, max_new_tokens=256, do_sample=False
        )

    # Strip the prompt tokens, decode only the freshly generated continuation.
    trimmed = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], generated)]
    text = caption_processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return text.strip()


def handler(job):
    load_models()

    job_input = job.get("input", {})
    image_b64 = job_input.get("image_b64")
    task = job_input.get("task", "all")

    if not image_b64:
        return {"error": "Missing required field: 'image_b64'"}
    if task not in ("embed", "caption", "all"):
        return {"error": f"Invalid task '{task}'; expected embed|caption|all"}

    try:
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:  # noqa: BLE001 - report decode failures to caller
        return {"error": f"Failed to decode image: {e}"}

    result: dict[str, Any] = {"models": {}}
    try:
        if task in ("embed", "all"):
            result["embedding"] = embed_image(image)
            result["embedding_dim"] = EMBED_DIM
            result["models"]["embed"] = EMBED_MODEL_ID
        if task in ("caption", "all"):
            result["description"] = caption_image(image)
            result["models"]["caption"] = CAPTION_MODEL_ID
    except Exception as e:  # noqa: BLE001
        return {"error": f"Inference failed: {e}"}

    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
