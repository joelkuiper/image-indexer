#!/usr/bin/env python3
"""Full E2E pipeline test with REAL models on CPU.

Indexes 10 images from /esther/data/Sync/Pictures/ using local SigLIP2 + Qwen3-VL-4B.
Downloads models on first run (~9GB total), caches in HuggingFace cache.

Usage:
    uv run python scripts/e2e_real.py

Runs in ~5-15 min on 6-core CPU, 64GB RAM.
Stores results in /tmp/idx_e2e_test.db for CLI testing.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from image_indexer.db import connect, upsert_image
from image_indexer.preprocess import preprocess
from image_indexer.text_embed import TextEmbedder

# 10 diverse images: Renaissance art, screenshot, fractal, photos.
IMAGE_PATHS = [
    "/esther/data/Sync/Pictures/Sanzio_01_Euclid.jpg",
    "/esther/data/Sync/Pictures/Screenshot_2026-02-05_11-20-47.png",
    "/esther/data/Sync/Pictures/fireworks-galaxy_36092440231_o.jpg",
    "/esther/data/Sync/Pictures/Apophysis-040902-116.png",
    "/esther/data/Sync/Pictures/img_6378jpg_5185559831_o.jpg",
    "/esther/data/Sync/Pictures/img_0071jpg_21774554382_o.jpg",
    "/esther/data/Sync/Pictures/dscf1738jpg_21966520485_o.jpg",
    "/esther/data/Sync/Pictures/img_2451jpg_16019365802_o.jpg",
    "/esther/data/Sync/Pictures/clues_4119939146_o.jpg",
    "/esther/data/Sync/Pictures/img_7580jpg_9954958036_o.jpg",
]

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "idx_e2e_test.db"
CAPTION_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
CAPTION_PROMPT = (
    "Describe this image in 2-4 sentences. Cover subjects, setting, mood, "
    "colours, and transcribe any readable text. Write plain descriptive prose."
)


def load_models(embedder: TextEmbedder):
    """Load Qwen3-VL-4B (SigLIP2 already loaded via text_embed)."""
    print(f"Loading {CAPTION_MODEL_ID} (~8GB) — this takes a few minutes on CPU...")
    t0 = time.time()
    caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID)
    caption_model = AutoModelForImageTextToText.from_pretrained(
        CAPTION_MODEL_ID,
        device_map="cpu",
        dtype=None,
    ).eval()
    caption_model.generation_config.pad_token_id = caption_processor.tokenizer.eos_token_id
    print(f"  Models loaded in {time.time() - t0:.0f}s")
    return caption_processor, caption_model


def caption_image(
    img: Image.Image, caption_processor: AutoProcessor, caption_model
) -> str:
    """Generate a caption using Qwen3-VL-4B."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
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
    )
    with torch.no_grad():
        generated = caption_model.generate(
            **inputs, max_new_tokens=256, do_sample=False
        )
    trimmed = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], generated)]
    return caption_processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def embed_image(img: Image.Image, embedder: TextEmbedder) -> list:
    """Generate SigLIP2 image embedding."""
    embed_input = embedder._processor(images=[img], return_tensors="pt")
    assert embedder._model is not None
    with torch.no_grad():
        feats = embedder._model.get_image_features(**embed_input).pooler_output
    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    return feats[0].cpu().tolist()


def main():
    print("=" * 60)
    print("  image-indexer — full E2E with real models (CPU, batch)")
    print("=" * 60)

    # 1. Load models
    print("\n[1/4] Loading SigLIP2 (~800MB)...")
    t0 = time.time()
    embedder = TextEmbedder()
    embedder._load()
    print(f"  SigLIP2 loaded in {time.time() - t0:.1f}s")

    print("\n[2/4] Loading Qwen3-VL-4B (~8GB)...")
    caption_processor, caption_model = load_models(embedder)

    # 2. Open DB
    DB_PATH.unlink(missing_ok=True)
    db = connect(DB_PATH)

    # 3. Index images
    print(f"\n[3/4] Indexing {len(IMAGE_PATHS)} images...")
    stats = {"indexed": 0, "failed": 0}
    total_t0 = time.time()

    for i, img_path in enumerate(IMAGE_PATHS, 1):
        p = Path(img_path)
        print(f"\n  [{i}/{len(IMAGE_PATHS)}] {p.name}")

        # Preprocess
        prep = preprocess(p)
        if prep.skipped:
            print(f"    SKIP: {prep.skip_reason}")
            stats["failed"] += 1
            continue
        print(
            f"    preprocess: {prep.orig_width}x{prep.orig_height} → "
            f"{prep.resized_width}x{prep.resized_height} "
            f"({len(prep.jpeg_bytes) // 1024}KB)"
        )

        try:
            img = Image.open(io.BytesIO(prep.jpeg_bytes)).convert("RGB")

            # Embed
            t0 = time.time()
            embedding = embed_image(img, embedder)
            print(f"    embed: {time.time() - t0:.1f}s ({len(embedding)} dims)")

            # Caption
            t0 = time.time()
            description = caption_image(img, caption_processor, caption_model)
            print(f"    caption: {time.time() - t0:.1f}s")
            print(f'    "{description[:120]}..."')

            # Store
            meta = {
                "path": str(prep.path),
                "sha256": prep.sha256,
                "file_size": prep.file_size,
                "format": prep.disk_format,
                "width": prep.orig_width,
                "height": prep.orig_height,
                "description": description,
                "model_caption": CAPTION_MODEL_ID,
                "model_embed": "google/siglip2-so400m-patch16-384",
            }
            upsert_image(db, meta, embedding=embedding)
            stats["indexed"] += 1
        except Exception as e:
            print(f"    FAIL: {e}")
            stats["failed"] += 1

    elapsed = time.time() - total_t0
    print(f"\n  Indexed {stats['indexed']}/{len(IMAGE_PATHS)} in {elapsed:.0f}s")
    print(f"  Failed: {stats['failed']}")

    # 4. Test semantic search
    print("\n[4/4] Semantic search test: 'ancient geometry diagram'")
    from image_indexer.db import search_semantic
    from typing import cast

    query_text = "ancient Greek philosopher with geometry"
    raw_vec = embedder.embed(query_text)
    query_vec = cast(list[float], raw_vec)
    results = search_semantic(db, query_vec, k=5)
    for r in results:
        print(f"  dist={r['distance']:.4f}  {r['path']}")
        print(f"    {r['description'][:100]}...")

    print("\n" + "=" * 60)
    print(f"  Done! DB at: {DB_PATH}")
    print(f"  Try: idx search --semantic 'water' --db {DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
