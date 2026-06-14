#!/usr/bin/env python3
"""Re-embed processed images in the local SQLite db using OpenAI CLIP.

Avoids re-running the heavy Qwen3-VL caption generator.
Loads local CLIP (~150MB, fast on CPU) to overwrite the 1152-d SigLIP2 vectors 
with 512-d CLIP vectors for optimal cosine-similarity search.
"""
from __future__ import annotations

import sqlite3
import struct
import time
from pathlib import Path

import torch
from PIL import Image

from image_indexer.db import connect, serialize_f32
from image_indexer.text_embed import TextEmbedder

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "idx_e2e_test.db"


def main():
    if not DB_PATH.exists():
        print(f"Error: DB not found at {DB_PATH}. Run scripts/e2e_real.py first.")
        return

    print("=" * 60)
    print("  image-indexer — Re-embedding Database with OpenAI CLIP")
    print("=" * 60)

    # 1. Load CLIP model
    print("\n[1/3] Loading CLIP embedder (~150MB)...")
    t0 = time.time()
    embedder = TextEmbedder()
    embedder._load()  # Force load CLIPModel and CLIPProcessor
    print(f"  CLIP loaded in {time.time() - t0:.1f}s")

    # 2. Connect to DB
    print("\n[2/3] Connecting to SQLite...")
    # Open normal connection
    db = connect(DB_PATH)

    # Recreate vec_images to support 512 dimensions instead of 1152
    db.execute("DROP TABLE IF EXISTS vec_images")
    db.execute(
        "CREATE VIRTUAL TABLE vec_images USING vec0("
        "image_id INTEGER PRIMARY KEY,"
        "embedding FLOAT[512] distance_metric=cosine"
        ")"
    )
    db.commit()

    rows = db.execute("SELECT id, path FROM images").fetchall()
    print(f"  Found {len(rows)} images to update.")

    # 3. Process and update
    print("\n[3/3] Re-embedding images...")
    for row_id, path_str in rows:
        path = Path(path_str)
        if not path.exists():
            print(f"  [Skip] File not found: {path}")
            continue

        print(f"  Processing {path.name}...")
        try:
            # Load and preprocess minimally (clip-processor expects PIL image)
            img = Image.open(path).convert("RGB")
            
            # Embed image in CLIP shared space
            inputs = embedder._processor(images=[img], return_tensors="pt")
            with torch.no_grad():
                outputs = embedder._model.get_image_features(**inputs)
                if hasattr(outputs, "pooler_output"):
                    feats = outputs.pooler_output
                else:
                    feats = outputs
            
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            embedding = feats[0].cpu().tolist()

            # Update DB
            # 1. Update model_embed registry
            db.execute(
                "UPDATE images SET model_embed = ? WHERE id = ?",
                ("openai/clip-vit-base-patch32", row_id),
            )
            # 2. Rewrite vector inside vec_images (idempotent delete then insert)
            db.execute("DELETE FROM vec_images WHERE image_id = ?", (row_id,))
            db.execute(
                "INSERT INTO vec_images (image_id, embedding) VALUES (?, ?)",
                (row_id, serialize_f32(embedding)),
            )
            db.commit()
            print(f"    ✓ Updated embedding (512-d) saved to database.")

        except Exception as e:
            print(f"    ✗ Failed to re-embed: {e}")

    print("\n" + "=" * 60)
    print("  Re-embedding complete! ✓")
    print(f"  Database updated in place at: {DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
