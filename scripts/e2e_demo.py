#!/usr/bin/env python3
"""End-to-end demo of the image-indexer pipeline.

Runs the full data flow WITHOUT real models or RunPod:

    test image → preprocess → base64 → handler (mocked inference)
    → SQLite upsert → semantic/lexical/structured search

Usage:
    python scripts/e2e_demo.py
"""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Ensure the project root is importable when run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "worker"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from image_indexer import db  # noqa: E402
from image_indexer.preprocess import preprocess  # noqa: E402


# ---------------------------------------------------------------------------
# Fake models: generate a deterministic embedding + caption from the image
# content so each "photo" gets a unique, queryable result.
# ---------------------------------------------------------------------------
EMBED_DIM = 512


def fake_embed_image(image: Image.Image) -> list[float]:
    """Generate a deterministic 1152-d embedding based on dominant colour."""
    # Sample the centre pixel to get a "colour signature".
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    px = arr[h // 2, w // 2].astype(float)
    # Tile the 3 RGB values across 1152 dims, add small noise for uniqueness.
    vec = np.tile(px, 1152 // 3 + 1)[:EMBED_DIM]
    rng = np.random.default_rng(seed=int(px.sum()))
    vec = vec + rng.normal(0, 0.01, EMBED_DIM)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def fake_caption_image(image: Image.Image) -> str:
    """Generate a deterministic caption based on the image's dominant colour."""
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    px = arr[h // 2, w // 2]
    r, g, b = int(px[0]), int(px[1]), int(px[2])
    # Simple colour-name heuristic for the demo.
    if r > max(g, b):
        colour = "red"
    elif g > max(r, b):
        colour = "green"
    elif b > max(r, g):
        colour = "blue"
    else:
        colour = "grey"
    return (
        f"A {colour} test image ({w}x{h}). Solid background, no subjects. "
        f"RGB centre pixel: ({r}, {g}, {b}). Studio lighting, no text visible."
    )


# ---------------------------------------------------------------------------
# Test image generator
# ---------------------------------------------------------------------------
def make_test_image(path: Path, colour: tuple[int, int, int], label: str) -> Path:
    """Create a labelled solid-colour JPEG for the demo."""
    img = Image.new("RGB", (3000, 2000), colour)  # 6 MP — triggers resize
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 120
        )
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((3000 - tw) // 2, (2000 - th) // 2), label, fill="white", font=font)
    img.save(path, "JPEG", quality=95)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  image-indexer  —  end-to-end pipeline demo")
    print("=" * 60)

    # 1. Create test images
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        photos = [
            ("red_sunset.jpg", (200, 60, 30), "RED SUNSET"),
            ("green_forest.jpg", (20, 160, 50), "GREEN FOREST"),
            ("blue_ocean.jpg", (30, 80, 200), "BLUE OCEAN"),
        ]

        print("\n[1/5] Creating test images...")
        image_paths = []
        for name, colour, label in photos:
            p = make_test_image(tmp_path / name, colour, label)
            size_kb = p.stat().st_size / 1024
            print(f"  {name:25s}  {size_kb:6.0f} KB   colour={colour}")
            image_paths.append(p)

        # 2. Preprocess
        print("\n[2/5] Preprocessing (resize + JPEG encode)...")
        preprocessed = []
        for p in image_paths:
            result = preprocess(p)
            assert not result.skipped, f"preprocess failed: {result.skip_reason}"
            out_kb = len(result.jpeg_bytes) / 1024
            b64_kb = len(base64.b64encode(result.jpeg_bytes)) / 1024
            print(
                f"  {p.name:25s}  "
                f"{result.orig_width}x{result.orig_height} → "
                f"{result.resized_width}x{result.resized_height}  "
                f"jpeg={out_kb:.0f} KB  b64={b64_kb:.0f} KB"
            )
            preprocessed.append(result)

        # 3. Send through handler (with mocked models)
        print("\n[3/5] Running handler (mocked inference)...")
        with (
            patch("handler.load_models"),
            patch("handler.embed_image", side_effect=fake_embed_image),
            patch("handler.caption_image", side_effect=fake_caption_image),
        ):
            from handler import handler  # ty: ignore[unresolved-import]

            responses = []
            for prep in preprocessed:
                b64_str = base64.b64encode(prep.jpeg_bytes).decode()
                job = {"input": {"image_b64": b64_str, "task": "all"}}
                resp = handler(job)
                assert "error" not in resp, f"Handler failed: {resp}"
                assert resp["embedding_dim"] == EMBED_DIM
                assert len(resp["embedding"]) == EMBED_DIM
                responses.append(resp)
                caption_preview = resp["description"][:60]
                print(
                    f"  {prep.path.name:25s}  dim={resp['embedding_dim']}  caption='{caption_preview}...'"
                )

        # 4. Store in SQLite
        print("\n[4/5] Storing in SQLite (in-memory DB)...")
        conn = db.connect(":memory:")
        ids = []
        for prep, resp in zip(preprocessed, responses):
            meta = {
                "path": str(prep.path),
                "sha256": prep.sha256,
                "file_size": prep.file_size,
                "format": prep.disk_format,
                "width": prep.orig_width,
                "height": prep.orig_height,
                "description": resp["description"],
                "model_caption": "fake-caption-model",
                "model_embed": "fake-embed-model",
            }
            row_id = db.upsert_image(conn, meta, embedding=resp["embedding"])
            ids.append(row_id)
            print(
                f"  upserted id={row_id}  sha={prep.sha256[:16]}...  path={prep.path.name}"
            )

        # 5. Query
        print("\n[5/5] Querying the database...")

        # Semantic: search with an embedding close to "red"
        red_vec = np.zeros(EMBED_DIM, dtype=np.float32)
        red_vec[0] = 200 / 255  # red channel dominates
        red_vec = (red_vec / np.linalg.norm(red_vec)).tolist()
        semantic = db.search_semantic(conn, red_vec, k=3)
        print("\n  Semantic search (query ≈ red):")
        for row in semantic:
            print(
                f"    id={row['image_id']}  dist={row['distance']:.4f}  path={row['path']}"
            )

        # Lexical: search for "green"
        lexical = db.search_lexical(conn, "green", k=3)
        print("\n  Lexical search ('green'):")
        for row in lexical:
            print(f"    id={row['id']}  score={row['score']:.3f}  path={row['path']}")

        # Structured: filter by file size
        structured = db.search_structured(conn, "file_size > 0")
        print("\n  Structured search (all rows, file_size > 0):")
        for row in structured:
            print(
                f"    id={row['id']}  {row['width']}x{row['height']}  {row['description'][:50]}..."
            )

    print(f"\n{'=' * 60}")
    print(f"  All {len(photos)} images processed end-to-end. Pipeline works! ✓")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
