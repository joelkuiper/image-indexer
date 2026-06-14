"""Image pre-processing for the on-device pipeline.

Responsibilities:
  1. Extract SHA-256 of the *original* file (dedup key).
  2. Read basic file-level metadata (size, mtime, format, original dimensions).
  3. Apply EXIF orientation so the pixels match what a human sees.
  4. Resize to a max ~1 MP bounding box (Lanczos) — keeps enough detail for
     both SigLIP2 (384 px input) and Qwen3-VL (dynamic tiles) while cutting
     a 40 MP photo to ~150 KB JPEG for the RunPod payload.
  5. Encode as optimised JPEG bytes (ready for base64 + HTTP POST).
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

# ~1 MP bounding box. 1024x1024 = 1 048 576 px.  Landscape photos will be
# 1024x_wide_, portrait _tall_, panoramic gets clipped to fit within the box.
MAX_DIMENSION = 1024
JPEG_QUALITY = 85
IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".heic",
        ".heif",
        ".webp",
        ".tiff",
        ".tif",
        ".bmp",
    }
)


@dataclass
class PreprocessedImage:
    """Everything the pipeline needs from one source photo."""

    path: Path
    sha256: str
    file_size: int
    orig_width: int
    orig_height: int
    resized_width: int
    resized_height: int
    format: str
    jpeg_bytes: bytes  # ready for base64 encoding
    skipped: bool = False
    skip_reason: str = ""


def sha256_file(path: Path, chunk_size: int = 1 << 16) -> str:
    """SHA-256 hex digest of the raw file (not the resized output)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def is_image_file(path: Path) -> bool:
    """Quick extension check before attempting to open."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _fit_within(w: int, h: int, max_dim: int) -> tuple[int, int]:
    """Scale (w, h) to fit within max_dim x max_dim, preserving aspect ratio."""
    if w <= max_dim and h <= max_dim:
        return w, h
    scale = max_dim / max(w, h)
    return round(w * scale), round(h * scale)


def preprocess(
    path: Path,
    max_dimension: int = MAX_DIMENSION,
    jpeg_quality: int = JPEG_QUALITY,
) -> PreprocessedImage:
    """Read, fix orientation, resize, encode as JPEG bytes.

    Returns a PreprocessedImage with jpeg_bytes ready for base64 encoding.
    If the file can't be opened, returns with skipped=True and skip_reason set.
    """
    path = Path(path)
    file_size = path.stat().st_size
    digest = sha256_file(path)

    try:
        with Image.open(path) as raw:
            # EXIF orientation — rotate/flip pixels so "up" is actually up.
            img = ImageOps.exif_transpose(raw)
            if img is None:
                return PreprocessedImage(
                    path=path,
                    sha256=digest,
                    file_size=file_size,
                    orig_width=0,
                    orig_height=0,
                    resized_width=0,
                    resized_height=0,
                    format="unknown",
                    jpeg_bytes=b"",
                    skipped=True,
                    skip_reason="exif_transpose returned None",
                )

            img = img.convert("RGB")
            orig_w, orig_h = img.size
            new_w, new_h = _fit_within(orig_w, orig_h, max_dimension)

            if (new_w, new_h) != (orig_w, orig_h):
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            img.save(
                buf, format="JPEG", quality=jpeg_quality, optimize=True, subsampling=0
            )
            jpeg_bytes = buf.getvalue()

            return PreprocessedImage(
                path=path,
                sha256=digest,
                file_size=file_size,
                orig_width=orig_w,
                orig_height=orig_h,
                resized_width=new_w,
                resized_height=new_h,
                format=raw.format or path.suffix.lstrip(".").upper(),
                jpeg_bytes=jpeg_bytes,
            )

    except Exception as e:
        return PreprocessedImage(
            path=path,
            sha256=digest,
            file_size=file_size,
            orig_width=0,
            orig_height=0,
            resized_width=0,
            resized_height=0,
            format="unknown",
            jpeg_bytes=b"",
            skipped=True,
            skip_reason=str(e),
        )


def scan_directory(directory: Path) -> list[Path]:
    """Recursively find all image files under *directory*, sorted by path."""
    return sorted(
        p for p in Path(directory).rglob("*") if p.is_file() and is_image_file(p)
    )
