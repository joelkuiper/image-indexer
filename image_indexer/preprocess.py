"""Image pre-processing for the on-device pipeline.

Responsibilities:
  1. Extract SHA-256 of the *original* file (dedup key).
  2. Read basic file-level metadata (size, mtime, on-disk format, original dimensions).
  3. Apply EXIF orientation so the pixels match what a human sees.
  4. Resize to a max ~1 MP bounding box (Lanczos) — keeps enough detail for
     both SigLIP2 (384 px input) and Qwen3-VL (dynamic tiles) while cutting
     a 40 MP photo to ~150 KB JPEG for the RunPod payload.
  5. Encode as optimised JPEG bytes (ready for base64 + HTTP POST).

The file is read once into memory; the same bytes serve both SHA-256 hashing
and Pillow decoding, so large raw/HEIC files aren't read from disk twice.

HEIC/HEIF formats (common for iPhone photos) require ``pillow-heif`` to be
installed as a Pillow plugin. If missing, those files are skipped gracefully.
Install the ``image-indexer[heif]`` extra to enable them.
"""
from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

log = logging.getLogger(__name__)

# Register pillow-heif plugin if available so HEIC/HEIF files can be decoded.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    pillow_heif.register_avif_opener()
    log.debug("pillow-heif loaded — HEIC/HEIF/AVIF support enabled")
except ImportError:
    log.info("pillow-heif not installed — HEIC/HEIF/AVIF files will be skipped")

# ~1 MP bounding box. 1024x1024 = 1 048 576 px. Landscape photos end up
# 1024x_wide_, portrait _tall_; panoramas are clipped to fit within the box.
MAX_DIMENSION = 1024
JPEG_QUALITY = 85
IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff", ".tif", ".bmp",
})


@dataclass
class PreprocessedImage:
    """Everything the pipeline needs from one source photo.

    Fields:
        path:            Absolute path to the on-disk source file.
        sha256:          Hex digest of the raw file bytes (dedup key).
        file_size:       On-disk size in bytes.
        orig_width:      Width as stored on disk (before EXIF transpose).
        orig_height:     Height as stored on disk (before EXIF transpose).
        resized_width:   Width after preprocess (post-EXIF + resize).
        resized_height:  Height after preprocess.
        disk_format:     Container format as reported by Pillow from the source
                         (e.g. "JPEG", "PNG", "HEIF"). Uppercase when known.
        jpeg_bytes:      Resized, JPEG-encoded bytes ready for base64/POST.
        skipped:         True when the file could not be processed.
        skip_reason:     Human-readable reason when ``skipped=True``.
    """
    path: Path
    sha256: str
    file_size: int
    orig_width: int
    orig_height: int
    resized_width: int
    resized_height: int
    disk_format: str
    jpeg_bytes: bytes
    skipped: bool = False
    skip_reason: str = ""


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def is_image_file(path: Path) -> bool:
    """Quick extension check before attempting to open."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _fit_within(w: int, h: int, max_dim: int) -> tuple[int, int]:
    """Scale (w, h) to fit within max_dim x max_dim, preserving aspect ratio.

    If both dimensions already fit, the original values are returned unchanged.
    """
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

    Returns a ``PreprocessedImage`` with ``jpeg_bytes`` ready for base64
    encoding. If the file cannot be opened or decoded, the returned object
    has ``skipped=True`` and ``skip_reason`` populated; the SHA-256 is still
    set to the on-disk file hash so the caller can record a rejected row.
    """
    path = Path(path)

    # Single-read: load bytes once, use for both hash and Pillow.
    try:
        raw_bytes = path.read_bytes()
    except OSError as e:
        return PreprocessedImage(
            path=path, sha256="", file_size=0,
            orig_width=0, orig_height=0, resized_width=0, resized_height=0,
            disk_format="unknown", jpeg_bytes=b"",
            skipped=True, skip_reason=f"cannot read file: {e}",
        )

    file_size = len(raw_bytes)
    digest = sha256_bytes(raw_bytes)

    try:
        raw = Image.open(io.BytesIO(raw_bytes))
    except Exception as e:
        return PreprocessedImage(
            path=path, sha256=digest, file_size=file_size,
            orig_width=0, orig_height=0, resized_width=0, resized_height=0,
            disk_format="unknown", jpeg_bytes=b"",
            skipped=True, skip_reason=f"cannot decode image: {e}",
        )

    try:
        # On-disk dimensions, reported before any transformation.
        orig_w, orig_h = raw.size
        disk_format = (raw.format or path.suffix.lstrip(".").lstrip(".").upper())

        # EXIF orientation: rotate/flip pixels so "up" is actually up.
        img = ImageOps.exif_transpose(raw)
        if img is None:
            return PreprocessedImage(
                path=path, sha256=digest, file_size=file_size,
                orig_width=orig_w, orig_height=orig_h,
                resized_width=0, resized_height=0,
                disk_format=disk_format, jpeg_bytes=b"",
                skipped=True, skip_reason="exif_transpose returned None",
            )

        img = img.convert("RGB")
        new_w, new_h = _fit_within(img.size[0], img.size[1], max_dimension)
        if (new_w, new_h) != img.size:
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Preserve ICC profile when present so colours aren't washed out.
        icc_profile = raw.info.get("icc_profile")

        buf = io.BytesIO()
        img.save(
            buf,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            subsampling=0,
            icc_profile=icc_profile,
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
            disk_format=disk_format,
            jpeg_bytes=jpeg_bytes,
        )

    except Exception as e:  # noqa: BLE001
        log.warning("preprocess failed for %s: %s", path, e)
        return PreprocessedImage(
            path=path, sha256=digest, file_size=file_size,
            orig_width=0, orig_height=0, resized_width=0, resized_height=0,
            disk_format="unknown", jpeg_bytes=b"",
            skipped=True, skip_reason=str(e),
        )


def scan_directory(directory: Path) -> list[Path]:
    """Recursively find all image files under *directory*, sorted by path.

    Silently skips paths that fail ``stat()`` (unreadable, broken symlink).
    """
    found: list[Path] = []
    for p in Path(directory).rglob("*"):
        try:
            if not p.is_file():
                continue
        except OSError:
            continue
        if is_image_file(p):
            found.append(p)
    found.sort()
    return found
