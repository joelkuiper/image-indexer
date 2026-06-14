"""Tests for image_indexer.preprocess."""

import io
from pathlib import Path

import pytest
from PIL import Image
from image_indexer.preprocess import (
    MAX_DIMENSION,
    PreprocessedImage,
    _fit_within,
    is_image_file,
    preprocess,
    scan_directory,
    sha256_bytes,
)


@pytest.fixture
def tiny_image(tmp_path: Path) -> Path:
    """A 100x80 RGB JPEG — small enough to not trigger resizing."""
    p = tmp_path / "tiny.jpg"
    Image.new("RGB", (100, 80), "red").save(p, "JPEG")
    return p


@pytest.fixture
def huge_image(tmp_path: Path) -> Path:
    """A 4000x3000 RGB JPEG — 12 MP, must be resized."""
    p = tmp_path / "huge.jpg"
    Image.new("RGB", (4000, 3000), "blue").save(p, "JPEG")
    return p


@pytest.fixture
def square_image(tmp_path: Path) -> Path:
    """A 2048x2048 PNG — should resize to 1024x1024."""
    p = tmp_path / "square.png"
    Image.new("RGB", (2048, 2048), "green").save(p, "PNG")
    return p


# --- _fit_within ---


def test_fit_within_no_resize():
    assert _fit_within(500, 500, 1024) == (500, 500)


def test_fit_within_landscape():
    w, h = _fit_within(4000, 3000, 1024)
    assert w == 1024
    assert h == 768  # 3000 * (1024/4000) = 768


def test_fit_within_portrait():
    w, h = _fit_within(3000, 4000, 1024)
    assert w == 768
    assert h == 1024


def test_fit_within_square():
    w, h = _fit_within(2048, 2048, 1024)
    assert w == 1024
    assert h == 1024


# --- is_image_file ---


def test_is_image_file_yes():
    assert is_image_file(Path("photo.jpg")) is True
    assert is_image_file(Path("photo.JPEG")) is True
    assert is_image_file(Path("scan.tiff")) is True
    assert is_image_file(Path("raw.heic")) is True


def test_is_image_file_no():
    assert is_image_file(Path("readme.md")) is False
    assert is_image_file(Path("script.py")) is False
    assert is_image_file(Path("")) is False


# --- sha256_file ---


def test_sha256_bytes_deterministic():
    data = b"hello world"
    assert sha256_bytes(data) == sha256_bytes(data)
    assert sha256_bytes(data) != sha256_bytes(b"different")


# --- preprocess ---


def test_preprocess_small_image_unchanged(tiny_image: Path):
    result = preprocess(tiny_image)
    assert not result.skipped
    assert result.orig_width == 100
    assert result.orig_height == 80
    assert result.resized_width == 100
    assert result.resized_height == 80
    assert len(result.jpeg_bytes) > 0
    assert result.sha256  # non-empty hex digest
    assert result.disk_format == "JPEG"


def test_preprocess_large_image_resized(huge_image: Path):
    result = preprocess(huge_image)
    assert not result.skipped
    assert result.orig_width == 4000
    assert result.orig_height == 3000
    assert result.resized_width == 1024
    assert result.resized_height == 768
    assert max(result.resized_width, result.resized_height) <= MAX_DIMENSION


def test_preprocess_square_resized(square_image: Path):
    result = preprocess(square_image)
    assert not result.skipped
    assert result.orig_width == 2048
    assert result.resized_width == 1024
    assert result.resized_height == 1024


def test_preprocess_output_is_valid_jpeg(tiny_image: Path):
    result = preprocess(tiny_image)
    # Pillow should be able to re-open the output bytes.
    img = Image.open(io.BytesIO(result.jpeg_bytes))
    assert img.format == "JPEG"
    assert img.size == (result.resized_width, result.resized_height)


def test_preprocess_compression_ratio(huge_image: Path):
    """The resized JPEG should be dramatically smaller than 12 MP source."""
    original_size = huge_image.stat().st_size
    result = preprocess(huge_image)
    assert not result.skipped
    # Even a solid-colour 12 MP JPEG is ~100KB; the resized one should be tiny.
    assert len(result.jpeg_bytes) < original_size


def test_preprocess_broken_file(tmp_path: Path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not a real image")
    result = preprocess(broken)
    assert result.skipped
    assert result.skip_reason
    # SHA-256 is still set so the caller can record a rejected row.
    assert result.sha256


def test_preprocess_unreadable_file(tmp_path: Path):
    """File we can't stat/read returns skipped with empty sha256."""
    missing = tmp_path / "does_not_exist.jpg"
    result = preprocess(missing)
    assert result.skipped
    assert "cannot read" in result.skip_reason
    assert result.sha256 == ""


def test_scan_directory_skips_unreadable(tmp_path: Path):
    """Broken symlinks should not crash the scanner."""
    (tmp_path / "good.jpg").write_bytes(b"fake")
    bad = tmp_path / "bad.jpg"
    bad.symlink_to(tmp_path / "nonexistent.jpg")
    found = scan_directory(tmp_path)
    names = [p.name for p in found]
    assert "good.jpg" in names
    # bad.jpg may or may not appear (depends on resolution); at least no crash.


def test_preprocess_sha256_is_of_original(huge_image: Path):
    """SHA-256 must be of the *source* file, not the resized output."""
    result = preprocess(huge_image)
    assert result.sha256 == sha256_bytes(huge_image.read_bytes())


# --- scan_directory ---


def test_scan_directory_finds_images(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"fake")
    (tmp_path / "b.png").write_bytes(b"fake")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.heic").write_bytes(b"fake")
    (tmp_path / "notes.txt").write_text("not an image")

    found = scan_directory(tmp_path)
    names = [p.name for p in found]
    assert "a.jpg" in names
    assert "b.png" in names
    assert "c.heic" in names
    assert "notes.txt" not in names
    assert len(found) == 3
