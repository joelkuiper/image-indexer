"""Tests for image_indexer CLI (idx)."""
import json
import subprocess
import tempfile
from pathlib import Path

import pytest
from PIL import Image


def run_idx(*args, **kwargs):
    """Run `idx` as a subprocess and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["uv", "run", "idx", *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        **kwargs,
    )
    return result.returncode, result.stdout, result.stderr


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def photo_dir(tmp_path):
    """A temp dir with 3 small test images."""
    d = tmp_path / "photos"
    d.mkdir()
    for name, colour in [("red.jpg", "red"), ("green.jpg", "green"), ("blue.jpg", "blue")]:
        Image.new("RGB", (100, 100), colour).save(d / name, "JPEG")
    return str(d)


class TestStatus:
    def test_empty_db_json(self, db_path):
        rc, out, err = run_idx("status", "--json", "--db", db_path)
        assert rc == 0
        data = json.loads(out)
        assert data["images"] == 0
        assert data["last_indexed"] is None

    def test_empty_db_human(self, db_path):
        rc, out, err = run_idx("status", "--db", db_path)
        assert rc == 0
        assert "Images:   0" in out


class TestSearch:
    def test_no_mode_exits_1(self, db_path):
        rc, out, err = run_idx("search", "hello", "--db", db_path)
        assert rc == 1
        assert "pick" in err

    def test_lexical_empty_db(self, db_path):
        rc, out, err = run_idx("search", "hello", "--lexical", "--json", "--db", db_path)
        assert rc == 0
        data = json.loads(out)
        assert data == []


class TestIndex:
    def test_dry_run(self, photo_dir, db_path):
        rc, out, err = run_idx("index", photo_dir, "--dry-run", "--json", "--db", db_path)
        assert rc == 0
        data = json.loads(out)
        assert data["indexed"] == 3
        assert data["failed"] == 0

    def test_dry_run_verbose(self, photo_dir, db_path):
        rc, out, err = run_idx(
            "index", photo_dir, "--dry-run", "--verbose", "--json", "--db", db_path
        )
        assert rc == 0
        assert "Scanning" in err
        assert "Found 3" in err

    def test_dry_run_idempotent(self, photo_dir, db_path):
        """Running index twice dry-run still shows 3 indexed (dry-run doesn't persist)."""
        run_idx("index", photo_dir, "--dry-run", "--json", "--db", db_path)
        rc, out, _ = run_idx("index", photo_dir, "--dry-run", "--json", "--db", db_path)
        assert rc == 0
        data = json.loads(out)
        assert data["indexed"] == 3

    def test_no_endpoint_exits_1(self, photo_dir, db_path):
        """Without --endpoint-id and not --dry-run, exits with user error."""
        rc, out, err = run_idx("index", photo_dir, "--json", "--db", db_path)
        assert rc == 1
        assert "endpoint-id" in err.lower()
