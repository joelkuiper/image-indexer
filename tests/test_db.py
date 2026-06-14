"""Database-layer tests. These hit a REAL sqlite-vec + FTS5 database (in a temp
file) — this is the part that must genuinely work, so we don't mock it.
"""

import math
import tempfile
import unittest
from pathlib import Path

from image_indexer import db
from image_indexer.db import EMBED_DIM


def _vec(seed: float) -> list[float]:
    """Deterministic unit-ish vector of the right dimension."""
    v = [math.sin(seed + i) for i in range(EMBED_DIM)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _meta(path: str, sha: str, **over) -> dict:
    base = {
        "path": path,
        "sha256": sha,
        "file_size": 1234,
        "format": "JPEG",
        "width": 4000,
        "height": 3000,
        "camera_make": "FUJIFILM",
        "camera_model": "X-T5",
        "lens_model": "XF23mmF1.4",
        "focal_length": 23.0,
        "aperture": 1.4,
        "iso": 200,
        "shutter_speed": "1/250",
        "datetime_original": "2026-06-14T09:00:00",
        "gps_lat": 52.37,
        "gps_lon": 4.90,
        "description": "A black cat sleeping in warm afternoon sunlight by a window.",
        "model_caption": "Qwen/Qwen3-VL-4B-Instruct",
        "model_embed": "google/siglip2-so400m-patch16-384",
        "file_mtime": "2026-06-14T08:00:00",
    }
    base.update(over)
    return base


class TestDbLayer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = db.connect(Path(self.tmp.name) / "test.db")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_migration_creates_all_surfaces(self):
        names = {
            r[0]
            for r in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            )
        }
        self.assertIn("images", names)
        self.assertIn("images_fts", names)
        self.assertIn("vec_images", names)
        version = self.db.execute(
            "SELECT max(version) FROM schema_migrations"
        ).fetchone()[0]
        self.assertEqual(version, 1)

    def test_upsert_and_dedup(self):
        i1 = db.upsert_image(self.db, _meta("/a.jpg", "sha-a"), _vec(0.0))
        # Same sha, different path content -> updates in place, no new row.
        i2 = db.upsert_image(self.db, _meta("/a.jpg", "sha-a", iso=400), _vec(0.0))
        self.assertEqual(i1, i2)
        count = self.db.execute("SELECT count(*) FROM images").fetchone()[0]
        self.assertEqual(count, 1)
        self.assertEqual(
            self.db.execute("SELECT iso FROM images WHERE id=?", (i1,)).fetchone()[0],
            400,
        )

    def test_embedding_dim_guard(self):
        with self.assertRaises(ValueError):
            db.upsert_image(self.db, _meta("/b.jpg", "sha-b"), [0.0] * 10)

    def test_semantic_search(self):
        db.upsert_image(self.db, _meta("/cat.jpg", "sha-cat"), _vec(0.0))
        db.upsert_image(self.db, _meta("/dog.jpg", "sha-dog"), _vec(50.0))
        # Query close to the cat vector should rank cat first.
        results = db.search_semantic(self.db, _vec(0.01), k=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["path"].endswith("cat.jpg"))

    def test_lexical_search(self):
        db.upsert_image(self.db, _meta("/cat.jpg", "sha-cat"))
        db.upsert_image(
            self.db,
            _meta(
                "/beach.jpg",
                "sha-beach",
                description="A sunny beach with blue waves and surfers.",
            ),
        )
        results = db.search_lexical(self.db, "cat", k=5)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["path"].endswith("cat.jpg"))

    def test_lexical_search_camera_field(self):
        db.upsert_image(self.db, _meta("/x.jpg", "sha-x"))
        results = db.search_lexical(self.db, "FUJIFILM", k=5)
        self.assertEqual(len(results), 1)

    def test_structured_search(self):
        db.upsert_image(self.db, _meta("/x.jpg", "sha-x", camera_model="X-T5"))
        db.upsert_image(self.db, _meta("/y.jpg", "sha-y", camera_model="X100VI"))
        results = db.search_structured(self.db, "camera_model = ?", ("X-T5",))
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["path"].endswith("x.jpg"))

    def test_fts_synced_on_delete(self):
        i = db.upsert_image(self.db, _meta("/gone.jpg", "sha-gone"))
        self.db.execute("DELETE FROM images WHERE id=?", (i,))
        self.db.commit()
        self.assertEqual(len(db.search_lexical(self.db, "cat", k=5)), 0)


if __name__ == "__main__":
    unittest.main()
