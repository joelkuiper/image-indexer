"""SQLite data layer for image-indexer.

Loads the sqlite-vec extension, applies SQL migrations, and exposes the three
search surfaces: structured (SQL), lexical (FTS5), and semantic (vec0).

The embedding dimension is fixed at EMBED_DIM to match
our local text embedder. If you swap encoders, bump the migration.
"""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Iterable, Sequence

import sqlite_vec

EMBED_DIM = 512
MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def serialize_f32(vector: Sequence[float]) -> bytes:
    """Pack a list of floats into the compact little-endian blob sqlite-vec wants."""
    return struct.pack(f"{len(vector)}f", *vector)


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with sqlite-vec loaded and the schema applied."""
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    apply_migrations(db)
    return db


def apply_migrations(db: sqlite3.Connection) -> None:
    """Run any .sql files in migrations/ not yet recorded in schema_migrations."""
    # Ensure the bookkeeping table exists before we query it.
    db.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, "
        "applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
    )
    applied = {row[0] for row in db.execute("SELECT version FROM schema_migrations")}

    for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
        version = int(sql_file.name.split("_", 1)[0])
        if version in applied:
            continue
        db.executescript(sql_file.read_text())
        db.commit()


def upsert_image(
    db: sqlite3.Connection, meta: dict, embedding: Sequence[float] | None = None
) -> int:
    """Insert or update one image row + its vector. Returns the image id.

    Dedup is on sha256: an unchanged file updates in place rather than duplicating.
    """
    if embedding is not None and len(embedding) != EMBED_DIM:
        raise ValueError(
            f"embedding has dim {len(embedding)}, expected {EMBED_DIM}"
        )

    cols = [
        "path",
        "sha256",
        "file_size",
        "format",
        "width",
        "height",
        "camera_make",
        "camera_model",
        "lens_model",
        "focal_length",
        "aperture",
        "iso",
        "shutter_speed",
        "datetime_original",
        "gps_lat",
        "gps_lon",
        "description",
        "model_caption",
        "model_embed",
        "file_mtime",
    ]
    values = [meta.get(c) for c in cols]
    placeholders = ", ".join("?" for _ in cols)
    update_clause = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "path")

    cur = db.execute(
        f"INSERT INTO images ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(sha256) DO UPDATE SET {update_clause}, updated_at=datetime('now') "
        f"RETURNING id",
        values,
    )
    image_id = cur.fetchone()[0]

    if embedding is not None:
        # vec0 has no UPSERT; delete-then-insert keeps it idempotent.
        db.execute("DELETE FROM vec_images WHERE image_id = ?", (image_id,))
        db.execute(
            "INSERT INTO vec_images (image_id, embedding) VALUES (?, ?)",
            (image_id, serialize_f32(embedding)),
        )
    db.commit()
    return image_id


def search_semantic(
    db: sqlite3.Connection, query_embedding: Sequence[float], k: int = 10
):
    """Vector KNN. Pass a CLIP text OR image embedding (same space)."""
    rows = db.execute(
        "SELECT v.image_id, v.distance, i.path, i.description "
        "FROM vec_images v JOIN images i ON i.id = v.image_id "
        "WHERE v.embedding MATCH ? AND k = ? "
        "ORDER BY v.distance",
        (serialize_f32(query_embedding), k),
    ).fetchall()
    return [dict(r) for r in rows]


def search_lexical(db: sqlite3.Connection, query: str, k: int = 10):
    """FTS5 full-text search over description + camera fields."""
    rows = db.execute(
        "SELECT i.id, i.path, i.description, bm25(images_fts) AS score "
        "FROM images_fts JOIN images i ON i.id = images_fts.rowid "
        "WHERE images_fts MATCH ? "
        "ORDER BY score LIMIT ?",
        (query, k),
    ).fetchall()
    return [dict(r) for r in rows]


def search_structured(
    db: sqlite3.Connection, where: str, params: Iterable = ()
):
    """Plain SQL filter over structured columns, e.g. "camera_model = ?"."""
    rows = db.execute(
        f"SELECT * FROM images WHERE {where} ORDER BY datetime_original DESC",
        tuple(params),
    ).fetchall()
    return [dict(r) for r in rows]
