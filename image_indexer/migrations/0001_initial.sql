-- image-indexer schema — migration 0001
-- Three search surfaces over one corpus of images:
--   1. structured / relational  -> `images` (EXIF, file metadata)  : plain SQL WHERE filters
--   2. lexical / full-text       -> `images_fts` (FTS5)             : MATCH over description + camera text
--   3. semantic / vector         -> `vec_images` (sqlite-vec vec0)  : SigLIP2 1152-d cosine/L2 search
--
-- Apply via image_indexer.db.connect() which loads the sqlite-vec extension first.
-- Embedding dim is fixed at 1152 to match google/siglip2-so400m-patch16-384.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Structured metadata. One row per indexed image file.
-- sha256 is the dedup key: re-indexing an unchanged file is a no-op upsert.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY,
    path            TEXT    NOT NULL UNIQUE,      -- absolute path on disk
    sha256          TEXT    NOT NULL UNIQUE,      -- content hash, dedup + change detection
    file_size       INTEGER NOT NULL,            -- bytes
    format          TEXT,                         -- JPEG / PNG / WEBP / ...
    width           INTEGER,
    height          INTEGER,

    -- EXIF / camera metadata (all nullable; not every image has these)
    camera_make     TEXT,
    camera_model    TEXT,
    lens_model      TEXT,
    focal_length    REAL,                         -- mm
    aperture        REAL,                         -- f-number
    iso             INTEGER,
    shutter_speed   TEXT,                         -- kept as text: "1/250", "30"
    datetime_original TEXT,                       -- ISO-8601 from EXIF DateTimeOriginal
    gps_lat         REAL,
    gps_lon         REAL,

    -- model-generated content
    description     TEXT,                         -- Qwen3-VL-4B caption
    model_caption   TEXT,                         -- which VLM produced `description`
    model_embed     TEXT,                         -- which encoder produced the vector

    -- bookkeeping
    file_mtime      TEXT,                         -- filesystem mtime, ISO-8601
    indexed_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_images_camera_model ON images(camera_model);
CREATE INDEX IF NOT EXISTS idx_images_datetime     ON images(datetime_original);
CREATE INDEX IF NOT EXISTS idx_images_format       ON images(format);

-- ---------------------------------------------------------------------------
-- Lexical full-text search (FTS5).
-- content='images' makes this an external-content table: the FTS index stores
-- only the inverted index, not a copy of the text, and rowid maps to images.id.
-- Kept in sync by the triggers below.
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
    description,
    camera_make,
    camera_model,
    lens_model,
    path,
    content='images',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Keep FTS in lockstep with the images table.
CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
    INSERT INTO images_fts(rowid, description, camera_make, camera_model, lens_model, path)
    VALUES (new.id, new.description, new.camera_make, new.camera_model, new.lens_model, new.path);
END;

CREATE TRIGGER IF NOT EXISTS images_ad AFTER DELETE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, description, camera_make, camera_model, lens_model, path)
    VALUES ('delete', old.id, old.description, old.camera_make, old.camera_model, old.lens_model, old.path);
END;

CREATE TRIGGER IF NOT EXISTS images_au AFTER UPDATE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, description, camera_make, camera_model, lens_model, path)
    VALUES ('delete', old.id, old.description, old.camera_make, old.camera_model, old.lens_model, old.path);
    INSERT INTO images_fts(rowid, description, camera_make, camera_model, lens_model, path)
    VALUES (new.id, new.description, new.camera_make, new.camera_model, new.lens_model, new.path);
END;

-- ---------------------------------------------------------------------------
-- Semantic vector index (sqlite-vec).
-- vec0 virtual table; rowid is the FK back to images.id (we set it explicitly).
-- float[1152] == google/siglip2-so400m-patch16-384 joint image/text space,
-- so text->image and image->image search both live here.
-- distance_metric=cosine is the right choice for SigLIP-style normalized embeds.
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS vec_images USING vec0(
    image_id INTEGER PRIMARY KEY,
    embedding FLOAT[1152] distance_metric=cosine
);

-- ---------------------------------------------------------------------------
-- Schema version bookkeeping.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT OR IGNORE INTO schema_migrations(version) VALUES (1);
