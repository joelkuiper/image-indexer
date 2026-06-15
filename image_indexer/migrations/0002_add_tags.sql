-- Add tags column for path-based auto-tagging
-- Migration 0002

ALTER TABLE images ADD COLUMN tags TEXT;

-- We also make sure the tags column is indexed for fast lookups
CREATE INDEX IF NOT EXISTS idx_images_tags ON images(tags);

-- We recreate the FTS5 virtual table and its triggers to also index the tags column
DROP TABLE IF EXISTS images_fts;

CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
    description,
    camera_make,
    camera_model,
    lens_model,
    path,
    tags,
    content='images',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Recreate triggers to sync fully with the new columns
DROP TRIGGER IF EXISTS images_ai;
DROP TRIGGER IF EXISTS images_ad;
DROP TRIGGER IF EXISTS images_au;

CREATE TRIGGER images_ai AFTER INSERT ON images BEGIN
    INSERT INTO images_fts(rowid, description, camera_make, camera_model, lens_model, path, tags)
    VALUES (new.id, new.description, new.camera_make, new.camera_model, new.lens_model, new.path, new.tags);
END;

CREATE TRIGGER images_ad AFTER DELETE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, description, camera_make, camera_model, lens_model, path, tags)
    VALUES ('delete', old.id, old.description, old.camera_make, old.camera_model, old.lens_model, old.path, old.tags);
END;

CREATE TRIGGER images_au AFTER UPDATE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, description, camera_make, camera_model, lens_model, path, tags)
    VALUES ('delete', old.id, old.description, old.camera_make, old.camera_model, old.lens_model, old.path, old.tags);
    INSERT INTO images_fts(rowid, description, camera_make, camera_model, lens_model, path, tags)
    VALUES (new.id, new.description, new.camera_make, new.camera_model, new.lens_model, new.path, new.tags);
END;
