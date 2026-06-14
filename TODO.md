# Pipeline TODO

## 1. Local & Production CLI Polish
- [x] Configure centralization of settings using `dynaconf` (remove magic dimensions and constants).
- [x] Refactor `idx index` into a highly parallel native `asyncio` event loop.
- [x] Integrate OCR-capturing and local CLIP text encoding (512-d).
- [ ] Add support for path-based auto-tagging upon ingestion (`/screenshots/` vs `/photos/` vs `/downloads/`).
- [ ] Implement query composition on CLI directly (combining `--semantic` and `--structured` with custom where clauses inside one single shell pipeline).

## 2. Ingestion & Sync Automation (Hetzner Box)
- [ ] Write thin rsync scripts on Mac to sync pictures, downloads, and screenshots to the Hetzner target directory.
- [ ] Set up secure key-based SSH parameters (using restricted config commands to bypass agent validation friction).
- [ ] Implement automated indexing via cron scheduler post-rsync sync.

## 3. Signal Integration
- [ ] Build Signal-based listener querying the database for "vibe" parameters.
- [ ] Implement media-attachment responder with transcribed text sent as `alt-text`.
- [ ] Add clean quality reports or curation checks ("You have 23 similar screenshots from today, want to clear them out?").
