# Image Indexer

An on-device visual memory search engine. Find anything with pixels by what you remember, not what you named it.

```bash
idx search "philosophers studying geometry fresco" --semantic
idx search "ssh config authorized_keys" --lexical
idx search "width > 3000 AND format = 'PNG'" --structured
```

It supports everything with pixels: screenshots of terminal errors, whiteboard scribbles, system diagrams, slides, photos, and academic quotes.

---

## Design Principles

1. **Vibes over Precision** — You don't need exact filenames. Ask for the visual vibe ("swirling cosmic light", "dark terminal screenshot") or textual cues ("error 500", "Bayes theorem").
2. **First-Class OCR** — Pixels are text. Screenshots and photos are parsed under Qwen3-VL descriptors, feeding local FTS5 search surfaces directly.
3. **Hybrid Query Engine** — Three composable search surfaces inside one single SQLite database:
   - **Semantic** — OpenAI CLIP ViT-B-32 visual space (512-d text → image cosine similarity).
   - **Lexical** — Pure FTS5 full-text indexing over Qwen3-VL details and transcribed OCR text.
   - **Structured** — Standard relational SQL filters over EXIF, camera, lens, dimensions, and and filesystem metadata.
4. **Local-First, Cloud-Compute** — Heavy generative descriptors (Qwen3-VL) are offloaded to RunPod Serverless GPUs during index periods. Retrieval is 100% on-device (local CLIP models of ~150MB run instantly on standard CPUs).
5. **rsync-Friendly Ingestion** — Point-and-sync. No file watchers required. Intact Mac screenshots, downloads, and pictures sync to your Hetzner box via rsync, indexing via periodic cron-jobs.
6. **Agentic & Pipe-Ready** — Returns parsed JSON formats via `--json` flags. No user-interactive prompts. Strict exit codes (0=Success, 1=User error, 2=System error, 3=Partial failure) for automated processes.

---

## Core Flow

```
[ Local Files ]
       │
       ▼   (Local Preprocessing)
  preprocess ──► 1MP downscale, extract original EXIF and SHA-256
       │
       ▼   (Non-blocking Concurrency — Async Semaphore limit: 10)
[ RunPod GPU ] ──► Compute CLIP embedding + Qwen3-VL captions & OCR
       │
       ▼   (Serialised SQLite Writes)
 [ SQLite DB ] ──► Store inside vec0 (embeddings), FTS5 (lexical), and images (relational)
       │
       ▼   (Local Retrieval — instant CPU execution)
  idx search ──► Local CLIP text-encode + sqlite-vec cosine KNN
```

---

## CLI Usage

Configure endpoints in `image_indexer/settings.toml` or override through environment variables.

### Indexing

The `index` command runs on a fully asynchronous pipeline, processing multiple images concurrently while safely serializing database writes in real-time.

```bash
# Dry run — test local downscaling & metadata parsing without RunPod
idx index /path/to/photos --dry-run --verbose

# Production ingestion (submits async blocks to RunPod Serverless)
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="rpa_your-api-key"
idx index /path/to/photos --json
```

### Searching

No credentials or internet connections are required to query your indices.

```bash
# Semantic "vibe" search (local text encoder)
idx search "starry night sky or galaxy" --semantic

# Lexical search over transcribed text or rich captions
idx search "docker port conflict Exception" --lexical --json

# Structured relational SQL filters
idx search "camera_model = 'X-T5' AND focal_length = 23" --structured

# Composed mixed search (lexical + structured)
idx search "whiteboard" --lexical --structured --where "format = 'PNG'"
```

### Statistics

```bash
idx status --json
```

---

## Verification & Testing

Tests cover downscaling thresholds, correct EXIF transpositions, database triggers, and HTTP client retry loops.

```bash
# Run full suite (50 passing tests)
uv run pytest -v

# Run local end-to-end pipeline validation (mocks OpenAI/RunPod calls)
uv run python scripts/e2e_demo.py
```
