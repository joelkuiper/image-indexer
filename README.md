# idx — Visual Memory Search

Find anything with pixels by what you remember, not what you named it.

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
   - **Structured** — Standard relational SQL filters over EXIF, camera, lens, dimensions, and filesystem metadata.

4. **Local-First, Cloud-Optional** — Heavy generative descriptors (Qwen3-VL) run locally on your GPU. Retrieval is 100% on-device (local CLIP models of ~150MB run instantly on standard CPUs).

5. **rsync-Friendly Ingestion** — Point-and-sync. No file watchers required. Intact Mac screenshots, downloads, and pictures sync to your index directory via rsync, indexing via periodic cron-jobs.

6. **Agentic & Pipe-Ready** — Returns parsed JSON formats via `--json` flags. No user-interactive prompts. Strict exit codes (0=Success, 1=User error, 2=System error, 3=Partial failure) for automated processes.

---

## Core Flow

```
[ Local Files ]
       │
       ▼   (Local Preprocessing)
  preprocess ──► 1MP downscale, extract original EXIF and SHA-256
       │
       ▼   (Non-blocking Concurrency — Async Semaphore limit: 5)
[ Inference ] ──► CLIP embedding + Qwen3-VL captions & OCR (local or remote)
       │
       ▼   (Serialised SQLite Writes)
  [ SQLite DB ] ──► Store inside vec0 (embeddings), FTS5 (lexical), and images (relational)
       │
       ▼   (Local Retrieval — instant CPU execution)
  idx search ──► Local CLIP text-encode + sqlite-vec cosine KNN
```

---

## Quick Start

### Installation

```bash
# Clone and install
cd /path/to/image-indexer
uv sync

# Or install from PyPI (when published)
pip install image-indexer
```

### Basic Usage

**Index a directory** (local mode, default):

```bash
# Just works! No flags needed.
idx index /path/to/photos

# Control concurrency
idx index /path/to/photos --workers 10

# Dry-run (test preprocessing without inference)
idx index /path/to/photos --dry-run

# Remote mode (upload to your server)
idx index /path/to/photos --mode remote --url https://your-vm.example.com/api

# Environment variables
export INDEXER_MODE=remote
export INDEXER_REMOTE_URL=https://your-vm.example.com/api
idx index /path/to/photos
```

**Search indexed images**:

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

**Check status**:

```bash
idx status --json
```

---

## Inference Modes

### Local Mode (Default)

Inference runs directly on your machine using PyTorch.

**Requirements**:
- GPU with ≥12GB VRAM recommended (for 5 concurrent workers)
- Or CPU-only (slow, ~0.1-0.5 images/second)

**Pros**:
- Zero network, zero authentication
- Works completely offline
- No external dependencies

**Cons**:
- Cold start: 60-90s (model load)
- Throughput limited by hardware
- GPU memory usage: ~12GB for 5 workers

### Remote Mode

Images uploaded to your self-hosted inference server via HTTPS.

**Setup**:
1. Build and push your container to `registry.joelkuiper.eu`
2. Deploy container on your VM exposing `/api/inference` endpoint
3. Set `INDEXER_REMOTE_URL` or use `--url` flag

**Pros**:
- Offload processing to remote hardware
- Scale independently from local machine
- Shared indexing across machines

**Cons**:
- Requires network connectivity
- Server must be online
- HTTP overhead

---

## Configuration

Edit `image_indexer/settings.toml` or use environment variables:

```ini
# Database
db_path = "~/.local/share/image-indexer/index.db"

# Models
embed_model_id = "openai/clip-vit-base-patch32"
caption_model_id = "Qwen/Qwen3-VL-4B-Instruct"

# Concurrency
max_workers = 5

# Remote server
remote_api_base = "https://your-vm.example.com/api"
remote_timeout = 300
```

---

## Performance

### Local Inference (Single GPU)
- **Cold start**: 60-90s (load CLIP + Qwen3-VL)
- **Embed**: 200-500ms per image
- **Caption**: 500-2000ms per image
- **Throughput**: ~0.5-2 images/second (single worker)

### With Concurrency (`--workers 5`)
- **Cold start**: Same (models load once)
- **Throughput**: ~2.5-10 images/second (5 parallel workers)
- **GPU memory**: ~12GB VRAM needed

### CPU-Only
- **Cold start**: 5-10 minutes
- **Throughput**: ~0.1-0.5 images/second
- **Use case**: Development/testing only

---

## Verification & Testing

Tests cover downscaling thresholds, correct EXIF transpositions, database triggers, and HTTP client retry loops.

```bash
# Run full suite (50+ passing tests)
uv run pytest -v

# Run local end-to-end pipeline validation
uv run python scripts/e2e_demo.py
```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design, component diagrams, and API contracts.
