# Image Indexer

Find anything with pixels by what you remember, not what you named it.

```bash
idx search "black-and-white waterfall in Iceland" --semantic
idx search "bayes theorem formula" --lexical
idx search "camera_model = 'Nikon'" --structured
```

Three composable search surfaces over one SQLite database:

- **Semantic** — SigLIP2 vectors (text → 1152-d → cosine KNN)
- **Lexical** — FTS5 over OCR'd text + Qwen3-VL captions
- **Structured** — SQL on EXIF and metadata columns

## How it works

```
[ Images on disk ]
       │
       ▼
  preprocess ──► resize 40MP → 1MP JPEG, extract EXIF, SHA-256 dedup
       │
       ▼
[ RunPod GPU ] ──► SigLIP2 embedding (1152-d) + Qwen3-VL caption / OCR
       │
       ▼
[ SQLite ] ──► vec0 (vector) + FTS5 (text) + EXIF columns
       │
       ▼
  idx search ──► local SigLIP2 text encoder + sqlite-vec cosine KNN
```

Indexing requires a RunPod GPU endpoint (SigLIP2 + Qwen3-VL run on GPU).
Searching runs entirely locally (text encoder is ~200MB, fits on CPU).

## Setup

```bash
uv install
uv run idx --help
```

## CLI

```bash
# Index a directory (requires RunPod endpoint)
export RUNPOD_ENDPOINT_ID=...
export RUNPOD_API_KEY=***idx index ~/Photos/2024 --verbose

# Semantic search (local, no RunPod needed)
idx search "sunset over mountains" --semantic --json

# Lexical search over captions + OCR'd text
idx search "error 500 gateway timeout" --lexical

# Structured filter on metadata
idx search "width > 3000 AND format = 'JPEG'" --structured

# Database stats
idx status --json
```

Exit codes: `0` ok · `1` user error · `2` system error · `3` partial failure.
All commands support `--json` for machine consumption.

## Tests

```bash
uv run pytest -v
# 50 passing: preprocess(17) db(8) handler(4) client(7) text_embed(6) cli(8)
```
