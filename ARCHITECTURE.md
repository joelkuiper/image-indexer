# Image Indexer Architecture

## Overview
A local-first visual memory search system with two indexing modes: local (direct inference) and remote (HTTP endpoint).

## Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Machine                               │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     idx CLI Tool                              │  │
│  │                                                               │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │              Indexing Client                            │  │  │
│  │  │  ┌──────────────┐         ┌──────────────┐            │  │  │
│  │  │  │   Local Mode │         │  Remote Mode │            │  │  │
│  │  │  │  (direct)    │         │  (HTTP)      │            │  │  │
│  │  │  └──────────────┘         └──────────────┘            │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              │ Async Pipeline                       │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     SQLite Database                           │  │
│  │  - images table (relational metadata)                         │  │
│  │  - vec0 extension (vector embeddings)                         │  │
│  │  - FTS5 extension (lexical text search)                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Indexing Server (Optional)                   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Self-hosted Container (registry.joelkuiper.eu)              │  │
│  │                                                               │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │              Async Inference Server                    │  │  │
│  │  │  - POST /api/inference (base64 image, task)            │  │  │
│  │  │  - GET  /api/status/{job_id} (poll results)           │  │  │
│  │  │  - Models: CLIP (embed) + Qwen3-VL (caption)          │  │  │
│  │  │  - Async processing with job queue                    │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Modes

### Local Mode (Default)
```bash
idx index /path/to/photos
```
- Inference runs directly on local machine
- No network, no authentication
- Best for: Development, local GPU available

### Remote Mode
```bash
idx index /path/to/photos --remote --url https://your-vm/api/inference
```
- Images uploaded to remote server
- Async processing (upload → poll results)
- Best for: Cloud processing, offloading work

## Querying (Always Local)
```bash
# Semantic search (vector similarity)
idx search "philosophers studying geometry" --semantic

# Lexical search (text in captions/OCR)
idx search "docker port conflict" --lexical

# Structured search (EXIF, metadata)
idx search "width > 3000" --structured
```

## API Contract (Remote Server)

### POST `/api/inference`
**Request:**
```json
{
  "image_b64": "base64-encoded-image-bytes",
  "task": "embed|caption|all"
}
```

**Response (async):**
```json
{
  "id": "job-uuid-123",
  "status": "queued|processing|completed|failed"
}
```

### GET `/api/status/{job_id}`
**Response (completed):**
```json
{
  "id": "job-uuid-123",
  "status": "completed",
  "output": {
    "embedding": [1152 floats],
    "embedding_dim": 1152,
    "description": "rich caption text",
    "models": {
      "embed": "openai/clip-vit-base-patch32",
      "caption": "Qwen/Qwen3-VL-4B-Instruct"
    }
  }
}
```

## Database Schema
- `images`: id, path, sha256, file_size, format, width, height, description, tags, embedding (vec0), created_at, updated_at
- FTS5 table: `images_lexical` for full-text search over descriptions and OCR text

## Concurrency
- `--workers N` flag controls parallel inference workers
- Default: 5 workers (adjust based on GPU VRAM)
- Async pipeline with semaphore limiting

## Performance Expectations

### Local Mode (Single GPU)
- Cold start: 60-90s (model load)
- Embed: 200-500ms/image
- Caption: 500-2000ms/image
- Throughput: ~0.5-2 images/sec

### Local Mode (5 Workers)
- Cold start: Same (models load once)
- Throughput: ~2.5-10 images/sec

### Remote Mode
- Network-bound (upload + download latency)
- Server capacity depends on host hardware
