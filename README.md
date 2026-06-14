# Image Indexer

An on-device CLI tool for indexing images and enabling hybrid structured/semantic search, backed by SQLite with vector-search support (`sqlite-vec`).

It processes raw image directories locally (extracting EXIF/metadata and local fallback tags/features), uploads images to a RunPod serverless worker for high-performance visual/description embeddings, and saves the final consolidated indexes back into an SQLite DB for immediate local queries.

## Architecture Flow

```
[Raw Folder of Images] 
       │
       ▼
 [Local Parsing] ──► Extract EXIF / Basic dimensions & details
       │ 
       ▼
 [Local Pipeline] ──(Fallback embeddings / check cached)
       │
       ▼
 [RunPod Serverless Worker] (Dockerized Inference Stack) 
       │ ──► Processes the image chunk / returns rich description & vectors
       ▼
 [Local SQLite DB] ──► Saves metadata / description / high-dim vector (sqlite-vec)
       │
       ▼
 [CLI Query Interface] ──► Full hybrid search (EXIF filters, lexical desc, sqlite-vec visual searches)
```

## Technology Stack

- **Database:** SQLite (v3.44+)
- **Semantic Layer:** `sqlite-vec` (`vec0` virtual table, 1152-d cosine distance)
- **Lexical Layer:** SQLite FTS5 (external-content trigger-synced tables)
- **Embedding Model:** `google/siglip2-so400m-patch16-384` (1152-d joint text/image space)
- **Caption Model:** `Qwen/Qwen3-VL-4B-Instruct` (generates rich textual descriptions and OCR)
- **Deployment Platform:** RunPod Serverless GPU Worker (custom Docker image)
- **Dependency Manager:** `uv`
