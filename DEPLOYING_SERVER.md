# Deploying the Inference Server

Self-host your own inference server for remote indexing.

---

## Overview

The inference server exposes a simple HTTP API:

```bash
POST /api/inference  # Submit job
GET  /api/status/{id} # Poll results
```

---

## Quick Start

### Build and Run

```bash
cd ~/Repositories/image-indexer/server

# Build image
docker build -t image-indexer-server .

# Run container
docker run -d \
  --name inference-server \
  -p 8080:8080 \
  -v ~/server-db:/app \
  image-indexer-server

# Verify it's running
curl http://localhost:8080/
```

### Test the API

```bash
# Submit a job (requires base64 image)
curl -X POST http://localhost:8080/api/inference \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "base64-encoded-image", "task": "all"}'

# Poll status
curl http://localhost:8080/api/status/{job-id}
```

---

## Deployment on Your VM

### 1. Build and Push

```bash
cd ~/Repositories/image-indexer/server

# Build
docker build -t image-indexer-server .

# Tag for your registry
docker tag image-indexer-server registry.joelkuiper.eu/joelkuiper/image-indexer-server:latest

# Push
docker push registry.joelkuiper.eu/joelkuiper/image-indexer-server:latest
```

### 2. Deploy on VM

```bash
# SSH to your VM
ssh joel@136.243.3.235

# Pull image
docker pull registry.joelkuiper.eu/joelkuiper/image-indexer-server:latest

# Run container
docker run -d \
  --name inference-server \
  -p 8080:8080 \
  -v ~/server-db:/app \
  registry.joelkuiper.eu/joelkuiper/image-indexer-server:latest
```

### 3. Expose to Network

Option A — Nginx reverse proxy (recommended):

```nginx
server {
    listen 80;
    server_name inference.your-domain.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Option B — Direct (less secure):

```bash
# On VM
sudo ufw allow 8080/tcp
```

### 4. Get Public URL

```bash
# If you have a domain
https://inference.your-domain.com/api/inference

# Or use tunneling (ngrok, cloudflare tunnels, etc.)
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `8080` | Port to listen on |
| `SERVER_DB_PATH` | `/app/server.db` | SQLite database path |
| `SERVER_MAX_WORKERS` | `3` | Max concurrent jobs |
| `HF_HOME` | `/app/model-cache` | HuggingFace cache |
| `LOG_LEVEL` | `INFO` | Logging level |

### Example: High-Performance Server

```bash
docker run -d \
  --name inference-server \
  -p 8080:8080 \
  -v ~/server-db:/app \
  -e SERVER_MAX_WORKERS=5 \
  -e HF_HOME=/app/model-cache \
  image-indexer-server
```

---

## API Reference

### POST `/api/inference`

Submit an inference job.

**Request:**
```json
{
  "image_b64": "base64-encoded-image-bytes",
  "task": "embed|caption|all"
}
```

**Response:**
```json
{
  "id": "job-uuid-123",
  "status": "queued"
}
```

### GET `/api/status/{job_id}`

Poll job status.

**Response (queued/processing):**
```json
{
  "id": "job-uuid-123",
  "status": "processing"
}
```

**Response (completed):**
```json
{
  "id": "job-uuid-123",
  "status": "completed",
  "output": {
    "embedding": [1152 floats],
    "embedding_dim": 512,
    "description": "rich caption",
    "models": {
      "embed": "openai/clip-vit-base-patch32",
      "caption": "Qwen/Qwen3-VL-4B-Instruct"
    }
  }
}
```

**Response (failed):**
```json
{
  "id": "job-uuid-123",
  "status": "failed",
  "error": "error message"
}
```

---

## Health Check

```bash
curl http://localhost:8080/
# {"status": "ok", "service": "inference-server"}
```

---

## Monitoring

### Check Container Logs

```bash
docker logs inference-server
```

### Check Database

```bash
sqlite3 ~/server-db/server.db "SELECT * FROM jobs WHERE status = 'queued';"
```

### Monitor GPU Usage

```bash
nvidia-smi
```

---

## Scaling

### Horizontal Scaling

Run multiple server instances:

```bash
# Instance 1
docker run -d --name server-1 -p 8081:8080 image-indexer-server

# Instance 2
docker run -d --name server-2 -p 8082:8080 image-indexer-server
```

Load balance across instances (requires load balancer).

### Vertical Scaling

Increase worker count:

```bash
docker run -d \
  --name inference-server \
  -e SERVER_MAX_WORKERS=10 \
  image-indexer-server
```

---

## Troubleshooting

### Job Queued Forever

- Check `SERVER_MAX_WORKERS` isn't too low
- Verify GPU has enough VRAM
- Check logs: `docker logs inference-server`

### Slow Inference

- Ensure models are cached in `/app/model-cache`
- Check GPU is not overloaded
- Reduce concurrent workers if memory constrained

### Container Won't Start

- Verify GPU drivers on VM
- Check port 8080 is not in use
- Ensure enough disk space for model cache
