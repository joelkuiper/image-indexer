# idx — Local Usage Guide

Zero setup. Just works.

---

## Installation

```bash
# Clone repository
cd ~/Repositories/image-indexer

# Install dependencies
uv sync

# Verify installation
idx --help
```

## Quick Start

### Index Images (Local Mode)

```bash
# Point it at a directory, it just works
idx index /path/to/photos

# Progress shows to stderr when --verbose
idx index /path/to/photos --verbose

# Dry-run: test preprocessing without inference
idx index /path/to/photos --dry-run
```

### Search Images

```bash
# Semantic search (vibe-based)
idx search "philosophers studying geometry" --semantic

# Text search (OCR + captions)
idx search "docker error" --lexical

# Structured filters (EXIF, metadata)
idx search "width > 3000" --structured

# JSON output (for agents/scripts)
idx search "geometry" --semantic --json
```

### Check Status

```bash
idx status
idx status --json
```

---

## Remote Indexing (Optional)

If you have a self-hosted inference server:

```bash
# Build and push container to your registry
cd worker
docker build -t registry.joelkuiper.eu/joelkuiper/image-indexer:latest .
docker push registry.joelkuiper.eu/joelkuiper/image-indexer:latest

# Deploy container on your VM with HTTP API exposed
# (see DEPLOYING_SERVER.md for details)

# Index via remote server
idx index /path/to/photos --mode remote --url https://your-vm.example.com/api
```

---

## Configuration

```bash
# Concurrency control
export INDEXER_WORKERS=10
idx index /path/to/photos

# Remote server
export INDEXER_MODE=remote
export INDEXER_REMOTE_URL=https://your-vm.example.com/api
idx index /path/to/photos

# Database path
export DB_PATH=~/my-index.db
idx index /path/to/photos
```

---

## Troubleshooting

### Inference Slow

- Check GPU VRAM: `nvidia-smi`
- Reduce workers: `--workers 2`
- Ensure models are cached: `~/.cache/huggingface`

### Database Errors

```bash
# Check database exists
ls -la ~/.local/share/image-indexer/index.db

# Verify schema
sqlite3 ~/.local/share/image-indexer/index.db ".tables"
```

### No Results

- Verify images are indexed: `idx status`
- Check search mode: use `--semantic`, `--lexical`, or `--structured`
- Try simpler query terms

---

## See Also

- [README.md](README.md) — Full documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) — System design
- [DEPLOYING_SERVER.md](DEPLOYING_SERVER.md) — Self-hosted inference server
