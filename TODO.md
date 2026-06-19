# TODO — Image Indexer

## Completed ✅

- [x] Refactor to local-first architecture
- [x] Remove RunPod dependencies
- [x] Create `inference.py` with local and remote modes
- [x] Update CLI with `--mode` and `--url` flags
- [x] Create inference server with HTTP API
- [x] Update README.md
- [x] Create LOCAL.md (usage guide)
- [x] Create DEPLOYING_SERVER.md (server deployment)
- [x] Create ARCHITECTURE.md (system design)
- [x] Update pyproject.toml (remove runpod)

## In Progress 🚧

- [ ] Test local mode end-to-end
- [ ] Test remote mode with server
- [ ] Add unit tests for inference client
- [ ] Add integration tests for CLI
- [ ] Optimize model loading (lazy initialization)
- [ ] Add progress tracking for long jobs

## To Do ⏳

- [ ] Add support for multiple inference backends
- [ ] Implement job cancellation
- [ ] Add metrics/monitoring endpoints
- [ ] Create Docker Compose setup for local dev
- [ ] Add GitHub Actions CI/CD
- [ ] Publish to PyPI

## Future Ideas 💡

- [ ] Support for additional models (SigLIP2, BLIP2)
- [ ] Image-to-image search (embed images, not just text)
- [ ] Batch upload optimization
- [ ] Web UI for browsing indexed images
- [ ] Plugin system for custom inference backends
