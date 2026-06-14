# Implementation & Testing TODO

## 1. Local Testing
- [ ] Write a local runner/test script that starts the Docker image/handler locally and sends payloads.
- [ ] Validate the integration between `handler.py` and real model loading/mocking locally.
- [ ] Test the SQLite + `sqlite-vec` integration via `pytest tests/test_db.py`.

## 2. Deployment
- [ ] Validate dependencies and CUDA compatibility in `worker/Dockerfile`.
- [ ] Build and tag the Docker image with the correct GHCR namespace.
- [ ] Push the vision-worker image to GitHub Packages (GHCR).
- [ ] Configure the RunPod Serverless Endpoint with the pushed image and the required GPU tier (16 GB / 24 GB VRAM).

## 3. RunPod E2E Test
- [ ] Write a client script (`image_indexer/client.py` or test script) that connects to the live RunPod endpoint API.
- [ ] Send a base64-encoded image to the endpoint and verify both the embedding (1152-d) and description are returned correctly.
- [ ] Integrate this result into the local database pipeline to validate data is persisted.

## 4. Agentic-Friendly CLI Options
- [ ] Design the CLI (`image_indexer/cli.py` or Click app) with machine-parsable outputs (e.g., `--json` flag on all commands).
- [ ] Provide non-interactive defaults and clear exit codes so Esther (or other AI agents) can easily run indexing or queries without human intervention.
- [ ] Implement robust error handling and clear output logging (e.g., `idx index --json --verbose <dir>`).
