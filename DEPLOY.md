# Deployment Guide (GHCR & RunPod)

- **Configuration & Login**:
  - Generate a classic GitHub token with the `write:packages` scope.
  - Save it to `~/.ghcr-token` and log in to Docker:
    ```bash
    docker login ghcr.io -u <github-username> --password-stdin < ~/.ghcr-token
    ```

- **Building (repeat for each new version)**:
  ```bash
  cd ~/Repositories/image-indexer
  docker build -t ghcr.io/<github-username>/image-indexer-worker:latest -f worker/Dockerfile .
  ```
  *(Note: Since SigLIP2 and Qwen3-VL weights are baked into the image, expect ~10–15 GB.)*

- **Pushing to GHCR**:
  ```bash
  docker push ghcr.io/<github-username>/image-indexer-worker:latest
  ```

- **GitHub Package Visibility**:
  - Set the package to **Public** via the GitHub Packages settings page (`image-indexer-worker` -> Package settings -> Change visibility -> Public).
  - *Alternative (Private)*: Keep it private and add your registry credentials in RunPod (Settings -> Container Registry Credentials).

- **Deploy to RunPod**:
  - Log in to RunPod -> Serverless -> New Endpoint.
  - **Container Image**: `ghcr.io/<github-username>/image-indexer-worker:latest`
  - **GPU**: Choose a 16 GB or 24 GB VRAM card (e.g., L4 or A10G).
  - **Workers**: Min: 0 (auto-scale down to save costs), Max: 1 (for initial testing).
  - **Test Payload** (via the RunPod Console):
    ```json
    { "input": { "image_b64": "<base64_string>", "task": "all" } }
    ```
