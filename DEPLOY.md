# Deployment Guide (GHCR & RunPod)

- **Configuratie & Login**:
  - Maak een classic GitHub token aan met `write:packages` scope.
  - Sla het op in `~/.ghcr-token` en log in op Docker:
    ```bash
    docker login ghcr.io -u <github-username> --password-stdin < ~/.ghcr-token
    ```

- **Bouwen (elke nieuwe versie)**:
  ```bash
  cd ~/Repositories/image-indexer
  docker build -t ghcr.io/<github-username>/image-indexer-worker:latest -f worker/Dockerfile .
  ```
  *(Opmerking: Omdat SigLIP2 en Qwen3-VL worden meegebacken, wordt de image ~10-15GB)*

- **Pushen naar GHCR**:
  ```bash
  docker push ghcr.io/<github-username>/image-indexer-worker:latest
  ```

- **Zichtbaarheid op GitHub**:
  - Maak de package **Public** via de GitHub settings pagina van de container (`image-indexer-worker` -> Package settings -> Change visibility -> Public).
  - *Alternatief (Private)*: Houd 'm private en voeg je registry credentials toe in RunPod (Settings -> Container Registry Credentials).

- **Deploy op RunPod**:
  - Log in op RunPod -> Serverless -> New Endpoint.
  - **Container Image**: `ghcr.io/<github-username>/image-indexer-worker:latest`
  - **GPU**: Kies een 16GB of 24GB VRAM kaart (bijv. L4 of A10G).
  - **Workers**: Min: 0 (automatisch afschalen om kosten te sparen), Max: 1 (voor initiële test).
  - **Test Payload** (binnen de RunPod Console):
    ```json
    { "input": { "image_b64": "<base64_string>", "task": "all" } }
    ```
