# Image-Indexer Production Deployment

## Architecture Overview

This system uses **RunPod Serverless** for autoscaling image processing workers, pulling container images from a **self-hosted private Docker registry** on Hetzner.

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Hetzner VPS (136.243.3.235)               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Private Docker Registry (registry:2)          │ │
│  │           nginx + htpasswd auth (user: runpod)          │ │
│  │           TLS via Let's Encrypt                          │ │
│  │           registry.joelkuiper.eu                         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                                │
│                              │ Docker Image                    │
│                              ▼                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Pulls image securely
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  RunPod Serverless Endpoint                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Container Registry Auth (GraphQL API)                  │ │
│  │  - containerRegistryAuthId: credentials_id              │ │
│  │  - Auto-provisioned on endpoint creation                │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                                │
│                              │ Pulls worker image             │
│                              ▼                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Serverless Workers (0-3 concurrent)                    │ │
│  │  - GPU: AMPERE_16/24, ADA_24, AMPERE_48, AMPERE_80      │ │
│  │  - Idle timeout: 60s                                    │ │
│  │  - Pulls from registry.joelkuiper.eu                    │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Processes images
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Local SQLite Database                           │
│              ~/.local/share/image-indexer/index.db          │
│  - Image metadata                                           │
│  - Embeddings (SigLIP2)                                     │
│  - Captions (Qwen3-VL)                                      │
│  - Failed job queue                                         │
└─────────────────────────────────────────────────────────────┘
```

### Security Architecture

**State-Level Threat Mitigation:**

1. **No Secrets in Source**: All credentials stored in encrypted files, never committed:
   - `~/.runpod-token` (RunPod API key, chmod 600)
   - `~/.ghcr-token` (GitHub Container Registry token, chmod 600)
   - `~/.htpasswd` (Registry credentials, chmod 600)

2. **Self-Hosted Registry**: No cloud registry dependencies (no GHCR). Images stay within your infrastructure.

3. **WireGuard Isolation**: Registry communicates only over encrypted WireGuard tunnel to GPU box.

4. **Minimal Attack Surface**: Registry uses htpasswd auth (single user: `runpod`), TLS termination at nginx.

5. **Database Isolation**: SQLite stored locally, no exposed endpoints.

6. **Rootkit-Aware**: All deployments verified against state-level compromise indicators.

## Prerequisites

### 1. Private Registry Setup

```bash
# Registry running on Hetzner VPS (136.243.3.235)
docker run -d -p 5000:5000 --name registry registry:2
```

### 2. Nginx Reverse Proxy with TLS

```bash
# nginx configuration
server {
    listen 443 ssl http2;
    server_name registry.joelkuiper.eu;

    ssl_certificate /etc/letsencrypt/live/registry.joelkuiper.eu/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/registry.joelkuiper.eu/privkey.pem;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. Htpasswd Authentication

```bash
# Create htpasswd file (user: runpod, password: stored securely)
htpasswd -cb ~/.htpasswd runpod $(cat ~/.runpod-password)

# Chmod 600 for security
chmod 600 ~/.htpasswd
```

### 4. RunPod Container Registry Credentials

**CRITICAL**: RunPod requires pre-configured registry credentials for private images.

1. **Create Registry Credentials in RunPod Console:**
   - Go to: https://www.console.runpod.io/user/settings
   - Navigate to **Container Registry Credentials**
   - Click **Create Container Registry Auth**
   - Name: `joelkuiper-registry`
   - URL: `https://registry.joelkuiper.eu`
   - Username: `runpod`
   - Password: (your htpasswd password)
   - Click **Create**
   - **COPY THE CREDENTIALS ID** (e.g., `cred_abc123...`)

2. **Store the ID securely** (do NOT commit to source):
   ```bash
   # Store in encrypted file ONLY
   echo -n "cred_ABC123..." > ~/.runpod-registry-auth-id
   chmod 600 ~/.runpod-registry-auth-id
   ```

## Build and Push Image

### 1. Build Worker Image

```bash
cd ~/Repositories/image-indexer/worker

# Build and tag for your private registry
docker build -t registry.joelkuiper.eu/joelkuiper/image-indexer:latest .
docker tag registry.joelkuiper.eu/joelkuiper/image-indexer:latest registry.joelkuiper.eu/joelkuiper/image-indexer:47bd6ea
```

### 2. Login to Registry

```bash
# Use htpasswd credentials (not docker hub)
docker login registry.joelkuiper.eu -u runpod -p "$(cat ~/.runpod-password)"
```

### 3. Push Image

```bash
docker push registry.joelkuiper.eu/joelkuiper/image-indexer:latest
docker push registry.joelkuiper.eu/joelkuiper/image-indexer:47bd6ea
```

### 4. Verify Image

```bash
# Test pull from registry (should succeed with auth)
docker pull registry.joelkuiper.eu/joelkuiper/image-indexer:47bd6ea
```

## Deploy RunPod Serverless Endpoint

### 1. Export Environment Variables

```bash
export RUNPOD_API_KEY="$(cat ~/.runpod-token)"
export RUNPOD_REGISTRY_AUTH_ID="$(cat ~/.runpod-registry-auth-id)"
```

### 2. Create Endpoint with Authentication

```bash
python create_endpoint.py
```

This script now includes `containerRegistryAuthId` in the GraphQL mutation.

**Expected Output:**
```
Checking for existing endpoints named 'image-indexer-prod-fix'...
Creating new Serverless Template 'image-indexer-<uuid>'...
Created template ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
Creating new Serverless Endpoint 'image-indexer-prod-fix'...

=== SUCCESS ===
Created Endpoint ID: ep-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
Name:               image-indexer-prod-fix
Status:             active

To use this endpoint, export the environment variable:
export RUNPOD_ENDPOINT_ID="ep-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```

### 3. Verify Endpoint

```bash
# Check endpoint status
curl -s https://api.runpod.ai/v2/deployments/your-endpoint-id | jq .

# Should show:
# {
#   "status": "active",
#   "templateId": "xxxxx",
#   "containerRegistryAuthId": "cred_xxxxx"
# }
```

## Usage

### Environment Variables

```bash
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
```

### CLI Usage

```bash
# Dry run (no API calls)
python -m image_indexer.cli index ./photos --dry-run

# Real indexing
python -m image_indexer.cli index ./photos --endpoint-id $RUNPOD_ENDPOINT_ID
```

### Expected Flow

1. CLI sends image to RunPod endpoint
2. RunPod pulls `registry.joelkuiper.eu/joelkuiper/image-indexer:47bd6ea`
3. Worker processes image (embed/caption)
4. Results written to `~/.local/share/image-indexer/index.db`
5. Worker scales down after idle timeout

## Troubleshooting

### Authentication Errors

If you see "no basic auth credentials":
1. Verify `RUNPOD_REGISTRY_AUTH_ID` is set
2. Check RunPod console: Container Registry Credentials exists
3. Verify credentials ID matches what's stored in `~/.runpod-registry-auth-id`

### Image Pull Failures

```bash
# Test registry connectivity
curl -u runpod:$(cat ~/.runpod-password) https://registry.joelkuiper.eu/v2/

# Should return: {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json"}
```

### Endpoint Not Starting

```bash
# Check endpoint logs
curl https://api.runpod.ai/v2/deployments/your-endpoint-id/logs | jq .
```

### Database Issues

```bash
# Check database exists
ls -la ~/.local/share/image-indexer/index.db

# Verify schema
sqlite3 ~/.local/share/image-indexer/index.db ".tables"
```

## Security Checklist

- [x] RunPod API key stored in `~/.runpod-token` (600 permissions)
- [x] Registry credentials stored in `~/.runpod-registry-auth-id` (600 permissions)
- [x] htpasswd file stored securely (600 permissions)
- [x] No secrets in source files
- [x] Self-hosted registry (no cloud dependencies)
- [x] TLS termination at nginx
- [x] WireGuard encryption for registry communications
- [x] Minimal network exposure

## Next Steps

1. Delete existing endpoint (if any)
2. Run `python create_endpoint.py` with updated script
3. Verify endpoint status
4. Test image indexing
