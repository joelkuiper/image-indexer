# RunPod Hello World — Stap-voor-stap

Deze guide helpt je een minimale RunPod worker deployen om te bewijzen dat de infra werkt.

**Wat je nodig hebt:**
- GitHub account (ingelogd)
- RunPod account (ingelogd)
- Docker lokaal geïnstalleerd
- ~5-10 minuten

---

## Stap 1: Lokaal testen

Eerst testen we de handler lokaal zonder Docker/RunPod:

```bash
cd ~/Repositories/image-indexer
uv run python scripts/test_hello_handler.py
```

Je zou moeten zien:
```
Test 1: Basic handler call (no image)...
  ✓ Greeting: Hello from RunPod! 🚀
  ✓ CUDA: True/False
  ✓ GPU: NVIDIA RTX ...
  ✓ Handler time: 12.34ms

Test 2: Handler with base64 image...
  ✓ Image decoded: (2, 2)
  ✓ Format: PNG
  ✓ Bytes decoded: 69

✅ All tests passed! Handler is ready for RunPod.
```

---

## Stap 2: GitHub Container Registry (GHCR) — Token maken

1. Ga naar https://github.com/settings/tokens
2. Klik **"Generate new token (classic)"**
3. Vul in:
   - **Note**: `runpod-deploy`
   - **Expiration**: 90 days (of wat je prettig vindt)
   - **Scopes**: vink aan `write:packages` (automatisch ook `read:packages`)
4. Klik onderaan **"Generate token"**
5. **Kopieer het token NU** (ghp_...) — je ziet het nooit meer!

Sla het veilig op:
```bash
# Plak je token hier (ghp_xxxx)
echo "ghp_YOUR_TOKEN_HERE" > ~/.ghcr-token
chmod 600 ~/.ghcr-token
```

---

## Stap 3: Docker login op GHCR

```bash
docker login ghcr.io -u joelkuiper --password-stdin < ~/.ghcr-token
```

Je zou moeten zien: `Login Succeeded`

---

## Stap 4: Docker image bouwen

```bash
cd ~/Repositories/image-indexer
docker build -t ghcr.io/joelkuiper/image-indexer-hello:latest -f worker/hello/Dockerfile .
```

**Verwacht:**
- Download van de PyTorch base image (~2-3 GB)
- `pip install runpod pillow`
- Kopie van handler.py

**Grootte:** ~3-4 GB (veel kleiner dan de 15GB echte worker)

Controleer:
```bash
docker images | grep image-indexer-hello
```

---

## Stap 5: Push naar GHCR

```bash
docker push ghcr.io/joelkuiper/image-indexer-hello:latest
```

**Duurt:** 5-15 minuten afhankelijk van je upload.

---

## Stap 6: GitHub Package public maken

1. Ga naar https://github.com/joelkuiper?tab=packages
2. Klik op **`image-indexer-hello`**
3. Klik rechts op **"Package settings"**
4. Scroll naar beneden → **"Change visibility"**
5. Kies **"Public"** → bevestig met je package naam
6. Klik **"Change visibility"** button

**Waarom public?** RunPod kan dan pullen zonder credentials configureren.

---

## Stap 7: RunPod — Endpoint maken

1. Ga naar https://www.runpod.io → login
2. Links in de sidebar: **"Serverless"**
3. Klik **"+ New Endpoint"**
4. Vul in:
   - **Endpoint name**: `image-indexer-hello`
   - **Template**: Selecteer **"Custom"** (onderaan de lijst)
   - **Container Image**: `ghcr.io/joelkuiper/image-indexer-hello:latest`
   - **Container Disk**: 20 GB (default is prima)
   - **Volume Disk**: 0 GB (niet nodig voor deze test)
   - **GPU Types**: Vink **1 of 2 GPU types** aan (bijv: RTX 3070, RTX 4000 Ada, L4)
     - *Niet alle types aanvinken — kies de goedkoopste voor deze test*
   - **Max Workers**: `1`
   - **Idle Timeout**: `60` (seconden)
   - **FlashBoot**: ✅ aan (snellere cold start)
5. Klik **"Deploy"**

**Wacht:** RunPod pullt de image (~1-2 min). Status gaat van "Deploying" → "Ready".

---

## Stap 8: Testen via RunPod UI

1. Klik op je endpoint naam (**`image-indexer-hello`**)
2. Je ziet een pagina met **"Test this endpoint"**
3. Plak in het JSON vak:

```json
{
  "input": {}
}
```

4. Klik **"Run"**
5. **Verwacht response** (na ~10-30 sec cold start):

```json
{
  "status": "COMPLETED",
  "output": {
    "greeting": "Hello from RunPod! 🚀",
    "compute": {
      "cuda_available": true,
      "gpu": "NVIDIA RTX 4000 Ada Generation",
      "torch_version": "2.1.0+cu118"
    },
    "image": null,
    "handler_time_ms": 142.57
  }
}
```

**🎉 Gefeliciteerd!** Je hebt nu:
- ✅ Een werkende RunPod serverless worker
- ✅ GPU access (CUDA)
- ✅ Een handler die JSON in/out doet
- ✅ Bewezen dat de infra werkt

---

## Stap 9: Test met een image

1. Op je lokale machine, maak een base64 string van een kleine foto:

```bash
# Als je een test.jpg hebt:
base64 -i test.jpg | tr -d '\n' > test.b64

# Of snel een test image maken:
python -c "from PIL import Image; Image.new('RGB', (100,100), 'red').save('/tmp/test.jpg')"
base64 -i /tmp/test.jpg | tr -d '\n' > /tmp/test.b64
```

2. Kopieer de inhoud van `test.b64` (één lange regel)
3. Ga terug naar RunPod UI → je endpoint
4. Plak in het JSON vak (vervang `<PASTE_BASE64_HERE>`):

```json
{
  "input": {
    "image_b64": "<PASTE_BASE64_HERE>"
  }
}
```

5. Klik **"Run"**
6. **Verwacht response**:

```json
{
  "status": "COMPLETED",
  "output": {
    "greeting": "Hello from RunPod! 🚀",
    "compute": {
      "cuda_available": true,
      "gpu": "NVIDIA RTX 4000 Ada Generation",
      "torch_version": "2.1.0+cu118"
    },
    "image": {
      "format": "JPEG",
      "size": [100, 100],
      "mode": "RGB",
      "bytes_decoded": 1234
    },
    "handler_time_ms": 89.23
  }
}
```

**🎉 Dubbel gefeliciteerd!** Je kunt nu:
- ✅ Base64 images versturen naar RunPod
- ✅ GPU inference-ready containers draaien
- ✅ Handler responses terugkrijgen

---

## Troubleshooting

### "Container failed to start"
- Check dat je `CMD ["python", "-u", "handler.py"]` hebt in je Dockerfile (`-u` voor unbuffered output)
- Check RunPod logs: klik op je endpoint → **"Monitoring"** tab

### "CUDA not available"
- Je hebt een CPU-only instance gepakt — maak een nieuw endpoint met GPU types aangevinkt
- Of de base image heeft geen CUDA support (onze `runpod/pytorch` image heeft het wel)

### "Handler timeout"
- Eerste request duurt ~30 sec (cold start + container pull)
- Volgende requests: ~5-10 sec (warm)

### "Image not found" bij push/pull
- Check dat je package **Public** is (Stap 6)
- Check dat je username klopt in de image tag

---

## Wat nu?

Als dit werkt, weten we dat:
1. Docker build/push werkt ✅
2. GHCR integratie werkt ✅  
3. RunPod serverless werkt ✅
4. GPU access werkt ✅
5. Handler contract werkt ✅

**Volgende stap:** De echte worker (SigLIP2 + Qwen3-VL) deployen. Die is groter (15 GB) maar het proces is identiek.

---

## Kosten

RunPod Serverless rekent per seconde:
- **Idle** (0 workers): $0
- **Active** (1 worker): ~$0.0002-0.0004/sec afhankelijk van GPU
- **Eerste test** (~2 min): ~$0.02 - verwaarloosbaar

Zet **Min Workers = 0** en **Idle Timeout = 60** dan kost het bijna niets als je het niet gebruikt.
