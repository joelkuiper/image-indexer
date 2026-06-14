# Implementatie & Test TODO

## 1. Lokaal Testen
- [ ] Schrijf een lokale runner/testscript die de Docker image/handler container lokaal start en payloads verstuurt.
- [ ] Valideer de integratie tussen `handler.py` en echte model loading/mocking lokaal.
- [ ] Test de SQLite + `sqlite-vec` integratie via `pytest tests/test_db.py`.

## 2. Deployment
- [ ] Valideer de dependencies en CUDA compatibiliteit in `worker/Dockerfile`.
- [ ] Bouw en tag de Docker image met de juiste GHCR naamvallen.
- [ ] Push de vision-worker image naar GitHub Packages (GHCR).
- [ ] Configureer RunPod Serverless Endpoint met de gepushte image en de vereiste GPU tier (16GB/24GB VRAM).

## 3. RunPod E2E Test
- [ ] Schrijf een client script (`image_indexer/client.py` of test script) dat verbinding maakt met de live RunPod endpoint API.
- [ ] Stuur een base64 gecodeerde afbeelding naar de endpoint en controleer of zowel de embedding (1152-d) als de beschrijving correct terugkomen.
- [ ] Integreer dit resultaat in de lokale database pipeline om te valideren dat data weggeschreven wordt.

## 4. Agentic-friendly CLI Opties
- [ ] Ontwerp de CLI (`image_indexer/cli.py` of click app) met machine-parsable outputs (bijv. `--json` flag op alle commando's).
- [ ] Zorg voor non-interactive defaults en duidelijke exit-codes zodat Esther (of andere AI-agents) makkelijk indexeringen kan draaien of queries kan uitvoeren zonder menselijke tussenkomst.
- [ ] Implementeer robuuste error handling en duidelijke output logging (bijv. `idx index --json --verbose <dir>`).
