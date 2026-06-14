# Image Indexer — Product Vision

> *"Dat plaatje met die Bayes formule... waar was dat ook alweer?"*

---

## The Problem

Het is niet alleen foto's. Het is alles wat pixels heeft:

- Screenshots van Slack conversations
- Whiteboard foto's uit meetings  
- Diagrams uit papers
- Memes die je ooit grappig vond
- Grafieken met loss curves
- Slides van talks
- Recipes geknipt uit websites
- Handgeschreven notities gefotografeerd
- UML diagrams die je nooit meer kan terugvinden
- Screenshots van error messages die je later nodig hebt

Duizenden afbeeldingen. Verspreid over je hele machine. En als je er eentje zoekt,
weet je alleen nog de *vibe*: "iets met Bayes", "die Docker meme", "een graph die 
omlaag ging", "iets met paarse achtergrond en witte tekst".

Huidige tools snappen dat niet. Ze zoeken op bestandsnaam of datum.
Wij zoeken op *wat je je herinnert*.

---

## The Dream

Een visuele knowledge base. Je gooit er alles in, en daarna kan je praten
tegen je collectie alsof je een vriend vraagt:

**"Dat diagram over attention mechanisms, met die pijlen en boxes"**
→ Qwen3-VL heeft het beschreven als "a flowchart showing attention mechanism 
   with query, key, value vectors and softmax layer". FTS5 matcht. SigLIP2 
   rankt visueel vergelijkbare diagrams erbij. → 4 resultaten.

**"Show me every screenshot where I captured an error"**
→ Lexical search over OCR'd text: "Error", "Exception", "Traceback", "FAILED".
   → 47 resultaten.

**"Die meme met de twee knoppen"**
→ Semantic search. SigLIP2 voelt de visuele compositie. → Drake meme.

**"Screenshots van terminal output"**
→ Scene classification (local, geen RunPod nodig): high contrast, monospace text,
   dark background. → Alle terminal screenshots.

**"Iets met een paarse gradient en witte tekst, denk ik"**
→ Colour histogram filter + semantic search. → Found it.

---

## Core Features

### 1. Vibes-Based Search

De killer feature. Je hoeft niet precies te weten wat je zoekt — 
je hoeft het je alleen vaag te herinneren.

**Hoe het werkt:**
- SigLIP2 begrijpt *wat een afbeelding voelt als* (compositionele similarity)
- Qwen3-VL beschrijft de inhoud in natuurlijke taal (voor FTS5)
- OCR vangt elk woord dat in de pixels staat
- Samen dekken ze het hele spectrum van "exact" naar "vaag"

**Voorbeelden:**

```bash
# Exact — je weet wat je zoekt
idx search "bayes theorem formula" --json

# Vaag — je herinnert je de vibe
idx search "something mathematical with a fraction and Greek letters" --semantic --json

# Visueel — je herinnert je hoe het eruit zag  
idx search "dark background green text code" --semantic --json

# Mixed — structureel filter + vage omschrijving
idx search "graph" --semantic --where "source_type = 'screenshot'" --json
```

### 2. Deep OCR & Text Understanding

Tekst in afbeeldingen is net zo doorzoekbaar als tekst in bestanden.

- **Qwen3-VL** leest elke pixel: handgeschreven, screenshot, foto van een whiteboard
- **FTS5** indexeert alles wat het vindt — full-text search werkt op OCR'd text
- Niet alleen Engels — Nederlands, code, formules, alles

```bash
# "Waar stond die SSH config ook alweer?"
idx search "ssh config authorized_keys" --lexical --json

# "Die ene Slack thread over the deployment"
idx search "deploy production Friday rollback" --lexical --json

# "Het recept met knoflook en citroen"  
idx search "knoflook citroen" --lexical --json
```

### 3. Universal Intake — rsync in, index out

Draait op de Hetzner box. Data komt binnen via rsync + cron van de Mac.

| Bron | Hoe het binnenkomt |
|------|-------------------|
| Screenshots | `rsync ~/Pictures/Screenshots/` → `/esther/data/images/screenshots/` |
| Downloads | `rsync ~/Downloads/*.{png,jpg}` → `/esther/data/images/downloads/` |
| Photos | `rsync ~/Photos/` → `/esther/data/images/photos/` (periodic, large) |
| URLs | `idx fetch https://...` — download + index direct |
| Bulk import | `idx import ~/old-backup/` — one-time migration |
| Clipboard | `ssh hetzner "idx capture" < image.png` — pipe via SSH |

Cron job op de Mac (draait 's nachts of handmatig):
```bash
# /etc/cron.d/sync-images
0 3 * * * joel rsync -avz ~/Pictures/Screenshots/ hetzner:/esther/data/images/screenshots/
0 3 * * * joel rsync -avz ~/Downloads/*.png hetzner:/esther/data/images/downloads/
```

En op de Hetzner box, na sync:
```bash
# /etc/cron.d/index-after-sync
30 3 * * * joel cd /esther/data/repos/image-indexer && uv run idx index /esther/data/images/
```

### 4. Source Awareness

Niet alle afbeeldingen zijn gelijk. De index onthoudt waar ze vandaan komen:

```json
```

### 5. The Agent Layer (where Esther comes in)

Esther kan actief curaten. Niet alleen zoeken — *organiseren*:

```bash
# Query vanuit Signal: "heb ik recent iets gescreenshot van docker errors?"
idx search "docker error" --type screenshot --days 7 --json
```

Periodieke curator runs (cron):
```bash
0 4 * * 0 joel cd /esther/data/repos/image-indexer && uv run idx curator report --json
```

### 6. Composable CLI

De Unix way. Kleine tools die samenwerken:

```bash
# Simpel
idx search "bayes" --json

# Pipe naar andere tools
idx search "loss curve" --json | jq -r '.[].path' | xargs feh -g 800x600

# Feed vanuit een bron
find ~/Downloads -name "*.png" -mtime -7 | xargs idx index

# Capture → index in één move  
wl-paste | idx capture --tags "clipboard"  # Wayland clipboard

# Export wat je vindt
idx search "meeting notes" --json | idx export --obsidian "Meeting Notes"
```

---

## Design Principles

1. **Anything with pixels is fair game.** Geen aannames over foto vs screenshot vs diagram.
2. **Vibes over precision.** Je hoeft niet exact te weten wat je zoekt.
3. **OCR everything.** Tekst in pixels is tekst. Punt.
4. **Local-first.** Je data verlaat je machine niet (behalve verkleinde previews naar RunPod).
5. **Agent-friendly.** JSON output, exit codes, geen prompts. Esther kan het draaien.
6. **Idempotent ingest.** Gooi dezelfde map 100 keer erin — alleen nieuwe/geupdate files worden verwerkt.

---

## What We Already Have

| Component | Status | Notes |
|-----------|--------|-------|
| `preprocess.py` | ✅ | Resize, EXIF fix, JPEG encode, SHA-256 |
| `handler.py` | ✅ | SigLIP2 + Qwen3-VL (incl. OCR via caption prompt) |
| `db.py` | ✅ | SQLite + vec0 + FTS5, drie search surfaces |
| `e2e_demo.py` | ✅ | Full pipeline validation |
| CLI (`idx`) | 📋 | Agentic command interface |
| RunPod client | 📋 | HTTP client, polling, error handling |
| rsync setup | ❌ | Mac → Hetzner sync scripts + cron |
| Source tagging | ❌ | Path-based classification (screenshots/ vs downloads/) |
| Colour search | ❌ | Dominant colour extraction + filtering |
| Duplicate detection | ❌ | Perceptual hash + vector similarity |
| Export | ❌ | Markdown, JSONL |

---

## What I'd Build Next

1. **CLI + RunPod client** — Use the tool, prove it works end-to-end
2. **rsync ingest pipeline** — Scripts om data van de Mac naar de Hetzner box te krijgen + cron
3. **Source tagging** — Classificeer based op path: `/screenshots/`, `/downloads/`, `/photos/`
4. **Colour extraction** — "Find that thing with the blue background"
5. **OCR-weighted search** — Boost text-rich results when query looks like text
6. **Duplicate detection** — Perceptual hash + SigLIP2 vector similarity

---

*"Je visuele geheugen, buiten je hoofd, doorzoekbaar."*
