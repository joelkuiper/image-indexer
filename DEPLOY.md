# E2E Test Plan: Social Media Folder Indexing

## 🎯 Test Doel
E2e testen van de image indexer op een gecontroleerde folder om de volledige pipeline te valideren voordat we de volledige Pictures folder indexeren.

## 📦 Test Context

### Source Data
- **Folder:** `/esther/data/Sync/Pictures/Social Media/`
- **Size:** 87MB
- **Files:** 489
- **Location:** VPS (Hetzner FSN1)

### Indexer Config
```toml
db_path = "~/.local/share/image-indexer/index.db"
embed_model_id = "openai/clip-vit-base-patch32"
caption_model_id = "Qwen/Qwen3-VL-4B-Instruct"
runpod_api_base = "https://api.runpod.ai/v2"
```

## 🔧 Pre-Test Checklist

### 1. RunPod Serverless Deployment
- [ ] Endpoint ID is gekregen en opgeslagen
- [ ] API key is geconfigureerd
- [ ] Endpoint status: **Deploying / Running / Failed?**
- [ ] GPU resources allocated (gebruik RunPod CLI of web portal)

### 2. Environment Setup
```bash
# Export credentials
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="rpa_your-api-key"

# Verify indexer CLI is available
which idx
idx --help

# Check settings
cat ~/.local/share/image-indexer/settings.toml
```

### 3. Database State
```bash
# Check existing index
idx status --json

# Verify DB location
ls -la ~/.local/share/image-indexer/index.db
```

## 🚀 Test Execution

### Step 1: Dry Run (Local Validation)
```bash
# Test local downscaling & metadata parsing
idx index /esther/data/Sync/Pictures/Social\ Media/ --dry-run --verbose
```

**Expected:**
- ✓ 489 files gevonden
- ✓ EXIF data geëxtraheerd
- ✓ SHA-256 hashes gegenereerd
- ✓ Geen RunPod calls (dry-run)

### Step 2: Production Indexing
```bash
# Full indexing (submits to RunPod)
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="rpa_your-api-key"

idx index /esther/data/Sync/Pictures/Social\ Media/ --json --verbose
```

**Expected Output:**
```json
{
  "total": 489,
  "processed": 489,
  "failed": 0,
  "queued": 0,
  "uploaded": 489,
  "indexed": 489,
  "searchable": 489
}
```

### Step 3: Search Validation

#### Semantic Search
```bash
idx search "social media screenshot" --semantic --json
idx search "profile picture" --semantic --json
```

**Expected:**
- ✓ Returns relevant images
- ✓ JSON output parseable
- ✓ Results ranked by cosine similarity

#### Lexical Search
```bash
idx search "screenshot" --lexical --json
idx search "profile" --lexical --json
```

**Expected:**
- ✓ OCR text gevonden in captions
- ✓ FTS5 full-text search werkt
- ✓ Relevante hits gerangschikt

#### Structured Search
```bash
idx search "format = 'PNG'" --structured --json
idx search "width > 1000" --structured --json
```

**Expected:**
- ✓ EXIF metadata correct
- ✓ SQL queries return results
- ✓ Filters combinable

### Step 4: Index Statistics
```bash
idx status --json
```

**Expected:**
```json
{
  "total_files": 489,
  "indexed_files": 489,
  "failed_files": 0,
  "db_size_mb": "X.XX",
  "embeddings_dim": 512,
  "last_indexed": "2026-06-16T..."
}
```

## ⚠️ Known Issues & Workarounds

### 1. RunPod Deploy Timeout
**Symptom:** Endpoint deployment crasht via wifi
**Workaround:**
```bash
# Check endpoint status
curl https://api.runpod.ai/v2/deployments/your-endpoint-id

# Retry deploy
runpod endpoint create --name esther-indexer --gpu A100 --docker-image ...
```

### 2. Database Corruption
**Symptom:** Index crashen of corrupt
**Workaround:**
```bash
# Backup & recreate
mv ~/.local/share/image-indexer/index.db ~/.local/share/image-indexer/index.db.backup
rm ~/.local/share/image-indexer/index.db
idx index /path/to/photos --dry-run
```

### 3. Memory Limit Exceeded
**Symptom:** Process killed during upload
**Workaround:**
```bash
# Reduce concurrency
export RUNPOD_CONCURRENCY=5
idx index /path/to/photos --concurrency 5
```

## 📊 Success Criteria

### Critical (Must Pass)
- [ ] 489/489 files geïndexed
- [ ] Geen failed uploads
- [ ] Search returns results voor alle modaliteiten
- [ ] JSON output consistent

### Important (Should Pass)
- [ ] Search response < 500ms
- [ ] DB size < 100MB
- [ ] No memory errors
- [ ] EXIF data complete

### Nice to Have (Could Pass)
- [ ] Semantic search > 80% accuracy
- [ ] OCR text extraction > 90%
- [ ] All image formats supported

## 🔍 Debug Tools

### Check RunPod Status
```bash
# Via API
curl -H "Authorization: Bearer rpa_your-api-key" \
  https://api.runpod.ai/v2/deployments/your-endpoint-id

# Via CLI (install if needed)
pip install runpod
runpod endpoint list
```

### Check Indexer Logs
```bash
# Tail index process
idx index /path/to/photos --verbose 2>&1 | tee index.log

# Check recent errors
grep -i "error\|fail\|timeout" index.log | tail -50
```

### Database Inspection
```bash
# List tables
sqlite3 ~/.local/share/image-indexer/index.db ".tables"

# Check embeddings
sqlite3 ~/.local/share/image-indexer/index.db "SELECT COUNT(*) FROM images;"

# Check failed entries
sqlite3 ~/.local/share/image-indexer/index.db "SELECT * FROM failed WHERE status = 'failed' LIMIT 10;"
```

## 🎬 Post-Test Actions

### If Successful
1. Document learnings
2. Update DEPLOY.md with endpoint config
3. Plan full Pictures folder indexing
4. Set up cron job for periodic updates

### If Failed
1. Review error logs
2. Check RunPod endpoint status
3. Test with single image first
4. Consider smaller test folder

## 📝 Notes

- **Timestamp:** 2026-06-16
- **Tester:** Esther
- **Priority:** High (gateway for full Pictures indexing)
- **Risk:** Medium (87MB is manageable, but RunPod connectivity uncertain)
