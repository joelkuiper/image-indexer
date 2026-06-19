"""Async inference server exposing HTTP API for remote indexing.

Endpoints:
  POST /api/inference  — Submit inference job
  GET  /api/status/{id} — Poll job status

Features:
  - Async job queue
  - Concurrency limiting
  - CLIP + Qwen3-VL models
  - Job persistence (SQLite)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sqlite3
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
)

import torch

# Configuration
DB_PATH = os.getenv("SERVER_DB_PATH", "/app/server.db")
MAX_WORKERS = int(os.getenv("SERVER_MAX_WORKERS", 3))
MODEL_CACHE_DIR = os.getenv("HF_HOME", "/app/model-cache")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=logging.getLevelName(LOG_LEVEL))
log = logging.getLogger(__name__)


# ── Database ──────────────────────────────────────────────────────────────────

@dataclass
class Job:
    """Job tracking."""
    job_id: str
    status: str = "queued"  # queued | processing | completed | failed
    input_data: dict = field(default_factory=dict)
    output_data: dict = field(default_factory=dict)
    error: str = ""
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    updated_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class JobManager:
    """Job queue with SQLite persistence."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self):
        """Create database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                input_data TEXT NOT NULL,
                output_data TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)")
        conn.commit()
        conn.close()

    def add_job(self, job_id: str, input_data: dict) -> Job:
        """Add new job to queue."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO jobs 
                (job_id, status, input_data, output_data, error, created_at, updated_at)
                VALUES (?, 'queued', ?, NULL, NULL, ?, ?)
                """,
                (job_id, json.dumps(input_data), job_id, job_id),
            )
            conn.commit()
            conn.close()
            return Job(
                job_id=job_id,
                status="queued",
                input_data=input_data,
            )

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT job_id, status, input_data, output_data, error, created_at, updated_at FROM jobs WHERE job_id = ?",
                (job_id,),
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return Job(
                    job_id=row[0],
                    status=row[1],
                    input_data=json.loads(row[2]) if row[2] else {},
                    output_data=json.loads(row[3]) if row[3] else {},
                    error=row[4] or "",
                    created_at=row[5],
                    updated_at=row[6],
                )
            return None

    def update_job(self, job_id: str, status: str, output_data: dict | None = None, error: str = ""):
        """Update job status."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            if output_data:
                cursor.execute(
                    """
                    UPDATE jobs SET status = ?, output_data = ?, error = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (status, json.dumps(output_data), error, job_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE jobs SET status = ?, error = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (status, error, job_id),
                )
            conn.commit()
            conn.close()

    def get_queued_jobs(self, max_jobs: int = MAX_WORKERS) -> list[Job]:
        """Get queued jobs up to concurrency limit."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT job_id, status, input_data, output_data, error, created_at, updated_at
                FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (max_jobs,),
            )
            rows = cursor.fetchall()
            conn.close()

            return [
                Job(
                    job_id=row[0],
                    status=row[1],
                    input_data=json.loads(row[2]) if row[2] else {},
                    output_data=json.loads(row[3]) if row[3] else {},
                    error=row[4] or "",
                    created_at=row[5],
                    updated_at=row[6],
                )
                for row in rows
            ]

    def complete_job(self, job_id: str, output_data: dict):
        """Mark job as completed."""
        self.update_job(job_id, "completed", output_data)

    def fail_job(self, job_id: str, error: str):
        """Mark job as failed."""
        self.update_job(job_id, "failed", error=error)


# ── Inference Models ──────────────────────────────────────────────────────────

class InferenceEngine:
    """CLIP + Qwen3-VL inference engine."""

    def __init__(self):
        self._initialized = False
        self._embed_model = None
        self._caption_model = None
        self._embed_processor = None
        self._caption_processor = None
        self._device = None
        self._dtype = None

    def _init_device(self):
        """Initialize device (CPU or GPU)."""
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = (
            torch.bfloat16
            if self._device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        log.info(f"Using device: {self._device}")

    def _load_models(self):
        """Load models lazily."""
        if self._initialized:
            return

        if self._embed_model is None:
            embed_model_id = "openai/clip-vit-base-patch32"
            self._embed_processor = CLIPProcessor.from_pretrained(embed_model_id)
            self._embed_model = CLIPModel.from_pretrained(
                embed_model_id,
                torch_dtype=torch.float32,
                device_map="auto" if self._device == "cuda" else None,
            ).eval()
            if self._device == "cpu":
                self._embed_model.to(self._device)

        if self._caption_model is None:
            caption_model_id = "Qwen/Qwen3-VL-4B-Instruct"
            self._caption_processor = AutoProcessor.from_pretrained(caption_model_id)
            self._caption_model = AutoModelForImageTextToText.from_pretrained(
                caption_model_id,
                torch_dtype=self._dtype,
                device_map="auto" if self._device == "cuda" else None,
            ).eval()
            if self._device == "cpu":
                self._caption_model.to(self._device)

        self._initialized = True
        log.info("Models loaded")

    def embed_image(self, image: Image.Image) -> list[float]:
        """Generate CLIP embedding."""
        self._load_models()

        inputs = self._embed_processor(images=[image], return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._embed_model.get_image_features(**inputs)
            feats = output.pooler_output if hasattr(output, "pooler_output") else output
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            return feats[0].cpu().to(torch.float32).tolist()

    def caption_image(self, image: Image.Image) -> str:
        """Generate caption."""
        self._load_models()

        prompt = "You are an expert photo curator building a searchable archive. " "Describe this image in 2-4 sentences. Cover the main subjects, the setting, " "the mood and lighting, dominant colours, and transcribe any visible text. " "Write plain descriptive prose with no preamble."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self._caption_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            generated = self._caption_model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )

        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs["input_ids"], generated)
        ]
        text = self._caption_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text.strip()

    async def run(self, image_bytes: bytes, task: str = "all") -> dict[str, Any]:
        """Run inference (blocking in event loop)."""
        import io

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result: dict[str, Any] = {"models": {}}

        if task in ("embed", "all"):
            embedding = self.embed_image(image)
            result["embedding"] = embedding
            result["embedding_dim"] = 512
            result["models"]["embed"] = "openai/clip-vit-base-patch32"

        if task in ("caption", "all"):
            description = self.caption_image(image)
            result["description"] = description
            result["models"]["caption"] = "Qwen/Qwen3-VL-4B-Instruct"

        return result


# ── Job Processor ────────────────────────────────────────────────────────────

async def process_job(job_manager: JobManager, job: Job, engine: InferenceEngine):
    """Process a single inference job."""
    log.info(f"Processing job {job.job_id}")

    try:
        image_b64 = job.input_data.get("input", {}).get("image_b64")
        task = job.input_data.get("input", {}).get("task", "all")

        if not image_b64:
            raise ValueError("Missing image_b64")

        image_bytes = base64.b64decode(image_b64)
        result = await engine.run(image_bytes, task)

        job_manager.complete_job(job.job_id, result)
        log.info(f"Completed job {job.job_id}")

    except Exception as e:
        log.error(f"Job {job.job_id} failed: {e}")
        job_manager.fail_job(job.job_id, str(e))


# ── API ──────────────────────────────────────────────────────────────────────

@dataclass
class InferenceRequest(BaseModel):
    image_b64: str
    task: str = "all"


@dataclass
class InferenceResponse(BaseModel):
    id: str
    status: str


@dataclass
class InferenceStatusResponse(BaseModel):
    id: str
    status: str
    output: dict | None = None
    error: str | None = None


# Global instances
job_manager = JobManager(DB_PATH)
engine = InferenceEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management."""
    # Startup
    log.info("Inference server starting...")
    engine._init_device()
    yield
    # Shutdown
    log.info("Inference server shutting down")


# ── Main App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Image Indexer Inference Server",
    description="Async inference server for visual memory search",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "inference-server"}


@app.post("/api/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Submit inference job."""
    job_id = f"job-{base64.b64encode(os.urandom(16)).decode()}"
    job_manager.add_job(job_id, {"input": request.model_dump(), "task": request.task})
    return InferenceResponse(id=job_id, status="queued")


@app.get("/api/status/{job_id}", response_model=InferenceStatusResponse)
async def status(job_id: str):
    """Poll job status."""
    job = job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == "completed":
        return InferenceStatusResponse(
            id=job.job_id,
            status=job.status,
            output=job.output_data,
        )

    if job.status == "failed":
        return InferenceStatusResponse(
            id=job.job_id,
            status=job.status,
            error=job.error,
        )

    return InferenceStatusResponse(
        id=job.job_id,
        status=job.status,
    )


# ── Worker Pool ──────────────────────────────────────────────────────────────

async def worker_pool():
    """Background worker pool processing queued jobs."""
    while True:
        await asyncio.sleep(1)  # Check every second

        queued_jobs = job_manager.get_queued_jobs()

        if not queued_jobs:
            continue

        for job in queued_jobs:
            asyncio.create_task(process_job(job_manager, job, engine))


# ── Run Server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", 8080))

    log.info(f"Starting inference server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, workers=1)
