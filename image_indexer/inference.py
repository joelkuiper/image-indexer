"""Asynchronous inference client supporting local and remote modes.

Two modes:
  - Local: Direct inference on local GPU/CPU (default)
  - Remote: Async HTTP API with job queue and polling

Design:
  - Unified interface for both modes
  - Async pipeline: upload → poll → results
  - Retry on transient failures
  - Concurrency limiting via semaphore
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from image_indexer.config import settings

log = logging.getLogger(__name__)

InferenceMode = Literal["local", "remote"]

# Configuration
LOCAL_WORKERS = int(settings.get("max_workers", 5))
REMOTE_API_BASE = settings.get("remote_api_base", "https://your-vm.example.com/api")
REMOTE_TIMEOUT = int(settings.get("remote_timeout", 300))
MAX_RETRIES: int = 3
RETRY_BACKOFF: float = 2.0
RETRYABLE_CODES: set[int] = {429, 502, 503}
POLL_INTERVAL: float = 2.0


@dataclass
class InferenceResult:
    """Result from inference (embedding + caption)."""
    embedding: list[float]
    embedding_dim: int
    description: str
    models: dict[str, str]


@dataclass
class InferenceJob:
    """Job tracking for remote mode."""
    job_id: str
    status: str  # queued | processing | completed | failed
    output: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class LocalInference:
    """Local inference client using PyTorch models."""

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
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = (
            torch.bfloat16
            if self._device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        log.info(f"Using device: {self._device}")

    def _load_models(self):
        """Load both models lazily on first call."""
        if self._initialized:
            return

        import torch
        from PIL import Image
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            CLIPModel,
            CLIPProcessor,
        )

        # CLIP embedder
        self._embed_model_id = settings.get(
            "embed_model_id", "openai/clip-vit-base-patch32"
        )
        self._embed_dim = settings.get("embed_dim", 512)

        self._embed_processor = CLIPProcessor.from_pretrained(self._embed_model_id)
        self._embed_model = CLIPModel.from_pretrained(
            self._embed_model_id,
            torch_dtype=torch.float32,
            device_map="auto" if self._device == "cuda" else None,
        ).eval()
        if self._device == "cpu":
            self._embed_model.to(self._device)

        # Qwen3-VL captioner
        self._caption_model_id = settings.get(
            "caption_model_id", "Qwen/Qwen3-VL-4B-Instruct"
        )
        self._caption_max_tokens = 256

        self._caption_processor = AutoProcessor.from_pretrained(self._caption_model_id)
        self._caption_model = AutoModelForImageTextToText.from_pretrained(
            self._caption_model_id,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
        ).eval()
        if self._device == "cpu":
            self._caption_model.to(self._device)

        self._initialized = True
        log.info(f"Models loaded on {self._device}")

    def embed_image(self, image: Image.Image) -> list[float]:
        """Generate CLIP embedding for an image."""
        self._load_models()

        inputs = self._embed_processor(images=[image], return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._embed_model.get_image_features(**inputs)
            feats = output.pooler_output if hasattr(output, "pooler_output") else output
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            return feats[0].cpu().to(torch.float32).tolist()

    def caption_image(self, image: Image.Image) -> str:
        """Generate rich caption using Qwen3-VL."""
        self._load_models()

        prompt = settings.get(
            "caption_prompt",
            "You are an expert photo curator building a searchable archive. "
            "Describe this image in 2-4 sentences. Cover the main subjects, the setting, "
            "the mood and lighting, dominant colours, and transcribe any visible text. "
            "Write plain descriptive prose with no preamble."
        )

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
                **inputs, max_new_tokens=self._caption_max_tokens, do_sample=False
            )

        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs["input_ids"], generated)
        ]
        text = self._caption_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text.strip()

    async def run(self, image_bytes: bytes, task: str = "all") -> InferenceResult:
        """Run inference locally (blocking call in event loop)."""
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result: dict[str, Any] = {"models": {}}

        if task in ("embed", "all"):
            embedding = self.embed_image(image)
            result["embedding"] = embedding
            result["embedding_dim"] = self._embed_dim
            result["models"]["embed"] = self._embed_model_id

        if task in ("caption", "all"):
            description = self.caption_image(image)
            result["description"] = description
            result["models"]["caption"] = self._caption_model_id

        return InferenceResult(
            embedding=result["embedding"],
            embedding_dim=result["embedding_dim"],
            description=result["description"],
            models=result["models"],
        )


@dataclass
class RemoteInference:
    """Remote inference client using async HTTP API."""

    url: str
    timeout: int = REMOTE_TIMEOUT

    def __post_init__(self):
        if not self.url.endswith("/"):
            self.url += "/"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self.url}{path}"

    async def _post_with_retry(self, endpoint: str, payload: dict) -> dict:
        """POST with exponential backoff on transient errors."""
        last_error: Exception | None = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(MAX_RETRIES):
                try:
                    resp = await client.post(
                        self._url(endpoint),
                        headers=self._headers(),
                        json=payload,
                    )

                    if (
                        resp.status_code in RETRYABLE_CODES
                        and attempt < MAX_RETRIES - 1
                    ):
                        await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
                        continue

                    if resp.status_code in RETRYABLE_CODES:
                        raise ConnectionError(
                            f"Server returned HTTP {resp.status_code} after {MAX_RETRIES} attempts"
                        )

                    if resp.status_code != 200:
                        raise RuntimeError(
                            f"Server HTTP {resp.status_code}: {resp.text}"
                        )

                    return resp.json()

                except (httpx.RequestError, ConnectionError) as e:
                    last_error = e
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
                        continue

        raise ConnectionError(
            f"Remote request failed after {MAX_RETRIES} attempts: {last_error}"
        )

    async def _poll_with_retry(self, job_id: str) -> dict:
        """Poll job status with retry on connection errors."""
        deadline = asyncio.get_running_loop().time() + self.timeout
        poll_interval = POLL_INTERVAL

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while asyncio.get_running_loop().time() < deadline:
                try:
                    resp = await client.get(
                        self._url(f"status/{job_id}"),
                        headers=self._headers(),
                    )
                    if resp.status_code == 200:
                        return resp.json()
                except httpx.RequestError:
                    # Transient error, retry
                    pass

                await asyncio.sleep(poll_interval)

        raise TimeoutError(
            f"Job {job_id} did not complete within {self.timeout}s"
        )

    async def run(self, image_bytes: bytes, task: str = "all") -> InferenceResult:
        """Submit job asynchronously and poll for results."""
        image_b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "input": {
                "image_b64": image_b64,
                "task": task,
            }
        }

        # Step 1: Submit job
        response_data = await self._post_with_retry("inference", payload)
        job_id = response_data.get("id")
        status = response_data.get("status")

        if not job_id:
            raise RuntimeError(f"Server failed to return job ID: {response_data}")

        if status == "FAILED":
            error = response_data.get("error", "unknown")
            raise RuntimeError(f"Job failed on submit: {error}")

        if status == "COMPLETED":
            # Rare but possible for cached jobs
            return self._extract_output(response_data)

        # Step 2: Poll for result
        return await self._poll_result(job_id)

    async def _poll_result(self, job_id: str) -> InferenceResult:
        """Poll job status until completed or failed."""
        while True:
            status_data = await self._poll_with_retry(job_id)
            status = status_data.get("status")
            output = status_data.get("output", {})

            if status == "COMPLETED":
                return self._extract_output(status_data)

            if status == "FAILED":
                error = output.get("error", "unknown")
                raise RuntimeError(f"Job failed: {error}")

            log.info(f"Job {job_id} status: {status}")
            await asyncio.sleep(POLL_INTERVAL)

    @staticmethod
    def _extract_output(data: dict) -> InferenceResult:
        """Extract inference result from server response."""
        output = data.get("output", {})
        if not output:
            output = data

        embedding = output.get("embedding")
        description = output.get("description", "")
        models = output.get("models", {})
        embedding_dim = output.get("embedding_dim", 512)

        if embedding is None:
            raise RuntimeError("Missing embedding in response")

        if "error" in output:
            raise RuntimeError(f"Inference error: {output['error']}")

        return InferenceResult(
            embedding=embedding,
            embedding_dim=embedding_dim,
            description=description,
            models=models,
        )


class InferenceClient:
    """Unified inference client supporting local and remote modes."""

    def __init__(
        self,
        mode: InferenceMode = "local",
        url: str | None = None,
        workers: int = LOCAL_WORKERS,
    ):
        self.mode = mode
        self._local_inference = LocalInference()
        self._remote_inference = RemoteInference(
            url=url or REMOTE_API_BASE,
            timeout=REMOTE_TIMEOUT,
        )

        if mode == "local":
            self._inference = self._local_inference
        else:
            self._inference = self._remote_inference

        self._semaphore = asyncio.Semaphore(workers)
        log.info(f"Inference client initialized: mode={mode}")

    async def run(
        self, image_bytes: bytes, task: str = "all"
    ) -> InferenceResult:
        """Run inference (local or remote)."""
        if self.mode == "local":
            # Local mode: blocking call in event loop
            return await asyncio.to_thread(
                self._inference.run, image_bytes, task
            )
        else:
            # Remote mode: async HTTP
            return await self._inference.run(image_bytes, task)

    async def run_concurrent(
        self, image_chunks: list[tuple[Path, bytes]]
    ) -> list[tuple[Path, InferenceResult]]:
        """Run inference on multiple images concurrently."""
        tasks = []

        async def process_one(path: Path, img_bytes: bytes) -> tuple[Path, InferenceResult]:
            async with self._semaphore:
                return path, await self.run(img_bytes)

        for path, img_bytes in image_chunks:
            tasks.append(process_one(path, img_bytes))

        results = await asyncio.gather(*tasks)
        return list(results)


# Convenience function for CLI
async def run_inference(
    image_bytes: bytes,
    task: str = "all",
    mode: InferenceMode = "local",
    url: str | None = None,
    workers: int = LOCAL_WORKERS,
) -> InferenceResult:
    """Run inference with given parameters."""
    client = InferenceClient(mode=mode, url=url, workers=workers)
    return await client.run(image_bytes, task)
