"""Asynchronous RunPod HTTP client.

Wraps the /runsync and /run endpoints with httpx:
- Asynchronous non-blocking network I/O
- Semaphore concurrency limiting to prevent rate-limiting or network over-saturation
- Retry on transient failures (429, 502, 503)
- Polling for async jobs
- Clean error handling (inference errors, timeouts)
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass

import httpx

from image_indexer.config import settings

log = logging.getLogger(__name__)

RUNPOD_API_BASE = settings.get("runpod_api_base", "https://api.runpod.ai/v2")

# Retry config for transient HTTP errors.
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds
RETRYABLE_CODES = {429, 502, 503}


@dataclass
class RunPodClient:
    """Async Client for a RunPod serverless endpoint."""

    endpoint_id: str
    api_key: str
    timeout: int = 300  # seconds, covers cold start + inference
    concurrency_limit: int = 10  # max concurrent HTTP uploads to RunPod

    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(self.concurrency_limit)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _url(self, path: str) -> str:
        return f"{RUNPOD_API_BASE}/{self.endpoint_id}/{path}"

    async def run(self, image_bytes: bytes, task: str = "all") -> dict:
        """Send an image and get back embeddings + caption concurrently.

        Uses /run to submit the job asynchronously, then polls the status
        endpoint. Respects concurrency_limit semaphore to prevent network choke.

        Returns the "output" dict from the RunPod response.
        Raises RuntimeError on API errors or inference failures.
        """
        image_b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "input": {
                "image_b64": image_b64,
                "task": task,
            }
        }

        async with self._semaphore:
            # Step 1: Submit the job asynchronously
            response_data = await self._post_with_retry("run", payload)
            job_id = response_data.get("id")
            status = response_data.get("status")

            if not job_id:
                raise RuntimeError(
                    f"RunPod failed to return a job ID. Response: {response_data}"
                )

            # If it's somehow already completed (cached / runsync-like response), return it
            if status == "COMPLETED":
                return self._extract_output(response_data)
            if status == "FAILED":
                raise RuntimeError(
                    f"RunPod job failed on submit: {response_data.get('error', 'unknown')}"
                )

            # Step 2: Poll for the result asynchronously
            return await self._poll_result(job_id)

    async def _post_with_retry(self, endpoint: str, payload: dict) -> dict:
        """Asynchronous HTTP POST with exponential backoff on transient errors."""
        last_error: Exception | None = None

        # We share/create an AsyncClient for retries or use single sessions
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(MAX_RETRIES):
                try:
                    resp = await client.post(
                        self._url(endpoint),
                        headers=self._headers,
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
                            f"RunPod returned HTTP {resp.status_code} after {MAX_RETRIES} attempts"
                        )
                    if resp.status_code != 200:
                        raise RuntimeError(
                            f"RunPod HTTP {resp.status_code}: {resp.text}"
                        )

                    return resp.json()
                except (httpx.RequestError, ConnectionError) as e:
                    last_error = e
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
                        continue

        raise ConnectionError(
            f"RunPod request failed after {MAX_RETRIES} attempts: {last_error}"
        )

    async def _poll_result(self, job_id: str) -> dict:
        """Poll /status/{id} asynchronously until the job completes or times out."""
        deadline = asyncio.get_running_loop().time() + self.timeout
        poll_interval = 2.0

        async with httpx.AsyncClient(timeout=30) as client:
            while asyncio.get_running_loop().time() < deadline:
                try:
                    resp = await client.get(
                        self._url(f"status/{job_id}"),
                        headers=self._headers,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        status = data.get("status")
                        if status == "COMPLETED":
                            return self._extract_output(data)
                        if status == "FAILED":
                            raise RuntimeError(
                                f"RunPod job failed: {data.get('error', 'unknown')}"
                            )
                except httpx.RequestError as e:
                    # Ignore transient status connection errors during polling, retry
                    log.warning(f"Transient polling request error for {job_id}: {e}")

                await asyncio.sleep(poll_interval)

        raise TimeoutError(
            f"RunPod job {job_id} did not complete within {self.timeout}s"
        )

    @staticmethod
    def _extract_output(data: dict) -> dict:
        """Pull the useful bits out of a RunPod response."""
        output = data.get("output", {})
        if not output:
            # Some endpoints nest under "result" instead.
            output = data
            # Strip RunPod metadata keys.
            output = {
                k: v
                for k, v in output.items()
                if k not in ("id", "status", "delayTime", "executionTime")
            }
        if "error" in output:
            raise RuntimeError(f"Inference error: {output['error']}")
        return output
