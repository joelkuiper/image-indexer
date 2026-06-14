"""RunPod HTTP client.

Wraps the /runsync and /run endpoints with:
- Retry on transient failures (429, 502, 503)
- Polling for async jobs
- Clean error handling

Usage:
    client = RunPodClient(endpoint_id="abc123", api_key="rpa_...")
    result = client.run(image_bytes, task="all")
    # result = {"embedding": [...], "description": "..."}
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import requests

RUNPOD_API_BASE = "https://api.runpod.ai/v2"

# Retry config for transient HTTP errors.
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds
RETRYable_CODES = {429, 502, 503}


@dataclass
class RunPodClient:
    """Client for a RunPod serverless endpoint."""

    endpoint_id: str
    api_key: str
    timeout: int = 300  # seconds, covers cold start + inference

    @property
    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _url(self, path: str) -> str:
        return f"{RUNPOD_API_BASE}/{self.endpoint_id}/{path}"

    def run(self, image_bytes: bytes, task: str = "all") -> dict:
        """Send an image and get back embeddings + caption.

        Tries /runsync first (simpler, faster). Falls back to /run + polling
        if runsync is not available for the endpoint.

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

        response = self._post_with_retry("runsync", payload)

        data = response.json()
        status = data.get("status")

        if status == "COMPLETED":
            return self._extract_output(data)
        if status == "FAILED":
            raise RuntimeError(f"RunPod job failed: {data.get('error', 'unknown')}")

        # runsync returned something unexpected — it shouldn't happen but
        # fall through to poll just in case.
        return self._poll_result(data.get("id", ""))

    def _post_with_retry(self, endpoint: str, payload: dict) -> requests.Response:
        """POST with exponential backoff on transient errors."""
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    self._url(endpoint),
                    headers=self._headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code in RETRYable_CODES and attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue
                if resp.status_code in RETRYable_CODES:
                    # Exhausted retries on a retryable status.
                    raise ConnectionError(
                        f"RunPod returned HTTP {resp.status_code} after {MAX_RETRIES} attempts"
                    )
                return resp
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue

        raise ConnectionError(
            f"RunPod request failed after {MAX_RETRIES} attempts: {last_error}"
        )

    def _poll_result(self, job_id: str) -> dict:
        """Poll /status/{id} until the job completes or times out."""
        deadline = time.time() + self.timeout
        poll_interval = 2.0

        while time.time() < deadline:
            resp = requests.get(
                self._url(f"status/{job_id}"),
                headers=self._headers,
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                if status == "COMPLETED":
                    return self._extract_output(data)
                if status == "FAILED":
                    raise RuntimeError(f"RunPod job failed: {data.get('error', 'unknown')}")

            time.sleep(poll_interval)

        raise TimeoutError(f"RunPod job {job_id} did not complete within {self.timeout}s")

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
