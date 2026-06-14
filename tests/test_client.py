"""Tests for image_indexer.client."""

from unittest.mock import MagicMock, patch

import pytest

from image_indexer.client import RunPodClient


@pytest.fixture
def client():
    return RunPodClient(endpoint_id="test123", api_key="rpa_test_key")


class TestRunPodClient:
    def test_url_construction(self, client):
        assert client._url("run") == "https://api.runpod.ai/v2/test123/run"

    @pytest.mark.anyio
    async def test_run_success(self, client):
        """Successful runsync-like COMPLETED submission returns the output dict."""
        # We Mock httpx.AsyncClient's post method
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "status": "COMPLETED",
                "id": "job_123",
                "output": {
                    "embedding": [0.1] * 512,
                    "description": "A serene lake at dawn.",
                },
            }
            mock_post.return_value = mock_resp

            result = await client.run(b"fake-image-bytes", task="all")

            assert result["description"] == "A serene lake at dawn."
            assert len(result["embedding"]) == 512
            mock_post.assert_called_once()

    @pytest.mark.anyio
    async def test_run_inference_error(self, client):
        """Inference errors in the output raise RuntimeError."""
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "status": "COMPLETED",
                "id": "job_123",
                "output": {"error": "CUDA out of memory"},
            }
            mock_post.return_value = mock_resp

            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                await client.run(b"fake")

    @pytest.mark.anyio
    async def test_run_job_failed(self, client):
        """Job-level failures raise RuntimeError."""
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            # Submitted but directly marked as failed
            mock_resp.json.return_value = {
                "status": "FAILED",
                "id": "job_123",
                "error": "Worker crashed",
            }
            mock_post.return_value = mock_resp

            with pytest.raises(RuntimeError, match="Worker crashed"):
                await client.run(b"fake")

    @pytest.mark.anyio
    @patch("asyncio.sleep")  # Don't actually wait
    async def test_run_polls_on_async(self, _sleep, client):
        """When runsync returns without COMPLETED, it falls back to polling."""
        # submit mock post
        with (
            patch("httpx.AsyncClient.post") as mock_post,
            patch("httpx.AsyncClient.get") as mock_get,
        ):
            mock_post_resp = MagicMock()
            mock_post_resp.status_code = 200
            mock_post_resp.json.return_value = {
                "status": "IN_PROGRESS",
                "id": "job_abc",
            }
            mock_post.return_value = mock_post_resp

            # First poll: still in progress. Second poll: done.
            mock_get_resp_in_progress = MagicMock()
            mock_get_resp_in_progress.status_code = 200
            mock_get_resp_in_progress.json.return_value = {
                "status": "IN_PROGRESS",
                "id": "job_abc",
            }
            mock_get_resp_done = MagicMock()
            mock_get_resp_done.status_code = 200
            mock_get_resp_done.json.return_value = {
                "status": "COMPLETED",
                "id": "job_abc",
                "output": {"embedding": [0.5] * 512, "description": "Done"},
            }

            # Since mock_get is async, we return future Mock objects
            mock_get.side_effect = [mock_get_resp_in_progress, mock_get_resp_done]

            result = await client.run(b"fake")
            assert result["description"] == "Done"
            assert mock_get.call_count == 2

    @pytest.mark.anyio
    @patch("asyncio.sleep")
    async def test_retry_on_429(self, _sleep, client):
        """429 triggers a retry with backoff."""
        with patch("httpx.AsyncClient.post") as mock_post:
            resp_429 = MagicMock()
            resp_429.status_code = 429
            resp_ok = MagicMock()
            resp_ok.status_code = 200
            resp_ok.json.return_value = {
                "status": "COMPLETED",
                "id": "job_123",
                "output": {"description": "after retry"},
            }
            mock_post.side_effect = [resp_429, resp_ok]

            result = await client.run(b"fake")
            assert result["description"] == "after retry"
            assert mock_post.call_count == 2
            _sleep.assert_called_once()

    @pytest.mark.anyio
    @patch("asyncio.sleep")
    async def test_retry_exhausted(self, _sleep, client):
        """After MAX_RETRIES, a ConnectionError is raised."""
        with patch("httpx.AsyncClient.post") as mock_post:
            resp = MagicMock()
            resp.status_code = 503
            mock_post.return_value = resp

            with pytest.raises(ConnectionError, match="503"):
                await client.run(b"fake")
