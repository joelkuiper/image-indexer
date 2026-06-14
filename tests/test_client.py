"""Tests for image_indexer.client."""
import json
from unittest.mock import MagicMock, patch

import pytest

from image_indexer.client import RunPodClient


@pytest.fixture
def client():
    return RunPodClient(endpoint_id="test123", api_key="rpa_test_key")


class TestRunPodClient:
    def test_url_construction(self, client):
        assert client._url("runsync") == "https://api.runpod.ai/v2/test123/runsync"

    @patch("image_indexer.client.requests.post")
    def test_run_success(self, mock_post, client):
        """Successful runsync returns the output dict."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "COMPLETED",
            "output": {
                "embedding": [0.1] * 1152,
                "description": "A serene lake at dawn.",
            },
        }
        mock_post.return_value = mock_resp

        result = client.run(b"fake-image-bytes", task="all")

        assert result["description"] == "A serene lake at dawn."
        assert len(result["embedding"]) == 1152

        # Verify the request was made correctly.
        call = mock_post.call_args
        assert "runsync" in call[1].get("url", "") or "runsync" in call.args[0]
        assert call[1]["headers"]["Authorization"] == "Bearer rpa_test_key"
        assert "image_b64" in call[1]["json"]["input"]

    @patch("image_indexer.client.requests.post")
    def test_run_inference_error(self, mock_post, client):
        """Inference errors in the output raise RuntimeError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "COMPLETED",
            "output": {"error": "CUDA out of memory"},
        }
        mock_post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            client.run(b"fake")

    @patch("image_indexer.client.requests.post")
    def test_run_job_failed(self, mock_post, client):
        """Job-level failures raise RuntimeError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "FAILED",
            "error": "Worker crashed",
        }
        mock_post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="Worker crashed"):
            client.run(b"fake")

    @patch("image_indexer.client.time.sleep")  # Don't actually wait
    @patch("image_indexer.client.requests.get")
    @patch("image_indexer.client.requests.post")
    def test_run_polls_on_async(self, mock_post, mock_get, _sleep, client):
        """When runsync returns without COMPLETED, it falls back to polling."""
        # runsync returns an IN_PROGRESS status with a job ID.
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
            "output": {"embedding": [0.5] * 1152, "description": "Done"},
        }
        mock_get.side_effect = [mock_get_resp_in_progress, mock_get_resp_done]

        result = client.run(b"fake")
        assert result["description"] == "Done"
        assert mock_get.call_count == 2

    @patch("image_indexer.client.time.sleep")
    @patch("image_indexer.client.requests.post")
    def test_retry_on_429(self, mock_post, _sleep, client):
        """429 triggers a retry with backoff."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = {
            "status": "COMPLETED",
            "output": {"description": "after retry"},
        }
        mock_post.side_effect = [resp_429, resp_ok]

        result = client.run(b"fake")
        assert result["description"] == "after retry"
        assert mock_post.call_count == 2
        _sleep.assert_called_once()

    @patch("image_indexer.client.time.sleep")
    @patch("image_indexer.client.requests.post")
    def test_retry_exhausted(self, mock_post, _sleep, client):
        """After MAX_RETRIES, a ConnectionError is raised."""
        resp = MagicMock()
        resp.status_code = 503
        mock_post.return_value = resp

        with pytest.raises(ConnectionError, match="503"):
            client.run(b"fake")
