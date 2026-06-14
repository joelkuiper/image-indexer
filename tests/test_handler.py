"""Handler tests. Models are mocked so this runs in milliseconds with no GPU
or network. We test the request contract and routing, not the model weights.
"""

import base64
import unittest
from unittest.mock import patch


def _b64():
    return base64.b64encode(b"fake-image-bytes").decode("utf-8")


class TestHandlerContract(unittest.TestCase):
    @patch("worker.handler.load_models")
    def test_missing_image(self, _load):
        from worker import handler

        resp = handler.handler({"input": {}})
        self.assertEqual(resp["error"], "Missing required field: 'image_b64'")

    @patch("worker.handler.load_models")
    def test_invalid_task(self, _load):
        from worker import handler

        resp = handler.handler({"input": {"image_b64": _b64(), "task": "bogus"}})
        self.assertIn("Invalid task", resp["error"])

    @patch("worker.handler.load_models")
    @patch("worker.handler.caption_image", return_value="A calm harbour at dusk.")
    @patch("worker.handler.embed_image", return_value=[0.1] * 512)
    @patch("PIL.Image.open")
    def test_task_all(self, mock_open, _embed, _caption, _load):
        from worker import handler

        mock_open.return_value.convert.return_value = object()
        resp = handler.handler({"input": {"image_b64": _b64(), "task": "all"}})
        self.assertEqual(len(resp["embedding"]), 512)
        self.assertEqual(resp["embedding_dim"], 512)
        self.assertEqual(resp["description"], "A calm harbour at dusk.")
        self.assertIn("embed", resp["models"])
        self.assertIn("caption", resp["models"])

    @patch("worker.handler.load_models")
    @patch("worker.handler.caption_image", return_value="X")
    @patch("worker.handler.embed_image", return_value=[0.1] * 512)
    @patch("PIL.Image.open")
    def test_task_embed_only(self, mock_open, _embed, mock_caption, _load):
        from worker import handler

        mock_open.return_value.convert.return_value = object()
        resp = handler.handler({"input": {"image_b64": _b64(), "task": "embed"}})
        self.assertIn("embedding", resp)
        self.assertNotIn("description", resp)
        mock_caption.assert_not_called()


if __name__ == "__main__":
    unittest.main()
