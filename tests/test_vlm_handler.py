import base64
import unittest
from unittest.mock import MagicMock, patch

# Mock torch, PIL, and transformers to prevent heavy downloads during unit-testing cycle
import sys

mock_modules = ["torch", "transformers", "PIL", "accelerate"]
for module in mock_modules:
    sys.modules[module] = MagicMock()

# Setup explicit mock bindings for our loader and execution targets
from worker import handler


class TestRunPodHandlerVLM(unittest.TestCase):
    @patch("worker.handler.processor")
    @patch("worker.handler.model")
    @patch("PIL.Image.open")
    def test_handler_vlm_inference_pipeline(
        self, mock_image_open, mock_model, mock_processor
    ):
        # Configure model attributes to skip Lazy Loading trigger in code
        handler.model = MagicMock()
        handler.processor = MagicMock()
        handler.device = "cpu"

        # Configure visual projection mocking
        mock_visual_output = MagicMock()
        mock_visual_output.mean.return_value.cpu.return_value.to.return_value.numpy.return_value.tolist.return_value = [
            0.1,
            0.2,
            0.3,
        ]
        handler.model.visual.return_value = mock_visual_output
        handler.model.visual.__bool__.return_value = True

        # Configure transcription mocking
        handler.model.generate.return_value = [[1, 2, 3, 4]]
        handler.processor.batch_decode.return_value = [
            "A beautiful landscape with deep warm sunset vibes."
        ]

        # Configure processor output mocking (mock dictionary return values)
        mock_processor_output = {
            "input_ids": MagicMock(),
            "pixel_values": MagicMock(),
            "image_grid_thw": MagicMock(),
        }
        handler.processor.return_value = mock_processor_output

        # Define base64 job payload
        dummy_b64 = base64.b64encode(b"fake-image-bytes").decode("utf-8")
        job = {"input": {"image_b64": dummy_b64}}

        # Trigger
        response = handler.handler(job)

        # Verification Contracts
        self.assertIn("description", response)
        self.assertIn("embedding", response)
        self.assertEqual(
            response["description"],
            "A beautiful landscape with deep warm sunset vibes.",
        )
        self.assertEqual(response["embedding"], [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
