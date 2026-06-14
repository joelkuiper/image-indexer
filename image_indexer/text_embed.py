"""Local text embedding via SigLIP2.

SigLIP2 is a CLIP-style dual encoder: images and text map to the same 1152-d space.
We load only the text encoder locally (~200MB, runs fine on CPU).
This enables semantic search without a RunPod endpoint.

Usage:
    embedder = TextEmbedder()
    vec = embedder.embed("a black and white photo of a waterfall")
    # vec = [float x 1152], L2-normalised
"""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip2-so400m-patch16-384"
TEXT_EMBED_DIM = 1152


class TextEmbedder:
    """Local SigLIP2 text encoder. Lazy-loads on first call."""

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
            ).eval()

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Embed one or more text queries into the SigLIP2 1152-d space.

        Returns L2-normalised vectors compatible with sqlite-vec cosine search.
        """
        self._load()
        assert self._processor is not None
        assert self._model is not None

        single = isinstance(text, str)
        texts = [text] if single else text

        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs).pooler_output

        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        vectors = feats.cpu().to(torch.float32).tolist()

        return vectors[0] if single else vectors
