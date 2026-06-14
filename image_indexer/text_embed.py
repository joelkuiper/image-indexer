"""Local text embedding via OpenAI CLIP.

CLIP (CLIP-ViT-B-32) has excellent direct cosine-similarity alignment between
image and text embeddings, making it perfect for native SQLite-vec searches.
The text encoder is extremely lightweight (~150MB, runs instantly on CPU).

Usage:
    embedder = TextEmbedder()
    vec = embedder.embed("a black and white photo of a waterfall")
    # vec = [float x 512], L2-normalised
"""
from __future__ import annotations

import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
TEXT_EMBED_DIM = 512


class TextEmbedder:
    """Local CLIP text encoder. Lazy-loads on first call."""

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            self._processor = CLIPProcessor.from_pretrained(self.model_id)
            self._model = CLIPModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
            ).eval()

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Embed one or more text queries into the CLIP 512-d space.

        Returns L2-normalised vectors compatible with sqlite-vec cosine search.
        """
        self._load()
        assert self._processor is not None
        assert self._model is not None

        single = isinstance(text, str)
        texts = [text] if single else text

        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            output_obj = self._model.get_text_features(**inputs)
            # In some transformers versions, get_text_features returns the raw Tensor,
            # in others (with newer AutoModel mappings) it returns BaseModelOutputWithPooling.
            # Handle both gracefully.
            if hasattr(output_obj, "pooler_output"):
                feats = output_obj.pooler_output
            else:
                feats = output_obj

        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        vectors = feats.cpu().to(torch.float32).tolist()

        return vectors[0] if single else vectors
