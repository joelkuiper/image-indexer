"""Tests for image_indexer.text_embed."""

from __future__ import annotations

import pytest

from image_indexer.text_embed import TEXT_EMBED_DIM, TextEmbedder


@pytest.fixture(scope="module")
def embedder():
    """Singleton embedder — loading the model is expensive, reuse across tests."""
    return TextEmbedder()


class TestTextEmbedder:
    def test_returns_correct_dim(self, embedder):
        vec = embedder.embed("a photo of a waterfall")
        assert isinstance(vec, list)
        assert len(vec) == TEXT_EMBED_DIM

    def test_returns_floats(self, embedder):
        vec = embedder.embed("sunset over mountains")
        assert all(isinstance(v, float) for v in vec)

    def test_l2_normalised(self, embedder):
        vec = embedder.embed("black and white photograph")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-4

    def test_similar_queries_similar_vectors(self, embedder):
        v1 = embedder.embed("a black and white photo of a waterfall in Iceland")
        v2 = embedder.embed("black and white photograph of Icelandic waterfall")
        v3 = embedder.embed("a recipe for chocolate cake")
        # Cosine similarity (vectors are L2-normed so dot product = cosine)
        sim_related = sum(a * b for a, b in zip(v1, v2))
        sim_unrelated = sum(a * b for a, b in zip(v1, v3))
        assert sim_related > sim_unrelated

    def test_batch_mode(self, embedder):
        vecs = embedder.embed(["waterfall", "mountain", "beach"])
        assert isinstance(vecs, list)
        assert len(vecs) == 3
        assert all(len(v) == TEXT_EMBED_DIM for v in vecs)

    def test_deterministic(self, embedder):
        v1 = embedder.embed("a red door on a grey wall")
        v2 = embedder.embed("a red door on a grey wall")
        assert v1 == v2
