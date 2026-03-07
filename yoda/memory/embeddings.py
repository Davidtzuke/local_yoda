"""Embedding pipeline for semantic memory search.

Uses sentence-transformers for local embedding generation.
Falls back to a simple hash-based approach if sentence-transformers is unavailable.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Module-level cache for the model (heavy to load)
_model_cache: dict[str, object] = {}


class EmbeddingPipeline:
    """Generates and compares text embeddings using a local model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: object | None = None
        self._dimension: int | None = None
        self._cache: dict[str, list[float]] = {}

    async def initialize(self) -> None:
        """Load the embedding model (model is cached across instances)."""
        if self._model_name in _model_cache:
            self._model = _model_cache[self._model_name]
            logger.info("Reusing cached embedding model: %s", self._model_name)
        else:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
                _model_cache[self._model_name] = self._model
                logger.info("Loaded embedding model: %s", self._model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed; "
                    "falling back to hash-based pseudo-embeddings"
                )
                self._model = None

        # Determine dimension
        if self._model is not None:
            dim = getattr(self._model, "get_sentence_embedding_dimension", None)
            self._dimension = dim() if callable(dim) else 384
        else:
            self._dimension = 384

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        return self._dimension or 384

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        if not text.strip():
            return [0.0] * self.dimension

        cache_key = hashlib.md5(text.encode()).hexdigest()  # noqa: S324
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._model is not None:
            vec: NDArray[np.float32] = self._model.encode(  # type: ignore[union-attr]
                text, normalize_embeddings=True, show_progress_bar=False
            )
            result = vec.tolist()
        else:
            result = self._hash_embed(text)

        self._cache[cache_key] = result
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts at once (more efficient with the real model)."""
        if not texts:
            return []

        if self._model is not None:
            vecs: NDArray[np.float32] = self._model.encode(  # type: ignore[union-attr]
                texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
            )
            return vecs.tolist()

        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Deterministic pseudo-embedding fallback using hashing.

        Not semantically meaningful, but allows the system to function
        without sentence-transformers installed.
        """
        dim = self.dimension
        h = hashlib.sha256(text.lower().encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        vec = rng.standard_normal(dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        dot = float(np.dot(va, vb))
        norm_a = float(np.linalg.norm(va))
        norm_b = float(np.linalg.norm(vb))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
