"""Embedding pipeline with sentence-transformers + API support, batch processing, and caching."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_single(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Local embeddings via sentence-transformers.

    Default model: all-MiniLM-L6-v2 (384 dims, fast, good quality).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None
        self._dimension: int = 384  # default, updated on load

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(self._model_name, device=self._device)
        # Get actual dimension
        test_emb = self._model.encode(["test"])
        self._dimension = test_emb.shape[1]
        logger.info("Loaded embedding model %s (dim=%d)", self._model_name, self._dimension)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        # Process in batches
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = self._model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI API embeddings (text-embedding-3-small, etc.)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 100,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._batch_size = batch_size
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    @property
    def dimension(self) -> int:
        return self._dimensions.get(self._model, 1536)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            import openai
        except ImportError:
            raise ImportError("openai is required: pip install openai")

        client = openai.AsyncOpenAI(api_key=self._api_key)
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = await client.embeddings.create(model=self._model, input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


class CachedEmbedder(EmbeddingProvider):
    """Wraps any EmbeddingProvider with disk-based caching."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_dir: str = "~/.yoda/memory/embed_cache",
        max_cache_size: int = 100_000,
    ) -> None:
        self._provider = provider
        self._cache_dir = Path(cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        self._max_cache_size = max_cache_size
        self._load_cache()

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _cache_file(self) -> Path:
        return self._cache_dir / "embeddings.json"

    def _load_cache(self) -> None:
        cache_file = self._cache_file()
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    self._cache = json.load(f)
                logger.debug("Loaded %d cached embeddings", len(self._cache))
            except Exception:
                logger.warning("Failed to load embedding cache, starting fresh")
                self._cache = {}

    def _save_cache(self) -> None:
        try:
            with open(self._cache_file(), "w") as f:
                json.dump(self._cache, f)
        except Exception:
            logger.warning("Failed to save embedding cache")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Embed uncached
        if uncached_texts:
            new_embeddings = await self._provider.embed(uncached_texts)
            for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                key = self._cache_key(text)
                self._cache[key] = new_embeddings[i]
                results[idx] = new_embeddings[i]

            # Evict if too large
            if len(self._cache) > self._max_cache_size:
                keys = list(self._cache.keys())
                for k in keys[: len(self._cache) - self._max_cache_size]:
                    del self._cache[k]

            self._save_cache()

        return [r for r in results if r is not None]


def create_embedder(
    model: str = "all-MiniLM-L6-v2",
    cache_dir: str = "~/.yoda/memory/embed_cache",
    use_cache: bool = True,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Factory to create an embedding provider.

    If model starts with 'text-embedding-' it uses OpenAI, otherwise sentence-transformers.
    """
    if model.startswith("text-embedding-"):
        provider: EmbeddingProvider = OpenAIEmbedder(model=model, **kwargs)
    else:
        provider = SentenceTransformerEmbedder(model_name=model, **kwargs)

    if use_cache:
        provider = CachedEmbedder(provider, cache_dir=cache_dir)

    return provider
