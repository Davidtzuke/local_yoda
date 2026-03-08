"""Vector store abstraction with ChromaDB and FAISS backends.

Supports multiple named collections for different memory types:
- episodic: conversation memories, events, experiences
- semantic: facts, knowledge, learned information
- procedural: how-to knowledge, workflows, patterns
- preferences: user preferences, settings, habits
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)

# Memory collection types
COLLECTION_TYPES = ("episodic", "semantic", "procedural", "preferences")


@dataclass
class Document:
    """A document stored in the vector store."""

    id: str = field(default_factory=lambda: uuid4().hex)
    content: str = ""
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    collection: str = "semantic"
    score: float = 0.0  # similarity score from retrieval


@dataclass
class SearchResult:
    """Result from a vector search."""

    documents: list[Document]
    query: str = ""
    total_found: int = 0


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def add(self, documents: list[Document], collection: str = "semantic") -> list[str]: ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        collection: str = "semantic",
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]: ...

    @abstractmethod
    async def delete(self, ids: list[str], collection: str = "semantic") -> int: ...

    @abstractmethod
    async def get(self, ids: list[str], collection: str = "semantic") -> list[Document]: ...

    @abstractmethod
    async def count(self, collection: str = "semantic") -> int: ...

    @abstractmethod
    async def list_collections(self) -> list[str]: ...

    @abstractmethod
    async def clear(self, collection: str | None = None) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store with persistent storage."""

    def __init__(self, persist_dir: str = "~/.yoda/memory/chroma") -> None:
        self._persist_dir = Path(persist_dir).expanduser()
        self._client: Any = None
        self._collections: dict[str, Any] = {}

    async def initialize(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required: pip install chromadb"
            )

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Create default collections
        for ctype in COLLECTION_TYPES:
            self._collections[ctype] = self._client.get_or_create_collection(
                name=f"yoda_{ctype}",
                metadata={"hnsw:space": "cosine"},
            )
        logger.info("ChromaDB initialized at %s with %d collections", self._persist_dir, len(self._collections))

    def _get_collection(self, name: str) -> Any:
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=f"yoda_{name}",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    async def add(self, documents: list[Document], collection: str = "semantic") -> list[str]:
        coll = self._get_collection(collection)
        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents if doc.embedding]
        contents = [doc.content for doc in documents]
        metadatas = [self._sanitize_metadata(doc.metadata) for doc in documents]

        kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": contents,
            "metadatas": metadatas,
        }
        if embeddings and len(embeddings) == len(documents):
            kwargs["embeddings"] = embeddings

        coll.add(**kwargs)
        return ids

    async def search(
        self,
        query_embedding: list[float],
        collection: str = "semantic",
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        coll = self._get_collection(collection)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, coll.count() or top_k),
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        if coll.count() == 0:
            return []

        results = coll.query(**kwargs)

        docs = []
        for i, doc_id in enumerate(results["ids"][0]):
            doc = Document(
                id=doc_id,
                content=results["documents"][0][i] if results["documents"] else "",
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                collection=collection,
                score=1.0 - (results["distances"][0][i] if results["distances"] else 0.0),
            )
            docs.append(doc)
        return docs

    async def delete(self, ids: list[str], collection: str = "semantic") -> int:
        coll = self._get_collection(collection)
        coll.delete(ids=ids)
        return len(ids)

    async def get(self, ids: list[str], collection: str = "semantic") -> list[Document]:
        coll = self._get_collection(collection)
        results = coll.get(ids=ids)
        docs = []
        for i, doc_id in enumerate(results["ids"]):
            docs.append(Document(
                id=doc_id,
                content=results["documents"][i] if results["documents"] else "",
                metadata=results["metadatas"][i] if results["metadatas"] else {},
                collection=collection,
            ))
        return docs

    async def count(self, collection: str = "semantic") -> int:
        return self._get_collection(collection).count()

    async def list_collections(self) -> list[str]:
        return list(self._collections.keys())

    async def clear(self, collection: str | None = None) -> None:
        if collection:
            coll = self._get_collection(collection)
            # ChromaDB doesn't have a clear method, delete and recreate
            self._client.delete_collection(f"yoda_{collection}")
            self._collections[collection] = self._client.get_or_create_collection(
                name=f"yoda_{collection}",
                metadata={"hnsw:space": "cosine"},
            )
        else:
            for name in list(self._collections.keys()):
                await self.clear(name)

    async def close(self) -> None:
        self._client = None
        self._collections.clear()

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """ChromaDB only supports str, int, float, bool metadata values."""
        clean: dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, list):
                clean[k] = json.dumps(v)
            elif v is None:
                clean[k] = ""
            else:
                clean[k] = str(v)
        return clean


class FAISSVectorStore(VectorStore):
    """FAISS-backed vector store for high-performance local search.

    Stores embeddings in FAISS indices and metadata in JSON sidecar files.
    """

    def __init__(self, persist_dir: str = "~/.yoda/memory/faiss", dimension: int = 384) -> None:
        self._persist_dir = Path(persist_dir).expanduser()
        self._dimension = dimension
        self._indices: dict[str, Any] = {}
        self._doc_stores: dict[str, dict[str, Document]] = {}
        self._id_maps: dict[str, list[str]] = {}  # ordered list of IDs per collection

    async def initialize(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")

        self._persist_dir.mkdir(parents=True, exist_ok=True)

        for ctype in COLLECTION_TYPES:
            self._init_collection(ctype)
            # Load existing data if available
            self._load_collection(ctype)

        logger.info("FAISS initialized at %s", self._persist_dir)

    def _init_collection(self, name: str) -> None:
        import faiss

        if name not in self._indices:
            index = faiss.IndexFlatIP(self._dimension)  # Inner product (cosine after normalization)
            self._indices[name] = index
            self._doc_stores[name] = {}
            self._id_maps[name] = []

    def _get_collection_path(self, name: str) -> Path:
        return self._persist_dir / name

    def _save_collection(self, name: str) -> None:
        import faiss

        coll_dir = self._get_collection_path(name)
        coll_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._indices[name], str(coll_dir / "index.faiss"))

        # Save document metadata
        docs_data = {}
        for doc_id, doc in self._doc_stores[name].items():
            docs_data[doc_id] = {
                "content": doc.content,
                "metadata": doc.metadata,
                "collection": doc.collection,
            }
        with open(coll_dir / "docs.json", "w") as f:
            json.dump(docs_data, f)

        # Save ID map
        with open(coll_dir / "id_map.json", "w") as f:
            json.dump(self._id_maps[name], f)

    def _load_collection(self, name: str) -> None:
        import faiss

        coll_dir = self._get_collection_path(name)
        index_path = coll_dir / "index.faiss"
        docs_path = coll_dir / "docs.json"
        id_map_path = coll_dir / "id_map.json"

        if not index_path.exists():
            return

        try:
            self._indices[name] = faiss.read_index(str(index_path))

            if docs_path.exists():
                with open(docs_path) as f:
                    docs_data = json.load(f)
                for doc_id, data in docs_data.items():
                    self._doc_stores[name][doc_id] = Document(
                        id=doc_id,
                        content=data["content"],
                        metadata=data.get("metadata", {}),
                        collection=data.get("collection", name),
                    )

            if id_map_path.exists():
                with open(id_map_path) as f:
                    self._id_maps[name] = json.load(f)

            logger.debug("Loaded FAISS collection %s with %d docs", name, len(self._doc_stores[name]))
        except Exception:
            logger.exception("Failed to load FAISS collection %s", name)

    async def add(self, documents: list[Document], collection: str = "semantic") -> list[str]:
        self._init_collection(collection)

        ids = []
        embeddings = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} has no embedding — FAISS requires pre-computed embeddings")
            self._doc_stores[collection][doc.id] = doc
            self._id_maps[collection].append(doc.id)
            ids.append(doc.id)
            embeddings.append(doc.embedding)

        # Normalize and add to FAISS
        vectors = np.array(embeddings, dtype=np.float32)
        faiss_module = __import__("faiss")
        faiss_module.normalize_L2(vectors)
        self._indices[collection].add(vectors)

        self._save_collection(collection)
        return ids

    async def search(
        self,
        query_embedding: list[float],
        collection: str = "semantic",
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        self._init_collection(collection)

        if not self._id_maps[collection]:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        faiss_module = __import__("faiss")
        faiss_module.normalize_L2(query)

        k = min(top_k, len(self._id_maps[collection]))
        scores, indices = self._indices[collection].search(query, k)

        docs = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._id_maps[collection]):
                continue
            doc_id = self._id_maps[collection][idx]
            doc = self._doc_stores[collection].get(doc_id)
            if doc is None:
                continue

            # Apply metadata filter
            if filter_metadata:
                if not all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue

            result_doc = Document(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                collection=collection,
                score=float(scores[0][i]),
            )
            docs.append(result_doc)
        return docs

    async def delete(self, ids: list[str], collection: str = "semantic") -> int:
        # FAISS doesn't support deletion natively, rebuild the index
        self._init_collection(collection)
        deleted = 0
        for doc_id in ids:
            if doc_id in self._doc_stores[collection]:
                del self._doc_stores[collection][doc_id]
                deleted += 1

        # Rebuild
        if deleted > 0:
            self._rebuild_index(collection)
        return deleted

    def _rebuild_index(self, collection: str) -> None:
        import faiss

        self._indices[collection] = faiss.IndexFlatIP(self._dimension)
        new_id_map = []

        for doc_id, doc in self._doc_stores[collection].items():
            if doc.embedding:
                vec = np.array([doc.embedding], dtype=np.float32)
                faiss.normalize_L2(vec)
                self._indices[collection].add(vec)
                new_id_map.append(doc_id)

        self._id_maps[collection] = new_id_map
        self._save_collection(collection)

    async def get(self, ids: list[str], collection: str = "semantic") -> list[Document]:
        self._init_collection(collection)
        return [
            self._doc_stores[collection][doc_id]
            for doc_id in ids
            if doc_id in self._doc_stores[collection]
        ]

    async def count(self, collection: str = "semantic") -> int:
        self._init_collection(collection)
        return len(self._doc_stores.get(collection, {}))

    async def list_collections(self) -> list[str]:
        return list(self._indices.keys())

    async def clear(self, collection: str | None = None) -> None:
        import faiss

        if collection:
            self._indices[collection] = faiss.IndexFlatIP(self._dimension)
            self._doc_stores[collection] = {}
            self._id_maps[collection] = []
            self._save_collection(collection)
        else:
            for name in list(self._indices.keys()):
                await self.clear(name)

    async def close(self) -> None:
        # Save all before closing
        for name in self._indices:
            self._save_collection(name)
        self._indices.clear()
        self._doc_stores.clear()
        self._id_maps.clear()


def create_vector_store(backend: str = "chromadb", persist_dir: str = "~/.yoda/memory", **kwargs: Any) -> VectorStore:
    """Factory to create a vector store backend."""
    if backend == "chromadb":
        return ChromaVectorStore(persist_dir=os.path.join(persist_dir, "chroma"))
    elif backend == "faiss":
        return FAISSVectorStore(persist_dir=os.path.join(persist_dir, "faiss"), **kwargs)
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")
