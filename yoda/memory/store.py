"""Concrete async memory store using SQLite + sentence-transformers embeddings."""

from __future__ import annotations

import json
import logging
from datetime import datetime

import aiosqlite

from yoda.config import MemorySettings
from yoda.memory.base import BaseMemoryStore
from yoda.memory.embeddings import EmbeddingPipeline
from yoda.types import MemoryEntry, MemorySearchResult

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT '',
    tags        TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL,
    embedding   BLOB,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
"""


class SQLiteMemoryStore(BaseMemoryStore):
    """Production memory store backed by SQLite with embedding-based search.

    - Stores entries in a local SQLite database via aiosqlite.
    - Generates embeddings with sentence-transformers (all-MiniLM-L6-v2 by default).
    - Performs cosine-similarity search in Python for simplicity and portability.
    """

    def __init__(self, settings: MemorySettings | None = None) -> None:
        self._settings = settings or MemorySettings()
        self._db: aiosqlite.Connection | None = None
        self._embedder = EmbeddingPipeline(self._settings.embedding_model)

    @property
    def embedder(self) -> EmbeddingPipeline:
        """Access the embedding pipeline."""
        return self._embedder

    # -- lifecycle --------------------------------------------------------

    async def initialize(self) -> None:
        """Create the database, tables, and load the embedding model."""
        db_path = self._settings.resolved_db_path
        logger.info("Opening memory database at %s", db_path)
        self._db = await aiosqlite.connect(str(db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        await self._embedder.initialize()
        logger.info("Memory store initialized (dim=%d)", self._embedder.dimension)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # -- helpers ----------------------------------------------------------

    def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            msg = "Memory store not initialized. Call initialize() first."
            raise RuntimeError(msg)
        return self._db

    @staticmethod
    def _row_to_entry(row: aiosqlite.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        embedding_raw = row["embedding"]
        embedding: list[float] | None = None
        if embedding_raw is not None:
            embedding = json.loads(embedding_raw)

        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            source=row["source"],
            tags=json.loads(row["tags"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            embedding=embedding,
            metadata=json.loads(row["metadata"]),
        )

    # -- CRUD -------------------------------------------------------------

    async def store(self, entry: MemoryEntry) -> str:
        """Persist a memory entry, generating an embedding if missing."""
        db = self._ensure_db()

        if entry.embedding is None:
            entry.embedding = self._embedder.embed(entry.content)

        embedding_blob = json.dumps(entry.embedding) if entry.embedding else None
        ts = entry.created_at.isoformat()

        await db.execute(
            """
            INSERT OR REPLACE INTO memories (id, content, source, tags, created_at, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.content,
                entry.source,
                json.dumps(entry.tags),
                ts,
                embedding_blob,
                json.dumps(entry.metadata),
            ),
        )
        await db.commit()
        logger.debug("Stored memory %s (%d chars)", entry.id[:8], len(entry.content))
        return entry.id

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a single memory entry by ID."""
        db = self._ensure_db()
        cursor = await db.execute("SELECT * FROM memories WHERE id = ?", (entry_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry by ID."""
        db = self._ensure_db()
        cursor = await db.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
        await db.commit()
        return cursor.rowcount > 0

    async def list_all(self, *, limit: int = 100, offset: int = 0) -> list[MemoryEntry]:
        """List memory entries ordered by creation time (newest first)."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_entry(r) for r in rows]

    # -- semantic search --------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        tags: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        """Semantic search: embed the query, compute cosine similarity against all entries.

        For a personal assistant with thousands of entries, brute-force
        cosine similarity is fast enough. For larger datasets, swap in an ANN index.
        """
        db = self._ensure_db()
        query_vec = self._embedder.embed(query)

        cursor = await db.execute(
            "SELECT * FROM memories WHERE embedding IS NOT NULL"
        )
        rows = await cursor.fetchall()
        scored: list[MemorySearchResult] = []

        for row in rows:
            entry = self._row_to_entry(row)

            # Tag filter
            if tags and not any(t in entry.tags for t in tags):
                continue

            if entry.embedding is None:
                continue

            score = EmbeddingPipeline.cosine_similarity(query_vec, entry.embedding)
            if score >= min_score:
                scored.append(MemorySearchResult(entry=entry, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    # -- bulk operations --------------------------------------------------

    async def count(self) -> int:
        """Return the total number of memory entries."""
        db = self._ensure_db()
        cursor = await db.execute("SELECT COUNT(*) FROM memories")
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def search_by_source(self, source: str) -> list[MemoryEntry]:
        """Find all memories from a specific source."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM memories WHERE source = ? ORDER BY created_at DESC",
            (source,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_entry(r) for r in rows]
