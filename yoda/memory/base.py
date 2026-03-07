"""Abstract base class for the memory storage backend."""

from __future__ import annotations

from abc import ABC, abstractmethod

from yoda.types import MemoryEntry, MemorySearchResult


class BaseMemoryStore(ABC):
    """Async interface for storing and retrieving memory entries.

    Downstream agents should subclass this to provide concrete
    implementations (e.g. SQLite + embeddings).
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Set up the storage backend (create tables, load models, etc.)."""

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """Persist a memory entry. Returns the entry ID."""

    @abstractmethod
    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a single memory entry by ID, or None if not found."""

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        tags: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        """Semantic search over stored memories.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results to return.
            min_score: Minimum similarity score (0.0–1.0) to include.
            tags: Optional tag filter — only return entries matching at least one tag.

        Returns:
            List of MemorySearchResult sorted by descending relevance.
        """

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry by ID. Returns True if it existed."""

    @abstractmethod
    async def list_all(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """List memory entries with pagination."""

    async def close(self) -> None:
        """Clean up resources (close DB connections, etc.). Override if needed."""
