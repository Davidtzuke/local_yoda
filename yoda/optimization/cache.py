"""Semantic response caching with fuzzy matching and TTL."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yoda.optimization.tokens import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached response with metadata."""

    key: str
    query: str
    response: str
    model: str
    tokens_saved: int
    created_at: float
    ttl: float
    hit_count: int = 0
    last_hit: float = 0.0
    metadata: dict[str, Any] | None = None

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class SemanticCache:
    """Response cache with exact and fuzzy matching, TTL, and SQLite persistence.

    Features:
    - Exact key matching (hash of normalized query)
    - Fuzzy matching via trigram similarity
    - TTL-based expiration
    - Hit counting for popularity tracking
    - SQLite persistence for cross-session caching
    - Token savings tracking
    """

    def __init__(
        self,
        persist_path: str | Path = "~/.yoda/cache.db",
        default_ttl: float = 3600.0,  # 1 hour
        max_entries: int = 10_000,
        similarity_threshold: float = 0.85,
        model: str = "default",
    ) -> None:
        self.persist_path = Path(persist_path).expanduser()
        self.default_ttl = default_ttl
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.counter = TokenCounter(model)
        self._db: sqlite3.Connection | None = None
        self._total_hits = 0
        self._total_misses = 0
        self._tokens_saved = 0

    # -- Lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        """Create database and tables."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.persist_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_saved INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                ttl REAL NOT NULL,
                hit_count INTEGER DEFAULT 0,
                last_hit REAL DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_created ON cache(created_at)
        """)
        self._db.commit()
        self._cleanup_expired()

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None

    # -- Key generation ----------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize query text for consistent hashing."""
        return " ".join(text.lower().strip().split())

    @staticmethod
    def _make_key(query: str, model: str = "") -> str:
        """Generate cache key from normalized query."""
        normalized = SemanticCache._normalize(query)
        raw = f"{model}:{normalized}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    # -- Trigram similarity ------------------------------------------------

    @staticmethod
    def _trigrams(text: str) -> set[str]:
        """Generate character trigrams from text."""
        text = f"  {text.lower().strip()}  "
        return {text[i : i + 3] for i in range(len(text) - 2)}

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Jaccard similarity of trigram sets."""
        ta, tb = SemanticCache._trigrams(a), SemanticCache._trigrams(b)
        if not ta or not tb:
            return 0.0
        intersection = ta & tb
        union = ta | tb
        return len(intersection) / len(union)

    # -- Get / Put ---------------------------------------------------------

    def get(self, query: str, model: str = "") -> CacheEntry | None:
        """Look up a cached response. Tries exact match first, then fuzzy."""
        if not self._db:
            return None

        # Exact match
        key = self._make_key(query, model)
        entry = self._get_by_key(key)
        if entry and not entry.is_expired:
            self._record_hit(entry)
            return entry

        # Fuzzy match
        entry = self._fuzzy_match(query, model)
        if entry:
            self._record_hit(entry)
            return entry

        self._total_misses += 1
        return None

    def put(
        self,
        query: str,
        response: str,
        model: str = "",
        ttl: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        """Cache a response."""
        if not self._db:
            self.initialize()
            assert self._db is not None

        key = self._make_key(query, model)
        tokens = self.counter.count(response)
        now = time.time()
        entry_ttl = ttl or self.default_ttl

        self._db.execute(
            """INSERT OR REPLACE INTO cache
               (key, query, response, model, tokens_saved, created_at, ttl, hit_count, last_hit, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?)""",
            (key, query, response, model, tokens, now, entry_ttl, json.dumps(metadata or {})),
        )
        self._db.commit()
        self._enforce_max_entries()

        return CacheEntry(
            key=key,
            query=query,
            response=response,
            model=model,
            tokens_saved=tokens,
            created_at=now,
            ttl=entry_ttl,
            metadata=metadata,
        )

    def invalidate(self, query: str, model: str = "") -> bool:
        """Remove a specific cache entry."""
        if not self._db:
            return False
        key = self._make_key(query, model)
        cursor = self._db.execute("DELETE FROM cache WHERE key = ?", (key,))
        self._db.commit()
        return cursor.rowcount > 0

    def clear(self) -> None:
        """Clear all cache entries."""
        if self._db:
            self._db.execute("DELETE FROM cache")
            self._db.commit()

    # -- Internal ----------------------------------------------------------

    def _get_by_key(self, key: str) -> CacheEntry | None:
        if not self._db:
            return None
        row = self._db.execute(
            "SELECT key, query, response, model, tokens_saved, created_at, ttl, hit_count, last_hit, metadata FROM cache WHERE key = ?",
            (key,),
        ).fetchone()
        if not row:
            return None
        return CacheEntry(
            key=row[0],
            query=row[1],
            response=row[2],
            model=row[3],
            tokens_saved=row[4],
            created_at=row[5],
            ttl=row[6],
            hit_count=row[7],
            last_hit=row[8],
            metadata=json.loads(row[9]) if row[9] else None,
        )

    def _fuzzy_match(self, query: str, model: str = "") -> CacheEntry | None:
        """Find best fuzzy match above similarity threshold."""
        if not self._db:
            return None

        now = time.time()
        rows = self._db.execute(
            "SELECT key, query, response, model, tokens_saved, created_at, ttl, hit_count, last_hit, metadata FROM cache WHERE created_at + ttl > ?",
            (now,),
        ).fetchall()

        best_score = 0.0
        best_entry: CacheEntry | None = None

        for row in rows:
            score = self._similarity(query, row[1])
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_entry = CacheEntry(
                    key=row[0],
                    query=row[1],
                    response=row[2],
                    model=row[3],
                    tokens_saved=row[4],
                    created_at=row[5],
                    ttl=row[6],
                    hit_count=row[7],
                    last_hit=row[8],
                    metadata=json.loads(row[9]) if row[9] else None,
                )

        return best_entry

    def _record_hit(self, entry: CacheEntry) -> None:
        """Update hit count and stats."""
        if not self._db:
            return
        now = time.time()
        self._db.execute(
            "UPDATE cache SET hit_count = hit_count + 1, last_hit = ? WHERE key = ?",
            (now, entry.key),
        )
        self._db.commit()
        self._total_hits += 1
        self._tokens_saved += entry.tokens_saved

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        if not self._db:
            return
        now = time.time()
        self._db.execute("DELETE FROM cache WHERE created_at + ttl < ?", (now,))
        self._db.commit()

    def _enforce_max_entries(self) -> None:
        """Remove oldest entries if over max."""
        if not self._db:
            return
        count = self._db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        if count > self.max_entries:
            excess = count - self.max_entries
            self._db.execute(
                "DELETE FROM cache WHERE key IN (SELECT key FROM cache ORDER BY last_hit ASC, created_at ASC LIMIT ?)",
                (excess,),
            )
            self._db.commit()

    # -- Stats -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        entry_count = 0
        if self._db:
            entry_count = self._db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        total_requests = self._total_hits + self._total_misses
        hit_rate = self._total_hits / total_requests if total_requests > 0 else 0.0
        return {
            "entries": entry_count,
            "max_entries": self.max_entries,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": f"{hit_rate:.1%}",
            "tokens_saved": self._tokens_saved,
        }
