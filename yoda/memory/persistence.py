"""SQLite metadata store + export/import/backup for memory persistence.

Stores document metadata, memory statistics, and user preferences in SQLite.
The actual embeddings live in the vector store; this is the metadata sidecar.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

from yoda.memory.vector_store import Document

logger = logging.getLogger(__name__)


class MemoryMetadataStore:
    """SQLite-backed metadata store for memory documents."""

    def __init__(self, db_path: str = "~/.yoda/memory/metadata.db") -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        """Create tables and indices."""
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                collection TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL,
                importance REAL DEFAULT 0.5,
                decay_rate REAL DEFAULT 0.01,
                source TEXT DEFAULT 'conversation',
                metadata_json TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_memories_collection ON memories(collection);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);

            CREATE TABLE IF NOT EXISTS memory_stats (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_relations (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at REAL NOT NULL,
                PRIMARY KEY (source_id, target_id, relation_type),
                FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS consolidation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                source_ids TEXT NOT NULL,
                result_id TEXT,
                timestamp REAL NOT NULL,
                details TEXT
            );
        """)
        self._conn.commit()
        logger.info("Memory metadata store initialized at %s", self._db_path)

    async def store(self, doc: Document) -> None:
        """Store or update document metadata."""
        if self._conn is None:
            raise RuntimeError("Store not initialized")

        now = time.time()
        self._conn.execute(
            """INSERT OR REPLACE INTO memories
            (id, collection, content, created_at, updated_at, importance, source, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc.id,
                doc.collection,
                doc.content,
                doc.metadata.get("created_at", now),
                now,
                doc.metadata.get("importance", 0.5),
                doc.metadata.get("source", "conversation"),
                json.dumps(doc.metadata),
            ),
        )
        self._conn.commit()

    async def store_batch(self, docs: list[Document]) -> None:
        """Store multiple documents efficiently."""
        if self._conn is None:
            raise RuntimeError("Store not initialized")

        now = time.time()
        rows = [
            (
                doc.id,
                doc.collection,
                doc.content,
                doc.metadata.get("created_at", now),
                now,
                doc.metadata.get("importance", 0.5),
                doc.metadata.get("source", "conversation"),
                json.dumps(doc.metadata),
            )
            for doc in docs
        ]
        self._conn.executemany(
            """INSERT OR REPLACE INTO memories
            (id, collection, content, created_at, updated_at, importance, source, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    async def get(self, doc_id: str) -> dict[str, Any] | None:
        """Get document metadata by ID."""
        if self._conn is None:
            return None

        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (doc_id,)
        ).fetchone()

        if row is None:
            return None

        return dict(row)

    async def record_access(self, doc_id: str) -> None:
        """Record that a document was accessed (for forgetting curve)."""
        if self._conn is None:
            return

        now = time.time()
        self._conn.execute(
            """UPDATE memories SET
            access_count = access_count + 1,
            last_accessed = ?
            WHERE id = ?""",
            (now, doc_id),
        )
        self._conn.commit()

    async def get_decayed_memories(
        self,
        collection: str | None = None,
        threshold: float = 0.1,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get memories that have decayed below the importance threshold.

        Uses a forgetting curve: effective_importance = importance * exp(-decay_rate * days_since_access)
        """
        if self._conn is None:
            return []

        now = time.time()
        query = """
            SELECT *,
                importance * EXP(-decay_rate * (? - COALESCE(last_accessed, created_at)) / 86400.0)
                AS effective_importance
            FROM memories
            WHERE 1=1
        """
        params: list[Any] = [now]

        if collection:
            query += " AND collection = ?"
            params.append(collection)

        query += " HAVING effective_importance < ? ORDER BY effective_importance ASC LIMIT ?"
        params.extend([threshold, limit])

        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    async def update_importance(self, doc_id: str, importance: float) -> None:
        """Update a memory's importance score."""
        if self._conn is None:
            return

        self._conn.execute(
            "UPDATE memories SET importance = ?, updated_at = ? WHERE id = ?",
            (importance, time.time(), doc_id),
        )
        self._conn.commit()

    async def delete(self, doc_ids: list[str]) -> int:
        """Delete memories by IDs."""
        if self._conn is None:
            return 0

        placeholders = ",".join("?" * len(doc_ids))
        cursor = self._conn.execute(
            f"DELETE FROM memories WHERE id IN ({placeholders})", doc_ids
        )
        self._conn.commit()
        return cursor.rowcount

    async def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        if self._conn is None:
            return {}

        stats: dict[str, Any] = {}

        # Collection counts
        rows = self._conn.execute(
            "SELECT collection, COUNT(*) as count FROM memories GROUP BY collection"
        ).fetchall()
        stats["collections"] = {row["collection"]: row["count"] for row in rows}

        # Total count
        row = self._conn.execute("SELECT COUNT(*) as total FROM memories").fetchone()
        stats["total_memories"] = row["total"] if row else 0

        # Average importance
        row = self._conn.execute("SELECT AVG(importance) as avg_imp FROM memories").fetchone()
        stats["avg_importance"] = round(row["avg_imp"], 3) if row and row["avg_imp"] else 0

        return stats

    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> None:
        """Add a relation between two memories."""
        if self._conn is None:
            return

        self._conn.execute(
            """INSERT OR REPLACE INTO memory_relations
            (source_id, target_id, relation_type, weight, created_at)
            VALUES (?, ?, ?, ?, ?)""",
            (source_id, target_id, relation_type, weight, time.time()),
        )
        self._conn.commit()

    async def get_related(self, doc_id: str, relation_type: str | None = None) -> list[dict[str, Any]]:
        """Get memories related to a given memory."""
        if self._conn is None:
            return []

        if relation_type:
            rows = self._conn.execute(
                """SELECT m.*, r.relation_type, r.weight
                FROM memory_relations r JOIN memories m ON r.target_id = m.id
                WHERE r.source_id = ? AND r.relation_type = ?""",
                (doc_id, relation_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT m.*, r.relation_type, r.weight
                FROM memory_relations r JOIN memories m ON r.target_id = m.id
                WHERE r.source_id = ?""",
                (doc_id,),
            ).fetchall()

        return [dict(r) for r in rows]

    async def log_consolidation(
        self,
        action: str,
        source_ids: list[str],
        result_id: str | None = None,
        details: str = "",
    ) -> None:
        """Log a memory consolidation event."""
        if self._conn is None:
            return

        self._conn.execute(
            """INSERT INTO consolidation_log (action, source_ids, result_id, timestamp, details)
            VALUES (?, ?, ?, ?, ?)""",
            (action, json.dumps(source_ids), result_id, time.time(), details),
        )
        self._conn.commit()

    # -- Export / Import ---------------------------------------------------

    async def export_all(self, output_path: str) -> int:
        """Export all memories to a JSON file."""
        if self._conn is None:
            return 0

        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        data = [dict(r) for r in rows]

        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump({"memories": data, "exported_at": time.time()}, f, indent=2)

        logger.info("Exported %d memories to %s", len(data), output)
        return len(data)

    async def import_from(self, input_path: str) -> int:
        """Import memories from a JSON file."""
        if self._conn is None:
            return 0

        with open(Path(input_path).expanduser()) as f:
            data = json.load(f)

        memories = data.get("memories", [])
        now = time.time()
        rows = [
            (
                m["id"],
                m["collection"],
                m["content"],
                m.get("created_at", now),
                now,
                m.get("importance", 0.5),
                m.get("source", "imported"),
                m.get("metadata_json", "{}"),
            )
            for m in memories
        ]
        self._conn.executemany(
            """INSERT OR REPLACE INTO memories
            (id, collection, content, created_at, updated_at, importance, source, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()
        logger.info("Imported %d memories from %s", len(rows), input_path)
        return len(rows)

    async def backup(self, backup_dir: str = "~/.yoda/backups") -> str:
        """Create a backup of the metadata database."""
        backup_path = Path(backup_dir).expanduser()
        backup_path.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        dest = backup_path / f"metadata_{timestamp}.db"
        shutil.copy2(str(self._db_path), str(dest))
        logger.info("Database backed up to %s", dest)
        return str(dest)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
