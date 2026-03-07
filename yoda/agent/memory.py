"""Persistent conversation memory backed by JSON files."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class MemoryMessage:
    """A single message in conversation history."""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)

    @property
    def timestamp_iso(self) -> str:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()


class ConversationMemory:
    """Persistent conversation memory stored as JSON.

    Each conversation is keyed by an ID. A default conversation ("default")
    is used for the CLI.
    """

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._dir = storage_dir or Path.home() / ".yoda" / "memory"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[MemoryMessage]] = {}

    def _file(self, conversation_id: str) -> Path:
        safe = conversation_id.replace("/", "_").replace("..", "_")
        return self._dir / f"{safe}.json"

    def _load(self, conversation_id: str) -> list[MemoryMessage]:
        if conversation_id in self._cache:
            return self._cache[conversation_id]
        path = self._file(conversation_id)
        if path.exists():
            data = json.loads(path.read_text())
            msgs = [MemoryMessage(**m) for m in data]
        else:
            msgs = []
        self._cache[conversation_id] = msgs
        return msgs

    def _save(self, conversation_id: str) -> None:
        msgs = self._cache.get(conversation_id, [])
        path = self._file(conversation_id)
        path.write_text(json.dumps([asdict(m) for m in msgs], indent=2))

    def add(self, conversation_id: str, role: str, content: str) -> None:
        msgs = self._load(conversation_id)
        msgs.append(MemoryMessage(role=role, content=content))
        self._save(conversation_id)

    def get(self, conversation_id: str, limit: int = 50) -> list[MemoryMessage]:
        msgs = self._load(conversation_id)
        return msgs[-limit:]

    def clear(self, conversation_id: str) -> None:
        self._cache[conversation_id] = []
        path = self._file(conversation_id)
        if path.exists():
            path.unlink()

    def get_context_messages(
        self, conversation_id: str, limit: int = 20
    ) -> list[dict[str, str]]:
        """Return messages formatted for LLM context."""
        msgs = self.get(conversation_id, limit)
        return [{"role": m.role, "content": m.content} for m in msgs]
