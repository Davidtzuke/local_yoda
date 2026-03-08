"""Notes and quick-capture tools with SQLite persistence."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from yoda.tools.registry import ToolPermission, tool

_DB_PATH: Path = Path("~/.yoda/notes.db").expanduser()


def _get_db() -> sqlite3.Connection:
    """Get or create the notes database."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT DEFAULT '[]',
            pinned INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def register_notes_tools() -> None:
    """No-op — importing this module registers tools via decorators."""


@tool(
    name="note_create",
    permission=ToolPermission.WRITE,
    category="notes",
    tags=["notes", "create", "write"],
    timeout=10.0,
)
async def note_create(title: str, content: str, tags: str = "", pinned: bool = False) -> str:
    """Create a new note.

    Args:
        title: Note title.
        content: Note content (markdown supported).
        tags: Comma-separated tags.
        pinned: Whether to pin the note.
    """
    tag_list = json.dumps([t.strip() for t in tags.split(",") if t.strip()])
    conn = _get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO notes (title, content, tags, pinned) VALUES (?, ?, ?, ?)",
            (title, content, tag_list, int(pinned)),
        )
        conn.commit()
        return f"Note created (id={cursor.lastrowid}): {title}"
    finally:
        conn.close()


@tool(
    name="note_read",
    permission=ToolPermission.READ,
    category="notes",
    tags=["notes", "read"],
    timeout=10.0,
)
async def note_read(note_id: int) -> str:
    """Read a note by ID.

    Args:
        note_id: Note ID.
    """
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT title, content, tags, pinned, created_at, updated_at "
            "FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return f"No note found with id={note_id}"

    title, content, tags, pinned, created, updated = row
    header = f"# {title}\n"
    if pinned:
        header += "[PINNED] "
    header += f"Tags: {tags}\nCreated: {created} | Updated: {updated}\n"
    return f"{header}\n{content}"


@tool(
    name="note_list",
    permission=ToolPermission.READ,
    category="notes",
    tags=["notes", "list"],
    timeout=10.0,
)
async def note_list(tag: str = "", pinned_only: bool = False, limit: int = 50) -> str:
    """List notes with optional filtering.

    Args:
        tag: Filter by tag.
        pinned_only: Only show pinned notes.
        limit: Maximum results.
    """
    conn = _get_db()
    try:
        query = "SELECT id, title, tags, pinned, created_at FROM notes WHERE 1=1"
        params: list[str | int] = []

        if tag:
            query += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')
        if pinned_only:
            query += " AND pinned = 1"

        query += " ORDER BY pinned DESC, updated_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    if not rows:
        return "No notes found"

    lines: list[str] = []
    for nid, title, tags, pinned, created in rows:
        pin = "[*] " if pinned else "    "
        lines.append(f"{pin}[{nid}] {title} ({created})")
    return "\n".join(lines)


@tool(
    name="note_search",
    permission=ToolPermission.READ,
    category="notes",
    tags=["notes", "search"],
    timeout=10.0,
)
async def note_search(query: str, limit: int = 20) -> str:
    """Search notes by title or content.

    Args:
        query: Search query.
        limit: Maximum results.
    """
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT id, title, substr(content, 1, 100), created_at "
            "FROM notes WHERE title LIKE ? OR content LIKE ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return f"No notes matching '{query}'"

    lines: list[str] = []
    for nid, title, snippet, created in rows:
        lines.append(f"[{nid}] {title}: {snippet}...")
    return "\n".join(lines)


@tool(
    name="note_update",
    permission=ToolPermission.WRITE,
    category="notes",
    tags=["notes", "update", "edit"],
    timeout=10.0,
)
async def note_update(note_id: int, title: str = "", content: str = "", tags: str = "") -> str:
    """Update an existing note.

    Args:
        note_id: Note ID to update.
        title: New title (empty = keep current).
        content: New content (empty = keep current).
        tags: New tags comma-separated (empty = keep current).
    """
    conn = _get_db()
    try:
        updates: list[str] = []
        params: list[str | int] = []

        if title:
            updates.append("title = ?")
            params.append(title)
        if content:
            updates.append("content = ?")
            params.append(content)
        if tags:
            tag_list = json.dumps([t.strip() for t in tags.split(",") if t.strip()])
            updates.append("tags = ?")
            params.append(tag_list)

        if not updates:
            return "No fields to update"

        updates.append("updated_at = datetime('now')")
        params.append(note_id)

        cursor = conn.execute(
            f"UPDATE notes SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()

        if cursor.rowcount == 0:
            return f"No note found with id={note_id}"
        return f"Updated note {note_id}"
    finally:
        conn.close()


@tool(
    name="note_delete",
    permission=ToolPermission.WRITE,
    category="notes",
    tags=["notes", "delete"],
    requires_approval=True,
    timeout=10.0,
)
async def note_delete(note_id: int) -> str:
    """Delete a note by ID.

    Args:
        note_id: Note ID to delete.
    """
    conn = _get_db()
    try:
        cursor = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        conn.commit()
        if cursor.rowcount == 0:
            return f"No note found with id={note_id}"
        return f"Deleted note {note_id}"
    finally:
        conn.close()
