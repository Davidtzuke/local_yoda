"""Calendar and scheduling tools with SQLite persistence."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from yoda.tools.registry import ToolPermission, tool

# Module-level DB path (configured at startup)
_DB_PATH: Path = Path("~/.yoda/calendar.db").expanduser()


def _get_db() -> sqlite3.Connection:
    """Get or create the calendar database."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            start_time TEXT NOT NULL,
            end_time TEXT,
            location TEXT DEFAULT '',
            tags TEXT DEFAULT '[]',
            recurring TEXT DEFAULT '',
            reminder_minutes INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def register_calendar_tools() -> None:
    """No-op — importing this module registers tools via decorators."""


@tool(
    name="calendar_add",
    permission=ToolPermission.WRITE,
    category="calendar",
    tags=["calendar", "event", "schedule"],
    timeout=10.0,
)
async def calendar_add(
    title: str,
    start_time: str,
    end_time: str = "",
    description: str = "",
    location: str = "",
    tags: str = "",
    reminder_minutes: int = 15,
) -> str:
    """Add a calendar event.

    Args:
        title: Event title.
        start_time: Start time (ISO format: YYYY-MM-DD HH:MM).
        end_time: End time (ISO format, optional).
        description: Event description.
        location: Event location.
        tags: Comma-separated tags.
        reminder_minutes: Reminder before event (minutes).
    """
    # Validate time format
    try:
        datetime.fromisoformat(start_time)
    except ValueError:
        raise ValueError(f"Invalid start_time format: {start_time}. Use YYYY-MM-DD HH:MM")

    tag_list = json.dumps([t.strip() for t in tags.split(",") if t.strip()])
    conn = _get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO events (title, description, start_time, end_time, location, tags, reminder_minutes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, description, start_time, end_time or None, location, tag_list, reminder_minutes),
        )
        conn.commit()
        return f"Event created (id={cursor.lastrowid}): {title} at {start_time}"
    finally:
        conn.close()


@tool(
    name="calendar_list",
    permission=ToolPermission.READ,
    category="calendar",
    tags=["calendar", "events", "list"],
    timeout=10.0,
)
async def calendar_list(
    start_date: str = "",
    end_date: str = "",
    days: int = 7,
) -> str:
    """List upcoming calendar events.

    Args:
        start_date: Start date (ISO format, default: today).
        end_date: End date (ISO format, default: start + days).
        days: Number of days to look ahead (default: 7).
    """
    if not start_date:
        start_dt = datetime.now()
        start_date = start_dt.strftime("%Y-%m-%d")
    else:
        start_dt = datetime.fromisoformat(start_date)

    if not end_date:
        end_dt = start_dt + timedelta(days=days)
        end_date = end_dt.strftime("%Y-%m-%d 23:59:59")

    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT id, title, start_time, end_time, location, description "
            "FROM events WHERE start_time >= ? AND start_time <= ? "
            "ORDER BY start_time",
            (start_date, end_date),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return f"No events from {start_date} to {end_date}"

    lines: list[str] = []
    for row in rows:
        eid, title, start, end, loc, desc = row
        line = f"[{eid}] {start} - {title}"
        if loc:
            line += f" @ {loc}"
        if desc:
            line += f" ({desc[:50]})"
        lines.append(line)

    return "\n".join(lines)


@tool(
    name="calendar_delete",
    permission=ToolPermission.WRITE,
    category="calendar",
    tags=["calendar", "delete"],
    requires_approval=True,
    timeout=10.0,
)
async def calendar_delete(event_id: int) -> str:
    """Delete a calendar event by ID.

    Args:
        event_id: Event ID to delete.
    """
    conn = _get_db()
    try:
        cursor = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        conn.commit()
        if cursor.rowcount == 0:
            return f"No event found with id={event_id}"
        return f"Deleted event {event_id}"
    finally:
        conn.close()


@tool(
    name="calendar_search",
    permission=ToolPermission.READ,
    category="calendar",
    tags=["calendar", "search"],
    timeout=10.0,
)
async def calendar_search(query: str, limit: int = 20) -> str:
    """Search calendar events by title or description.

    Args:
        query: Search query.
        limit: Maximum results.
    """
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT id, title, start_time, location, description "
            "FROM events WHERE title LIKE ? OR description LIKE ? "
            "ORDER BY start_time DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return f"No events matching '{query}'"

    lines: list[str] = []
    for eid, title, start, loc, desc in rows:
        line = f"[{eid}] {start} - {title}"
        if loc:
            line += f" @ {loc}"
        lines.append(line)
    return "\n".join(lines)
