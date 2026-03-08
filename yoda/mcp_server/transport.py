"""SSE transport utilities for MCP server."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SSETransport:
    """Server-Sent Events transport for MCP protocol.

    Provides helper methods for formatting SSE messages and managing
    client connections.
    """

    @staticmethod
    def format_event(event: str, data: Any, event_id: str | None = None) -> bytes:
        """Format a Server-Sent Event message."""
        parts = []
        if event_id:
            parts.append(f"id: {event_id}")
        parts.append(f"event: {event}")
        if isinstance(data, (dict, list)):
            data = json.dumps(data)
        parts.append(f"data: {data}")
        parts.append("")
        parts.append("")
        return "\n".join(parts).encode("utf-8")

    @staticmethod
    def format_jsonrpc_event(response: dict[str, Any], event_id: str | None = None) -> bytes:
        """Format a JSON-RPC response as an SSE message event."""
        return SSETransport.format_event("message", response, event_id)

    @staticmethod
    def format_endpoint_event(endpoint: str) -> bytes:
        """Format the initial endpoint event sent to SSE clients."""
        return SSETransport.format_event("endpoint", endpoint)


class ConnectionManager:
    """Manages SSE client connections with heartbeat."""

    def __init__(self, heartbeat_interval: float = 30.0) -> None:
        self._connections: dict[str, asyncio.Queue[bytes]] = {}
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_task: asyncio.Task[Any] | None = None

    def add(self, client_id: str) -> asyncio.Queue[bytes]:
        """Register a new client connection."""
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._connections[client_id] = queue
        if not self._heartbeat_task and self._connections:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        return queue

    def remove(self, client_id: str) -> None:
        """Remove a client connection."""
        self._connections.pop(client_id, None)
        if not self._connections and self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

    async def send(self, client_id: str, data: bytes) -> bool:
        """Send data to a specific client."""
        queue = self._connections.get(client_id)
        if queue:
            await queue.put(data)
            return True
        return False

    async def broadcast(self, data: bytes) -> int:
        """Send data to all connected clients."""
        count = 0
        for queue in self._connections.values():
            await queue.put(data)
            count += 1
        return count

    @property
    def client_count(self) -> int:
        return len(self._connections)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to keep connections alive."""
        heartbeat = SSETransport.format_event("ping", "")
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                await self.broadcast(heartbeat)
        except asyncio.CancelledError:
            pass
