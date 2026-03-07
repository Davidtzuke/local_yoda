"""Agent protocol that the Integration agent will implement."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolInfo:
    """Describes a tool available to the agent."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEntry:
    """A single memory/conversation entry."""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str = ""


class AgentProtocol(abc.ABC):
    """Protocol that any agent backend must implement for the CLI to use."""

    @abc.abstractmethod
    async def send_message(self, message: str) -> str:
        """Send a message and get a complete response."""
        ...

    @abc.abstractmethod
    async def stream_message(self, message: str) -> AsyncIterator[str]:
        """Send a message and stream response tokens."""
        ...

    @abc.abstractmethod
    async def get_tools(self) -> list[ToolInfo]:
        """Return list of available tools."""
        ...

    @abc.abstractmethod
    async def get_memory(self, limit: int = 20) -> list[MemoryEntry]:
        """Return recent conversation history."""
        ...

    @abc.abstractmethod
    async def clear_memory(self) -> None:
        """Clear conversation history."""
        ...


class StubAgent(AgentProtocol):
    """A stub agent for development/testing before Integration wires in the real one."""

    async def send_message(self, message: str) -> str:
        return f"Echo: {message}\n\n_(Stub agent - real agent not yet connected)_"

    async def stream_message(self, message: str) -> AsyncIterator[str]:
        response = f"Echo: {message}\n\n_(Stub agent - real agent not yet connected)_"
        for word in response.split(" "):
            yield word + " "

    async def get_tools(self) -> list[ToolInfo]:
        return [
            ToolInfo(name="filesystem", description="Read, write, and list files"),
            ToolInfo(name="websearch", description="Search the web"),
        ]

    async def get_memory(self, limit: int = 20) -> list[MemoryEntry]:
        return []

    async def clear_memory(self) -> None:
        pass
