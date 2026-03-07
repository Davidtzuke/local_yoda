"""Shared types and data models used across all Yoda modules."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Role(StrEnum):
    """Message author role."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in a conversation."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A request from the LLM to invoke a tool."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = Field(default_factory=lambda: uuid.uuid4().hex)


class ToolResult(BaseModel):
    """The result of executing a tool."""

    call_id: str
    tool_name: str
    output: str
    success: bool = True
    error: str | None = None


class Conversation(BaseModel):
    """An ordered sequence of messages forming a conversation."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_message(self, role: Role, content: str, **kwargs: Any) -> Message:
        """Append a message and return it."""
        msg = Message(role=role, content=content, **kwargs)
        self.messages.append(msg)
        return msg

    @property
    def last_user_message(self) -> Message | None:
        """Return the most recent user message, if any."""
        for msg in reversed(self.messages):
            if msg.role == Role.USER:
                return msg
        return None


class MemoryEntry(BaseModel):
    """A single unit of stored memory (fact, summary, or note)."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    content: str
    source: str = ""  # e.g. "conversation:<id>", "user_note", "tool_output"
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchResult(BaseModel):
    """A memory entry with a relevance score from search."""

    entry: MemoryEntry
    score: float  # 0.0–1.0, higher is more relevant


class AgentResponse(BaseModel):
    """The agent's full response after processing a user turn."""

    message: Message
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    memories_used: list[MemorySearchResult] = Field(default_factory=list)
