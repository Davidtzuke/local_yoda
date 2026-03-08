"""Typed message classes with token counting for the Yoda conversation protocol."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Role(StrEnum):
    """Message roles in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


class ToolCall(BaseModel):
    """A tool invocation requested by the assistant."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result returned from a tool execution."""

    tool_call_id: str
    output: Any = None
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


class Message(BaseModel):
    """Base message in the Yoda conversation protocol."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    role: Role
    content: str = ""
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Token counts — populated after provider encoding
    token_count: int | None = None

    def estimate_tokens(self, chars_per_token: float = 3.5) -> int:
        """Fast heuristic token estimate (no tokenizer needed)."""
        return max(1, int(len(self.content) / chars_per_token))

    def count_tokens(self, model: str = "gpt-4") -> int:
        """Accurate token count using tiktoken. Caches result."""
        if self.token_count is not None:
            return self.token_count
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(model)
            self.token_count = len(enc.encode(self.content))
        except Exception:
            self.token_count = self.estimate_tokens()
        return self.token_count


class SystemMessage(Message):
    """System prompt message."""

    role: Role = Role.SYSTEM


class UserMessage(Message):
    """User input message."""

    role: Role = Role.USER


class AssistantMessage(Message):
    """Assistant response, optionally containing tool calls."""

    role: Role = Role.ASSISTANT
    tool_calls: list[ToolCall] = Field(default_factory=list)
    thinking: str | None = None  # Chain-of-thought / scratchpad

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class ToolResultMessage(Message):
    """Result of a tool execution fed back into the conversation."""

    role: Role = Role.TOOL_RESULT
    tool_results: list[ToolResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Conversation history helper
# ---------------------------------------------------------------------------

class Conversation(BaseModel):
    """Ordered list of messages forming a conversation."""

    messages: list[Message] = Field(default_factory=list)
    system_prompt: str = ""

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def add_user(self, content: str, **kwargs: Any) -> UserMessage:
        msg = UserMessage(content=content, **kwargs)
        self.add(msg)
        return msg

    def add_assistant(self, content: str, **kwargs: Any) -> AssistantMessage:
        msg = AssistantMessage(content=content, **kwargs)
        self.add(msg)
        return msg

    def total_tokens(self, model: str = "gpt-4") -> int:
        return sum(m.count_tokens(model) for m in self.messages)

    def to_provider_format(self) -> list[dict[str, Any]]:
        """Convert to the list-of-dicts format most LLM APIs expect."""
        out: list[dict[str, Any]] = []
        if self.system_prompt:
            out.append({"role": "system", "content": self.system_prompt})
        for msg in self.messages:
            entry: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
            if isinstance(msg, AssistantMessage) and msg.has_tool_calls:
                entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            if isinstance(msg, ToolResultMessage):
                entry["tool_results"] = [tr.model_dump() for tr in msg.tool_results]
            out.append(entry)
        return out

    def last(self, n: int = 1) -> list[Message]:
        return self.messages[-n:]

    def __len__(self) -> int:
        return len(self.messages)
