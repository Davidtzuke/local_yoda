"""Abstract LLM provider interface and factory."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

from yoda.core.config import ProviderConfig
from yoda.core.messages import AssistantMessage, ToolCall
from yoda.core.plugins import ToolSchema


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

class StreamChunk(BaseModel):
    """A single chunk in a streaming response."""

    delta: str = ""
    tool_call: ToolCall | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    """Complete response from an LLM provider."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    thinking: str | None = None
    finish_reason: str = "stop"
    model: str = ""
    usage: dict[str, int] = Field(default_factory=dict)  # input_tokens, output_tokens
    latency_ms: float = 0.0

    def to_assistant_message(self) -> AssistantMessage:
        return AssistantMessage(
            content=self.content,
            tool_calls=self.tool_calls,
            thinking=self.thinking,
            token_count=self.usage.get("output_tokens"),
            metadata={"model": self.model, "latency_ms": self.latency_ms},
        )


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract interface for LLM providers (Anthropic, OpenAI, local, etc.)."""

    name: str = "base"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send messages and return a complete response."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Send messages and yield streaming chunks."""
        ...
        # Make this an async generator
        if False:  # pragma: no cover
            yield StreamChunk()  # type: ignore[misc]

    async def health_check(self) -> bool:
        """Return True if the provider is reachable."""
        try:
            resp = await self.complete(
                [{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(resp.content)
        except Exception:
            return False

    def _timed(self) -> float:
        """Return current time in ms for latency tracking."""
        return time.monotonic() * 1000


# ---------------------------------------------------------------------------
# Provider registry & factory
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {}


def register_provider(name: str):  # noqa: ANN201
    """Decorator to register a provider class."""
    def decorator(cls: type[LLMProvider]) -> type[LLMProvider]:
        _PROVIDER_REGISTRY[name] = cls
        return cls
    return decorator


def create_provider(config: ProviderConfig) -> LLMProvider:
    """Instantiate a provider by name from config."""
    # Lazy-import concrete providers to register them
    _ensure_providers_imported()

    name = config.name.lower()
    if name not in _PROVIDER_REGISTRY:
        available = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return _PROVIDER_REGISTRY[name](config)


def _ensure_providers_imported() -> None:
    """Import all concrete provider modules so they self-register."""
    import yoda.core.providers.anthropic_provider  # noqa: F401
    import yoda.core.providers.openai_provider  # noqa: F401
    import yoda.core.providers.local_provider  # noqa: F401
