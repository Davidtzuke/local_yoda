"""Anthropic / Claude LLM provider."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from yoda.core.config import ProviderConfig
from yoda.core.messages import ToolCall
from yoda.core.plugins import ToolSchema
from yoda.core.providers.base import (
    LLMProvider,
    ProviderResponse,
    StreamChunk,
    register_provider,
)


@register_provider("anthropic")
class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude models."""

    name = "anthropic"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("pip install anthropic") from e

        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=config.timeout,
            **({"base_url": config.base_url} if config.base_url else {}),
        )

    def _convert_tools(self, tools: list[ToolSchema] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        p.name: {"type": p.type, "description": p.description}
                        for p in t.parameters
                    },
                    "required": [p.name for p in t.parameters if p.required],
                },
            }
            for t in tools
        ]

    def _build_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Extract system from messages
        system = ""
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                api_messages.append({"role": m["role"], "content": m["content"]})

        params: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens),
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "messages": api_messages,
        }
        if system:
            params["system"] = system
        converted = self._convert_tools(tools)
        if converted:
            params["tools"] = converted
        params.update(kwargs)
        return params

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        start = self._timed()
        params = self._build_params(messages, tools, **kwargs)
        response = await self._client.messages.create(**params)
        latency = self._timed() - start

        content = ""
        thinking = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
            elif hasattr(block, "type") and block.type == "thinking":
                thinking += getattr(block, "thinking", "")
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            thinking=thinking or None,
            finish_reason=response.stop_reason or "stop",
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            latency_ms=latency,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        params = self._build_params(messages, tools, **kwargs)
        async with self._client.messages.stream(**params) as stream:
            async for event in stream:
                if hasattr(event, "type"):
                    if event.type == "content_block_delta":
                        delta = getattr(event.delta, "text", "")
                        yield StreamChunk(delta=delta)
                    elif event.type == "message_stop":
                        final = await stream.get_final_message()
                        yield StreamChunk(
                            finish_reason=final.stop_reason or "stop",
                            usage={
                                "input_tokens": final.usage.input_tokens,
                                "output_tokens": final.usage.output_tokens,
                            },
                        )
