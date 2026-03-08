"""OpenAI-compatible LLM provider (GPT-4, GPT-4o, etc.)."""

from __future__ import annotations

import json
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


@register_provider("openai")
class OpenAIProvider(LLMProvider):
    """Provider for OpenAI and OpenAI-compatible APIs."""

    name = "openai"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            import openai
        except ImportError as e:
            raise ImportError("pip install openai") from e

        api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "")
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": config.timeout}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = openai.AsyncOpenAI(**kwargs)

    def _convert_tools(self, tools: list[ToolSchema] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            p.name: {"type": p.type, "description": p.description}
                            for p in t.parameters
                        },
                        "required": [p.name for p in t.parameters if p.required],
                    },
                },
            }
            for t in tools
        ]

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        start = self._timed()
        params: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens),
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "messages": messages,
        }
        converted = self._convert_tools(tools)
        if converted:
            params["tools"] = converted
        params.update(kwargs)

        response = await self._client.chat.completions.create(**params)
        latency = self._timed() - start

        choice = response.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments) if tc.function.arguments else {},
                    )
                )

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return ProviderResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            model=response.model,
            usage=usage,
            latency_ms=latency,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        params: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens),
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "messages": messages,
            "stream": True,
        }
        converted = self._convert_tools(tools)
        if converted:
            params["tools"] = converted
        params.update(kwargs)

        response = await self._client.chat.completions.create(**params)
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            yield StreamChunk(
                delta=delta.content or "",
                finish_reason=chunk.choices[0].finish_reason,
            )
