"""Local model provider via Ollama."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from yoda.core.config import ProviderConfig
from yoda.core.plugins import ToolSchema
from yoda.core.providers.base import (
    LLMProvider,
    ProviderResponse,
    StreamChunk,
    register_provider,
)


@register_provider("local")
@register_provider("ollama")
class LocalProvider(LLMProvider):
    """Provider for local models via Ollama HTTP API."""

    name = "local"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise ImportError("pip install httpx") from e

        import httpx

        self._base_url = config.base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=config.timeout)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        start = self._timed()
        payload = {
            "model": kwargs.pop("model", self.config.model),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.pop("temperature", self.config.temperature),
                "num_predict": kwargs.pop("max_tokens", self.config.max_tokens),
            },
        }
        payload.update(kwargs)

        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        latency = self._timed() - start

        msg = data.get("message", {})
        usage = {}
        if "eval_count" in data:
            usage["output_tokens"] = data["eval_count"]
        if "prompt_eval_count" in data:
            usage["input_tokens"] = data["prompt_eval_count"]

        return ProviderResponse(
            content=msg.get("content", ""),
            finish_reason="stop",
            model=data.get("model", self.config.model),
            usage=usage,
            latency_ms=latency,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        payload = {
            "model": kwargs.pop("model", self.config.model),
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": kwargs.pop("temperature", self.config.temperature),
                "num_predict": kwargs.pop("max_tokens", self.config.max_tokens),
            },
        }
        payload.update(kwargs)

        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            import json

            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                msg = data.get("message", {})
                done = data.get("done", False)
                yield StreamChunk(
                    delta=msg.get("content", ""),
                    finish_reason="stop" if done else None,
                )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False
