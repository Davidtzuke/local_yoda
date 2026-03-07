"""Core Yoda agent — orchestrates LLM, memory, and tools.

Supports pluggable LLM backends. Ships with a built-in local backend
that works without any API keys (rule-based + tool dispatch). Set
YODA_LLM_BACKEND=openai and OPENAI_API_KEY to use OpenAI, or
YODA_LLM_BACKEND=ollama with OLLAMA_MODEL for local Ollama models.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator
from typing import Any

from yoda.agent.memory import ConversationMemory
from yoda.agent.tools import ToolRegistry, create_default_registry
from yoda.cli.protocol import AgentProtocol, MemoryEntry, ToolInfo

# Default conversation ID used by the CLI
_CLI_CONV = "default"

SYSTEM_PROMPT = """\
You are Yoda, a helpful local personal AI assistant. You have access to tools
for file system operations and web search. Be concise, helpful, and friendly.

Available tools (use JSON to call them):
{tools}

To use a tool, respond with:
[TOOL_CALL] {{"tool": "<name>", "args": {{...}}}}

After receiving tool output, incorporate it into your response.
"""


class YodaAgent(AgentProtocol):
    """Unified agent implementation satisfying both CLI and Web protocols."""

    def __init__(
        self,
        memory: ConversationMemory | None = None,
        tool_registry: ToolRegistry | None = None,
        llm_backend: str | None = None,
    ) -> None:
        self.memory = memory or ConversationMemory()
        self.tools = tool_registry or create_default_registry()
        self._backend = llm_backend or os.getenv("YODA_LLM_BACKEND", "local")

    # ----- CLI AgentProtocol -----

    async def send_message(self, message: str) -> str:
        """Send a message and get a complete response (CLI protocol)."""
        return await self._process(message, _CLI_CONV)

    async def stream_message(self, message: str) -> AsyncIterator[str]:
        """Stream response tokens (CLI protocol)."""
        async for token in self._stream(message, _CLI_CONV):
            yield token

    async def get_tools(self) -> list[ToolInfo]:
        return [
            ToolInfo(name=t.name, description=t.description, parameters=t.parameters)
            for t in self.tools.list_all()
        ]

    async def get_memory(self, limit: int = 20) -> list[MemoryEntry]:
        msgs = self.memory.get(_CLI_CONV, limit)
        return [
            MemoryEntry(role=m.role, content=m.content, timestamp=m.timestamp_iso)
            for m in msgs
        ]

    async def clear_memory(self) -> None:
        self.memory.clear(_CLI_CONV)

    # ----- Web AgentProtocol -----

    async def stream(self, message: str, conversation_id: str) -> AsyncIterator[str]:
        """Stream response tokens (Web protocol)."""
        async for token in self._stream(message, conversation_id):
            yield token

    # ----- Internal processing -----

    async def _process(self, message: str, conversation_id: str) -> str:
        """Process a message and return the full response."""
        self.memory.add(conversation_id, "user", message)

        response = await self._call_llm(message, conversation_id)

        # Handle tool calls in the response
        response = await self._handle_tool_calls(response, message, conversation_id)

        self.memory.add(conversation_id, "assistant", response)
        return response

    async def _stream(self, message: str, conversation_id: str) -> AsyncIterator[str]:
        """Stream response tokens."""
        response = await self._process(message, conversation_id)
        words = response.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

    async def _call_llm(self, message: str, conversation_id: str) -> str:
        """Route to the configured LLM backend."""
        if self._backend == "openai":
            return await self._call_openai(message, conversation_id)
        if self._backend == "ollama":
            return await self._call_ollama(message, conversation_id)
        return await self._call_local(message, conversation_id)

    async def _call_local(self, message: str, conversation_id: str) -> str:
        """Built-in rule-based backend — works without any API keys."""
        lower = message.lower().strip()

        # Tool dispatch: file operations
        if any(kw in lower for kw in ("read file", "show file", "cat ")):
            path = _extract_path(message)
            if path:
                result = await self.tools.execute("read_file", path=path)
                return f"Here's the content of `{path}`:\n\n```\n{result}\n```"

        if any(kw in lower for kw in ("write file", "save file", "create file")):
            path = _extract_path(message)
            content = _extract_content(message)
            if path and content:
                result = await self.tools.execute(
                    "write_file", path=path, content=content
                )
                return result
            return (
                "Please specify a file path and content. "
                "Example: `write file hello.txt with content: Hello World`"
            )

        if any(kw in lower for kw in ("list files", "ls", "show files", "list dir")):
            path = _extract_path(message) or "."
            result = await self.tools.execute("list_files", path=path)
            return f"Files in `{path}`:\n\n```\n{result}\n```"

        # Tool dispatch: web search
        if any(
            kw in lower
            for kw in ("search", "look up", "find info", "google", "what is", "who is")
        ):
            query = _extract_search_query(message)
            if query:
                result = await self.tools.execute("web_search", query=query)
                return f"Search results for *{query}*:\n\n{result}"

        # Conversational responses
        if lower in ("hi", "hello", "hey", "yo"):
            return (
                "Hello! How can I help you today? I can search the web, "
                "read/write files, and more. Type `/help` for commands."
            )

        if "help" in lower and not lower.startswith("/"):
            tools = self.tools.list_all()
            tool_list = "\n".join(f"- **{t.name}**: {t.description}" for t in tools)
            return (
                f"I'm Yoda, your local AI assistant. Here's what I can do:\n\n"
                f"{tool_list}\n\n"
                "Just ask me naturally — e.g., \"search for Python tutorials\" "
                'or "list files".'
            )

        return (
            f"I received your message: *{message}*\n\n"
            "I'm running in **local mode** (no LLM API configured). "
            "I can still help with:\n"
            '- **File operations**: "list files", "read file example.txt", '
            '"write file test.txt with content: hello"\n'
            '- **Web search**: "search for Python tutorials"\n\n'
            "Set `YODA_LLM_BACKEND=openai` and `OPENAI_API_KEY` for full AI "
            "capabilities, or `YODA_LLM_BACKEND=ollama` for local LLM support."
        )

    async def _call_openai(self, message: str, conversation_id: str) -> str:
        """OpenAI API backend."""
        try:
            import httpx
        except ImportError:
            return (
                "Error: `httpx` is required for OpenAI backend. "
                "Install with: `pip install httpx`"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY environment variable not set."

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        history = self.memory.get_context_messages(conversation_id, limit=20)

        tool_desc = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools.list_all()
        )
        system = SYSTEM_PROMPT.format(tools=tool_desc)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system}
        ] + history + [{"role": "user", "content": message}]

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": messages, "max_tokens": 2048},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _call_ollama(self, message: str, conversation_id: str) -> str:
        """Ollama local LLM backend."""
        try:
            import httpx
        except ImportError:
            return (
                "Error: `httpx` is required for Ollama backend. "
                "Install with: `pip install httpx`"
            )

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        history = self.memory.get_context_messages(conversation_id, limit=20)

        tool_desc = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools.list_all()
        )
        system = SYSTEM_PROMPT.format(tools=tool_desc)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system}
        ] + history + [{"role": "user", "content": message}]

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{base_url}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

    async def _handle_tool_calls(
        self, response: str, original_msg: str, conversation_id: str
    ) -> str:
        """Parse and execute any [TOOL_CALL] blocks in the LLM response."""
        pattern = r"\[TOOL_CALL\]\s*(\{.*?\})"
        matches = re.findall(pattern, response, re.DOTALL)
        if not matches:
            return response

        for match in matches:
            try:
                call = json.loads(match)
                tool_name = call.get("tool", "")
                args = call.get("args", {})
                result = await self.tools.execute(tool_name, **args)
                response = response.replace(
                    f"[TOOL_CALL] {match}",
                    f"\n**Tool result ({tool_name}):**\n```\n{result}\n```\n",
                )
            except (json.JSONDecodeError, TypeError):
                continue
        return response


# ----- Helpers for local mode parsing -----


def _extract_path(message: str) -> str | None:
    """Extract a file path from user message."""
    quoted = re.search(r'["\']([^"\']+)["\']', message)
    if quoted:
        return quoted.group(1)
    for word in message.split():
        if ("." in word or "/" in word) and not word.startswith("http"):
            return word.strip(".,;:")
    return None


def _extract_content(message: str) -> str | None:
    """Extract file content from messages like 'write file X with content: Y'."""
    for sep in ("content:", "with:", "containing:"):
        if sep in message.lower():
            idx = message.lower().index(sep) + len(sep)
            return message[idx:].strip()
    return None


def _extract_search_query(message: str) -> str | None:
    """Extract search query from user message."""
    lower = message.lower()
    for prefix in (
        "search for ",
        "search ",
        "look up ",
        "find info about ",
        "find info on ",
        "google ",
        "what is ",
        "who is ",
    ):
        if prefix in lower:
            idx = lower.index(prefix) + len(prefix)
            query = message[idx:].strip().rstrip("?.!")
            if query:
                return query
    return message.strip()
