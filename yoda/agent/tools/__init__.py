"""Tool registry for the Yoda agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


@dataclass
class ToolDef:
    """Definition of a tool the agent can invoke."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., Coroutine[Any, Any, str]] | None = None


class ToolRegistry:
    """Central registry of available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_all(self) -> list[ToolDef]:
        return list(self._tools.values())

    async def execute(self, name: str, **kwargs: Any) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: Unknown tool '{name}'"
        if tool.handler is None:
            return f"Error: Tool '{name}' has no handler"
        try:
            return await tool.handler(**kwargs)
        except Exception as e:
            return f"Error executing '{name}': {e}"


def create_default_registry() -> ToolRegistry:
    """Build a registry with all built-in tools."""
    from yoda.agent.tools.filesystem import get_filesystem_tools
    from yoda.agent.tools.websearch import get_websearch_tools

    registry = ToolRegistry()
    for tool in get_filesystem_tools():
        registry.register(tool)
    for tool in get_websearch_tools():
        registry.register(tool)
    return registry
