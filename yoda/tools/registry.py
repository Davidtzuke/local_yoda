"""Tool registry and @tool decorator for the Yoda tool framework."""

from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

from yoda.tools.base import (
    Tool,
    ToolCapability,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central catalogue of available tools.

    Supports registration, lookup, listing, and manifest generation.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # --- registration ---

    def register(self, tool_instance: Tool) -> None:
        """Register a tool instance. Raises ValueError on duplicate name."""
        name = tool_instance.name
        if name in self._tools:
            raise ValueError(f"Tool already registered: '{name}'")
        self._tools[name] = tool_instance
        logger.info("Registered tool: %s", name)

    def unregister(self, name: str) -> None:
        """Remove a tool by name. Raises KeyError if not found."""
        if name not in self._tools:
            raise KeyError(f"Tool not found: '{name}'")
        del self._tools[name]
        logger.info("Unregistered tool: %s", name)

    # --- lookup ---

    def get(self, name: str) -> Tool:
        """Return the tool with the given name, or raise KeyError."""
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(
                f"Tool '{name}' not found. Available: {list(self._tools)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # --- listing ---

    def list_tools(self) -> list[Tool]:
        """Return all registered tools sorted by name."""
        return sorted(self._tools.values(), key=lambda t: t.name)

    def list_names(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools)

    # --- manifest ---

    def manifest(self) -> list[dict[str, Any]]:
        """Return a JSON-serialisable manifest of all registered tools."""
        return [t.manifest() for t in self.list_tools()]

    # --- bulk registration ---

    def register_all(self, tools: list[Tool]) -> None:
        """Register multiple tools at once."""
        for t in tools:
            self.register(t)


# ---------------------------------------------------------------------------
# @tool decorator – turn an async function into a Tool
# ---------------------------------------------------------------------------

def tool(
    name: str,
    description: str,
    parameters: list[ToolParameter] | None = None,
    capabilities: ToolCapability | None = None,
) -> Callable[
    [Callable[..., Awaitable[ToolResult]]],
    type[Tool],
]:
    """Decorator that converts an async function into a discoverable Tool class.

    Usage::

        @tool(
            name="greet",
            description="Say hello",
            parameters=[ToolParameter("user", "Name to greet")],
        )
        async def greet(user: str) -> ToolResult:
            return ToolResult.ok(f"Hello, {user}!")

    The returned object is a *Tool subclass* (not an instance). Instantiate it
    to get a registerable tool::

        registry.register(greet())
    """

    def decorator(
        fn: Callable[..., Awaitable[ToolResult]],
    ) -> type[Tool]:
        _params = parameters or []
        _caps = capabilities or ToolCapability()

        class _DecoratedTool(Tool):
            nonlocal name, description

            def __init__(self) -> None:
                self.name = name  # type: ignore[assignment]
                self.description = description  # type: ignore[assignment]
                self.parameters = _params  # type: ignore[assignment]
                self.capabilities = _caps  # type: ignore[assignment]

            async def execute(self, **kwargs: Any) -> ToolResult:
                return await fn(**kwargs)

        _DecoratedTool.__name__ = fn.__name__
        _DecoratedTool.__qualname__ = fn.__qualname__
        _DecoratedTool.__doc__ = fn.__doc__
        # Stash metadata so the class is self-describing before instantiation
        _DecoratedTool._tool_name = name  # type: ignore[attr-defined]
        return _DecoratedTool

    return decorator


# ---------------------------------------------------------------------------
# Global default registry (convenience)
# ---------------------------------------------------------------------------

default_registry = ToolRegistry()
"""Module-level default registry for quick prototyping."""
