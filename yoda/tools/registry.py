"""Decorator-based tool registry with auto JSON schema generation, permissions, and rate limiting."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Coroutine, get_type_hints

from pydantic import BaseModel, Field

from yoda.core.plugins import ToolParameter, ToolSchema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permission levels
# ---------------------------------------------------------------------------

class ToolPermission(str, Enum):
    """Permission level required to execute a tool."""
    SAFE = "safe"              # No side effects, always allowed
    READ = "read"              # Reads data from filesystem/network
    WRITE = "write"            # Writes data to filesystem
    EXECUTE = "execute"        # Executes commands/processes
    DANGEROUS = "dangerous"    # Destructive or privileged operations
    COMPUTER = "computer"      # Computer control (mouse/keyboard/screen)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter per tool."""

    def __init__(self, calls_per_minute: int = 60) -> None:
        self.calls_per_minute = calls_per_minute
        self._timestamps: list[float] = []

    def check(self) -> bool:
        """Return True if the call is allowed."""
        now = time.monotonic()
        window = 60.0
        self._timestamps = [t for t in self._timestamps if now - t < window]
        if len(self._timestamps) >= self.calls_per_minute:
            return False
        self._timestamps.append(now)
        return True

    @property
    def remaining(self) -> int:
        now = time.monotonic()
        recent = sum(1 for t in self._timestamps if now - t < 60.0)
        return max(0, self.calls_per_minute - recent)


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------

class ToolMetadata(BaseModel):
    """Metadata for a registered tool function."""

    name: str
    description: str
    permission: ToolPermission = ToolPermission.SAFE
    rate_limit: int = 60  # calls per minute
    tags: list[str] = Field(default_factory=list)
    requires_approval: bool = False
    timeout: float = 30.0  # seconds
    retries: int = 0
    category: str = "general"


# ---------------------------------------------------------------------------
# Type mapping for schema generation
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    bytes: "string",
}


def _python_type_to_json(tp: Any) -> str:
    """Convert a Python type annotation to a JSON schema type string."""
    origin = getattr(tp, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    if tp in _TYPE_MAP:
        return _TYPE_MAP[tp]
    # Handle Optional[X] -> type of X
    if origin is type(None):
        return "string"
    return "string"


def _extract_param_descriptions(func: Callable[..., Any]) -> dict[str, str]:
    """Extract parameter descriptions from docstring (Google-style)."""
    doc = inspect.getdoc(func) or ""
    descriptions: dict[str, str] = {}
    in_args = False
    for line in doc.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("args:") or stripped.lower().startswith("parameters:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped.startswith("-") and ":" not in stripped:
                in_args = False
                continue
            if ":" in stripped:
                param_part = stripped.lstrip("- ")
                name_part, _, desc_part = param_part.partition(":")
                # Handle type annotations in docstring like "name (type): desc"
                param_name = name_part.split("(")[0].strip()
                if param_name:
                    descriptions[param_name] = desc_part.strip()
    return descriptions


def _func_to_schema(func: Callable[..., Any], metadata: ToolMetadata) -> ToolSchema:
    """Generate a ToolSchema from a function's signature and type hints."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    param_docs = _extract_param_descriptions(func)
    parameters: list[ToolParameter] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        tp = hints.get(name, str)
        json_type = _python_type_to_json(tp)
        required = param.default is inspect.Parameter.empty
        default = None if required else param.default
        desc = param_docs.get(name, "")

        parameters.append(ToolParameter(
            name=name,
            type=json_type,
            description=desc,
            required=required,
            default=default,
        ))

    # Get return type
    return_type = _python_type_to_json(hints.get("return", str))

    # Use first line of docstring as description if not set in metadata
    doc = inspect.getdoc(func) or metadata.description
    description = doc.split("\n")[0] if doc else metadata.description

    return ToolSchema(
        name=metadata.name,
        description=description,
        parameters=parameters,
        returns=return_type,
    )


# ---------------------------------------------------------------------------
# Tool decorator
# ---------------------------------------------------------------------------

class _RegisteredTool:
    """Wraps a tool function with its metadata, schema, and rate limiter."""

    __slots__ = ("func", "metadata", "schema", "rate_limiter")

    def __init__(self, func: Callable[..., Any], metadata: ToolMetadata) -> None:
        self.func = func
        self.metadata = metadata
        self.schema = _func_to_schema(func, metadata)
        self.rate_limiter = RateLimiter(metadata.rate_limit)


# Module-level registry for decorator-collected tools
_TOOL_REGISTRY: dict[str, _RegisteredTool] = {}


def tool(
    name: str | None = None,
    *,
    description: str = "",
    permission: ToolPermission = ToolPermission.SAFE,
    rate_limit: int = 60,
    tags: list[str] | None = None,
    requires_approval: bool = False,
    timeout: float = 30.0,
    retries: int = 0,
    category: str = "general",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a function as a Yoda tool.

    Usage:
        @tool(name="read_file", permission=ToolPermission.READ)
        async def read_file(path: str) -> str:
            '''Read a file from disk.

            Args:
                path: Path to the file to read.
            '''
            ...
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        meta = ToolMetadata(
            name=tool_name,
            description=description or inspect.getdoc(func) or "",
            permission=permission,
            rate_limit=rate_limit,
            tags=tags or [],
            requires_approval=requires_approval,
            timeout=timeout,
            retries=retries,
            category=category,
        )
        registered = _RegisteredTool(func, meta)
        _TOOL_REGISTRY[tool_name] = registered
        # Attach metadata to the function for introspection
        func._tool_metadata = meta  # type: ignore[attr-defined]
        func._tool_schema = registered.schema  # type: ignore[attr-defined]
        return func
    return decorator


# ---------------------------------------------------------------------------
# Tool Registry class
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Central registry for tools with schema generation, permissions, and rate limiting.

    Can collect tools from:
    1. @tool decorated functions (auto-collected)
    2. Programmatic registration
    3. MCP server discovery
    """

    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool] = {}
        self._execution_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "errors": 0, "total_time": 0.0}
        )

    # -- Registration -------------------------------------------------------

    def register_function(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        *,
        description: str = "",
        permission: ToolPermission = ToolPermission.SAFE,
        rate_limit: int = 60,
        tags: list[str] | None = None,
        requires_approval: bool = False,
        timeout: float = 30.0,
        retries: int = 0,
        category: str = "general",
    ) -> None:
        """Register a function as a tool programmatically."""
        tool_name = name or func.__name__
        meta = ToolMetadata(
            name=tool_name,
            description=description,
            permission=permission,
            rate_limit=rate_limit,
            tags=tags or [],
            requires_approval=requires_approval,
            timeout=timeout,
            retries=retries,
            category=category,
        )
        self._tools[tool_name] = _RegisteredTool(func, meta)

    def collect_decorated(self) -> None:
        """Import all tools registered via the @tool decorator."""
        for name, registered in _TOOL_REGISTRY.items():
            if name not in self._tools:
                self._tools[name] = registered

    def register_from_schema(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter],
        handler: Callable[..., Any],
        *,
        permission: ToolPermission = ToolPermission.SAFE,
        category: str = "mcp",
    ) -> None:
        """Register a tool from an external schema (e.g., MCP)."""
        meta = ToolMetadata(
            name=name,
            description=description,
            permission=permission,
            category=category,
        )
        schema = ToolSchema(
            name=name,
            description=description,
            parameters=parameters,
        )
        reg = _RegisteredTool(handler, meta)
        reg.schema = schema
        self._tools[name] = reg

    # -- Lookup -------------------------------------------------------------

    def get(self, name: str) -> _RegisteredTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSchema]:
        """Return schemas for all registered tools."""
        return [t.schema for t in self._tools.values()]

    def list_by_category(self, category: str) -> list[ToolSchema]:
        return [t.schema for t in self._tools.values() if t.metadata.category == category]

    def list_by_permission(self, max_level: ToolPermission) -> list[ToolSchema]:
        """List tools up to a given permission level."""
        levels = list(ToolPermission)
        max_idx = levels.index(max_level)
        return [
            t.schema for t in self._tools.values()
            if levels.index(t.metadata.permission) <= max_idx
        ]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def stats(self) -> dict[str, dict[str, Any]]:
        return dict(self._execution_stats)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # -- Execution tracking --------------------------------------------------

    def record_execution(self, name: str, duration: float, error: bool = False) -> None:
        """Record execution stats for a tool."""
        stats = self._execution_stats[name]
        stats["calls"] += 1
        stats["total_time"] += duration
        if error:
            stats["errors"] += 1
