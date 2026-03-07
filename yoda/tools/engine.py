"""Tool execution engine with validation, error handling, timeouts, and sandboxing."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from yoda.tools.base import Tool, ToolCapability, ToolResult
from yoda.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Default timeout for tool execution (seconds)
DEFAULT_TIMEOUT: float = 30.0


class SandboxViolation(Exception):
    """Raised when a tool attempts an operation not declared in its capabilities."""


class ToolEngine:
    """Executes tools through the registry with validation, timeouts,
    error handling, and capability-based sandboxing.

    Args:
        registry: The tool registry to pull tools from.
        default_timeout: Max seconds a tool is allowed to run.
        allowed_capabilities: If set, restricts which capabilities tools
            may declare. Tools with undeclared capabilities are blocked.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        default_timeout: float = DEFAULT_TIMEOUT,
        allowed_capabilities: ToolCapability | None = None,
    ) -> None:
        self.registry = registry
        self.default_timeout = default_timeout
        self.allowed_capabilities = allowed_capabilities

    # --- public API ---

    async def run(
        self,
        tool_name: str,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Look up a tool by name and execute it safely.

        Steps:
        1. Resolve tool from registry.
        2. Validate parameters.
        3. Check sandbox capabilities.
        4. Execute with timeout.
        5. Catch and wrap any unhandled exceptions.

        Returns a ``ToolResult`` (never raises).
        """
        start = time.monotonic()

        # 1. Resolve tool
        try:
            tool_obj = self.registry.get(tool_name)
        except KeyError as exc:
            return ToolResult.fail(str(exc))

        # 2. Validate parameters
        errors = tool_obj.validate_params(kwargs)
        if errors:
            return ToolResult.fail(
                f"Validation failed: {'; '.join(errors)}",
                tool=tool_name,
            )

        # 3. Sandbox check
        violation = self._check_sandbox(tool_obj)
        if violation:
            return ToolResult.fail(
                f"Sandbox violation: {violation}",
                tool=tool_name,
            )

        # 4. Apply defaults and execute with timeout
        resolved_kwargs = tool_obj.apply_defaults(kwargs)
        effective_timeout = timeout if timeout is not None else self.default_timeout

        try:
            result = await asyncio.wait_for(
                tool_obj.execute(**resolved_kwargs),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            logger.warning(
                "Tool '%s' timed out after %.1fs", tool_name, elapsed
            )
            return ToolResult.fail(
                f"Tool '{tool_name}' timed out after {effective_timeout}s",
                tool=tool_name,
                elapsed=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.exception("Tool '%s' raised an exception", tool_name)
            return ToolResult.fail(
                f"Tool '{tool_name}' error: {type(exc).__name__}: {exc}",
                tool=tool_name,
                elapsed=elapsed,
            )

        # 5. Attach timing metadata
        elapsed = time.monotonic() - start
        result.metadata.setdefault("tool", tool_name)
        result.metadata.setdefault("elapsed", round(elapsed, 4))
        return result

    async def run_many(
        self,
        calls: list[dict[str, Any]],
        timeout: float | None = None,
    ) -> list[ToolResult]:
        """Execute multiple tool calls concurrently.

        Each item in *calls* must be a dict with ``"name"`` (str) and
        optional ``"kwargs"`` (dict).

        Returns results in the same order as *calls*.
        """
        tasks = [
            self.run(
                call["name"],
                timeout=timeout,
                **call.get("kwargs", {}),
            )
            for call in calls
        ]
        return list(await asyncio.gather(*tasks))

    # --- introspection ---

    def available_tools(self) -> list[dict[str, Any]]:
        """Return the full manifest of all registered tools."""
        return self.registry.manifest()

    # --- internal ---

    def _check_sandbox(self, tool_obj: Tool) -> str | None:
        """Return a violation message if the tool's capabilities exceed
        what is allowed, or None if OK."""
        allowed = self.allowed_capabilities
        if allowed is None:
            return None  # no restrictions

        caps = tool_obj.capabilities
        violations: list[str] = []

        if caps.reads_filesystem and not allowed.reads_filesystem:
            violations.append("reads_filesystem")
        if caps.writes_filesystem and not allowed.writes_filesystem:
            violations.append("writes_filesystem")
        if caps.network_access and not allowed.network_access:
            violations.append("network_access")
        if caps.subprocess_access and not allowed.subprocess_access:
            violations.append("subprocess_access")

        if violations:
            return (
                f"Tool '{tool_obj.name}' requires capabilities not allowed: "
                f"{', '.join(violations)}"
            )
        return None
