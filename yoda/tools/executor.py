"""Tool executor with parallel execution, retry, error recovery, and approval flow."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Coroutine

from pydantic import BaseModel, Field

from yoda.tools.registry import ToolPermission, ToolRegistry, _RegisteredTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Approval policy
# ---------------------------------------------------------------------------

class ApprovalPolicy(str, Enum):
    """How to handle tool approval requests."""
    ALWAYS_ALLOW = "always_allow"         # No approval needed
    REQUIRE_DANGEROUS = "require_dangerous"  # Only dangerous tools need approval
    REQUIRE_WRITE = "require_write"       # Write+ tools need approval
    REQUIRE_ALL = "require_all"           # All tools need approval


# Approval callback type: receives tool name + args, returns True if approved
ApprovalCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, bool]]


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

class ExecutionResult(BaseModel):
    """Result of a tool execution."""

    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    duration: float = 0.0
    retries_used: int = 0
    approved: bool = True


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Executes tools with retry, timeout, rate limiting, parallel execution, and approval.

    Usage:
        executor = ToolExecutor(registry)
        result = await executor.execute("read_file", {"path": "/tmp/test.txt"})
        results = await executor.execute_parallel([
            ("read_file", {"path": "/tmp/a.txt"}),
            ("read_file", {"path": "/tmp/b.txt"}),
        ])
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        approval_policy: ApprovalPolicy = ApprovalPolicy.REQUIRE_DANGEROUS,
        approval_callback: ApprovalCallback | None = None,
        max_parallel: int = 10,
        default_timeout: float = 30.0,
    ) -> None:
        self.registry = registry
        self.approval_policy = approval_policy
        self._approval_callback = approval_callback
        self._max_parallel = max_parallel
        self._default_timeout = default_timeout
        self._semaphore = asyncio.Semaphore(max_parallel)

    # -- Approval -----------------------------------------------------------

    def _needs_approval(self, tool: _RegisteredTool) -> bool:
        """Check if a tool requires user approval based on policy."""
        if tool.metadata.requires_approval:
            return True
        perm = tool.metadata.permission
        match self.approval_policy:
            case ApprovalPolicy.ALWAYS_ALLOW:
                return False
            case ApprovalPolicy.REQUIRE_DANGEROUS:
                return perm == ToolPermission.DANGEROUS
            case ApprovalPolicy.REQUIRE_WRITE:
                return perm in (ToolPermission.WRITE, ToolPermission.EXECUTE, ToolPermission.DANGEROUS)
            case ApprovalPolicy.REQUIRE_ALL:
                return True

    async def _request_approval(self, name: str, arguments: dict[str, Any]) -> bool:
        """Request approval from the user."""
        if self._approval_callback:
            return await self._approval_callback(name, arguments)
        # Default: auto-approve (in CLI, this would prompt the user)
        logger.warning("Tool %s requires approval but no callback set — auto-approving", name)
        return True

    # -- Single execution ---------------------------------------------------

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Execute a single tool with retry, timeout, and rate limiting."""
        arguments = arguments or {}

        registered = self.registry.get(tool_name)
        if registered is None:
            return ExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}",
            )

        # Check rate limit
        if not registered.rate_limiter.check():
            return ExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"Rate limit exceeded for {tool_name} "
                      f"({registered.metadata.rate_limit} calls/min)",
            )

        # Check approval
        if self._needs_approval(registered):
            approved = await self._request_approval(tool_name, arguments)
            if not approved:
                return ExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error="Tool execution denied by user",
                    approved=False,
                )

        # Execute with retry
        timeout = registered.metadata.timeout or self._default_timeout
        max_retries = registered.metadata.retries
        last_error: str | None = None

        for attempt in range(max_retries + 1):
            start = time.monotonic()
            try:
                async with self._semaphore:
                    result = await asyncio.wait_for(
                        self._call_tool(registered, arguments),
                        timeout=timeout,
                    )
                duration = time.monotonic() - start
                self.registry.record_execution(tool_name, duration)
                return ExecutionResult(
                    tool_name=tool_name,
                    success=True,
                    output=result,
                    duration=duration,
                    retries_used=attempt,
                )
            except asyncio.TimeoutError:
                last_error = f"Tool {tool_name} timed out after {timeout}s"
                logger.warning("%s (attempt %d/%d)", last_error, attempt + 1, max_retries + 1)
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    "Tool %s failed (attempt %d/%d): %s",
                    tool_name, attempt + 1, max_retries + 1, last_error,
                )

            # Brief backoff before retry
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))

        duration = time.monotonic() - start  # type: ignore[possibly-undefined]
        self.registry.record_execution(tool_name, duration, error=True)
        return ExecutionResult(
            tool_name=tool_name,
            success=False,
            error=last_error,
            duration=duration,
            retries_used=max_retries,
        )

    async def _call_tool(self, tool: _RegisteredTool, arguments: dict[str, Any]) -> Any:
        """Call the tool function, handling both sync and async."""
        func = tool.func
        if asyncio.iscoroutinefunction(func):
            return await func(**arguments)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**arguments))

    # -- Parallel execution -------------------------------------------------

    async def execute_parallel(
        self,
        calls: list[tuple[str, dict[str, Any]]],
    ) -> list[ExecutionResult]:
        """Execute multiple tools in parallel."""
        tasks = [self.execute(name, args) for name, args in calls]
        return await asyncio.gather(*tasks)

    # -- Batch execution with dependencies ----------------------------------

    async def execute_chain(
        self,
        steps: list[tuple[str, dict[str, Any] | Callable[[Any], dict[str, Any]]]],
    ) -> list[ExecutionResult]:
        """Execute tools sequentially, passing previous output to next step.

        Each step is (tool_name, args_or_callable). If callable, it receives
        the previous step's output and returns the arguments dict.
        """
        results: list[ExecutionResult] = []
        prev_output: Any = None

        for tool_name, args_or_fn in steps:
            if callable(args_or_fn) and not isinstance(args_or_fn, dict):
                arguments = args_or_fn(prev_output)
            else:
                arguments = args_or_fn  # type: ignore[assignment]

            result = await self.execute(tool_name, arguments)
            results.append(result)

            if not result.success:
                # Chain breaks on error
                break
            prev_output = result.output

        return results
