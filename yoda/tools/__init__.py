"""Tool & Computer Access subsystem for Yoda.

Provides:
- Decorator-based tool registry with auto JSON schema generation
- Built-in tools (file ops, shell, web, calendar, notes)
- MCP client for stdio+SSE server connections
- Tool executor with parallel execution, retry, and approval flow
- Computer access (screenshot, OCR, mouse/keyboard, app launcher)
"""

from yoda.tools.registry import ToolRegistry, tool, ToolPermission, ToolMetadata
from yoda.tools.executor import ToolExecutor, ApprovalPolicy, ExecutionResult

__all__ = [
    "ToolRegistry",
    "tool",
    "ToolPermission",
    "ToolMetadata",
    "ToolExecutor",
    "ApprovalPolicy",
    "ExecutionResult",
]
