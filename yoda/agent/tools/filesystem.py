"""File system tools — read, write, list files in a sandboxed directory."""

from __future__ import annotations

import asyncio
from pathlib import Path

from yoda.agent.tools import ToolDef

# Default sandbox directory
DEFAULT_SANDBOX = Path.home() / ".yoda" / "sandbox"


def _ensure_sandbox(sandbox: Path) -> Path:
    sandbox.mkdir(parents=True, exist_ok=True)
    return sandbox


def _resolve_safe(sandbox: Path, path: str) -> Path | None:
    """Resolve path within sandbox; return None if it escapes."""
    resolved = (sandbox / path).resolve()
    if not str(resolved).startswith(str(sandbox.resolve())):
        return None
    return resolved


async def read_file(path: str, sandbox: str | None = None) -> str:
    """Read a file from the sandbox directory."""
    sb = _ensure_sandbox(Path(sandbox) if sandbox else DEFAULT_SANDBOX)
    target = _resolve_safe(sb, path)
    if target is None:
        return "Error: Path escapes sandbox directory."
    if not target.exists():
        return f"Error: File not found: {path}"
    if not target.is_file():
        return f"Error: Not a file: {path}"
    content = await asyncio.to_thread(target.read_text, encoding="utf-8")
    return content


async def write_file(path: str, content: str, sandbox: str | None = None) -> str:
    """Write content to a file in the sandbox directory."""
    sb = _ensure_sandbox(Path(sandbox) if sandbox else DEFAULT_SANDBOX)
    target = _resolve_safe(sb, path)
    if target is None:
        return "Error: Path escapes sandbox directory."
    target.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(target.write_text, content, encoding="utf-8")
    return f"Written {len(content)} bytes to {path}"


async def list_files(path: str = ".", sandbox: str | None = None) -> str:
    """List files and directories in the sandbox."""
    sb = _ensure_sandbox(Path(sandbox) if sandbox else DEFAULT_SANDBOX)
    target = _resolve_safe(sb, path)
    if target is None:
        return "Error: Path escapes sandbox directory."
    if not target.exists():
        return f"Error: Directory not found: {path}"
    if not target.is_dir():
        return f"Error: Not a directory: {path}"

    entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    lines = []
    for entry in entries:
        rel = entry.relative_to(sb)
        suffix = "/" if entry.is_dir() else ""
        size = entry.stat().st_size if entry.is_file() else 0
        lines.append(f"{'DIR ' if entry.is_dir() else f'{size:>8} '} {rel}{suffix}")
    return "\n".join(lines) if lines else "(empty directory)"


def get_filesystem_tools() -> list[ToolDef]:
    """Return filesystem tool definitions."""
    return [
        ToolDef(
            name="read_file",
            description="Read a file from the workspace. Args: path (str)",
            parameters={"path": {"type": "string", "description": "Relative file path"}},
            handler=read_file,
        ),
        ToolDef(
            name="write_file",
            description="Write content to a file. Args: path (str), content (str)",
            parameters={
                "path": {"type": "string", "description": "Relative file path"},
                "content": {"type": "string", "description": "File content"},
            },
            handler=write_file,
        ),
        ToolDef(
            name="list_files",
            description="List files and directories. Args: path (str, default='.')",
            parameters={"path": {"type": "string", "description": "Directory path"}},
            handler=list_files,
        ),
    ]
