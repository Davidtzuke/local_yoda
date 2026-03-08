"""File operation tools: read, write, list, search, copy, move, delete."""

from __future__ import annotations

import glob as glob_mod
import os
import shutil
from pathlib import Path

from yoda.tools.registry import ToolPermission, tool


def register_file_tools() -> None:
    """No-op — importing this module registers tools via decorators."""


@tool(
    name="read_file",
    permission=ToolPermission.READ,
    category="file",
    tags=["file", "read"],
    timeout=10.0,
)
async def read_file(path: str, encoding: str = "utf-8", max_lines: int = 0) -> str:
    """Read a file and return its contents.

    Args:
        path: Path to the file to read.
        encoding: File encoding (default utf-8).
        max_lines: Maximum lines to return (0 = all).
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")

    content = p.read_text(encoding=encoding)
    if max_lines > 0:
        lines = content.split("\n")
        content = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            content += f"\n... ({len(lines) - max_lines} more lines)"
    return content


@tool(
    name="write_file",
    permission=ToolPermission.WRITE,
    category="file",
    tags=["file", "write"],
    requires_approval=True,
    timeout=10.0,
)
async def write_file(path: str, content: str, encoding: str = "utf-8", append: bool = False) -> str:
    """Write content to a file, creating directories if needed.

    Args:
        path: Path to the file to write.
        content: Content to write.
        encoding: File encoding (default utf-8).
        append: If True, append instead of overwrite.
    """
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    p.open(mode, encoding=encoding).write(content)
    return f"Written {len(content)} chars to {p}"


@tool(
    name="list_directory",
    permission=ToolPermission.READ,
    category="file",
    tags=["file", "list", "directory"],
    timeout=10.0,
)
async def list_directory(
    path: str = ".",
    pattern: str = "*",
    recursive: bool = False,
    include_hidden: bool = False,
) -> str:
    """List files and directories.

    Args:
        path: Directory path (default: current directory).
        pattern: Glob pattern to filter (default: *).
        recursive: If True, list recursively.
        include_hidden: If True, include hidden files.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise ValueError(f"Not a directory: {p}")

    if recursive:
        entries = sorted(p.rglob(pattern))
    else:
        entries = sorted(p.glob(pattern))

    results: list[str] = []
    for entry in entries:
        if not include_hidden and entry.name.startswith("."):
            continue
        rel = entry.relative_to(p)
        suffix = "/" if entry.is_dir() else ""
        size = entry.stat().st_size if entry.is_file() else 0
        results.append(f"{rel}{suffix} ({size} bytes)" if entry.is_file() else f"{rel}/")

    if not results:
        return f"No entries matching '{pattern}' in {p}"
    return "\n".join(results[:500])  # Cap output


@tool(
    name="search_files",
    permission=ToolPermission.READ,
    category="file",
    tags=["file", "search", "grep"],
    timeout=30.0,
)
async def search_files(
    directory: str,
    pattern: str,
    file_pattern: str = "**/*",
    max_results: int = 50,
) -> str:
    """Search for a text pattern in files (like grep -r).

    Args:
        directory: Root directory to search.
        pattern: Text pattern to search for (case-insensitive).
        file_pattern: Glob pattern for files to search (default: all).
        max_results: Maximum number of matches to return.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    matches: list[str] = []
    pattern_lower = pattern.lower()

    for file_path in root.rglob(file_pattern):
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text(errors="ignore")
            for i, line in enumerate(text.split("\n"), 1):
                if pattern_lower in line.lower():
                    rel = file_path.relative_to(root)
                    matches.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(matches) >= max_results:
                        break
        except (PermissionError, OSError):
            continue
        if len(matches) >= max_results:
            break

    if not matches:
        return f"No matches for '{pattern}' in {root}"
    return "\n".join(matches)


@tool(
    name="copy_file",
    permission=ToolPermission.WRITE,
    category="file",
    tags=["file", "copy"],
    timeout=30.0,
)
async def copy_file(source: str, destination: str) -> str:
    """Copy a file or directory.

    Args:
        source: Source path.
        destination: Destination path.
    """
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return f"Copied {src} -> {dst}"


@tool(
    name="move_file",
    permission=ToolPermission.WRITE,
    category="file",
    tags=["file", "move", "rename"],
    requires_approval=True,
    timeout=10.0,
)
async def move_file(source: str, destination: str) -> str:
    """Move or rename a file or directory.

    Args:
        source: Source path.
        destination: Destination path.
    """
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return f"Moved {src} -> {dst}"


@tool(
    name="delete_file",
    permission=ToolPermission.DANGEROUS,
    category="file",
    tags=["file", "delete"],
    requires_approval=True,
    timeout=10.0,
)
async def delete_file(path: str, recursive: bool = False) -> str:
    """Delete a file or directory.

    Args:
        path: Path to delete.
        recursive: If True, delete directories recursively.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    if p.is_dir():
        if not recursive:
            raise ValueError(f"Use recursive=True to delete directory: {p}")
        shutil.rmtree(p)
    else:
        p.unlink()
    return f"Deleted {p}"


@tool(
    name="file_info",
    permission=ToolPermission.READ,
    category="file",
    tags=["file", "info", "stat"],
    timeout=5.0,
)
async def file_info(path: str) -> str:
    """Get file metadata (size, permissions, timestamps).

    Args:
        path: Path to the file.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    stat = p.stat()
    return (
        f"Path: {p}\n"
        f"Type: {'directory' if p.is_dir() else 'file'}\n"
        f"Size: {stat.st_size} bytes\n"
        f"Permissions: {oct(stat.st_mode)}\n"
        f"Modified: {stat.st_mtime}\n"
        f"Created: {stat.st_ctime}"
    )
