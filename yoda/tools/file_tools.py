"""File system tools for the Yoda agent.

Provides read_file, write_file, list_directory, search_files (glob + grep),
and edit_file (diff-based editing). All operations are path-sandboxed to a
configurable workspace root.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from yoda.tools.base import (
    ParameterType,
    Tool,
    ToolCapability,
    ToolParameter,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Blocked system directories (never allow reads/writes here)
# ---------------------------------------------------------------------------

_BLOCKED_PREFIXES: tuple[str, ...] = (
    "/bin",
    "/sbin",
    "/usr/bin",
    "/usr/sbin",
    "/usr/lib",
    "/usr/local/bin",
    "/usr/local/sbin",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
    "/etc",
    "/var/run",
    "/var/lock",
    "/lib",
    "/lib64",
)


def _resolve_and_sandbox(path_str: str, workspace: Path) -> Path:
    """Resolve *path_str* relative to *workspace* and ensure it stays inside.

    Raises ``ValueError`` on sandbox escape or blocked system paths.
    """
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()

    # Must remain under the workspace root
    workspace_resolved = workspace.resolve()
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError:
        raise ValueError(
            f"Path '{resolved}' is outside the workspace '{workspace_resolved}'"
        ) from None

    # Must not target blocked system directories
    resolved_str = str(resolved)
    for prefix in _BLOCKED_PREFIXES:
        if resolved_str == prefix or resolved_str.startswith(prefix + "/"):
            raise ValueError(
                f"Access denied: '{resolved}' is inside system directory '{prefix}'"
            )

    return resolved


# ---------------------------------------------------------------------------
# Helpers: async file I/O via threadpool
# ---------------------------------------------------------------------------

async def _read_text(path: Path, encoding: str = "utf-8") -> str:
    """Read file contents asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, path.read_text, encoding)


async def _write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write content to a file asynchronously, creating parent dirs."""
    loop = asyncio.get_running_loop()

    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)

    await loop.run_in_executor(None, _write)


async def _listdir(path: Path) -> list[dict[str, Any]]:
    """List directory contents asynchronously."""
    loop = asyncio.get_running_loop()

    def _scan() -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for entry in sorted(path.iterdir()):
            info: dict[str, Any] = {
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
            }
            if entry.is_file():
                try:
                    info["size"] = entry.stat().st_size
                except OSError:
                    info["size"] = None
            entries.append(info)
        return entries

    return await loop.run_in_executor(None, _scan)


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------

class ReadFileTool(Tool):
    """Read the contents of a file within the workspace."""

    name = "read_file"
    description = "Read the text contents of a file at the given path."
    parameters = [
        ToolParameter(
            name="path",
            description="File path (absolute or relative to workspace).",
        ),
        ToolParameter(
            name="encoding",
            description="Text encoding to use.",
            required=False,
            default="utf-8",
        ),
    ]
    capabilities = ToolCapability(reads_filesystem=True)

    def __init__(self, workspace: str | Path = ".") -> None:
        self.workspace = Path(workspace).resolve()

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs["path"]
        encoding: str = kwargs.get("encoding", "utf-8")
        try:
            resolved = _resolve_and_sandbox(path_str, self.workspace)
        except ValueError as exc:
            return ToolResult.fail(str(exc))

        if not resolved.exists():
            return ToolResult.fail(f"File not found: '{resolved}'")
        if not resolved.is_file():
            return ToolResult.fail(f"Not a file: '{resolved}'")

        try:
            content = await _read_text(resolved, encoding=encoding)
        except (OSError, UnicodeDecodeError) as exc:
            return ToolResult.fail(f"Read error: {exc}")

        return ToolResult.ok(content, path=str(resolved), size=len(content))


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------

class WriteFileTool(Tool):
    """Write (create or overwrite) a file within the workspace."""

    name = "write_file"
    description = "Write text content to a file, creating parent directories as needed."
    parameters = [
        ToolParameter(
            name="path",
            description="File path (absolute or relative to workspace).",
        ),
        ToolParameter(
            name="content",
            description="Text content to write.",
        ),
        ToolParameter(
            name="encoding",
            description="Text encoding to use.",
            required=False,
            default="utf-8",
        ),
    ]
    capabilities = ToolCapability(reads_filesystem=True, writes_filesystem=True)

    def __init__(self, workspace: str | Path = ".") -> None:
        self.workspace = Path(workspace).resolve()

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs["path"]
        content: str = kwargs["content"]
        encoding: str = kwargs.get("encoding", "utf-8")

        try:
            resolved = _resolve_and_sandbox(path_str, self.workspace)
        except ValueError as exc:
            return ToolResult.fail(str(exc))

        try:
            await _write_text(resolved, content, encoding=encoding)
        except OSError as exc:
            return ToolResult.fail(f"Write error: {exc}")

        return ToolResult.ok(
            f"Wrote {len(content)} characters to {resolved}",
            path=str(resolved),
            size=len(content),
        )


# ---------------------------------------------------------------------------
# ListDirectoryTool
# ---------------------------------------------------------------------------

class ListDirectoryTool(Tool):
    """List the contents of a directory within the workspace."""

    name = "list_directory"
    description = "List files and subdirectories in a directory, with optional name pattern filtering."
    parameters = [
        ToolParameter(
            name="path",
            description="Directory path (absolute or relative to workspace). Defaults to workspace root.",
            required=False,
            default=".",
        ),
        ToolParameter(
            name="pattern",
            description="Optional glob/fnmatch pattern to filter entries (e.g. '*.py').",
            required=False,
        ),
    ]
    capabilities = ToolCapability(reads_filesystem=True)

    def __init__(self, workspace: str | Path = ".") -> None:
        self.workspace = Path(workspace).resolve()

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs.get("path", ".")
        pattern: str | None = kwargs.get("pattern")

        try:
            resolved = _resolve_and_sandbox(path_str, self.workspace)
        except ValueError as exc:
            return ToolResult.fail(str(exc))

        if not resolved.exists():
            return ToolResult.fail(f"Directory not found: '{resolved}'")
        if not resolved.is_dir():
            return ToolResult.fail(f"Not a directory: '{resolved}'")

        try:
            entries = await _listdir(resolved)
        except PermissionError as exc:
            return ToolResult.fail(f"Permission denied: {exc}")
        except OSError as exc:
            return ToolResult.fail(f"OS error: {exc}")

        if pattern:
            entries = [e for e in entries if fnmatch.fnmatch(e["name"], pattern)]

        return ToolResult.ok(entries, path=str(resolved), count=len(entries))


# ---------------------------------------------------------------------------
# SearchFilesTool  (glob + content grep)
# ---------------------------------------------------------------------------

class SearchFilesTool(Tool):
    """Search for files by glob pattern and optionally grep their contents."""

    name = "search_files"
    description = (
        "Search for files matching a glob pattern within the workspace. "
        "Optionally filter by content with a regex or plain-text query."
    )
    parameters = [
        ToolParameter(
            name="glob_pattern",
            description="Glob pattern to match files (e.g. '**/*.py', 'src/**/*.json').",
        ),
        ToolParameter(
            name="content_query",
            description="Optional text or regex to search inside matched files.",
            required=False,
        ),
        ToolParameter(
            name="regex",
            description="If true, treat content_query as a regex. Default false.",
            param_type=ParameterType.BOOLEAN,
            required=False,
            default=False,
        ),
        ToolParameter(
            name="max_results",
            description="Maximum number of matching files to return.",
            param_type=ParameterType.INTEGER,
            required=False,
            default=100,
        ),
    ]
    capabilities = ToolCapability(reads_filesystem=True)

    def __init__(self, workspace: str | Path = ".") -> None:
        self.workspace = Path(workspace).resolve()

    async def execute(self, **kwargs: Any) -> ToolResult:
        glob_pattern: str = kwargs["glob_pattern"]
        content_query: str | None = kwargs.get("content_query")
        use_regex: bool = kwargs.get("regex", False)
        max_results: int = kwargs.get("max_results", 100)

        loop = asyncio.get_running_loop()

        def _search() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            compiled_re: re.Pattern[str] | None = None

            if content_query and use_regex:
                try:
                    compiled_re = re.compile(content_query)
                except re.error as exc:
                    raise ValueError(f"Invalid regex: {exc}") from exc

            for path in sorted(self.workspace.glob(glob_pattern)):
                if len(results) >= max_results:
                    break
                if not path.is_file():
                    continue
                # Ensure within sandbox
                try:
                    _resolve_and_sandbox(str(path), self.workspace)
                except ValueError:
                    continue

                rel = path.relative_to(self.workspace)
                entry: dict[str, Any] = {
                    "path": str(rel),
                    "absolute_path": str(path),
                    "size": path.stat().st_size,
                }

                # Content grep
                if content_query:
                    try:
                        text = path.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue

                    matching_lines: list[dict[str, Any]] = []
                    for line_no, line in enumerate(text.splitlines(), start=1):
                        matched = False
                        if compiled_re:
                            matched = bool(compiled_re.search(line))
                        else:
                            matched = content_query in line

                        if matched:
                            matching_lines.append(
                                {"line": line_no, "text": line.rstrip()}
                            )
                            if len(matching_lines) >= 10:
                                break  # cap per file

                    if not matching_lines:
                        continue  # content filter active, no matches
                    entry["matches"] = matching_lines

                results.append(entry)

            return results

        try:
            matches = await loop.run_in_executor(None, _search)
        except ValueError as exc:
            return ToolResult.fail(str(exc))
        except OSError as exc:
            return ToolResult.fail(f"Search error: {exc}")

        return ToolResult.ok(matches, count=len(matches))


# ---------------------------------------------------------------------------
# EditFileTool  (diff-based editing)
# ---------------------------------------------------------------------------

class EditFileTool(Tool):
    """Edit a file by specifying text replacements (find-and-replace)."""

    name = "edit_file"
    description = (
        "Edit an existing file by providing a list of find-and-replace operations. "
        "Each operation replaces the first (or all) occurrences of old_text with new_text."
    )
    parameters = [
        ToolParameter(
            name="path",
            description="File path (absolute or relative to workspace).",
        ),
        ToolParameter(
            name="edits",
            description=(
                "List of edit operations. Each is a dict with keys: "
                "'old_text' (str), 'new_text' (str), and optional 'replace_all' (bool, default false)."
            ),
            param_type=ParameterType.LIST,
        ),
        ToolParameter(
            name="encoding",
            description="Text encoding to use.",
            required=False,
            default="utf-8",
        ),
    ]
    capabilities = ToolCapability(reads_filesystem=True, writes_filesystem=True)

    def __init__(self, workspace: str | Path = ".") -> None:
        self.workspace = Path(workspace).resolve()

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs["path"]
        edits: list[dict[str, Any]] = kwargs["edits"]
        encoding: str = kwargs.get("encoding", "utf-8")

        try:
            resolved = _resolve_and_sandbox(path_str, self.workspace)
        except ValueError as exc:
            return ToolResult.fail(str(exc))

        if not resolved.exists():
            return ToolResult.fail(f"File not found: '{resolved}'")
        if not resolved.is_file():
            return ToolResult.fail(f"Not a file: '{resolved}'")

        try:
            content = await _read_text(resolved, encoding=encoding)
        except (OSError, UnicodeDecodeError) as exc:
            return ToolResult.fail(f"Read error: {exc}")

        # Validate edits structure
        if not isinstance(edits, list) or not edits:
            return ToolResult.fail("'edits' must be a non-empty list of edit operations.")

        applied: list[str] = []
        modified = content

        for i, edit in enumerate(edits):
            old_text = edit.get("old_text")
            new_text = edit.get("new_text")
            if old_text is None or new_text is None:
                return ToolResult.fail(
                    f"Edit #{i + 1}: 'old_text' and 'new_text' are required."
                )
            if old_text == new_text:
                applied.append(f"Edit #{i + 1}: skipped (old_text == new_text)")
                continue

            if old_text not in modified:
                return ToolResult.fail(
                    f"Edit #{i + 1}: 'old_text' not found in file."
                )

            replace_all = edit.get("replace_all", False)
            if replace_all:
                count = modified.count(old_text)
                modified = modified.replace(old_text, new_text)
                applied.append(f"Edit #{i + 1}: replaced {count} occurrence(s)")
            else:
                modified = modified.replace(old_text, new_text, 1)
                applied.append(f"Edit #{i + 1}: replaced 1 occurrence")

        if modified == content:
            return ToolResult.ok("No changes made.", path=str(resolved))

        try:
            await _write_text(resolved, modified, encoding=encoding)
        except OSError as exc:
            return ToolResult.fail(f"Write error: {exc}")

        return ToolResult.ok(
            {"message": "File edited successfully.", "edits_applied": applied},
            path=str(resolved),
        )


# ---------------------------------------------------------------------------
# Convenience: register all file tools
# ---------------------------------------------------------------------------

def create_file_tools(workspace: str | Path = ".") -> list[Tool]:
    """Create instances of all file tools bound to the given workspace."""
    ws = Path(workspace)
    return [
        ReadFileTool(ws),
        WriteFileTool(ws),
        ListDirectoryTool(ws),
        SearchFilesTool(ws),
        EditFileTool(ws),
    ]


def register_file_tools(
    registry: "ToolRegistry",  # noqa: F821
    workspace: str | Path = ".",
) -> list[Tool]:
    """Create and register all file tools with the given registry.

    Returns the list of registered tool instances.
    """
    tools = create_file_tools(workspace)
    registry.register_all(tools)
    return tools
