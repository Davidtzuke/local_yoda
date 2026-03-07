# Yoda - Local Personal AI Agent

## About This Project
Yoda is a Python-based local personal AI assistant with memory, tools (file system, web search), a CLI interface, and a web chat UI. It runs entirely locally and uses async throughout.

## Tech Stack
- **Python 3.12+** with full async/await patterns
- **Tool Framework**: Abstract base classes (`Tool`, `ToolResult`, `ToolParameter`, `ToolCapability`) in `yoda/tools/base.py`, a `ToolRegistry` with `@tool` decorator in `registry.py`, and a `ToolEngine` with validation/timeouts/sandboxing in `engine.py`
- **File Tools**: 5 filesystem tools built on the framework in `yoda/tools/file_tools.py`
- **Testing**: pytest + pytest-asyncio, 55 tests total

## What This Branch Does
Implements the extensible tool framework and file system tools for Yoda. The tool framework provides base classes, a registry, and an execution engine with parameter validation, timeouts, and capability-based sandboxing. The file tools provide `read_file`, `write_file`, `list_directory`, `search_files` (glob + content grep with regex), and `edit_file` (diff-based find-and-replace). All file operations are path-sandboxed to a configurable workspace, blocking symlink escapes and system directory access. Async file I/O uses `asyncio.run_in_executor`.

## Key Files
- `yoda/tools/base.py` - Tool ABC, ToolResult, ToolParameter, ToolCapability
- `yoda/tools/registry.py` - ToolRegistry, @tool decorator, default_registry
- `yoda/tools/engine.py` - ToolEngine with validation, timeouts, sandboxing
- `yoda/tools/file_tools.py` - ReadFileTool, WriteFileTool, ListDirectoryTool, SearchFilesTool, EditFileTool
- `yoda/tools/__init__.py` - Public API re-exports
- `tests/test_tools.py` - 20 framework tests
- `tests/test_file_tools.py` - 35 file tools tests
- `pyproject.toml` - Project config with dev/web optional deps
