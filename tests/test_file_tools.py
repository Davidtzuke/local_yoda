"""Tests for yoda.tools.file_tools."""

from __future__ import annotations

import pytest
import pytest_asyncio

from pathlib import Path

from yoda.tools.file_tools import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    SearchFilesTool,
    EditFileTool,
    create_file_tools,
    register_file_tools,
    _resolve_and_sandbox,
)
from yoda.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a workspace with sample files."""
    (tmp_path / "hello.txt").write_text("Hello, world!")
    (tmp_path / "data.json").write_text('{"key": "value"}')
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "notes.txt").write_text("line one\nline two\nline three")
    (sub / "code.py").write_text("def foo():\n    return 42\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Sandbox tests
# ---------------------------------------------------------------------------

class TestSandbox:
    def test_resolve_relative(self, workspace: Path) -> None:
        result = _resolve_and_sandbox("hello.txt", workspace)
        assert result == workspace / "hello.txt"

    def test_resolve_absolute_inside(self, workspace: Path) -> None:
        p = str(workspace / "hello.txt")
        result = _resolve_and_sandbox(p, workspace)
        assert result == workspace / "hello.txt"

    def test_reject_escape(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="outside the workspace"):
            _resolve_and_sandbox("../../etc/passwd", workspace)

    def test_reject_system_dir(self, tmp_path: Path) -> None:
        # If workspace is somehow at root, block system dirs
        with pytest.raises(ValueError, match="outside the workspace"):
            _resolve_and_sandbox("/etc/passwd", tmp_path)

    def test_reject_symlink_escape(self, workspace: Path) -> None:
        link = workspace / "sneaky"
        link.symlink_to("/tmp")
        with pytest.raises(ValueError, match="outside the workspace"):
            _resolve_and_sandbox("sneaky/something", workspace)


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------

class TestReadFileTool:
    @pytest.mark.asyncio
    async def test_read_existing(self, workspace: Path) -> None:
        tool = ReadFileTool(workspace)
        result = await tool.execute(path="hello.txt")
        assert result.success
        assert result.data == "Hello, world!"
        assert result.metadata["size"] == 13

    @pytest.mark.asyncio
    async def test_read_missing(self, workspace: Path) -> None:
        tool = ReadFileTool(workspace)
        result = await tool.execute(path="nope.txt")
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_directory_fails(self, workspace: Path) -> None:
        tool = ReadFileTool(workspace)
        result = await tool.execute(path="subdir")
        assert not result.success
        assert "Not a file" in result.error

    @pytest.mark.asyncio
    async def test_read_sandbox_escape(self, workspace: Path) -> None:
        tool = ReadFileTool(workspace)
        result = await tool.execute(path="../../etc/passwd")
        assert not result.success
        assert "outside" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_absolute_path(self, workspace: Path) -> None:
        tool = ReadFileTool(workspace)
        abs_path = str(workspace / "data.json")
        result = await tool.execute(path=abs_path)
        assert result.success
        assert '"key"' in result.data


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------

class TestWriteFileTool:
    @pytest.mark.asyncio
    async def test_write_new(self, workspace: Path) -> None:
        tool = WriteFileTool(workspace)
        result = await tool.execute(path="new.txt", content="fresh content")
        assert result.success
        assert (workspace / "new.txt").read_text() == "fresh content"

    @pytest.mark.asyncio
    async def test_write_overwrite(self, workspace: Path) -> None:
        tool = WriteFileTool(workspace)
        result = await tool.execute(path="hello.txt", content="overwritten")
        assert result.success
        assert (workspace / "hello.txt").read_text() == "overwritten"

    @pytest.mark.asyncio
    async def test_write_creates_parents(self, workspace: Path) -> None:
        tool = WriteFileTool(workspace)
        result = await tool.execute(path="a/b/c/deep.txt", content="deep")
        assert result.success
        assert (workspace / "a" / "b" / "c" / "deep.txt").read_text() == "deep"

    @pytest.mark.asyncio
    async def test_write_sandbox_escape(self, workspace: Path) -> None:
        tool = WriteFileTool(workspace)
        result = await tool.execute(path="/etc/evil.txt", content="bad")
        assert not result.success


# ---------------------------------------------------------------------------
# ListDirectoryTool
# ---------------------------------------------------------------------------

class TestListDirectoryTool:
    @pytest.mark.asyncio
    async def test_list_root(self, workspace: Path) -> None:
        tool = ListDirectoryTool(workspace)
        result = await tool.execute(path=".")
        assert result.success
        names = [e["name"] for e in result.data]
        assert "hello.txt" in names
        assert "subdir" in names

    @pytest.mark.asyncio
    async def test_list_with_pattern(self, workspace: Path) -> None:
        tool = ListDirectoryTool(workspace)
        result = await tool.execute(path=".", pattern="*.txt")
        assert result.success
        names = [e["name"] for e in result.data]
        assert "hello.txt" in names
        assert "data.json" not in names

    @pytest.mark.asyncio
    async def test_list_subdirectory(self, workspace: Path) -> None:
        tool = ListDirectoryTool(workspace)
        result = await tool.execute(path="subdir")
        assert result.success
        names = [e["name"] for e in result.data]
        assert "notes.txt" in names
        assert "code.py" in names

    @pytest.mark.asyncio
    async def test_list_missing(self, workspace: Path) -> None:
        tool = ListDirectoryTool(workspace)
        result = await tool.execute(path="nonexistent")
        assert not result.success

    @pytest.mark.asyncio
    async def test_list_file_as_dir(self, workspace: Path) -> None:
        tool = ListDirectoryTool(workspace)
        result = await tool.execute(path="hello.txt")
        assert not result.success
        assert "Not a directory" in result.error

    @pytest.mark.asyncio
    async def test_entry_types(self, workspace: Path) -> None:
        tool = ListDirectoryTool(workspace)
        result = await tool.execute(path=".")
        assert result.success
        for entry in result.data:
            if entry["name"] == "subdir":
                assert entry["type"] == "directory"
            elif entry["name"] == "hello.txt":
                assert entry["type"] == "file"
                assert entry["size"] == 13


# ---------------------------------------------------------------------------
# SearchFilesTool
# ---------------------------------------------------------------------------

class TestSearchFilesTool:
    @pytest.mark.asyncio
    async def test_glob_all_txt(self, workspace: Path) -> None:
        tool = SearchFilesTool(workspace)
        result = await tool.execute(glob_pattern="**/*.txt")
        assert result.success
        paths = [m["path"] for m in result.data]
        assert any("hello.txt" in p for p in paths)
        assert any("notes.txt" in p for p in paths)

    @pytest.mark.asyncio
    async def test_glob_with_content(self, workspace: Path) -> None:
        tool = SearchFilesTool(workspace)
        result = await tool.execute(glob_pattern="**/*.txt", content_query="line two")
        assert result.success
        assert len(result.data) == 1
        assert result.data[0]["matches"][0]["line"] == 2

    @pytest.mark.asyncio
    async def test_glob_with_regex(self, workspace: Path) -> None:
        tool = SearchFilesTool(workspace)
        result = await tool.execute(
            glob_pattern="**/*.py", content_query=r"def \w+\(\)", regex=True
        )
        assert result.success
        assert len(result.data) == 1

    @pytest.mark.asyncio
    async def test_invalid_regex(self, workspace: Path) -> None:
        tool = SearchFilesTool(workspace)
        result = await tool.execute(
            glob_pattern="**/*.py", content_query="[invalid", regex=True
        )
        assert not result.success
        assert "regex" in result.error.lower()

    @pytest.mark.asyncio
    async def test_max_results(self, workspace: Path) -> None:
        tool = SearchFilesTool(workspace)
        result = await tool.execute(glob_pattern="**/*", max_results=2)
        assert result.success
        assert len(result.data) <= 2

    @pytest.mark.asyncio
    async def test_no_content_match(self, workspace: Path) -> None:
        tool = SearchFilesTool(workspace)
        result = await tool.execute(
            glob_pattern="**/*.txt", content_query="XYZNOEXIST"
        )
        assert result.success
        assert len(result.data) == 0


# ---------------------------------------------------------------------------
# EditFileTool
# ---------------------------------------------------------------------------

class TestEditFileTool:
    @pytest.mark.asyncio
    async def test_single_edit(self, workspace: Path) -> None:
        tool = EditFileTool(workspace)
        result = await tool.execute(
            path="hello.txt",
            edits=[{"old_text": "world", "new_text": "Yoda"}],
        )
        assert result.success
        assert (workspace / "hello.txt").read_text() == "Hello, Yoda!"

    @pytest.mark.asyncio
    async def test_multiple_edits(self, workspace: Path) -> None:
        tool = EditFileTool(workspace)
        result = await tool.execute(
            path="subdir/notes.txt",
            edits=[
                {"old_text": "line one", "new_text": "LINE 1"},
                {"old_text": "line three", "new_text": "LINE 3"},
            ],
        )
        assert result.success
        content = (workspace / "subdir" / "notes.txt").read_text()
        assert "LINE 1" in content
        assert "line two" in content
        assert "LINE 3" in content

    @pytest.mark.asyncio
    async def test_replace_all(self, workspace: Path) -> None:
        (workspace / "repeat.txt").write_text("aaa bbb aaa ccc aaa")
        tool = EditFileTool(workspace)
        result = await tool.execute(
            path="repeat.txt",
            edits=[{"old_text": "aaa", "new_text": "XXX", "replace_all": True}],
        )
        assert result.success
        assert (workspace / "repeat.txt").read_text() == "XXX bbb XXX ccc XXX"

    @pytest.mark.asyncio
    async def test_old_text_not_found(self, workspace: Path) -> None:
        tool = EditFileTool(workspace)
        result = await tool.execute(
            path="hello.txt",
            edits=[{"old_text": "NOPE", "new_text": "X"}],
        )
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_file(self, workspace: Path) -> None:
        tool = EditFileTool(workspace)
        result = await tool.execute(
            path="ghost.txt",
            edits=[{"old_text": "a", "new_text": "b"}],
        )
        assert not result.success

    @pytest.mark.asyncio
    async def test_empty_edits(self, workspace: Path) -> None:
        tool = EditFileTool(workspace)
        result = await tool.execute(path="hello.txt", edits=[])
        assert not result.success


# ---------------------------------------------------------------------------
# Integration / registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_create_file_tools(self, workspace: Path) -> None:
        tools = create_file_tools(workspace)
        assert len(tools) == 5
        names = {t.name for t in tools}
        assert names == {"read_file", "write_file", "list_directory", "search_files", "edit_file"}

    def test_register_file_tools(self, workspace: Path) -> None:
        registry = ToolRegistry()
        tools = register_file_tools(registry, workspace)
        assert len(tools) == 5
        assert "read_file" in registry
        assert "edit_file" in registry

    @pytest.mark.asyncio
    async def test_engine_integration(self, workspace: Path) -> None:
        from yoda.tools.engine import ToolEngine

        registry = ToolRegistry()
        register_file_tools(registry, workspace)
        engine = ToolEngine(registry)

        result = await engine.run("read_file", path="hello.txt")
        assert result.success
        assert result.data == "Hello, world!"
