"""Tests for the Yoda tool framework."""

import asyncio
import pytest
from yoda.tools import (
    Tool,
    ToolParameter,
    ToolResult,
    ToolCapability,
    ToolRegistry,
    ToolEngine,
    tool,
)
from yoda.tools.base import ParameterType


# --- Concrete tool for testing ---

class EchoTool(Tool):
    name = "echo"
    description = "Echoes back the input message."
    parameters = [
        ToolParameter("message", "The message to echo"),
        ToolParameter("shout", "Uppercase output", ParameterType.BOOLEAN, required=False, default=False),
    ]
    capabilities = ToolCapability()

    async def execute(self, **kwargs) -> ToolResult:
        msg = kwargs["message"]
        if kwargs.get("shout"):
            msg = msg.upper()
        return ToolResult.ok(msg)


class SlowTool(Tool):
    name = "slow"
    description = "Sleeps forever."
    parameters = []

    async def execute(self, **kwargs) -> ToolResult:
        await asyncio.sleep(100)
        return ToolResult.ok("done")


class BrokenTool(Tool):
    name = "broken"
    description = "Always raises."
    parameters = []

    async def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("kaboom")


class FileTool(Tool):
    name = "file_reader"
    description = "Reads files."
    parameters = []
    capabilities = ToolCapability(reads_filesystem=True)

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult.ok("file contents")


# --- ToolResult ---

def test_result_ok():
    r = ToolResult.ok("hello", source="test")
    assert r.success is True
    assert r.data == "hello"
    assert r.error is None
    assert r.metadata == {"source": "test"}


def test_result_fail():
    r = ToolResult.fail("oops")
    assert r.success is False
    assert r.error == "oops"


def test_result_to_dict():
    r = ToolResult.ok(42)
    d = r.to_dict()
    assert d == {"success": True, "data": 42}


# --- Tool ---

def test_tool_validate_params():
    t = EchoTool()
    assert t.validate_params({"message": "hi"}) == []
    assert t.validate_params({}) == ["Missing required parameter: 'message'"]
    assert "Unknown parameter: 'foo'" in t.validate_params({"message": "hi", "foo": 1})


def test_tool_apply_defaults():
    t = EchoTool()
    result = t.apply_defaults({"message": "hi"})
    assert result == {"message": "hi", "shout": False}


def test_tool_manifest():
    t = EchoTool()
    m = t.manifest()
    assert m["name"] == "echo"
    assert len(m["parameters"]) == 2
    assert m["capabilities"]["reads_filesystem"] is False


# --- ToolRegistry ---

def test_registry_register_and_get():
    reg = ToolRegistry()
    reg.register(EchoTool())
    assert "echo" in reg
    assert reg.get("echo").name == "echo"


def test_registry_duplicate_raises():
    reg = ToolRegistry()
    reg.register(EchoTool())
    with pytest.raises(ValueError):
        reg.register(EchoTool())


def test_registry_get_missing_raises():
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


def test_registry_list_and_manifest():
    reg = ToolRegistry()
    reg.register_all([EchoTool(), SlowTool()])
    assert reg.list_names() == ["echo", "slow"]
    assert len(reg.manifest()) == 2


def test_registry_unregister():
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.unregister("echo")
    assert "echo" not in reg


# --- @tool decorator ---

@tool(
    name="greet",
    description="Greet someone",
    parameters=[ToolParameter("user", "Name to greet")],
)
async def greet(user: str) -> ToolResult:
    return ToolResult.ok(f"Hello, {user}!")


@pytest.mark.asyncio
async def test_decorator_tool():
    t = greet()  # instantiate the class
    assert t.name == "greet"
    result = await t.execute(user="Alice")
    assert result.success
    assert result.data == "Hello, Alice!"


# --- ToolEngine ---

@pytest.mark.asyncio
async def test_engine_run_success():
    reg = ToolRegistry()
    reg.register(EchoTool())
    engine = ToolEngine(reg)
    result = await engine.run("echo", message="hi")
    assert result.success
    assert result.data == "hi"
    assert "elapsed" in result.metadata


@pytest.mark.asyncio
async def test_engine_run_validation_error():
    reg = ToolRegistry()
    reg.register(EchoTool())
    engine = ToolEngine(reg)
    result = await engine.run("echo")  # missing message
    assert not result.success
    assert "Validation failed" in result.error


@pytest.mark.asyncio
async def test_engine_run_unknown_tool():
    engine = ToolEngine(ToolRegistry())
    result = await engine.run("nope")
    assert not result.success
    assert "not found" in result.error


@pytest.mark.asyncio
async def test_engine_timeout():
    reg = ToolRegistry()
    reg.register(SlowTool())
    engine = ToolEngine(reg, default_timeout=0.1)
    result = await engine.run("slow")
    assert not result.success
    assert "timed out" in result.error


@pytest.mark.asyncio
async def test_engine_exception_handling():
    reg = ToolRegistry()
    reg.register(BrokenTool())
    engine = ToolEngine(reg)
    result = await engine.run("broken")
    assert not result.success
    assert "kaboom" in result.error


@pytest.mark.asyncio
async def test_engine_sandbox_block():
    reg = ToolRegistry()
    reg.register(FileTool())
    # Only allow network, not filesystem
    engine = ToolEngine(
        reg,
        allowed_capabilities=ToolCapability(network_access=True),
    )
    result = await engine.run("file_reader")
    assert not result.success
    assert "Sandbox violation" in result.error


@pytest.mark.asyncio
async def test_engine_sandbox_allow():
    reg = ToolRegistry()
    reg.register(FileTool())
    engine = ToolEngine(
        reg,
        allowed_capabilities=ToolCapability(reads_filesystem=True),
    )
    result = await engine.run("file_reader")
    assert result.success


@pytest.mark.asyncio
async def test_engine_run_many():
    reg = ToolRegistry()
    reg.register(EchoTool())
    engine = ToolEngine(reg)
    results = await engine.run_many([
        {"name": "echo", "kwargs": {"message": "a"}},
        {"name": "echo", "kwargs": {"message": "b"}},
    ])
    assert len(results) == 2
    assert all(r.success for r in results)
    assert {r.data for r in results} == {"a", "b"}
