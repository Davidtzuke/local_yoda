"""MCP (Model Context Protocol) client for connecting to stdio and SSE MCP servers.

Supports:
- stdio transport (spawn subprocess, communicate via stdin/stdout JSON-RPC)
- SSE transport (HTTP Server-Sent Events)
- Auto-discovery of tools from connected servers
- Proxy discovered tools through Yoda's tool registry
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from yoda.core.plugins import ToolParameter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP protocol types
# ---------------------------------------------------------------------------

class MCPTransport(str, Enum):
    STDIO = "stdio"
    SSE = "sse"


class MCPToolSchema(BaseModel):
    """Tool schema as defined by MCP protocol."""
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server connection."""
    name: str
    transport: MCPTransport = MCPTransport.STDIO
    command: str = ""          # For stdio: command to run
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""              # For SSE: server URL
    headers: dict[str, str] = Field(default_factory=dict)
    auto_connect: bool = True


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

def _jsonrpc_request(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 request."""
    req: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
    }
    if params is not None:
        req["params"] = params
    return req


# ---------------------------------------------------------------------------
# Stdio transport
# ---------------------------------------------------------------------------

class StdioTransport:
    """MCP transport over subprocess stdin/stdout."""

    def __init__(self, command: str, args: list[str], env: dict[str, str] | None = None) -> None:
        self.command = command
        self.args = args
        self.env = env
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Start the subprocess."""
        import os
        full_env = {**os.environ, **(self.env or {})}
        self._process = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )
        logger.info("MCP stdio process started: %s %s", self.command, self.args)

    async def send(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and read the response."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("Transport not connected")

        async with self._lock:
            data = json.dumps(message) + "\n"
            self._process.stdin.write(data.encode())
            await self._process.stdin.drain()

            line = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=30.0,
            )
            if not line:
                raise RuntimeError("MCP server closed connection")

            return json.loads(line.decode())

    async def disconnect(self) -> None:
        """Terminate the subprocess."""
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
            self._process = None

    @property
    def is_connected(self) -> bool:
        return self._process is not None and self._process.returncode is None


# ---------------------------------------------------------------------------
# SSE transport
# ---------------------------------------------------------------------------

class SSETransport:
    """MCP transport over HTTP Server-Sent Events."""

    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self.url = url.rstrip("/")
        self.headers = headers or {}
        self._connected = False

    async def connect(self) -> None:
        """Verify server is reachable."""
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.url}/health", headers=self.headers)
            # Accept any 2xx status
            if 200 <= response.status_code < 300:
                self._connected = True
                logger.info("MCP SSE server connected: %s", self.url)
            else:
                # Try without /health endpoint
                self._connected = True
                logger.info("MCP SSE server assumed connected: %s", self.url)

    async def send(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request via HTTP POST."""
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.url}/rpc",
                json=message,
                headers={**self.headers, "Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()

    async def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# MCP Server Connection
# ---------------------------------------------------------------------------

class MCPConnection:
    """A connection to a single MCP server with tool discovery."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.tools: list[MCPToolSchema] = []
        self._transport: StdioTransport | SSETransport | None = None

    async def connect(self) -> None:
        """Establish connection and discover tools."""
        if self.config.transport == MCPTransport.STDIO:
            self._transport = StdioTransport(
                self.config.command,
                self.config.args,
                self.config.env,
            )
        else:
            self._transport = SSETransport(
                self.config.url,
                self.config.headers,
            )

        await self._transport.connect()

        # Initialize MCP session
        try:
            init_req = _jsonrpc_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "yoda", "version": "0.1.0"},
            })
            init_resp = await self._transport.send(init_req)
            logger.debug("MCP init response: %s", init_resp)

            # Send initialized notification
            notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            if isinstance(self._transport, StdioTransport):
                await self._transport.send(notif)
        except Exception as e:
            logger.warning("MCP initialization failed (non-fatal): %s", e)

        # Discover tools
        await self._discover_tools()

    async def _discover_tools(self) -> None:
        """List tools from the MCP server."""
        if not self._transport:
            return
        try:
            req = _jsonrpc_request("tools/list")
            resp = await self._transport.send(req)
            result = resp.get("result", {})
            raw_tools = result.get("tools", [])
            self.tools = [MCPToolSchema(**t) for t in raw_tools]
            logger.info(
                "Discovered %d tools from MCP server '%s': %s",
                len(self.tools),
                self.config.name,
                [t.name for t in self.tools],
            )
        except Exception as e:
            logger.warning("Failed to discover tools from %s: %s", self.config.name, e)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if not self._transport:
            raise RuntimeError(f"Not connected to {self.config.name}")

        req = _jsonrpc_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        resp = await self._transport.send(req)

        if "error" in resp:
            error = resp["error"]
            raise RuntimeError(f"MCP tool error: {error.get('message', str(error))}")

        result = resp.get("result", {})
        # MCP returns content array
        content = result.get("content", [])
        if content:
            texts = [c.get("text", str(c)) for c in content if isinstance(c, dict)]
            return "\n".join(texts) if texts else str(content)
        return result

    async def disconnect(self) -> None:
        if self._transport:
            await self._transport.disconnect()
            self._transport = None

    @property
    def is_connected(self) -> bool:
        return self._transport is not None and self._transport.is_connected


# ---------------------------------------------------------------------------
# MCP Client Manager
# ---------------------------------------------------------------------------

class MCPClient:
    """Manages multiple MCP server connections and proxies their tools.

    Usage:
        client = MCPClient()
        client.add_server(MCPServerConfig(
            name="filesystem",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        ))
        await client.connect_all()
        tools = client.get_all_tool_schemas()
    """

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._tool_to_server: dict[str, str] = {}

    def add_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server configuration."""
        self._connections[config.name] = MCPConnection(config)

    async def connect_all(self) -> None:
        """Connect to all configured servers."""
        tasks = []
        for name, conn in self._connections.items():
            if conn.config.auto_connect:
                tasks.append(self._connect_one(name, conn))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _connect_one(self, name: str, conn: MCPConnection) -> None:
        try:
            await conn.connect()
            # Map tools to server
            for tool in conn.tools:
                qualified = f"{name}__{tool.name}"
                self._tool_to_server[qualified] = name
                self._tool_to_server[tool.name] = name
        except Exception as e:
            logger.error("Failed to connect to MCP server '%s': %s", name, e)

    async def disconnect_all(self) -> None:
        """Disconnect all servers."""
        for conn in self._connections.values():
            try:
                await conn.disconnect()
            except Exception as e:
                logger.warning("Error disconnecting %s: %s", conn.config.name, e)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool, routing to the correct MCP server."""
        server_name = self._tool_to_server.get(name)
        if not server_name:
            raise ValueError(f"Unknown MCP tool: {name}")

        conn = self._connections.get(server_name)
        if not conn or not conn.is_connected:
            raise RuntimeError(f"MCP server '{server_name}' not connected")

        # Strip server prefix if present
        tool_name = name.split("__", 1)[-1] if "__" in name else name
        return await conn.call_tool(tool_name, arguments)

    def get_all_tool_schemas(self) -> list[tuple[str, MCPToolSchema]]:
        """Return (server_name, tool_schema) for all discovered tools."""
        result: list[tuple[str, MCPToolSchema]] = []
        for name, conn in self._connections.items():
            for tool in conn.tools:
                result.append((name, tool))
        return result

    def get_yoda_tool_parameters(self, mcp_schema: MCPToolSchema) -> list[ToolParameter]:
        """Convert MCP input_schema to Yoda ToolParameter list."""
        params: list[ToolParameter] = []
        properties = mcp_schema.input_schema.get("properties", {})
        required = set(mcp_schema.input_schema.get("required", []))

        for prop_name, prop_def in properties.items():
            params.append(ToolParameter(
                name=prop_name,
                type=prop_def.get("type", "string"),
                description=prop_def.get("description", ""),
                required=prop_name in required,
                default=prop_def.get("default"),
            ))
        return params

    @property
    def connected_servers(self) -> list[str]:
        return [n for n, c in self._connections.items() if c.is_connected]

    @property
    def all_tool_names(self) -> list[str]:
        return list(self._tool_to_server.keys())
