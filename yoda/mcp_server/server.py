"""MCP Server — expose Yoda as an MCP tool server for Claude Code integration.

Implements the Model Context Protocol using JSON-RPC over SSE (Server-Sent Events).
Tools exposed: remember, recall, graph_query, get_preferences.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

def jsonrpc_response(id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def jsonrpc_error(id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id, "error": err}


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

MCP_TOOLS = [
    {
        "name": "remember",
        "description": "Store information in Yoda's long-term memory. Use for facts, preferences, or important user details.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Information to remember"},
                "importance": {"type": "number", "description": "Importance 0.0-1.0", "default": 0.5},
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall",
        "description": "Search Yoda's memory for relevant information about a topic.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "graph_query",
        "description": "Query Yoda's knowledge graph about entities and relationships.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Natural language question about known entities"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "get_preferences",
        "description": "Get learned user preferences from Yoda's knowledge base.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Category filter (e.g., coding, tools, general)",
                    "default": "all",
                },
            },
        },
    },
]


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

@dataclass
class SSEClient:
    """Connected SSE client."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    connected_at: float = field(default_factory=time.time)


class YodaMCPServer:
    """MCP Server that exposes Yoda's memory and knowledge graph as tools.

    Transport: SSE (Server-Sent Events) with JSON-RPC 2.0
    Tools: remember, recall, graph_query, get_preferences
    """

    def __init__(self, orchestrator: Any, host: str = "localhost", port: int = 8765) -> None:
        self.orchestrator = orchestrator
        self.host = host
        self.port = port
        self._clients: dict[str, SSEClient] = {}
        self._server: Any = None

    # -- Tool execution ----------------------------------------------------

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute an MCP tool by routing to the appropriate plugin."""
        if name == "remember":
            return await self._tool_remember(arguments)
        elif name == "recall":
            return await self._tool_recall(arguments)
        elif name == "graph_query":
            return await self._tool_graph_query(arguments)
        elif name == "get_preferences":
            return await self._tool_get_preferences(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _tool_remember(self, args: dict[str, Any]) -> str:
        plugin = self.orchestrator.get_plugin("memory")
        if not plugin:
            return "Memory system not available"
        return await plugin.execute("memory_store", {
            "content": args["content"],
            "importance": args.get("importance", 0.5),
        })

    async def _tool_recall(self, args: dict[str, Any]) -> str:
        plugin = self.orchestrator.get_plugin("memory")
        if not plugin:
            return "Memory system not available"
        return await plugin.execute("memory_search", {
            "query": args["query"],
            "top_k": args.get("top_k", 5),
        })

    async def _tool_graph_query(self, args: dict[str, Any]) -> str:
        plugin = self.orchestrator.get_plugin("knowledge_graph")
        if not plugin:
            return "Knowledge graph not available"
        return await plugin.execute("kg_query", {
            "question": args["question"],
        })

    async def _tool_get_preferences(self, args: dict[str, Any]) -> str:
        plugin = self.orchestrator.get_plugin("memory")
        if not plugin:
            return "Memory system not available"
        category = args.get("category", "all")
        query = f"user preferences {category}" if category != "all" else "user preferences"
        return await plugin.execute("memory_search", {
            "query": query,
            "collection": "preferences",
            "top_k": 10,
        })

    # -- JSON-RPC handling -------------------------------------------------

    async def handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a JSON-RPC 2.0 request."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        try:
            if method == "initialize":
                return jsonrpc_response(req_id, {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "yoda",
                        "version": "0.1.0",
                    },
                })

            elif method == "tools/list":
                return jsonrpc_response(req_id, {"tools": MCP_TOOLS})

            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                result = await self.execute_tool(tool_name, tool_args)
                return jsonrpc_response(req_id, {
                    "content": [{"type": "text", "text": str(result)}],
                })

            elif method == "notifications/initialized":
                # Client notification — no response needed
                return jsonrpc_response(req_id, {})

            elif method == "ping":
                return jsonrpc_response(req_id, {})

            else:
                return jsonrpc_error(req_id, -32601, f"Method not found: {method}")

        except Exception as e:
            logger.exception("MCP request failed: %s", method)
            return jsonrpc_error(req_id, -32603, str(e))

    # -- HTTP/SSE server ---------------------------------------------------

    async def start(self) -> None:
        """Start the SSE server."""
        try:
            from aiohttp import web
        except ImportError:
            logger.error("aiohttp required for MCP server: pip install aiohttp")
            return

        app = web.Application()
        app.router.add_get("/sse", self._handle_sse)
        app.router.add_post("/message", self._handle_message)
        app.router.add_get("/health", self._handle_health)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self._server = runner
        logger.info("MCP server started on http://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stop the SSE server."""
        if self._server:
            await self._server.cleanup()
            logger.info("MCP server stopped")

    async def _handle_sse(self, request: Any) -> Any:
        """SSE endpoint — maintains persistent connection for server-to-client events."""
        from aiohttp import web

        client = SSEClient()
        self._clients[client.id] = client

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Client-Id": client.id,
            },
        )
        await response.prepare(request)

        # Send endpoint info
        endpoint_event = f"event: endpoint\ndata: /message?clientId={client.id}\n\n"
        await response.write(endpoint_event.encode())

        try:
            while True:
                msg = await client.queue.get()
                await response.write(f"event: message\ndata: {msg}\n\n".encode())
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            self._clients.pop(client.id, None)

        return response

    async def _handle_message(self, request: Any) -> Any:
        """HTTP POST endpoint for JSON-RPC messages from client."""
        from aiohttp import web

        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                jsonrpc_error(None, -32700, "Parse error"),
                status=400,
            )

        client_id = request.query.get("clientId", "")
        response = await self.handle_jsonrpc(body)

        # If client is connected via SSE, also send through SSE
        if client_id and client_id in self._clients:
            await self._clients[client_id].queue.put(json.dumps(response))

        return web.json_response(response)

    async def _handle_health(self, request: Any) -> Any:
        """Health check endpoint."""
        from aiohttp import web

        return web.json_response({
            "status": "ok",
            "server": "yoda-mcp",
            "version": "0.1.0",
            "clients": len(self._clients),
            "tools": len(MCP_TOOLS),
        })


# ---------------------------------------------------------------------------
# Stdio transport (alternative for Claude Code)
# ---------------------------------------------------------------------------

class YodaMCPStdioServer:
    """MCP server using stdio transport (stdin/stdout JSON-RPC).

    This is the preferred transport for Claude Code integration.
    """

    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator
        self._sse_server = YodaMCPServer(orchestrator)

    async def run(self) -> None:
        """Run the stdio server, reading JSON-RPC from stdin and writing to stdout."""
        import sys

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_event_loop())

        logger.info("MCP stdio server started")

        try:
            while True:
                line = await reader.readline()
                if not line:
                    break

                line_str = line.decode().strip()
                if not line_str:
                    continue

                try:
                    request = json.loads(line_str)
                    response = await self._sse_server.handle_jsonrpc(request)
                    response_bytes = (json.dumps(response) + "\n").encode()
                    writer.write(response_bytes)
                    await writer.drain()
                except json.JSONDecodeError:
                    error = jsonrpc_error(None, -32700, "Parse error")
                    writer.write((json.dumps(error) + "\n").encode())
                    await writer.drain()
        except (asyncio.CancelledError, BrokenPipeError):
            pass

        logger.info("MCP stdio server stopped")
