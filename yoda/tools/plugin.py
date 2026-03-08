"""Plugin integration: exposes all tools to the Yoda agent via the Plugin interface."""

from __future__ import annotations

import logging
from typing import Any

from yoda.core.config import YodaConfig
from yoda.core.plugins import Plugin, ToolParameter, ToolSchema
from yoda.tools.executor import ApprovalPolicy, ExecutionResult, ToolExecutor
from yoda.tools.mcp_client import MCPClient, MCPServerConfig, MCPTransport
from yoda.tools.registry import ToolPermission, ToolRegistry

logger = logging.getLogger(__name__)


class ToolAccessPlugin(Plugin):
    """Plugin that provides tool & computer access to the Yoda agent.

    Integrates:
    - Built-in tools (file ops, shell, web, calendar, notes)
    - Computer access tools (screenshot, OCR, mouse/keyboard, app launcher)
    - MCP server tools (auto-discovered from connected servers)
    """

    name = "tool_access"
    version = "0.1.0"
    description = "Tool & Computer Access: file ops, shell, web, calendar, notes, MCP, computer control"

    def __init__(self, config: YodaConfig) -> None:
        super().__init__(config)
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(
            self.registry,
            approval_policy=ApprovalPolicy.REQUIRE_DANGEROUS,
        )
        self.mcp_client = MCPClient()
        self._tool_schemas: list[ToolSchema] = []

    async def on_load(self) -> None:
        """Load all built-in tools, computer tools, and connect MCP servers."""
        await super().on_load()

        # 1. Import built-in tools (triggers @tool decorators)
        try:
            from yoda.tools.builtins import register_all_builtins
            register_all_builtins()
        except Exception:
            logger.exception("Failed to load built-in tools")

        # 2. Import computer tools
        try:
            from yoda.tools.computer import register_all_computer_tools
            register_all_computer_tools()
        except Exception:
            logger.exception("Failed to load computer tools")

        # 3. Collect all @tool decorated functions
        self.registry.collect_decorated()
        logger.info("Registered %d tools from decorators", len(self.registry))

        # 4. Connect MCP servers from config
        await self._connect_mcp_servers()

        # 5. Build tool schemas
        self._build_schemas()

        logger.info(
            "ToolAccessPlugin loaded: %d tools (%d built-in, %d MCP)",
            len(self._tool_schemas),
            len(self.registry),
            len(self.mcp_client.all_tool_names),
        )

    async def _connect_mcp_servers(self) -> None:
        """Connect to MCP servers defined in config or environment."""
        # Check for MCP server configs in the data directory
        import json
        from pathlib import Path

        config_path = Path(self.config.data_dir).expanduser() / "mcp_servers.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    servers = json.load(f)
                for server_def in servers:
                    config = MCPServerConfig(**server_def)
                    self.mcp_client.add_server(config)
                await self.mcp_client.connect_all()

                # Register MCP tools in our registry
                for server_name, mcp_tool in self.mcp_client.get_all_tool_schemas():
                    params = self.mcp_client.get_yoda_tool_parameters(mcp_tool)
                    qualified_name = f"{server_name}__{mcp_tool.name}"

                    # Create a closure for the tool handler
                    async def _mcp_handler(
                        _name: str = qualified_name,
                        **kwargs: Any,
                    ) -> Any:
                        return await self.mcp_client.call_tool(_name, kwargs)

                    self.registry.register_from_schema(
                        name=qualified_name,
                        description=f"[MCP:{server_name}] {mcp_tool.description}",
                        parameters=params,
                        handler=_mcp_handler,
                        permission=ToolPermission.READ,
                        category="mcp",
                    )

                logger.info(
                    "Connected %d MCP servers, discovered %d tools",
                    len(self.mcp_client.connected_servers),
                    len(self.mcp_client.all_tool_names),
                )
            except Exception:
                logger.exception("Failed to connect MCP servers")

    def _build_schemas(self) -> None:
        """Build ToolSchema list from all registered tools."""
        self._tool_schemas = self.registry.list_tools()

    def tools(self) -> list[ToolSchema]:
        """Return all tool schemas."""
        return self._tool_schemas

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name."""
        result: ExecutionResult = await self.executor.execute(tool_name, arguments)
        if result.success:
            return result.output
        raise RuntimeError(result.error or f"Tool {tool_name} failed")

    async def on_unload(self) -> None:
        """Disconnect MCP servers."""
        await self.mcp_client.disconnect_all()
        await super().on_unload()

    # -- Management tools (meta) -------------------------------------------

    def get_tool_stats(self) -> dict[str, Any]:
        """Get execution statistics for all tools."""
        return {
            "total_tools": len(self._tool_schemas),
            "categories": self._count_by_category(),
            "execution_stats": self.registry.stats,
            "mcp_servers": self.mcp_client.connected_servers,
        }

    def _count_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tool_name in self.registry.tool_names:
            reg = self.registry.get(tool_name)
            if reg:
                cat = reg.metadata.category
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    def set_approval_callback(self, callback: Any) -> None:
        """Set the approval callback for dangerous tool operations."""
        self.executor._approval_callback = callback

    def add_mcp_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server configuration (call connect_all() after)."""
        self.mcp_client.add_server(config)
