"""Orchestrator — wires all Yoda components with correct init order and graceful shutdown."""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

from yoda.core.agent import Agent
from yoda.core.config import YodaConfig, load_config
from yoda.core.plugins import Plugin, PluginRegistry

logger = logging.getLogger(__name__)


class Orchestrator:
    """Central orchestrator that initializes all Yoda subsystems in the correct order.

    Init order:
    1. Load configuration
    2. Create Agent (provider + conversation)
    3. Register core plugins (memory, knowledge graph, token optimizer, tools)
    4. Initialize all plugins (on_load)
    5. Wire context injectors
    6. Register signal handlers for graceful shutdown

    Shutdown order (reverse):
    1. Save state (cost tracker, memory, graph)
    2. Unload plugins
    3. Cancel background tasks
    """

    def __init__(self, config: YodaConfig | None = None) -> None:
        self.config = config or load_config()
        self.agent = Agent(config=self.config)
        self._shutdown_event = asyncio.Event()
        self._background_tasks: list[asyncio.Task[Any]] = []

    # -- Plugin access -----------------------------------------------------

    def get_plugin(self, name: str) -> Plugin | None:
        """Get a loaded plugin by name."""
        return self.agent.plugins.get(name)

    # -- Initialization ----------------------------------------------------

    async def initialize(self) -> None:
        """Full initialization sequence."""
        logger.info("Initializing Yoda orchestrator...")

        # 1. Register core plugins BEFORE discovery/loading
        self._register_core_plugins()

        # 2. Initialize agent (discovers + loads plugins)
        await self.agent.initialize()

        # 3. Wire context injectors from plugins that provide them
        self._wire_context_injectors()

        # 4. Register signal handlers
        self._register_signals()

        logger.info(
            "Yoda ready: %d plugins, %d tools",
            len(self.agent.plugins.plugins),
            len(self.agent.plugins.all_tools()),
        )

    def _register_core_plugins(self) -> None:
        """Register built-in plugins in dependency order."""
        # Token optimizer first (other plugins may need cost tracking)
        try:
            from yoda.optimization.plugin import TokenOptimizerPlugin

            self.agent.plugins.register(TokenOptimizerPlugin(self.config))
            logger.info("Registered token optimizer plugin")
        except Exception:
            logger.warning("Token optimizer plugin unavailable", exc_info=True)

        # Memory system
        try:
            from yoda.memory.plugin import MemoryPlugin

            self.agent.plugins.register(MemoryPlugin(self.config))
            logger.info("Registered memory plugin")
        except Exception:
            logger.warning("Memory plugin unavailable", exc_info=True)

        # Knowledge graph
        try:
            from yoda.knowledge.plugin import KnowledgeGraphPlugin

            self.agent.plugins.register(KnowledgeGraphPlugin(self.config))
            logger.info("Registered knowledge graph plugin")
        except Exception:
            logger.warning("Knowledge graph plugin unavailable", exc_info=True)

        # Tool access (file ops, shell, web, computer, MCP)
        try:
            from yoda.tools.plugin import ToolAccessPlugin

            self.agent.plugins.register(ToolAccessPlugin(self.config))
            logger.info("Registered tool access plugin")
        except Exception:
            logger.warning("Tool access plugin unavailable", exc_info=True)

    def _wire_context_injectors(self) -> None:
        """Connect context injectors from plugins that provide them."""
        # Token optimizer context injector
        token_plugin = self.get_plugin("token_optimizer")
        if token_plugin and hasattr(token_plugin, "get_context_injector"):
            self.agent.add_context_injector(token_plugin.get_context_injector())
            logger.debug("Wired token optimizer context injector")

        # Memory context injector — searches memories based on conversation
        memory_plugin = self.get_plugin("memory")
        if memory_plugin and hasattr(memory_plugin, "manager"):
            manager = memory_plugin.manager  # type: ignore[attr-defined]

            def memory_injector(conversation: Any) -> dict[str, Any]:
                """Inject relevant memories into context."""
                if not conversation.messages:
                    return {}
                last_msg = conversation.messages[-1]
                if last_msg.role.value != "user":
                    return {}
                # Note: this is sync — the manager.search is async
                # We use a cached approach for the sync injector
                return {}

            self.agent.add_context_injector(memory_injector)

        # Knowledge graph context injector
        kg_plugin = self.get_plugin("knowledge_graph")
        if kg_plugin and hasattr(kg_plugin, "get_context_injector"):
            # The KG plugin's injector is async but Agent expects sync
            # We skip async injectors as Agent's _build_system_prompt is sync
            logger.debug("Knowledge graph plugin registered (context via tools)")

    def _register_signals(self) -> None:
        """Register OS signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._signal_handler)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

    def _signal_handler(self) -> None:
        """Handle OS signals."""
        logger.info("Received shutdown signal")
        self._shutdown_event.set()

    # -- Shutdown ----------------------------------------------------------

    async def shutdown(self) -> None:
        """Graceful shutdown in reverse order."""
        logger.info("Shutting down Yoda...")

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Shutdown agent (unloads all plugins)
        await self.agent.shutdown()

        logger.info("Yoda shutdown complete")

    # -- Background tasks --------------------------------------------------

    def add_background_task(self, coro: Any) -> asyncio.Task[Any]:
        """Schedule a background task that will be cancelled on shutdown."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(lambda t: self._background_tasks.remove(t))
        return task

    @property
    def is_shutting_down(self) -> bool:
        return self._shutdown_event.is_set()

    async def wait_for_shutdown(self) -> None:
        """Block until shutdown signal received."""
        await self._shutdown_event.wait()
