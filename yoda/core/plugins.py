"""Plugin framework with base class, registry, lifecycle hooks, and auto-discovery."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from yoda.core.config import YodaConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schema (plugins expose tools to the agent)
# ---------------------------------------------------------------------------

class ToolParameter(BaseModel):
    """Single parameter in a tool schema."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    default: Any = None


class ToolSchema(BaseModel):
    """Schema describing a tool the agent can invoke."""

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    returns: str = "string"


# ---------------------------------------------------------------------------
# Plugin base class
# ---------------------------------------------------------------------------

class Plugin(ABC):
    """Base class for all Yoda plugins.

    Lifecycle:
        1. ``__init__`` — called with config
        2. ``on_load`` — async setup (DB connections, etc.)
        3. Tool calls happen via ``execute``
        4. ``on_unload`` — async teardown
    """

    name: str = "unnamed_plugin"
    version: str = "0.1.0"
    description: str = ""

    def __init__(self, config: YodaConfig) -> None:
        self.config = config
        self._loaded = False

    # -- Lifecycle hooks --------------------------------------------------

    async def on_load(self) -> None:
        """Called once when the plugin is loaded. Override for async init."""
        self._loaded = True

    async def on_unload(self) -> None:
        """Called when the plugin is unloaded. Override for cleanup."""
        self._loaded = False

    # -- Tool interface ----------------------------------------------------

    @abstractmethod
    def tools(self) -> list[ToolSchema]:
        """Return the list of tools this plugin provides."""
        ...

    @abstractmethod
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name with the given arguments."""
        ...

    # -- Hooks into the agent loop (optional overrides) --------------------

    async def on_user_message(self, content: str) -> str | None:
        """Called before every user message. Return modified content or None."""
        return None

    async def on_assistant_response(self, content: str) -> str | None:
        """Called after every assistant response. Return modified content or None."""
        return None

    async def on_context_build(self, context: dict[str, Any]) -> dict[str, Any]:
        """Inject additional context before the LLM call. Return augmented context."""
        return context

    # -- Utilities ---------------------------------------------------------

    def __repr__(self) -> str:
        return f"<Plugin {self.name} v{self.version}>"


# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------

class PluginRegistry:
    """Central registry that discovers, loads, and manages plugins."""

    def __init__(self, config: YodaConfig) -> None:
        self.config = config
        self._plugins: dict[str, Plugin] = {}

    # -- Registration ------------------------------------------------------

    def register(self, plugin: Plugin) -> None:
        """Register a plugin instance."""
        if plugin.name in self._plugins:
            logger.warning("Plugin %s already registered — replacing", plugin.name)
        self._plugins[plugin.name] = plugin
        logger.info("Registered plugin: %s", plugin)

    def unregister(self, name: str) -> Plugin | None:
        return self._plugins.pop(name, None)

    # -- Lookup ------------------------------------------------------------

    def get(self, name: str) -> Plugin | None:
        return self._plugins.get(name)

    @property
    def plugins(self) -> dict[str, Plugin]:
        return dict(self._plugins)

    def all_tools(self) -> list[tuple[str, ToolSchema]]:
        """Return (plugin_name, tool_schema) for every registered tool."""
        out: list[tuple[str, ToolSchema]] = []
        for name, plugin in self._plugins.items():
            for tool in plugin.tools():
                out.append((name, tool))
        return out

    def find_tool(self, tool_name: str) -> tuple[Plugin, ToolSchema] | None:
        """Find which plugin owns a tool by name."""
        for plugin in self._plugins.values():
            for tool in plugin.tools():
                if tool.name == tool_name:
                    return plugin, tool
        return None

    # -- Lifecycle ---------------------------------------------------------

    async def load_all(self) -> None:
        """Call on_load for every registered plugin."""
        for plugin in self._plugins.values():
            try:
                await plugin.on_load()
                logger.info("Loaded plugin: %s", plugin.name)
            except Exception:
                logger.exception("Failed to load plugin: %s", plugin.name)

    async def unload_all(self) -> None:
        """Call on_unload for every registered plugin."""
        for plugin in self._plugins.values():
            try:
                await plugin.on_unload()
            except Exception:
                logger.exception("Failed to unload plugin: %s", plugin.name)

    # -- Auto-discovery ----------------------------------------------------

    def discover(self) -> None:
        """Discover plugins from configured directories and entry points."""
        settings = self.config.plugins
        if not settings.auto_discover:
            return

        # 1. Discover from plugin directories
        for dir_str in settings.plugin_dirs:
            plugin_dir = Path(dir_str).expanduser()
            if not plugin_dir.is_dir():
                continue
            self._discover_from_directory(plugin_dir)

        # 2. Discover from entry points (installed packages)
        self._discover_from_entry_points()

    def _discover_from_directory(self, directory: Path) -> None:
        """Import all .py files in a directory, looking for Plugin subclasses."""
        for py_file in sorted(directory.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            module_name = f"yoda_plugin_{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)  # type: ignore[union-attr]
                    self._register_from_module(module)
            except Exception:
                logger.exception("Failed to load plugin from %s", py_file)

    def _discover_from_entry_points(self) -> None:
        """Discover plugins via the ``yoda.plugins`` entry-point group."""
        try:
            from importlib.metadata import entry_points

            eps = entry_points(group="yoda.plugins")
            for ep in eps:
                try:
                    plugin_cls = ep.load()
                    if isinstance(plugin_cls, type) and issubclass(plugin_cls, Plugin):
                        instance = plugin_cls(self.config)
                        self.register(instance)
                except Exception:
                    logger.exception("Failed to load entry-point plugin: %s", ep.name)
        except Exception:
            logger.debug("Entry-point discovery not available")

    def _register_from_module(self, module: Any) -> None:
        """Find and register all Plugin subclasses in a module."""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Plugin)
                and attr is not Plugin
            ):
                settings = self.config.plugins
                if settings.disabled and attr.name in settings.disabled:
                    continue
                if settings.enabled and attr.name not in settings.enabled:
                    continue
                try:
                    instance = attr(self.config)
                    self.register(instance)
                except Exception:
                    logger.exception("Failed to instantiate plugin %s", attr_name)
