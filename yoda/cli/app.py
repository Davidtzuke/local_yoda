"""Rich terminal UI for Yoda with streaming markdown and slash commands."""

from __future__ import annotations

import asyncio
import logging
import shlex
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

YODA_THEME = Theme(
    {
        "yoda": "bold green",
        "user": "bold cyan",
        "info": "dim white",
        "warning": "bold yellow",
        "error": "bold red",
        "success": "bold green",
        "cost": "magenta",
        "slash": "bold blue",
    }
)


# ---------------------------------------------------------------------------
# Slash command registry
# ---------------------------------------------------------------------------

SlashHandler = Any  # async callable


class SlashCommandRegistry:
    """Registry for /slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, tuple[SlashHandler, str]] = {}

    def register(self, name: str, handler: SlashHandler, help_text: str = "") -> None:
        self._commands[name] = (handler, help_text)

    def get(self, name: str) -> SlashHandler | None:
        entry = self._commands.get(name)
        return entry[0] if entry else None

    def list_commands(self) -> list[tuple[str, str]]:
        return [(name, desc) for name, (_, desc) in sorted(self._commands.items())]


# ---------------------------------------------------------------------------
# CLI App
# ---------------------------------------------------------------------------

class YodaCLI:
    """Rich terminal interface for the Yoda agent.

    Features:
    - Streaming markdown rendering
    - Slash commands (/remember, /forget, /search, /graph, /status, /cost)
    - Token usage display
    - Graceful shutdown
    """

    def __init__(self, orchestrator: Any) -> None:
        """Initialize with a running orchestrator instance."""
        self.orchestrator = orchestrator
        self.console = Console(theme=YODA_THEME)
        self.slash = SlashCommandRegistry()
        self._running = False
        self._setup_slash_commands()

    def _setup_slash_commands(self) -> None:
        """Register built-in slash commands."""
        self.slash.register("help", self._cmd_help, "Show available commands")
        self.slash.register("remember", self._cmd_remember, "Store a memory: /remember <text>")
        self.slash.register("forget", self._cmd_forget, "Forget a memory: /forget <memory_id>")
        self.slash.register("search", self._cmd_search, "Search memories: /search <query>")
        self.slash.register("graph", self._cmd_graph, "Query knowledge graph: /graph <query>")
        self.slash.register("status", self._cmd_status, "Show agent status and stats")
        self.slash.register("cost", self._cmd_cost, "Show token usage and cost report")
        self.slash.register("reset", self._cmd_reset, "Reset conversation history")
        self.slash.register("setup", self._cmd_setup, "Set API key: /setup <api-key>")
        self.slash.register("claude", self._cmd_claude, "Generate CLAUDE.md from knowledge")
        self.slash.register("quit", self._cmd_quit, "Exit Yoda")

    # -- Main loop ---------------------------------------------------------

    async def run(self) -> None:
        """Main interactive loop."""
        self._running = True
        self._print_banner()

        while self._running:
            try:
                user_input = await self._async_input("\n🟢 You > ")
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    await self._handle_slash(user_input)
                else:
                    await self._handle_chat(user_input)

            except (KeyboardInterrupt, EOFError):
                self._running = False
            except Exception:
                logger.exception("Error in CLI loop")
                self.console.print("[error]An error occurred. Check logs for details.[/error]")

        self.console.print("\n[yoda]May the Force be with you. ✨[/yoda]\n")

    async def _async_input(self, prompt: str) -> str:
        """Read input that can be cancelled by Ctrl+C."""
        import sys, select
        sys.stdout.write(prompt)
        sys.stdout.flush()

        loop = asyncio.get_event_loop()
        while self._running and not self.orchestrator.is_shutting_down:
            # Poll stdin with a short timeout so we can check _running / catch signals
            ready = await loop.run_in_executor(
                None, lambda: select.select([sys.stdin], [], [], 0.3)[0]
            )
            if ready:
                line = sys.stdin.readline()
                if not line:
                    raise EOFError
                return line.strip()
        raise EOFError

    # -- Chat with streaming -----------------------------------------------

    async def _handle_chat(self, user_input: str) -> None:
        """Send user input to agent and render streaming response."""
        agent = self.orchestrator.agent

        self.console.print()
        accumulated = ""

        try:
            try:
                with Live(console=self.console, refresh_per_second=8) as live:
                    async for chunk in agent.chat_stream(user_input):
                        accumulated += chunk.delta
                        live.update(Panel(
                            Markdown(accumulated),
                            title="[yoda]Yoda[/yoda]",
                            border_style="green",
                            padding=(0, 1),
                        ))
            except Exception:
                # Fallback to non-streaming
                logger.debug("Streaming failed, falling back to non-streaming")
                response = await agent.chat(user_input)
                accumulated = response.content
                self.console.print(Panel(
                    Markdown(accumulated),
                    title="[yoda]Yoda[/yoda]",
                    border_style="green",
                    padding=(0, 1),
                ))
        except Exception as e:
            self.console.print(f"[error]Chat failed: {e}[/error]")
            if "api_key" in str(e).lower() or "auth" in str(e).lower():
                self.console.print(
                    "[warning]Set ANTHROPIC_API_KEY or configure a provider in ~/.yoda/config.yaml[/warning]"
                )
            return

        # Show token usage inline
        usage = agent.usage_summary
        self.console.print(
            f"  [info]tokens: {usage['total_tokens']:,} "
            f"(in: {usage['total_input_tokens']:,}, out: {usage['total_output_tokens']:,}) | "
            f"messages: {usage['conversation_messages']}[/info]"
        )

    # -- Slash command handling ---------------------------------------------

    async def _handle_slash(self, raw: str) -> None:
        """Parse and execute a slash command."""
        parts = raw.split(maxsplit=1)
        cmd_name = parts[0][1:]  # strip leading /
        args_str = parts[1] if len(parts) > 1 else ""

        handler = self.slash.get(cmd_name)
        if handler is None:
            self.console.print(f"[warning]Unknown command: /{cmd_name}. Type /help for list.[/warning]")
            return

        try:
            await handler(args_str)
        except Exception as e:
            self.console.print(f"[error]Command failed: {e}[/error]")
            logger.exception("Slash command /%s failed", cmd_name)

    # -- Slash command implementations -------------------------------------

    async def _cmd_help(self, _args: str) -> None:
        table = Table(title="Yoda Commands", show_header=True, header_style="bold")
        table.add_column("Command", style="slash")
        table.add_column("Description")
        for name, desc in self.slash.list_commands():
            table.add_row(f"/{name}", desc)
        self.console.print(table)

    async def _cmd_remember(self, args: str) -> None:
        if not args:
            self.console.print("[warning]Usage: /remember <text to store>[/warning]")
            return
        plugin = self.orchestrator.get_plugin("memory")
        if not plugin:
            self.console.print("[warning]Memory plugin not loaded.[/warning]")
            return
        result = await plugin.execute("memory_store", {"content": args, "importance": 0.7})
        self.console.print(f"[success]✓ {result}[/success]")

    async def _cmd_forget(self, args: str) -> None:
        if not args:
            self.console.print("[warning]Usage: /forget <memory_id>[/warning]")
            return
        plugin = self.orchestrator.get_plugin("memory")
        if not plugin:
            self.console.print("[warning]Memory plugin not loaded.[/warning]")
            return
        result = await plugin.execute("memory_forget", {"memory_id": args.strip()})
        self.console.print(f"[success]✓ {result}[/success]")

    async def _cmd_search(self, args: str) -> None:
        if not args:
            self.console.print("[warning]Usage: /search <query>[/warning]")
            return
        plugin = self.orchestrator.get_plugin("memory")
        if not plugin:
            self.console.print("[warning]Memory plugin not loaded.[/warning]")
            return
        result = await plugin.execute("memory_search", {"query": args, "top_k": 5})
        self.console.print(Panel(result, title="Memory Search Results", border_style="cyan"))

    async def _cmd_graph(self, args: str) -> None:
        if not args:
            self.console.print("[warning]Usage: /graph <query>[/warning]")
            return
        plugin = self.orchestrator.get_plugin("knowledge_graph")
        if not plugin:
            self.console.print("[warning]Knowledge graph plugin not loaded.[/warning]")
            return
        result = await plugin.execute("kg_query", {"question": args})
        self.console.print(Panel(result, title="Knowledge Graph", border_style="yellow"))

    async def _cmd_status(self, _args: str) -> None:
        agent = self.orchestrator.agent
        table = Table(title="Yoda Status", show_header=False)
        table.add_column("Key", style="bold")
        table.add_column("Value")

        usage = agent.usage_summary
        table.add_row("Model", agent.config.provider.model)
        table.add_row("Provider", agent.config.provider.name)
        table.add_row("Total tokens", f"{usage['total_tokens']:,}")
        table.add_row("Messages", str(usage["conversation_messages"]))
        table.add_row("Plugins", str(len(agent.plugins.plugins)))
        table.add_row("Tools", str(len(agent.plugins.all_tools())))

        # Plugin list
        for name, plugin in agent.plugins.plugins.items():
            table.add_row(f"  Plugin: {name}", plugin.description)

        self.console.print(table)

    async def _cmd_cost(self, _args: str) -> None:
        plugin = self.orchestrator.get_plugin("token_optimizer")
        if not plugin:
            # Fallback to basic usage
            usage = self.orchestrator.agent.usage_summary
            self.console.print(f"[cost]Tokens used: {usage['total_tokens']:,}[/cost]")
            return

        report = await plugin.execute("cost_report", {"period": "session"})
        table = Table(title="Cost Report (Session)", show_header=False)
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="cost")
        if isinstance(report, dict):
            for k, v in report.items():
                if k != "model_breakdown":
                    table.add_row(str(k), str(v))
        self.console.print(table)

    async def _cmd_setup(self, args: str) -> None:
        from yoda.core.config import save_config

        if not args:
            self.console.print("[warning]Usage: /setup <your-api-key>[/warning]")
            self.console.print("[info]  Saves your API key to ~/.yoda/config.yaml so you never need to export it again.[/info]")
            return

        key = args.strip()
        config = self.orchestrator.config
        if key.startswith("sk-ant-"):
            config.provider.name = "anthropic"
        elif key.startswith("sk-"):
            config.provider.name = "openai"
        config.provider.api_key = key
        path = save_config(config)
        self.console.print(f"[success]✓ API key saved to {path}[/success]")
        self.console.print("[info]  Restart Yoda to use the new key.[/info]")

    async def _cmd_reset(self, _args: str) -> None:
        self.orchestrator.agent.reset_conversation()
        self.console.print("[success]✓ Conversation history cleared.[/success]")

    async def _cmd_claude(self, _args: str) -> None:
        from yoda.cli.claude_gen import generate_claude_md

        content = await generate_claude_md(self.orchestrator)
        self.console.print(Panel(Markdown(content), title="Generated CLAUDE.md", border_style="blue"))
        self.console.print("[info]Written to CLAUDE.md in current directory.[/info]")

    async def _cmd_quit(self, _args: str) -> None:
        self._running = False

    # -- UI helpers --------------------------------------------------------

    def _print_banner(self) -> None:
        import time as _time

        LOGO = [
            "██╗   ██╗ ██████╗ ██████╗  █████╗ ",
            "╚██╗ ██╔╝██╔═══██╗██╔══██╗██╔══██╗",
            " ╚████╔╝ ██║   ██║██║  ██║███████║",
            "  ╚██╔╝  ██║   ██║██║  ██║██╔══██║",
            "   ██║   ╚██████╔╝██████╔╝██║  ██║",
            "   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝  ╚═╝",
        ]

        TAGLINE = "  Personal AI Assistant — Infinite Memory"
        BUILT = "  Built with Orcha  ·  orcha.nl"

        self.console.print()

        # Animate logo line by line
        for line in LOGO:
            self.console.print(f"  [bold green]{line}[/bold green]")
            _time.sleep(0.05)

        self.console.print()
        self.console.print(f"[dim]{TAGLINE}[/dim]")
        self.console.print(f"[dim cyan]{BUILT}[/dim cyan]")
        self.console.print()

        model = self.orchestrator.agent.config.provider.model
        plugins = len(self.orchestrator.agent.plugins.plugins)
        tools = len(self.orchestrator.agent.plugins.all_tools())
        self.console.print(f"  [dim]Model:[/dim] [bold]{model}[/bold]")
        self.console.print(f"  [dim]Plugins:[/dim] {plugins}  [dim]Tools:[/dim] {tools}")
        self.console.print(f"  [dim]Type [bold]/help[/bold] for commands[/dim]")
        self.console.print()
