"""Rich-based rendering utilities for the CLI."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from yoda.cli.protocol import MemoryEntry, ToolInfo

# Custom theme
YODA_THEME = Theme(
    {
        "yoda.prompt": "bold green",
        "yoda.user": "bold cyan",
        "yoda.info": "dim",
        "yoda.warning": "bold yellow",
        "yoda.error": "bold red",
        "yoda.success": "bold green",
        "yoda.command": "bold magenta",
    }
)

console = Console(theme=YODA_THEME)


def print_banner() -> None:
    """Print the Yoda startup banner."""
    banner = Text()
    banner.append("  __  __           _       \n", style="green")
    banner.append(" |  \\/  |         | |      \n", style="green")
    banner.append(" | \\  / | __ _ ___| |_ ___ _ __ \n", style="green")
    banner.append(" | |\\/| |/ _` / __| __/ _ \\ '__|\n", style="green")
    banner.append(" | |  | | (_| \\__ \\ ||  __/ |   \n", style="green")
    banner.append(" |_|  |_|\\__,_|___/\\__\\___|_|   \n", style="green")

    console.print(
        Panel(
            banner,
            title="[bold green]Yoda[/] [dim]v0.1.0[/]",
            subtitle="[dim]Type /help for commands, Ctrl+C to exit[/]",
            border_style="green",
            padding=(0, 2),
        )
    )


def render_markdown(text: str) -> None:
    """Render markdown content with syntax highlighting."""
    md = Markdown(text, code_theme="monokai")
    console.print(md)


def render_streaming_token(token: str) -> None:
    """Print a single token without newline for streaming effect."""
    console.print(token, end="", highlight=False)


def render_streaming_complete() -> None:
    """Finalize streaming output."""
    console.print()  # final newline


def render_response(text: str) -> None:
    """Render a full agent response as markdown."""
    console.print()
    console.print(
        Panel(
            Markdown(text, code_theme="monokai"),
            title="[bold green]Yoda[/]",
            border_style="green",
            padding=(0, 1),
        )
    )


def render_code(code: str, language: str = "python") -> None:
    """Render a syntax-highlighted code block."""
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)


def render_tools(tools: list[ToolInfo]) -> None:
    """Render available tools as a table."""
    table = Table(title="Available Tools", border_style="green")
    table.add_column("Tool", style="bold cyan")
    table.add_column("Description", style="dim")
    for tool in tools:
        table.add_row(tool.name, tool.description)
    console.print(table)


def render_memory(entries: list[MemoryEntry]) -> None:
    """Render conversation history."""
    if not entries:
        console.print("[yoda.info]No conversation history.[/]")
        return
    table = Table(title="Conversation History", border_style="green")
    table.add_column("Role", style="bold", width=10)
    table.add_column("Content", ratio=1)
    table.add_column("Time", style="dim", width=20)
    for entry in entries:
        style = "cyan" if entry.role == "user" else "green"
        content = entry.content[:100] + "..." if len(entry.content) > 100 else entry.content
        table.add_row(
            Text(entry.role, style=style),
            content,
            entry.timestamp or "-",
        )
    console.print(table)


def render_help() -> None:
    """Render the help message."""
    help_table = Table(title="Commands", border_style="green", show_header=True)
    help_table.add_column("Command", style="bold magenta", width=16)
    help_table.add_column("Description")
    commands = [
        ("/help", "Show this help message"),
        ("/clear", "Clear conversation history"),
        ("/history", "Show conversation history"),
        ("/tools", "List available tools"),
        ("/memory", "Show memory entries"),
        ("/reset", "Clear the terminal screen"),
        ("exit / quit", "Exit Yoda"),
    ]
    for cmd, desc in commands:
        help_table.add_row(cmd, desc)
    console.print(help_table)


def render_error(message: str) -> None:
    """Render an error message."""
    console.print(f"[yoda.error]Error:[/] {message}")


def render_info(message: str) -> None:
    """Render an info message."""
    console.print(f"[yoda.info]{message}[/]")


def render_success(message: str) -> None:
    """Render a success message."""
    console.print(f"[yoda.success]{message}[/]")
