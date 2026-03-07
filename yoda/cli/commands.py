"""Slash command handling for the CLI REPL."""

from __future__ import annotations

from yoda.cli.protocol import AgentProtocol
from yoda.cli.renderer import (
    console,
    render_help,
    render_info,
    render_memory,
    render_success,
    render_tools,
)


async def handle_slash_command(command: str, agent: AgentProtocol) -> bool:
    """Handle a slash command. Returns True if the command was handled.

    Returns False if the input is not a slash command (should be sent to agent).
    """
    cmd = command.strip().lower()

    if cmd == "/help":
        render_help()
        return True

    if cmd == "/clear":
        await agent.clear_memory()
        render_success("Conversation history cleared.")
        return True

    if cmd in ("/history", "/memory"):
        entries = await agent.get_memory()
        render_memory(entries)
        return True

    if cmd == "/tools":
        tools = await agent.get_tools()
        render_tools(tools)
        return True

    if cmd == "/reset":
        console.clear()
        render_info("Screen cleared.")
        return True

    if cmd.startswith("/"):
        render_info(f"Unknown command: {cmd}. Type /help for available commands.")
        return True

    return False
