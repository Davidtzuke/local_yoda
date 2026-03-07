"""Main CLI REPL for Yoda - async interactive loop with Rich rendering."""

from __future__ import annotations

import asyncio
import readline  # noqa: F401 — imported for history side-effects
import sys
from pathlib import Path

from yoda.cli.commands import handle_slash_command
from yoda.cli.protocol import AgentProtocol, StubAgent
from yoda.cli.renderer import (
    console,
    print_banner,
    render_error,
    render_info,
    render_response,
    render_streaming_complete,
    render_streaming_token,
)

# Readline history file
HISTORY_FILE = Path.home() / ".yoda_history"
HISTORY_LENGTH = 1000


def _setup_readline() -> None:
    """Configure readline for command history."""
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass
    readline.set_history_length(HISTORY_LENGTH)


def _save_readline() -> None:
    """Save readline history to file."""
    try:
        readline.write_history_file(HISTORY_FILE)
    except OSError:
        pass


def _get_input() -> str | None:
    """Get user input with a styled prompt. Returns None on EOF."""
    try:
        return input("\033[1;32myoda>\033[0m ")
    except EOFError:
        return None


async def _process_message(agent: AgentProtocol, message: str, stream: bool = True) -> None:
    """Send a message to the agent and display the response."""
    if stream:
        try:
            collected: list[str] = []
            console.print()
            console.print("[bold green]Yoda:[/] ", end="")
            async for token in agent.stream_message(message):
                render_streaming_token(token)
                collected.append(token)
            render_streaming_complete()
            console.print()
        except Exception as e:
            render_streaming_complete()
            render_error(f"Streaming failed: {e}")
            # Fall back to non-streaming
            try:
                response = await agent.send_message(message)
                render_response(response)
            except Exception as e2:
                render_error(f"Request failed: {e2}")
    else:
        try:
            response = await agent.send_message(message)
            render_response(response)
        except Exception as e:
            render_error(f"Request failed: {e}")


async def repl_loop(agent: AgentProtocol, stream: bool = True) -> None:
    """Run the main REPL loop."""
    _setup_readline()
    print_banner()
    console.print()

    try:
        while True:
            # Get input (run in executor to not block event loop)
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(None, _get_input)
            except KeyboardInterrupt:
                console.print()
                render_info("Interrupted. Press Ctrl+C again or type 'exit' to quit.")
                continue

            if user_input is None:
                # EOF
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Exit commands
            if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                break

            # Slash commands
            if user_input.startswith("/"):
                await handle_slash_command(user_input, agent)
                continue

            # Send to agent
            await _process_message(agent, user_input, stream=stream)

    except KeyboardInterrupt:
        pass
    finally:
        _save_readline()
        console.print()
        console.print("[bold green]May the force be with you.[/]")
        console.print()


async def run_cli(agent: AgentProtocol | None = None, stream: bool = True) -> None:
    """Entry point for the CLI. Integration agent calls this with a real agent."""
    if agent is None:
        agent = StubAgent()
    await repl_loop(agent, stream=stream)


def main() -> None:
    """Standalone entry point for development/testing."""
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
