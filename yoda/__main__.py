"""Entry point for the Yoda agent. Supports CLI and web server modes."""

from __future__ import annotations

import asyncio
import sys

import click


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--debug", is_flag=True, help="Enable debug logging.")
def cli(ctx: click.Context, debug: bool) -> None:
    """Yoda — your local personal AI agent."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    if ctx.invoked_subcommand is None:
        # Default to chat mode
        ctx.invoke(chat)


@cli.command()
@click.pass_context
def chat(ctx: click.Context) -> None:
    """Start an interactive CLI chat session."""
    from yoda.ui.cli import run_cli  # noqa: E402 — lazy import

    asyncio.run(run_cli(debug=ctx.obj.get("debug", False)))


@cli.command()
@click.option("--host", default=None, help="Bind host (default from config).")
@click.option("--port", default=None, type=int, help="Bind port (default from config).")
@click.pass_context
def serve(ctx: click.Context, host: str | None, port: int | None) -> None:
    """Start the web chat UI server."""
    from yoda.ui.web import run_server  # noqa: E402 — lazy import

    asyncio.run(run_server(host=host, port=port, debug=ctx.obj.get("debug", False)))


def main() -> None:
    """Package entry point."""
    cli()


if __name__ == "__main__":
    main()
