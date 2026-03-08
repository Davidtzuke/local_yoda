"""CLI entry point for Yoda — routes to interactive CLI or MCP server mode."""

from __future__ import annotations

import asyncio
import logging
import sys

import click


@click.command()
@click.option("--config", "-c", default=None, help="Path to yoda.yaml config file")
@click.option("--provider", "-p", default=None, help="LLM provider name override")
@click.option("--model", "-m", default=None, help="Model name override")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--mcp", is_flag=True, help="Run as MCP server instead of interactive CLI")
@click.option("--mcp-stdio", is_flag=True, help="Run MCP server with stdio transport")
@click.option("--port", default=8765, help="MCP server port (SSE mode)")
@click.option("--host", default="localhost", help="MCP server host")
def main(
    config: str | None,
    provider: str | None,
    model: str | None,
    debug: bool,
    mcp: bool,
    mcp_stdio: bool,
    port: int,
    host: str,
) -> None:
    """Yoda — your personal AI assistant with infinite memory."""
    if mcp or mcp_stdio:
        asyncio.run(_run_mcp(config, provider, model, debug, mcp_stdio, host, port))
    else:
        asyncio.run(_run_cli(config, provider, model, debug))


async def _run_cli(
    config_path: str | None,
    provider_name: str | None,
    model_name: str | None,
    debug: bool,
) -> None:
    """Run the interactive Rich CLI."""
    from yoda.cli.app import YodaCLI
    from yoda.cli.orchestrator import Orchestrator
    from yoda.core.config import load_config

    if debug:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    cfg = load_config(config_path)
    if provider_name:
        cfg.provider.name = provider_name
    if model_name:
        cfg.provider.model = model_name
    if debug:
        cfg.debug = True

    orchestrator = Orchestrator(config=cfg)

    try:
        await orchestrator.initialize()
        cli = YodaCLI(orchestrator)
        await cli.run()
    finally:
        await orchestrator.shutdown()


async def _run_mcp(
    config_path: str | None,
    provider_name: str | None,
    model_name: str | None,
    debug: bool,
    stdio: bool,
    host: str,
    port: int,
) -> None:
    """Run Yoda as an MCP server."""
    from yoda.cli.orchestrator import Orchestrator
    from yoda.core.config import load_config

    if debug:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    cfg = load_config(config_path)
    if provider_name:
        cfg.provider.name = provider_name
    if model_name:
        cfg.provider.model = model_name

    orchestrator = Orchestrator(config=cfg)

    try:
        await orchestrator.initialize()

        if stdio:
            from yoda.mcp_server.server import YodaMCPStdioServer

            server = YodaMCPStdioServer(orchestrator)
            await server.run()
        else:
            from yoda.mcp_server.server import YodaMCPServer

            server = YodaMCPServer(orchestrator, host=host, port=port)
            await server.start()
            click.echo(f"MCP server running on http://{host}:{port}")
            click.echo("Press Ctrl+C to stop.")
            await orchestrator.wait_for_shutdown()
            await server.stop()
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    main()
