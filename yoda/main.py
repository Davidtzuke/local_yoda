"""Main entry point for Yoda — launch CLI or Web mode."""

from __future__ import annotations

import argparse
import asyncio


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="yoda",
        description="Yoda — Local personal AI agent",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="cli",
        choices=["cli", "web"],
        help="Launch mode: 'cli' for terminal REPL (default), 'web' for browser UI",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Web server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web server port (default: 8000)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output in CLI mode",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "openai", "ollama"],
        default=None,
        help="LLM backend (default: local, or set YODA_LLM_BACKEND env var)",
    )
    return parser.parse_args(argv)


def _create_agent(backend: str | None = None) -> "YodaAgent":  # noqa: F821
    from yoda.agent.core import YodaAgent

    return YodaAgent(llm_backend=backend)


def run_cli_mode(backend: str | None = None, stream: bool = True) -> None:
    """Launch the CLI REPL."""
    from yoda.cli import run_cli

    agent = _create_agent(backend)
    asyncio.run(run_cli(agent, stream=stream))


def run_web_mode(
    host: str = "127.0.0.1",
    port: int = 8000,
    backend: str | None = None,
) -> None:
    """Launch the web UI."""
    import uvicorn

    from yoda.web.app import create_app

    agent = _create_agent(backend)
    app = create_app(agent)
    uvicorn.run(app, host=host, port=port)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    if args.mode == "web":
        run_web_mode(host=args.host, port=args.port, backend=args.backend)
    else:
        run_cli_mode(backend=args.backend, stream=not args.no_stream)


if __name__ == "__main__":
    main()
