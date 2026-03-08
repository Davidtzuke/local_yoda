"""Yoda CLI package — Rich terminal interface with slash commands."""

import asyncio
import logging
import os
import sys


def main() -> None:
    """Entry point for the ``yoda`` console script."""
    # Suppress noisy library loggers
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Only show yoda logs at INFO level
    logging.getLogger("yoda").setLevel(logging.INFO)

    # Pre-flight check for API key (env or saved config)
    from pathlib import Path
    has_env_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    has_saved_key = Path("~/.yoda/config.yaml").expanduser().exists()
    if not has_env_key and not has_saved_key:
        print(
            "⚠️  No API key found. Either:\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'   (auto-saved on first run)\n"
            "  or run:  /setup <your-api-key>          (inside Yoda)\n"
        )

    from yoda.cli.orchestrator import Orchestrator
    from yoda.cli.app import YodaCLI

    async def _run() -> None:
        orchestrator = Orchestrator()
        cli = None
        try:
            try:
                await orchestrator.initialize()
            except Exception as e:
                print(f"⚠️  Initialization warning: {e}")
                print("Starting with limited functionality...\n")
            cli = YodaCLI(orchestrator)
            await cli.run()
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            if cli:
                cli._running = False
            try:
                await orchestrator.shutdown()
            except Exception:
                pass

    try:
        asyncio.run(_run())
    except (KeyboardInterrupt, EOFError, SystemExit):
        print("\nGoodbye!")
