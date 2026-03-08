"""Minimal CLI entry point for Yoda (placeholder for Integration Architect)."""

from __future__ import annotations

import asyncio

import click


@click.command()
@click.option("--config", "-c", default=None, help="Path to yoda.yaml config file")
@click.option("--provider", "-p", default=None, help="LLM provider name override")
@click.option("--model", "-m", default=None, help="Model name override")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(
    config: str | None,
    provider: str | None,
    model: str | None,
    debug: bool,
) -> None:
    """Yoda — your personal AI assistant."""
    asyncio.run(_run(config, provider, model, debug))


async def _run(
    config_path: str | None,
    provider_name: str | None,
    model_name: str | None,
    debug: bool,
) -> None:
    import logging

    from yoda.core.agent import Agent
    from yoda.core.config import load_config

    if debug:
        logging.basicConfig(level=logging.DEBUG)

    cfg = load_config(config_path)
    if provider_name:
        cfg.provider.name = provider_name
    if model_name:
        cfg.provider.model = model_name
    if debug:
        cfg.debug = True

    agent = Agent(config=cfg)
    await agent.initialize()

    click.echo("🟢 Yoda is ready. Type 'quit' to exit.\n")

    try:
        while True:
            user_input = click.prompt("You", prompt_suffix=" > ")
            if user_input.strip().lower() in ("quit", "exit", "q"):
                break
            response = await agent.chat(user_input)
            click.echo(f"\nYoda > {response.content}\n")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await agent.shutdown()
        click.echo("\nMay the Force be with you. 🌟")


if __name__ == "__main__":
    main()
