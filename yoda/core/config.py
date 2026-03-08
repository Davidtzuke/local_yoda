"""YAML-based configuration with environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    name: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 120.0
    extra: dict[str, Any] = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    """Configuration for the memory / RAG subsystem."""

    backend: str = "chromadb"
    persist_dir: str = "~/.yoda/memory"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 64


class KnowledgeGraphConfig(BaseModel):
    """Configuration for the knowledge graph."""

    backend: str = "networkx"
    persist_path: str = "~/.yoda/kg.json"
    max_hops: int = 3


class TokenConfig(BaseModel):
    """Token budget and optimization settings."""

    max_context_tokens: int = 128_000
    max_output_tokens: int = 4096
    compression_enabled: bool = True
    sliding_window_size: int = 50  # number of recent messages to keep
    cost_tracking: bool = True


class PluginSettings(BaseModel):
    """Plugin discovery and loading settings."""

    auto_discover: bool = True
    plugin_dirs: list[str] = Field(default_factory=lambda: ["~/.yoda/plugins"])
    enabled: list[str] = Field(default_factory=list)
    disabled: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class YodaConfig(BaseModel):
    """Root configuration for the Yoda agent."""

    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    tokens: TokenConfig = Field(default_factory=TokenConfig)
    plugins: PluginSettings = Field(default_factory=PluginSettings)

    system_prompt: str = (
        "You are Yoda, a wise and helpful personal AI assistant. "
        "You remember everything the user tells you and learn their preferences over time. "
        "Think step-by-step, use tools when needed, and be concise."
    )
    user_name: str = "User"
    data_dir: str = "~/.yoda"
    debug: bool = False


# ---------------------------------------------------------------------------
# Env-var override helpers
# ---------------------------------------------------------------------------

_ENV_PREFIX = "YODA_"


def _apply_env_overrides(config: dict[str, Any], prefix: str = _ENV_PREFIX) -> dict[str, Any]:
    """Recursively override config values from environment variables.

    Convention: YODA_PROVIDER_MODEL -> config["provider"]["model"]
    """
    for key, value in list(config.items()):
        env_key = f"{prefix}{key}".upper()
        if isinstance(value, dict):
            config[key] = _apply_env_overrides(value, prefix=f"{env_key}_")
        else:
            env_val = os.environ.get(env_key)
            if env_val is not None:
                # Coerce to the original type
                if isinstance(value, bool):
                    config[key] = env_val.lower() in ("1", "true", "yes")
                elif isinstance(value, int):
                    config[key] = int(env_val)
                elif isinstance(value, float):
                    config[key] = float(env_val)
                else:
                    config[key] = env_val
    return config


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATHS = [
    Path("yoda.yaml"),
    Path("yoda.yml"),
    Path.home() / ".yoda" / "config.yaml",
    Path.home() / ".yoda" / "config.yml",
]


def load_config(path: str | Path | None = None) -> YodaConfig:
    """Load configuration from YAML file with env-var overrides.

    Search order:
    1. Explicit *path* argument
    2. ``YODA_CONFIG`` environment variable
    3. ``yoda.yaml`` / ``yoda.yml`` in CWD
    4. ``~/.yoda/config.yaml``
    5. Defaults
    """
    raw: dict[str, Any] = {}

    config_path: Path | None = None
    if path:
        config_path = Path(path)
    elif env_path := os.environ.get("YODA_CONFIG"):
        config_path = Path(env_path)
    else:
        for candidate in _DEFAULT_CONFIG_PATHS:
            if candidate.expanduser().exists():
                config_path = candidate
                break

    if config_path and config_path.expanduser().exists():
        with open(config_path.expanduser()) as f:
            raw = yaml.safe_load(f) or {}

    raw = _apply_env_overrides(raw)
    return YodaConfig.model_validate(raw)
