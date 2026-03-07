"""Configuration schema for Yoda, loaded from environment or config file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM provider configuration (OpenAI-compatible API)."""

    model_config = SettingsConfigDict(env_prefix="YODA_LLM_")

    base_url: str = "http://localhost:11434/v1"  # Ollama default
    api_key: str = "ollama"  # Ollama doesn't need a real key
    model: str = "llama3.1:8b"
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = (
        "You are Yoda, a helpful local personal AI assistant. "
        "You have access to tools and long-term memory. "
        "Be concise, helpful, and proactive."
    )


class MemorySettings(BaseSettings):
    """Memory subsystem configuration."""

    model_config = SettingsConfigDict(env_prefix="YODA_MEMORY_")

    db_path: Path = Field(default=Path("~/.yoda/memory.db"))
    embedding_model: str = "all-MiniLM-L6-v2"
    search_top_k: int = 5
    similarity_threshold: float = 0.3

    @property
    def resolved_db_path(self) -> Path:
        """Expand ~ and ensure parent directory exists."""
        p = self.db_path.expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


class ServerSettings(BaseSettings):
    """Web UI server configuration."""

    model_config = SettingsConfigDict(env_prefix="YODA_SERVER_")

    host: str = "127.0.0.1"
    port: int = 8420
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class Settings(BaseSettings):
    """Top-level application settings, composing all sub-configs."""

    model_config = SettingsConfigDict(
        env_prefix="YODA_",
        env_nested_delimiter="__",
    )

    debug: bool = False
    data_dir: Path = Field(default=Path("~/.yoda"))

    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @property
    def resolved_data_dir(self) -> Path:
        p = self.data_dir.expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p


def get_settings() -> Settings:
    """Create and return the application settings (reads from env vars)."""
    return Settings()
