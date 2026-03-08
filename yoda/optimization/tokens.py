"""Per-model token counting via tiktoken with budget tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelTokenProfile:
    """Token limits and encoding info for a specific model."""

    name: str
    max_context: int
    max_output: int
    encoding_name: str  # tiktoken encoding
    tokens_per_message: int = 3  # overhead per message
    tokens_per_name: int = 1  # if role name is included

    @property
    def effective_context(self) -> int:
        """Context budget minus output reservation."""
        return self.max_context - self.max_output


# Well-known model profiles
MODEL_PROFILES: dict[str, ModelTokenProfile] = {
    # Anthropic
    "claude-sonnet-4-20250514": ModelTokenProfile(
        "claude-sonnet-4-20250514", 200_000, 8_192, "cl100k_base"
    ),
    "claude-3-5-sonnet-20241022": ModelTokenProfile(
        "claude-3-5-sonnet-20241022", 200_000, 8_192, "cl100k_base"
    ),
    "claude-3-haiku-20240307": ModelTokenProfile(
        "claude-3-haiku-20240307", 200_000, 4_096, "cl100k_base"
    ),
    "claude-3-opus-20240229": ModelTokenProfile(
        "claude-3-opus-20240229", 200_000, 4_096, "cl100k_base"
    ),
    # OpenAI
    "gpt-4": ModelTokenProfile("gpt-4", 8_192, 4_096, "cl100k_base"),
    "gpt-4-turbo": ModelTokenProfile("gpt-4-turbo", 128_000, 4_096, "cl100k_base"),
    "gpt-4o": ModelTokenProfile("gpt-4o", 128_000, 16_384, "o200k_base"),
    "gpt-4o-mini": ModelTokenProfile("gpt-4o-mini", 128_000, 16_384, "o200k_base"),
    "gpt-3.5-turbo": ModelTokenProfile("gpt-3.5-turbo", 16_385, 4_096, "cl100k_base"),
    # Local / fallback
    "default": ModelTokenProfile("default", 128_000, 4_096, "cl100k_base"),
}


def get_model_profile(model: str) -> ModelTokenProfile:
    """Get token profile for a model, with fuzzy matching."""
    if model in MODEL_PROFILES:
        return MODEL_PROFILES[model]
    # Fuzzy: check if any known model name is a prefix
    for key, profile in MODEL_PROFILES.items():
        if model.startswith(key) or key.startswith(model):
            return profile
    return MODEL_PROFILES["default"]


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _get_encoder(encoding_name: str) -> Any:
    """Cached tiktoken encoder."""
    try:
        import tiktoken
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return None


class TokenCounter:
    """Accurate per-model token counting with budget tracking.

    Usage:
        counter = TokenCounter("gpt-4o")
        count = counter.count("Hello, world!")
        msg_tokens = counter.count_message({"role": "user", "content": "Hi"})
        total = counter.count_messages(messages)
    """

    def __init__(self, model: str = "default") -> None:
        self.profile = get_model_profile(model)
        self._encoder = _get_encoder(self.profile.encoding_name)
        self._budget_used: int = 0
        self._budget_limit: int = self.profile.effective_context

    # -- Counting ----------------------------------------------------------

    def count(self, text: str) -> int:
        """Count tokens in a string."""
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        # Heuristic fallback
        return max(1, int(len(text) / 3.5))

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens for a single message dict (OpenAI format)."""
        tokens = self.profile.tokens_per_message
        for key, value in message.items():
            if isinstance(value, str):
                tokens += self.count(value)
            elif isinstance(value, list):
                # Tool calls, tool results, etc.
                tokens += self.count(str(value))
        if message.get("name"):
            tokens += self.profile.tokens_per_name
        return tokens

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count total tokens for a list of messages."""
        total = sum(self.count_message(m) for m in messages)
        total += 3  # every reply is primed with <|start|>assistant<|message|>
        return total

    # -- Budget tracking ---------------------------------------------------

    @property
    def budget_limit(self) -> int:
        return self._budget_limit

    @budget_limit.setter
    def budget_limit(self, value: int) -> None:
        self._budget_limit = value

    @property
    def budget_used(self) -> int:
        return self._budget_used

    @property
    def budget_remaining(self) -> int:
        return max(0, self._budget_limit - self._budget_used)

    @property
    def budget_utilization(self) -> float:
        """Budget used as fraction (0.0 to 1.0)."""
        if self._budget_limit == 0:
            return 1.0
        return min(1.0, self._budget_used / self._budget_limit)

    def consume(self, tokens: int) -> None:
        """Record token usage against budget."""
        self._budget_used += tokens

    def reset_budget(self) -> None:
        """Reset budget tracking."""
        self._budget_used = 0

    def fits_in_budget(self, tokens: int) -> bool:
        """Check if additional tokens fit in remaining budget."""
        return (self._budget_used + tokens) <= self._budget_limit

    def tokens_until_compression(self, threshold: float = 0.8) -> int:
        """How many tokens until we should trigger compression."""
        trigger_point = int(self._budget_limit * threshold)
        return max(0, trigger_point - self._budget_used)
