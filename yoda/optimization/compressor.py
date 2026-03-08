"""Context compression: summarize old turns, progressive compression."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from yoda.optimization.tokens import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of a compression operation."""

    original_tokens: int
    compressed_tokens: int
    messages: list[dict[str, Any]]
    summary: str | None = None  # Summary of compressed turns

    @property
    def ratio(self) -> float:
        """Compression ratio (0.0 = no reduction, 1.0 = fully removed)."""
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.original_tokens)

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens


class ContextCompressor:
    """Multi-strategy context compression for conversation history.

    Strategies (applied progressively as context fills up):
    1. Trim tool results to summaries
    2. Drop old tool call/result pairs
    3. Summarize old conversation turns into a condensed "memory" message
    4. Truncate long individual messages
    """

    def __init__(
        self,
        model: str = "default",
        max_tokens: int | None = None,
        summary_fn: Any | None = None,
    ) -> None:
        self.counter = TokenCounter(model)
        self.max_tokens = max_tokens or self.counter.profile.effective_context
        self._summary_fn = summary_fn  # async callable(messages) -> str
        self._cached_summaries: dict[str, str] = {}

    def compress(
        self,
        messages: list[dict[str, Any]],
        target_tokens: int | None = None,
        preserve_last_n: int = 6,
    ) -> CompressionResult:
        """Compress messages to fit within token budget.

        Args:
            messages: Full message list (including system).
            target_tokens: Target token count (default: 80% of max).
            preserve_last_n: Number of recent messages to never compress.

        Returns:
            CompressionResult with compressed messages.
        """
        target = target_tokens or int(self.max_tokens * 0.8)
        original_tokens = self.counter.count_messages(messages)

        if original_tokens <= target:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                messages=messages,
            )

        # Separate system message from conversation
        system_msg = None
        conversation = messages
        if messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            conversation = messages[1:]

        # Split into compressible (old) and preserved (recent)
        if len(conversation) > preserve_last_n:
            old = conversation[:-preserve_last_n]
            recent = conversation[-preserve_last_n:]
        else:
            old = []
            recent = conversation

        compressed_old = list(old)

        # Strategy 1: Trim tool results
        compressed_old = self._trim_tool_results(compressed_old)

        # Strategy 2: Drop old tool call/result pairs
        current = self._count_with_system(system_msg, compressed_old + recent)
        if current > target:
            compressed_old = self._drop_tool_pairs(compressed_old)

        # Strategy 3: Truncate long messages
        current = self._count_with_system(system_msg, compressed_old + recent)
        if current > target:
            compressed_old = self._truncate_long_messages(
                compressed_old, max_per_message=500
            )

        # Strategy 4: Summarize old turns into a single message
        current = self._count_with_system(system_msg, compressed_old + recent)
        if current > target and compressed_old:
            summary = self._create_summary(compressed_old)
            compressed_old = [
                {"role": "system", "content": f"[Conversation summary]: {summary}"}
            ]

        # Rebuild
        result_messages: list[dict[str, Any]] = []
        if system_msg:
            result_messages.append(system_msg)
        result_messages.extend(compressed_old)
        result_messages.extend(recent)

        compressed_tokens = self.counter.count_messages(result_messages)

        # Last resort: aggressive truncation of even recent messages
        if compressed_tokens > target:
            result_messages = self._aggressive_truncate(result_messages, target)
            compressed_tokens = self.counter.count_messages(result_messages)

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            messages=result_messages,
            summary=compressed_old[0].get("content") if compressed_old else None,
        )

    def _count_with_system(
        self, system_msg: dict[str, Any] | None, messages: list[dict[str, Any]]
    ) -> int:
        all_msgs = ([system_msg] if system_msg else []) + messages
        return self.counter.count_messages(all_msgs)

    def _trim_tool_results(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Replace long tool results with truncated versions."""
        result = []
        for msg in messages:
            if msg.get("role") == "tool_result":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 500:
                    msg = dict(msg)
                    msg["content"] = content[:200] + "\n...[truncated]...\n" + content[-100:]
            result.append(msg)
        return result

    def _drop_tool_pairs(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove old tool call + result pairs, keeping only non-tool messages."""
        result = []
        skip_next_tool_result = False
        for msg in messages:
            role = msg.get("role", "")
            has_tool_calls = bool(msg.get("tool_calls"))
            if has_tool_calls:
                skip_next_tool_result = True
                continue
            if role == "tool_result" and skip_next_tool_result:
                skip_next_tool_result = False
                continue
            skip_next_tool_result = False
            result.append(msg)
        return result

    def _truncate_long_messages(
        self, messages: list[dict[str, Any]], max_per_message: int = 500
    ) -> list[dict[str, Any]]:
        """Truncate individual messages that exceed max token count."""
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                tokens = self.counter.count(content)
                if tokens > max_per_message:
                    # Keep first portion and last portion
                    chars_to_keep = int(max_per_message * 3.5)
                    half = chars_to_keep // 2
                    msg = dict(msg)
                    msg["content"] = (
                        content[:half] + "\n...[compressed]...\n" + content[-half:]
                    )
            result.append(msg)
        return result

    def _create_summary(self, messages: list[dict[str, Any]]) -> str:
        """Create a text summary of conversation messages."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not content:
                continue
            # Extract key info
            preview = content[:150].strip()
            if len(content) > 150:
                preview += "..."
            parts.append(f"{role}: {preview}")

        if not parts:
            return "Previous conversation context."
        return " | ".join(parts[:20])  # Cap at 20 entries

    def _aggressive_truncate(
        self, messages: list[dict[str, Any]], target: int
    ) -> list[dict[str, Any]]:
        """Last resort: drop messages from the front until we fit."""
        if not messages:
            return messages

        # Always keep system message and last few messages
        system = messages[0] if messages[0].get("role") == "system" else None
        rest = messages[1:] if system else messages

        while self.counter.count_messages(
            ([system] if system else []) + rest
        ) > target and len(rest) > 2:
            rest = rest[1:]

        return ([system] if system else []) + rest
