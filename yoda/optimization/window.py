"""Sliding window with priority queue for context items, auto-eviction, importance-weighted truncation."""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from yoda.optimization.tokens import TokenCounter

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Priority levels for context items (higher = more important)."""

    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10  # Never auto-evicted (e.g., system prompt)


@dataclass(order=True)
class ContextItem:
    """A prioritized item in the context window.

    Ordering: lower score = evicted first (min-heap).
    Score = priority * recency_weight * importance.
    """

    score: float = field(compare=True)
    timestamp: float = field(compare=False, default_factory=time.time)
    message: dict[str, Any] = field(compare=False, default_factory=dict)
    priority: Priority = field(compare=False, default=Priority.NORMAL)
    importance: float = field(compare=False, default=1.0)
    tokens: int = field(compare=False, default=0)
    pinned: bool = field(compare=False, default=False)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


class SlidingWindow:
    """Token-budget-aware sliding window with importance-weighted eviction.

    Features:
    - Priority queue: high-priority items survive longer
    - Recency weighting: recent items get higher effective scores
    - Pinning: critical messages are never evicted
    - Auto-eviction: when budget is exceeded, lowest-scored items are dropped
    - Token-aware: tracks exact token counts per item
    """

    def __init__(
        self,
        model: str = "default",
        max_tokens: int | None = None,
        max_messages: int = 100,
        recency_decay: float = 0.95,  # per-message decay factor
    ) -> None:
        self.counter = TokenCounter(model)
        self.max_tokens = max_tokens or self.counter.profile.effective_context
        self.max_messages = max_messages
        self.recency_decay = recency_decay
        self._items: list[ContextItem] = []
        self._total_tokens: int = 0
        self._eviction_count: int = 0

    # -- Scoring -----------------------------------------------------------

    def _compute_score(
        self, priority: Priority, importance: float, position: int
    ) -> float:
        """Compute eviction score. Higher = harder to evict."""
        recency = self.recency_decay ** position  # older = lower
        return priority * importance * recency

    def _rescore_all(self) -> None:
        """Recompute scores for all items based on current positions."""
        n = len(self._items)
        for i, item in enumerate(self._items):
            position = n - 1 - i  # 0 = most recent
            item.score = self._compute_score(item.priority, item.importance, position)

    # -- Add / remove ------------------------------------------------------

    def add(
        self,
        message: dict[str, Any],
        priority: Priority = Priority.NORMAL,
        importance: float = 1.0,
        pinned: bool = False,
    ) -> ContextItem:
        """Add a message to the window. Auto-evicts if over budget."""
        tokens = self.counter.count_message(message)
        score = self._compute_score(priority, importance, 0)

        item = ContextItem(
            score=score,
            message=message,
            priority=priority,
            importance=importance,
            tokens=tokens,
            pinned=pinned,
        )

        self._items.append(item)
        self._total_tokens += tokens
        self._rescore_all()
        self._auto_evict()

        return item

    def add_messages(
        self,
        messages: list[dict[str, Any]],
        priority: Priority = Priority.NORMAL,
    ) -> list[ContextItem]:
        """Add multiple messages. System messages are auto-pinned."""
        items = []
        for msg in messages:
            p = Priority.CRITICAL if msg.get("role") == "system" else priority
            pin = msg.get("role") == "system"
            items.append(self.add(msg, priority=p, pinned=pin))
        return items

    # -- Eviction ----------------------------------------------------------

    def _auto_evict(self) -> None:
        """Evict lowest-priority items until within budget."""
        while (
            self._total_tokens > self.max_tokens
            or len(self._items) > self.max_messages
        ) and len(self._items) > 1:
            evicted = self._evict_one()
            if evicted is None:
                break  # All items are pinned

    def _evict_one(self) -> ContextItem | None:
        """Remove and return the lowest-scored non-pinned item."""
        # Find lowest-scored non-pinned item
        candidates = [
            (i, item)
            for i, item in enumerate(self._items)
            if not item.pinned
        ]
        if not candidates:
            return None

        # Sort by score, evict the lowest
        candidates.sort(key=lambda x: x[1].score)
        idx, item = candidates[0]
        self._items.pop(idx)
        self._total_tokens -= item.tokens
        self._eviction_count += 1
        logger.debug("Evicted context item (score=%.2f, tokens=%d)", item.score, item.tokens)
        return item

    # -- Access ------------------------------------------------------------

    def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages in the window, ordered by timestamp."""
        sorted_items = sorted(self._items, key=lambda x: x.timestamp)
        return [item.message for item in sorted_items]

    def get_messages_within_budget(self, budget: int | None = None) -> list[dict[str, Any]]:
        """Get messages that fit within a specific token budget.

        Prioritizes high-scored items, then fills with remaining.
        """
        target = budget or self.max_tokens
        # Sort by score descending (keep highest-scored first)
        sorted_items = sorted(self._items, key=lambda x: x.score, reverse=True)

        selected: list[ContextItem] = []
        used = 0
        for item in sorted_items:
            if used + item.tokens <= target:
                selected.append(item)
                used += item.tokens

        # Re-sort by timestamp for correct ordering
        selected.sort(key=lambda x: x.timestamp)
        return [item.message for item in selected]

    # -- Stats -------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def message_count(self) -> int:
        return len(self._items)

    @property
    def utilization(self) -> float:
        if self.max_tokens == 0:
            return 1.0
        return self._total_tokens / self.max_tokens

    @property
    def eviction_count(self) -> int:
        return self._eviction_count

    def stats(self) -> dict[str, Any]:
        return {
            "messages": len(self._items),
            "total_tokens": self._total_tokens,
            "max_tokens": self.max_tokens,
            "utilization": f"{self.utilization:.1%}",
            "evictions": self._eviction_count,
            "pinned": sum(1 for i in self._items if i.pinned),
        }

    # -- Utilities ---------------------------------------------------------

    def clear(self) -> None:
        """Clear all items."""
        self._items.clear()
        self._total_tokens = 0

    def pin_message(self, index: int) -> None:
        """Pin a message by index (prevents eviction)."""
        if 0 <= index < len(self._items):
            self._items[index].pinned = True

    def unpin_message(self, index: int) -> None:
        """Unpin a message by index."""
        if 0 <= index < len(self._items):
            self._items[index].pinned = False
