"""Context window management and conversation history compaction.

Handles:
- Trimming conversation history to fit LLM context windows
- Summarizing old messages to preserve important context
- Surfacing relevant memories for the current conversation turn
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from yoda.memory.store import SQLiteMemoryStore
from yoda.types import Conversation, MemoryEntry, MemorySearchResult, Message, Role

logger = logging.getLogger(__name__)

# Rough token estimation: ~4 chars per token for English text
_CHARS_PER_TOKEN = 4


@dataclass
class ContextWindow:
    """Manages what gets sent to the LLM within token budget.

    Assembles a prompt from: system message + relevant memories +
    conversation history, trimmed to fit the context window.
    """

    max_tokens: int = 4096
    reserved_for_response: int = 1024
    memory_token_budget: int = 512
    system_prompt: str = ""

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimate."""
        return max(1, len(text) // _CHARS_PER_TOKEN)

    def build_messages(
        self,
        conversation: Conversation,
        memories: list[MemorySearchResult] | None = None,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Build the message list for the LLM API call.

        Returns a list of {"role": ..., "content": ...} dicts,
        trimmed to fit within the token budget.
        """
        prompt = system_prompt or self.system_prompt
        available = self.max_tokens - self.reserved_for_response

        result: list[dict[str, str]] = []

        # 1. System prompt (always included)
        if prompt:
            result.append({"role": "system", "content": prompt})
            available -= self._estimate_tokens(prompt)

        # 2. Inject relevant memories as system context
        if memories:
            memory_text = self._format_memories(memories)
            mem_tokens = self._estimate_tokens(memory_text)
            if mem_tokens <= min(self.memory_token_budget, available):
                result.append({
                    "role": "system",
                    "content": f"Relevant context from memory:\n{memory_text}",
                })
                available -= mem_tokens

        # 3. Conversation history (fill from newest, then reverse)
        history_msgs: list[dict[str, str]] = []
        for msg in reversed(conversation.messages):
            msg_dict = {"role": msg.role.value, "content": msg.content}
            msg_tokens = self._estimate_tokens(msg.content)
            if msg_tokens > available:
                break
            history_msgs.append(msg_dict)
            available -= msg_tokens

        history_msgs.reverse()
        result.extend(history_msgs)

        return result

    def _format_memories(self, memories: list[MemorySearchResult]) -> str:
        """Format memory search results into a concise text block."""
        lines: list[str] = []
        for m in memories:
            score_pct = int(m.score * 100)
            tags = ", ".join(m.entry.tags) if m.entry.tags else "none"
            lines.append(f"- [{score_pct}% match, tags: {tags}] {m.entry.content}")
        return "\n".join(lines)


@dataclass
class ConversationSummarizer:
    """Compacts long conversation histories by summarizing older messages.

    Uses a simple extractive approach (keeps important messages, drops filler).
    For LLM-based abstractive summarization, the Agent Core can call the LLM
    and store the result via this class.
    """

    max_messages_before_compact: int = 50
    keep_recent: int = 10

    def needs_compaction(self, conversation: Conversation) -> bool:
        """Check if the conversation should be compacted."""
        return len(conversation.messages) > self.max_messages_before_compact

    def compact(self, conversation: Conversation) -> tuple[Conversation, MemoryEntry | None]:
        """Compact a conversation by summarizing old messages.

        Returns:
            (compacted conversation, optional memory entry with the summary).
            The memory entry can be stored for future retrieval.
        """
        if not self.needs_compaction(conversation):
            return conversation, None

        old_messages = conversation.messages[: -self.keep_recent]
        recent_messages = conversation.messages[-self.keep_recent :]

        # Build extractive summary from old messages
        summary_parts: list[str] = []
        for msg in old_messages:
            if msg.role in (Role.USER, Role.ASSISTANT) and len(msg.content) > 20:
                prefix = "User" if msg.role == Role.USER else "Assistant"
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                summary_parts.append(f"{prefix}: {content}")

        summary_text = (
            f"Summary of earlier conversation ({len(old_messages)} messages):\n"
            + "\n".join(summary_parts[-20:])  # Keep last 20 significant exchanges
        )

        memory_entry = MemoryEntry(
            content=summary_text,
            source=f"conversation:{conversation.id}",
            tags=["conversation_summary", "auto_compacted"],
        )

        summary_msg = Message(role=Role.SYSTEM, content=summary_text)

        compacted = Conversation(
            id=conversation.id,
            messages=[summary_msg, *recent_messages],
            created_at=conversation.created_at,
            title=conversation.title,
            metadata={**conversation.metadata, "compacted": True},
        )

        logger.info(
            "Compacted conversation %s: %d -> %d messages",
            conversation.id[:8],
            len(conversation.messages),
            len(compacted.messages),
        )

        return compacted, memory_entry


@dataclass
class MemoryRetriever:
    """Retrieves relevant memories for the current conversation context.

    Combines semantic search with recency weighting.
    """

    store: SQLiteMemoryStore
    top_k: int = 5
    min_score: float = 0.3
    recency_boost: float = 0.1

    async def retrieve(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[MemorySearchResult]:
        """Find memories relevant to the query with recency boosting."""
        k = top_k or self.top_k
        results = await self.store.search(
            query, top_k=k * 2, min_score=self.min_score, tags=tags
        )

        if not results:
            return []

        # Apply recency boost: newer entries get a small score bump
        if len(results) > 1:
            by_date = sorted(results, key=lambda r: r.entry.created_at)
            for i, r in enumerate(by_date):
                recency_factor = (i / len(by_date)) * self.recency_boost
                r.score = min(1.0, r.score + recency_factor)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    async def retrieve_for_conversation(
        self,
        conversation: Conversation,
        *,
        top_k: int | None = None,
    ) -> list[MemorySearchResult]:
        """Retrieve memories relevant to the latest user message in a conversation."""
        last_msg = conversation.last_user_message
        if last_msg is None:
            return []
        return await self.retrieve(last_msg.content, top_k=top_k)
