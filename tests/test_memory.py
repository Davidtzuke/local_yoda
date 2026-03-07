"""Tests for the memory subsystem: store, search, context, and embeddings."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from yoda.config import MemorySettings
from yoda.memory.context import ContextWindow, ConversationSummarizer, MemoryRetriever
from yoda.memory.embeddings import EmbeddingPipeline
from yoda.memory.store import SQLiteMemoryStore
from yoda.types import Conversation, MemoryEntry, MemorySearchResult, Role


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_settings(tmp_path: Path) -> MemorySettings:
    return MemorySettings(db_path=tmp_path / "test_memory.db")


@pytest_asyncio.fixture
async def store(memory_settings: MemorySettings):
    s = SQLiteMemoryStore(memory_settings)
    await s.initialize()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def embedder() -> EmbeddingPipeline:
    pipe = EmbeddingPipeline("all-MiniLM-L6-v2")
    await pipe.initialize()
    return pipe


# ---------------------------------------------------------------------------
# EmbeddingPipeline tests
# ---------------------------------------------------------------------------


class TestEmbeddingPipeline:
    async def test_embed_returns_vector(self, embedder: EmbeddingPipeline):
        vec = embedder.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == embedder.dimension
        assert all(isinstance(v, float) for v in vec)

    async def test_embed_empty_string(self, embedder: EmbeddingPipeline):
        vec = embedder.embed("")
        assert all(v == 0.0 for v in vec)

    async def test_embed_deterministic(self, embedder: EmbeddingPipeline):
        v1 = embedder.embed("test input")
        v2 = embedder.embed("test input")
        assert v1 == v2

    async def test_embed_batch(self, embedder: EmbeddingPipeline):
        texts = ["hello", "world", "test"]
        vecs = embedder.embed_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == embedder.dimension for v in vecs)

    async def test_cosine_similarity_identical(self, embedder: EmbeddingPipeline):
        vec = embedder.embed("hello")
        score = EmbeddingPipeline.cosine_similarity(vec, vec)
        assert abs(score - 1.0) < 0.01

    async def test_cosine_similarity_different(self, embedder: EmbeddingPipeline):
        v1 = embedder.embed("the cat sat on the mat")
        v2 = embedder.embed("quantum physics equations")
        score = EmbeddingPipeline.cosine_similarity(v1, v2)
        assert score < 1.0

    async def test_cache_clear(self, embedder: EmbeddingPipeline):
        embedder.embed("cached text")
        assert len(embedder._cache) > 0
        embedder.clear_cache()
        assert len(embedder._cache) == 0


# ---------------------------------------------------------------------------
# SQLiteMemoryStore tests
# ---------------------------------------------------------------------------


class TestSQLiteMemoryStore:
    async def test_store_and_get(self, store: SQLiteMemoryStore):
        entry = MemoryEntry(content="Python is a programming language", source="test")
        entry_id = await store.store(entry)

        retrieved = await store.get(entry_id)
        assert retrieved is not None
        assert retrieved.content == "Python is a programming language"
        assert retrieved.source == "test"

    async def test_get_nonexistent(self, store: SQLiteMemoryStore):
        result = await store.get("nonexistent-id")
        assert result is None

    async def test_delete(self, store: SQLiteMemoryStore):
        entry = MemoryEntry(content="to be deleted")
        entry_id = await store.store(entry)
        assert await store.delete(entry_id) is True
        assert await store.get(entry_id) is None

    async def test_delete_nonexistent(self, store: SQLiteMemoryStore):
        assert await store.delete("nonexistent") is False

    async def test_list_all(self, store: SQLiteMemoryStore):
        for i in range(5):
            await store.store(MemoryEntry(content=f"Memory {i}"))

        entries = await store.list_all(limit=3)
        assert len(entries) == 3

        all_entries = await store.list_all()
        assert len(all_entries) == 5

    async def test_list_all_with_offset(self, store: SQLiteMemoryStore):
        for i in range(5):
            await store.store(MemoryEntry(content=f"Memory {i}"))

        page1 = await store.list_all(limit=2, offset=0)
        page2 = await store.list_all(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].id != page2[0].id

    async def test_count(self, store: SQLiteMemoryStore):
        assert await store.count() == 0
        await store.store(MemoryEntry(content="one"))
        await store.store(MemoryEntry(content="two"))
        assert await store.count() == 2

    async def test_store_with_tags(self, store: SQLiteMemoryStore):
        entry = MemoryEntry(content="tagged entry", tags=["important", "test"])
        entry_id = await store.store(entry)
        retrieved = await store.get(entry_id)
        assert retrieved is not None
        assert "important" in retrieved.tags
        assert "test" in retrieved.tags

    async def test_store_with_metadata(self, store: SQLiteMemoryStore):
        entry = MemoryEntry(
            content="meta entry",
            metadata={"key": "value", "num": 42},
        )
        entry_id = await store.store(entry)
        retrieved = await store.get(entry_id)
        assert retrieved is not None
        assert retrieved.metadata["key"] == "value"
        assert retrieved.metadata["num"] == 42

    async def test_search(self, store: SQLiteMemoryStore):
        await store.store(MemoryEntry(content="Python is great for data science"))
        await store.store(MemoryEntry(content="JavaScript runs in the browser"))
        await store.store(MemoryEntry(content="Rust is a systems programming language"))

        results = await store.search("data analysis programming", top_k=2)
        assert len(results) <= 2
        assert all(r.score >= 0.0 for r in results)

    async def test_search_with_tags_filter(self, store: SQLiteMemoryStore):
        await store.store(MemoryEntry(content="Work meeting notes", tags=["work"]))
        await store.store(MemoryEntry(content="Personal journal entry", tags=["personal"]))

        results = await store.search("meeting", tags=["work"])
        for r in results:
            assert "work" in r.entry.tags

    async def test_search_min_score(self, store: SQLiteMemoryStore):
        await store.store(MemoryEntry(content="Specific technical content about embeddings"))
        results = await store.search("Specific technical content about embeddings", min_score=0.5)
        assert all(r.score >= 0.5 for r in results)

    async def test_search_by_source(self, store: SQLiteMemoryStore):
        await store.store(MemoryEntry(content="From conv 1", source="conversation:abc"))
        await store.store(MemoryEntry(content="From conv 2", source="conversation:xyz"))

        results = await store.search_by_source("conversation:abc")
        assert len(results) == 1
        assert results[0].content == "From conv 1"

    async def test_upsert_replaces(self, store: SQLiteMemoryStore):
        entry = MemoryEntry(id="fixed-id", content="version 1")
        await store.store(entry)

        entry2 = MemoryEntry(id="fixed-id", content="version 2")
        await store.store(entry2)

        retrieved = await store.get("fixed-id")
        assert retrieved is not None
        assert retrieved.content == "version 2"
        assert await store.count() == 1

    async def test_embedding_generated_on_store(self, store: SQLiteMemoryStore):
        entry = MemoryEntry(content="Auto-embed this text")
        await store.store(entry)
        retrieved = await store.get(entry.id)
        assert retrieved is not None
        assert retrieved.embedding is not None
        assert len(retrieved.embedding) > 0


# ---------------------------------------------------------------------------
# ContextWindow tests
# ---------------------------------------------------------------------------


class TestContextWindow:
    def test_build_messages_basic(self):
        ctx = ContextWindow(max_tokens=2000, reserved_for_response=256, system_prompt="You are helpful.")
        conv = Conversation()
        conv.add_message(Role.USER, "Hello!")
        conv.add_message(Role.ASSISTANT, "Hi there!")

        msgs = ctx.build_messages(conv)
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."
        assert msgs[-1]["role"] == "assistant"

    def test_build_messages_with_memories(self):
        ctx = ContextWindow(max_tokens=2000, memory_token_budget=500)
        conv = Conversation()
        conv.add_message(Role.USER, "What's Python?")

        memories = [
            MemorySearchResult(
                entry=MemoryEntry(content="Python is a programming language"),
                score=0.9,
            )
        ]

        msgs = ctx.build_messages(conv, memories=memories, system_prompt="Be helpful.")
        assert len(msgs) >= 2
        memory_msg = [m for m in msgs if "memory" in m["content"].lower()]
        assert len(memory_msg) == 1

    def test_truncation_respects_budget(self):
        ctx = ContextWindow(max_tokens=50, reserved_for_response=10)
        conv = Conversation()
        for i in range(100):
            conv.add_message(Role.USER, f"Message number {i} with some padding text here")

        msgs = ctx.build_messages(conv)
        assert len(msgs) < 100


# ---------------------------------------------------------------------------
# ConversationSummarizer tests
# ---------------------------------------------------------------------------


class TestConversationSummarizer:
    def test_no_compaction_needed(self):
        summarizer = ConversationSummarizer(max_messages_before_compact=50)
        conv = Conversation()
        for i in range(10):
            conv.add_message(Role.USER, f"msg {i}")
        assert not summarizer.needs_compaction(conv)

    def test_compaction_needed(self):
        summarizer = ConversationSummarizer(max_messages_before_compact=10)
        conv = Conversation()
        for i in range(20):
            conv.add_message(Role.USER, f"Message {i} with some content here")
        assert summarizer.needs_compaction(conv)

    def test_compact_produces_summary(self):
        summarizer = ConversationSummarizer(max_messages_before_compact=5, keep_recent=2)
        conv = Conversation()
        for i in range(10):
            role = Role.USER if i % 2 == 0 else Role.ASSISTANT
            conv.add_message(role, f"This is message number {i} in the conversation")

        compacted, memory = summarizer.compact(conv)

        # Should have: 1 summary + 2 recent = 3 messages
        assert len(compacted.messages) == 3
        assert compacted.messages[0].role == Role.SYSTEM
        assert "Summary" in compacted.messages[0].content

        assert memory is not None
        assert "conversation_summary" in memory.tags

    def test_compact_preserves_id(self):
        summarizer = ConversationSummarizer(max_messages_before_compact=5, keep_recent=2)
        conv = Conversation()
        for i in range(10):
            conv.add_message(Role.USER, f"Message {i} with enough content here")

        compacted, _ = summarizer.compact(conv)
        assert compacted.id == conv.id

    def test_no_compact_returns_same(self):
        summarizer = ConversationSummarizer(max_messages_before_compact=50)
        conv = Conversation()
        conv.add_message(Role.USER, "hello")

        result, memory = summarizer.compact(conv)
        assert result is conv
        assert memory is None


# ---------------------------------------------------------------------------
# MemoryRetriever tests
# ---------------------------------------------------------------------------


class TestMemoryRetriever:
    async def test_retrieve_empty_store(self, store: SQLiteMemoryStore):
        retriever = MemoryRetriever(store=store, min_score=0.0)
        results = await retriever.retrieve("anything")
        assert results == []

    async def test_retrieve_returns_results(self, store: SQLiteMemoryStore):
        await store.store(MemoryEntry(content="Python is used for AI"))
        await store.store(MemoryEntry(content="TypeScript is for web"))

        retriever = MemoryRetriever(store=store, top_k=2, min_score=0.0)
        results = await retriever.retrieve("artificial intelligence programming")
        assert len(results) >= 1

    async def test_retrieve_for_conversation(self, store: SQLiteMemoryStore):
        await store.store(MemoryEntry(content="The user likes hiking"))

        conv = Conversation()
        conv.add_message(Role.USER, "What outdoor activities do I enjoy?")

        retriever = MemoryRetriever(store=store, min_score=0.0)
        results = await retriever.retrieve_for_conversation(conv)
        assert isinstance(results, list)

    async def test_retrieve_for_empty_conversation(self, store: SQLiteMemoryStore):
        retriever = MemoryRetriever(store=store)
        conv = Conversation()
        results = await retriever.retrieve_for_conversation(conv)
        assert results == []
