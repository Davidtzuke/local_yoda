"""Memory Manager — orchestrates storage, retrieval, fact extraction, and memory lifecycle.

Key features:
- Auto fact extraction from conversations
- Importance scoring for memories
- Forgetting curve with memory consolidation
- Preference learning over time
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any
from uuid import uuid4

from yoda.memory.chunking import ChunkingStrategy, create_chunker
from yoda.memory.embeddings import EmbeddingProvider, create_embedder
from yoda.memory.persistence import MemoryMetadataStore
from yoda.memory.retrieval import RetrievalPipeline
from yoda.memory.vector_store import Document, VectorStore, create_vector_store

logger = logging.getLogger(__name__)

# Patterns for extracting different types of facts
FACT_PATTERNS = {
    "preference": [
        r"(?:I |my )(prefer|like|love|enjoy|hate|dislike|want|need|use|always|never)\b(.+?)(?:\.|$)",
        r"(?:I'm |I am )(a |an )(.+?)(?:\.|$)",
        r"my (?:favorite|preferred|usual) (.+?) is (.+?)(?:\.|$)",
    ],
    "fact": [
        r"(?:I |my )(name|age|job|work|live|born|from|study|studied)\b(.+?)(?:\.|$)",
        r"(?:I )(have|had|own|bought|sold|made|created|built)\b(.+?)(?:\.|$)",
        r"(?:my )(wife|husband|partner|dog|cat|pet|car|house|phone)\b(.+?)(?:\.|$)",
    ],
    "procedural": [
        r"(?:to |how to |the way to )((?:do|make|create|build|fix|solve|handle).+?)(?:\.|$)",
        r"(?:the process|steps|procedure) (?:for|to) (.+?)(?:\.|$)",
        r"(?:when you |if you )(need to|want to|have to) (.+?)(?:\.|$)",
    ],
}


class MemoryManager:
    """Orchestrates the full memory lifecycle.

    Manages:
    - Document ingestion (chunking → embedding → storage)
    - Retrieval (hybrid search → re-ranking → compression)
    - Fact extraction from conversations
    - Memory consolidation and forgetting
    - Preference tracking
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: EmbeddingProvider | None = None,
        chunker: ChunkingStrategy | None = None,
        metadata_store: MemoryMetadataStore | None = None,
        retrieval_pipeline: RetrievalPipeline | None = None,
        persist_dir: str = "~/.yoda/memory",
        embedding_model: str = "all-MiniLM-L6-v2",
        backend: str = "chromadb",
        top_k: int = 10,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self._persist_dir = persist_dir
        self._top_k = top_k

        # Create components with defaults
        self.vector_store = vector_store or create_vector_store(
            backend=backend, persist_dir=persist_dir
        )
        self.embedder = embedder or create_embedder(
            model=embedding_model,
            cache_dir=f"{persist_dir}/embed_cache",
        )
        self.chunker = chunker or create_chunker(
            strategy="semantic",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.metadata_store = metadata_store or MemoryMetadataStore(
            db_path=f"{persist_dir}/metadata.db"
        )

        self._pipeline: RetrievalPipeline | None = retrieval_pipeline
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all memory components."""
        await self.vector_store.initialize()
        await self.metadata_store.initialize()

        if self._pipeline is None:
            self._pipeline = RetrievalPipeline(
                vector_store=self.vector_store,
                embedder=self.embedder,
            )

        self._initialized = True
        logger.info("MemoryManager initialized")

    async def close(self) -> None:
        """Clean up resources."""
        await self.vector_store.close()
        await self.metadata_store.close()
        self._initialized = False

    # -- Core operations ---------------------------------------------------

    async def store(
        self,
        content: str,
        collection: str = "semantic",
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        source: str = "conversation",
        chunk: bool = False,
    ) -> list[str]:
        """Store content in memory.

        Args:
            content: Text content to store
            collection: Memory collection type
            metadata: Additional metadata
            importance: Importance score [0, 1]
            source: Where this memory came from
            chunk: Whether to chunk the content first
        """
        self._ensure_initialized()
        meta = metadata or {}
        meta.update({
            "created_at": time.time(),
            "importance": importance,
            "source": source,
        })

        if chunk and len(content) > 300:
            chunks = self.chunker.chunk(content)
            docs = []
            for c in chunks:
                doc = Document(
                    content=c.content,
                    metadata={**meta, **c.metadata, "chunk_index": c.index},
                    collection=collection,
                )
                docs.append(doc)
        else:
            docs = [Document(content=content, metadata=meta, collection=collection)]

        # Embed all documents
        texts = [doc.content for doc in docs]
        embeddings = await self.embedder.embed(texts)
        for doc, emb in zip(docs, embeddings):
            doc.embedding = emb

        # Store in vector store
        ids = await self.vector_store.add(docs, collection=collection)

        # Store metadata
        await self.metadata_store.store_batch(docs)

        # Update BM25 index
        if self._pipeline:
            self._pipeline.update_bm25_corpus(docs)

        logger.debug("Stored %d memory items in %s", len(ids), collection)
        return ids

    async def search(
        self,
        query: str,
        collections: list[str] | None = None,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Search memories using the hybrid retrieval pipeline."""
        self._ensure_initialized()
        k = top_k or self._top_k

        if self._pipeline:
            results = await self._pipeline.retrieve(
                query=query,
                collections=collections,
                top_k=k,
                filter_metadata=filter_metadata,
            )
        else:
            # Fallback to simple vector search
            embedding = await self.embedder.embed_single(query)
            results = await self.vector_store.search(
                query_embedding=embedding,
                collection=(collections or ["semantic"])[0],
                top_k=k,
                filter_metadata=filter_metadata,
            )

        # Record access for forgetting curve
        for doc in results:
            await self.metadata_store.record_access(doc.id)

        return results

    async def delete(self, ids: list[str], collection: str = "semantic") -> int:
        """Delete memories by IDs."""
        self._ensure_initialized()
        count = await self.vector_store.delete(ids, collection=collection)
        await self.metadata_store.delete(ids)
        return count

    # -- Fact extraction ---------------------------------------------------

    async def extract_and_store_facts(
        self,
        user_message: str,
        assistant_response: str = "",
    ) -> list[str]:
        """Extract facts from a conversation turn and store them.

        Automatically categorizes facts into appropriate collections:
        - preferences → preferences collection
        - personal facts → semantic collection
        - how-to knowledge → procedural collection
        - events/experiences → episodic collection
        """
        stored_ids: list[str] = []

        # Extract different types of facts
        for fact_type, patterns in FACT_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, user_message, re.IGNORECASE)
                for match in matches:
                    fact = match.group(0).strip()
                    if len(fact) < 10:
                        continue

                    collection = self._fact_type_to_collection(fact_type)
                    importance = self._score_importance(fact, fact_type)

                    ids = await self.store(
                        content=fact,
                        collection=collection,
                        importance=importance,
                        source="extracted",
                        metadata={"fact_type": fact_type, "original_message": user_message[:200]},
                    )
                    stored_ids.extend(ids)

        # Also store the full message as an episodic memory if it's substantial
        if len(user_message) > 50:
            ids = await self.store(
                content=f"User said: {user_message}",
                collection="episodic",
                importance=0.4,
                source="conversation",
                metadata={"type": "conversation_turn"},
            )
            stored_ids.extend(ids)

        return stored_ids

    def _fact_type_to_collection(self, fact_type: str) -> str:
        return {
            "preference": "preferences",
            "fact": "semantic",
            "procedural": "procedural",
        }.get(fact_type, "semantic")

    def _score_importance(self, fact: str, fact_type: str) -> float:
        """Score the importance of an extracted fact."""
        base_scores = {"preference": 0.7, "fact": 0.6, "procedural": 0.5}
        score = base_scores.get(fact_type, 0.5)

        # Boost for personal information
        personal_keywords = {"name", "birthday", "age", "live", "work", "family"}
        if any(kw in fact.lower() for kw in personal_keywords):
            score = min(1.0, score + 0.2)

        # Boost for strong preferences
        strong_words = {"love", "hate", "always", "never", "favorite"}
        if any(w in fact.lower() for w in strong_words):
            score = min(1.0, score + 0.15)

        return round(score, 2)

    # -- Memory consolidation ----------------------------------------------

    async def consolidate(self, max_items: int = 50) -> int:
        """Consolidate memories: merge similar, prune decayed ones.

        This implements a simplified version of memory consolidation:
        1. Find memories that have decayed below threshold
        2. Either delete them or merge with similar active memories
        """
        self._ensure_initialized()
        pruned = 0

        # Get decayed memories
        decayed = await self.metadata_store.get_decayed_memories(threshold=0.1, limit=max_items)

        if not decayed:
            return 0

        # Delete low-importance decayed memories
        ids_to_delete = [
            m["id"] for m in decayed
            if m.get("importance", 0) < 0.3
        ]

        if ids_to_delete:
            for collection in ["episodic", "semantic", "procedural", "preferences"]:
                await self.vector_store.delete(ids_to_delete, collection=collection)
            await self.metadata_store.delete(ids_to_delete)
            pruned = len(ids_to_delete)

            await self.metadata_store.log_consolidation(
                action="prune_decayed",
                source_ids=ids_to_delete,
                details=f"Pruned {pruned} decayed memories",
            )

        logger.info("Consolidated memories: pruned %d", pruned)
        return pruned

    async def boost_importance(self, doc_id: str, boost: float = 0.1) -> None:
        """Boost a memory's importance (e.g., when user references it again)."""
        meta = await self.metadata_store.get(doc_id)
        if meta:
            new_importance = min(1.0, meta.get("importance", 0.5) + boost)
            await self.metadata_store.update_importance(doc_id, new_importance)

    # -- Context injection -------------------------------------------------

    def get_context_injector(self) -> Any:
        """Return a context injector function for the Agent.

        This retrieves relevant memories based on the current conversation
        and injects them into the system prompt.
        """
        import asyncio

        manager = self

        def injector(conversation: Any) -> dict[str, Any]:
            """Inject relevant memories into the agent context."""
            # Get the last user message
            recent_messages = conversation.messages[-3:] if conversation.messages else []
            user_messages = [
                m.content for m in recent_messages
                if hasattr(m, 'role') and str(m.role) == "user"
            ]

            if not user_messages:
                return {}

            query = user_messages[-1]

            # Run async search in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're already in an async context
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        results = pool.submit(
                            lambda: asyncio.run(manager.search(query, top_k=5))
                        ).result(timeout=5)
                else:
                    results = loop.run_until_complete(manager.search(query, top_k=5))
            except Exception:
                logger.debug("Memory context injection failed", exc_info=True)
                return {}

            if not results:
                return {}

            memory_context = "\n".join(
                f"- [{doc.collection}] {doc.content}" for doc in results
            )
            return {"relevant_memories": memory_context}

        return injector

    # -- Stats & utilities -------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        self._ensure_initialized()
        meta_stats = await self.metadata_store.get_stats()

        # Add vector store counts
        collections = await self.vector_store.list_collections()
        vector_counts = {}
        for coll in collections:
            vector_counts[coll] = await self.vector_store.count(coll)

        return {
            **meta_stats,
            "vector_store_counts": vector_counts,
        }

    async def export_memories(self, output_path: str) -> int:
        """Export all memories to JSON."""
        return await self.metadata_store.export_all(output_path)

    async def import_memories(self, input_path: str) -> int:
        """Import memories from JSON."""
        return await self.metadata_store.import_from(input_path)

    async def backup(self) -> str:
        """Create a backup of the memory database."""
        return await self.metadata_store.backup()

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")
