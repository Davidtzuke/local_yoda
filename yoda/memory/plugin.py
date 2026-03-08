"""Memory Plugin — integrates the memory system with the Yoda agent framework.

Exposes memory operations as tools and injects relevant memories as context.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from yoda.core.config import YodaConfig
from yoda.core.plugins import Plugin, ToolParameter, ToolSchema
from yoda.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


class MemoryPlugin(Plugin):
    """Plugin that provides memory tools and context injection to the agent.

    Tools:
    - memory_store: Store a piece of information in memory
    - memory_search: Search memories for relevant information
    - memory_recall: Get specific memories by type/topic
    - memory_forget: Remove a specific memory
    - memory_stats: Get memory statistics
    """

    name = "memory"
    version = "0.1.0"
    description = "Infinite memory with RAG — stores, retrieves, and manages user knowledge"

    def __init__(self, config: YodaConfig) -> None:
        super().__init__(config)
        mem_cfg = config.memory
        self.manager = MemoryManager(
            persist_dir=mem_cfg.persist_dir,
            embedding_model=mem_cfg.embedding_model,
            backend=mem_cfg.backend,
            top_k=mem_cfg.top_k,
            chunk_size=mem_cfg.chunk_size,
            chunk_overlap=mem_cfg.chunk_overlap,
        )

    async def on_load(self) -> None:
        await super().on_load()
        await self.manager.initialize()
        logger.info("Memory plugin loaded")

    async def on_unload(self) -> None:
        await self.manager.close()
        await super().on_unload()

    def tools(self) -> list[ToolSchema]:
        return [
            ToolSchema(
                name="memory_store",
                description="Store information in long-term memory. Use for facts, preferences, or important details the user shares.",
                parameters=[
                    ToolParameter(name="content", type="string", description="The information to remember", required=True),
                    ToolParameter(name="collection", type="string", description="Memory type: episodic, semantic, procedural, or preferences", required=False, default="semantic"),
                    ToolParameter(name="importance", type="number", description="Importance score from 0.0 to 1.0", required=False, default=0.5),
                ],
            ),
            ToolSchema(
                name="memory_search",
                description="Search through stored memories for relevant information. Use when you need to recall something the user told you.",
                parameters=[
                    ToolParameter(name="query", type="string", description="What to search for", required=True),
                    ToolParameter(name="collection", type="string", description="Which memory collection to search (or 'all')", required=False, default="all"),
                    ToolParameter(name="top_k", type="integer", description="Number of results to return", required=False, default=5),
                ],
            ),
            ToolSchema(
                name="memory_recall",
                description="Recall memories of a specific type or about a specific topic.",
                parameters=[
                    ToolParameter(name="topic", type="string", description="Topic to recall memories about", required=True),
                    ToolParameter(name="collection", type="string", description="Memory collection to search", required=False, default="all"),
                ],
            ),
            ToolSchema(
                name="memory_forget",
                description="Remove a specific memory. Use when the user asks you to forget something.",
                parameters=[
                    ToolParameter(name="memory_id", type="string", description="ID of the memory to forget", required=True),
                    ToolParameter(name="collection", type="string", description="Collection the memory belongs to", required=False, default="semantic"),
                ],
            ),
            ToolSchema(
                name="memory_stats",
                description="Get statistics about stored memories.",
                parameters=[],
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "memory_store":
            return await self._store(arguments)
        elif tool_name == "memory_search":
            return await self._search(arguments)
        elif tool_name == "memory_recall":
            return await self._recall(arguments)
        elif tool_name == "memory_forget":
            return await self._forget(arguments)
        elif tool_name == "memory_stats":
            return await self._stats()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def _store(self, args: dict[str, Any]) -> str:
        content = args["content"]
        collection = args.get("collection", "semantic")
        importance = float(args.get("importance", 0.5))

        ids = await self.manager.store(
            content=content,
            collection=collection,
            importance=importance,
            source="explicit",
        )
        return f"Stored in {collection} memory (id: {ids[0]})"

    async def _search(self, args: dict[str, Any]) -> str:
        query = args["query"]
        collection = args.get("collection", "all")
        top_k = int(args.get("top_k", 5))

        collections = None if collection == "all" else [collection]
        results = await self.manager.search(
            query=query,
            collections=collections,
            top_k=top_k,
        )

        if not results:
            return "No relevant memories found."

        output = []
        for doc in results:
            output.append(
                f"[{doc.collection}] (score: {doc.score:.2f}, id: {doc.id})\n{doc.content}"
            )
        return "\n---\n".join(output)

    async def _recall(self, args: dict[str, Any]) -> str:
        topic = args["topic"]
        collection = args.get("collection", "all")
        return await self._search({"query": topic, "collection": collection, "top_k": 5})

    async def _forget(self, args: dict[str, Any]) -> str:
        memory_id = args["memory_id"]
        collection = args.get("collection", "semantic")
        count = await self.manager.delete([memory_id], collection=collection)
        return f"Forgot {count} memory item(s)."

    async def _stats(self) -> str:
        stats = await self.manager.get_stats()
        return json.dumps(stats, indent=2)

    # -- Hooks for auto fact extraction ------------------------------------

    async def on_user_message(self, content: str) -> str | None:
        """Extract facts from user messages automatically."""
        try:
            await self.manager.extract_and_store_facts(content)
        except Exception:
            logger.debug("Auto fact extraction failed", exc_info=True)
        return None  # Don't modify the message

    async def on_assistant_response(self, content: str) -> str | None:
        """No modification needed for assistant responses."""
        return None

    async def on_context_build(self, context: dict[str, Any]) -> dict[str, Any]:
        """Inject relevant memories into the context."""
        # The context injector handles this via the agent's add_context_injector
        return context
