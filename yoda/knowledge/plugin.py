"""Knowledge Graph plugin — integrates the graph into the Yoda agent loop."""

from __future__ import annotations

import json
import logging
from typing import Any

from yoda.core.config import YodaConfig
from yoda.core.plugins import Plugin, ToolSchema, ToolParameter
from yoda.knowledge.graph import KnowledgeGraph
from yoda.knowledge.extractor import EntityExtractor
from yoda.knowledge.queries import GraphQueryEngine
from yoda.knowledge.reasoning import ReasoningEngine
from yoda.knowledge.updater import GraphUpdater
from yoda.knowledge.visualization import GraphVisualizer

logger = logging.getLogger(__name__)


class KnowledgeGraphPlugin(Plugin):
    """Plugin that exposes knowledge graph tools to the Yoda agent."""

    name = "knowledge_graph"
    version = "0.1.0"
    description = "Knowledge graph for entity relationships, reasoning, and inference"

    def __init__(self, config: YodaConfig) -> None:
        super().__init__(config)
        kg_config = config.knowledge_graph
        self._graph = KnowledgeGraph(persist_path=kg_config.persist_path)
        self._extractor = EntityExtractor()
        self._query_engine: GraphQueryEngine | None = None
        self._reasoning: ReasoningEngine | None = None
        self._updater: GraphUpdater | None = None
        self._visualizer: GraphVisualizer | None = None
        self._max_hops = kg_config.max_hops

    # -- Lifecycle ---------------------------------------------------------

    async def on_load(self) -> None:
        await self._graph.initialize()
        self._query_engine = GraphQueryEngine(self._graph)
        self._reasoning = ReasoningEngine(self._graph, max_hops=self._max_hops)
        self._updater = GraphUpdater(self._graph, self._extractor)
        self._visualizer = GraphVisualizer(self._graph)
        self._loaded = True
        logger.info(
            "Knowledge graph loaded: %d entities, %d relationships",
            self._graph.num_entities, self._graph.num_relationships,
        )

    async def on_unload(self) -> None:
        await self._graph.close()
        self._loaded = False

    # -- Tools -------------------------------------------------------------

    def tools(self) -> list[ToolSchema]:
        return [
            ToolSchema(
                name="kg_query",
                description="Query the knowledge graph with a natural language question. Use for lookups like 'What do you know about X?' or 'How are X and Y related?'",
                parameters=[
                    ToolParameter(name="question", type="string", description="Natural language question", required=True),
                ],
            ),
            ToolSchema(
                name="kg_add_entity",
                description="Add a new entity (person, place, concept, etc.) to the knowledge graph",
                parameters=[
                    ToolParameter(name="name", type="string", description="Entity name", required=True),
                    ToolParameter(name="entity_type", type="string", description="Type: person, place, organization, concept, event, preference"),
                    ToolParameter(name="properties", type="object", description="Additional properties as key-value pairs"),
                ],
            ),
            ToolSchema(
                name="kg_add_relation",
                description="Add a relationship between two entities in the knowledge graph",
                parameters=[
                    ToolParameter(name="source", type="string", description="Source entity name", required=True),
                    ToolParameter(name="target", type="string", description="Target entity name", required=True),
                    ToolParameter(name="relation_type", type="string", description="Relationship type (e.g., works_at, knows, prefers)", required=True),
                ],
            ),
            ToolSchema(
                name="kg_reason",
                description="Perform multi-hop reasoning about an entity or between two entities",
                parameters=[
                    ToolParameter(name="entity", type="string", description="Entity to reason about", required=True),
                    ToolParameter(name="target", type="string", description="Optional second entity for relationship inference"),
                ],
            ),
            ToolSchema(
                name="kg_visualize",
                description="Generate a visualization of the knowledge graph (Mermaid diagram or ASCII tree)",
                parameters=[
                    ToolParameter(name="format", type="string", description="Output format: mermaid, ascii, d3"),
                    ToolParameter(name="entity", type="string", description="Center visualization on this entity"),
                ],
            ),
            ToolSchema(
                name="kg_stats",
                description="Get knowledge graph statistics (entity count, relationship count, types)",
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        handlers = {
            "kg_query": self._handle_query,
            "kg_add_entity": self._handle_add_entity,
            "kg_add_relation": self._handle_add_relation,
            "kg_reason": self._handle_reason,
            "kg_visualize": self._handle_visualize,
            "kg_stats": self._handle_stats,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"
        return await handler(arguments)

    # -- Handlers ----------------------------------------------------------

    async def _handle_query(self, args: dict[str, Any]) -> str:
        assert self._query_engine
        question = args.get("question", "")
        result = await self._query_engine.query(question)
        return result.to_text()

    async def _handle_add_entity(self, args: dict[str, Any]) -> str:
        from yoda.knowledge.graph import Entity
        entity = Entity(
            name=args["name"],
            entity_type=args.get("entity_type", "concept"),
            properties=args.get("properties", {}),
            source="manual",
        )
        added = self._graph.add_entity(entity)
        return f"Added entity: {added.name} ({added.entity_type}) [id: {added.id}]"

    async def _handle_add_relation(self, args: dict[str, Any]) -> str:
        from yoda.knowledge.graph import Relationship
        source = self._graph.find_entity(args["source"])
        target = self._graph.find_entity(args["target"])

        if not source:
            # Auto-create source
            source = self._graph.add_entity(Entity(name=args["source"], source="manual"))
        if not target:
            # Auto-create target
            target = self._graph.add_entity(Entity(name=args["target"], source="manual"))

        rel = Relationship(
            source_id=source.id,
            target_id=target.id,
            relation_type=args.get("relation_type", "related_to"),
            source="manual",
        )
        added = self._graph.add_relationship(rel)
        return f"Added: {source.name} --[{added.relation_type}]--> {target.name}"

    async def _handle_reason(self, args: dict[str, Any]) -> str:
        assert self._reasoning
        entity_name = args.get("entity", "")
        target_name = args.get("target")

        if target_name:
            result = self._reasoning.infer_relationship(entity_name, target_name)
        else:
            result = self._reasoning.reason_about(entity_name)

        return result.to_text()

    async def _handle_visualize(self, args: dict[str, Any]) -> str:
        assert self._visualizer
        fmt = args.get("format", "mermaid")
        entity_name = args.get("entity")

        if fmt == "ascii" and entity_name:
            entity = self._graph.find_entity(entity_name)
            if entity:
                return self._visualizer.to_ascii(entity.id)
            return f"Entity '{entity_name}' not found."

        if fmt == "mermaid":
            entity_ids = None
            if entity_name:
                entity = self._graph.find_entity(entity_name)
                if entity:
                    neighbors = self._graph.get_neighbors(entity.id, max_hops=2)
                    entity_ids = [entity.id] + [n[0].id for n in neighbors]
            return self._visualizer.to_mermaid(entity_ids=entity_ids)

        if fmt == "d3":
            data = self._visualizer.to_d3_json()
            return json.dumps(data, indent=2, default=str)

        return f"Unknown format: {fmt}. Use: mermaid, ascii, d3"

    async def _handle_stats(self, args: dict[str, Any]) -> str:
        stats = self._graph.get_stats()
        lines = [
            f"Entities: {stats['entities']}",
            f"Relationships: {stats['relationships']}",
            f"Connected components: {stats['connected_components']}",
        ]
        if stats["entity_types"]:
            lines.append("Entity types:")
            for t, c in sorted(stats["entity_types"].items()):
                lines.append(f"  {t}: {c}")
        if stats["relationship_types"]:
            lines.append("Relationship types:")
            for t, c in sorted(stats["relationship_types"].items()):
                lines.append(f"  {t}: {c}")
        return "\n".join(lines)

    # -- Agent hooks -------------------------------------------------------

    async def on_user_message(self, content: str) -> str | None:
        """Auto-extract entities from user messages."""
        if self._updater and len(content) > 10:
            try:
                updates = await self._updater.process_message(content)
                if updates:
                    logger.debug("Graph updates: %s", updates)
            except Exception:
                logger.exception("Failed to process message for graph updates")
        return None

    async def on_context_build(self, context: dict[str, Any]) -> dict[str, Any]:
        """Inject relevant graph context before LLM calls."""
        return context

    def get_context_injector(self):
        """Return an async function that injects graph context."""

        async def inject_graph_context(messages: list[dict[str, Any]]) -> str:
            """Search the knowledge graph based on recent messages."""
            if not messages:
                return ""

            # Get last user message
            last_user = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user = msg.get("content", "")
                    break

            if not last_user or not self._query_engine:
                return ""

            result = await self._query_engine.query(last_user)
            if not result.entities and not result.relationships:
                return ""

            context = result.to_context()
            if context:
                return f"<knowledge_graph>\n{context}\n</knowledge_graph>"
            return ""

        return inject_graph_context
