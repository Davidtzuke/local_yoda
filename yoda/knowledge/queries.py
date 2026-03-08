"""Graph query engine — natural language to graph queries, path finding, temporal queries."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from yoda.knowledge.graph import KnowledgeGraph, Entity, Relationship

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query result types
# ---------------------------------------------------------------------------

class QueryResult:
    """Structured result from a graph query."""

    def __init__(
        self,
        entities: list[Entity] | None = None,
        relationships: list[Relationship] | None = None,
        paths: list[list[tuple[Entity, Relationship | None]]] | None = None,
        answer: str = "",
        confidence: float = 1.0,
    ) -> None:
        self.entities = entities or []
        self.relationships = relationships or []
        self.paths = paths or []
        self.answer = answer
        self.confidence = confidence

    def to_text(self) -> str:
        """Convert result to human-readable text."""
        parts: list[str] = []
        if self.answer:
            parts.append(self.answer)

        if self.entities:
            entity_strs = [f"- {e.name} ({e.entity_type})" for e in self.entities]
            parts.append("Entities:\n" + "\n".join(entity_strs))

        if self.relationships:
            rel_strs = []
            for r in self.relationships:
                rel_strs.append(f"- {r.source_id} --[{r.relation_type}]--> {r.target_id}")
            parts.append("Relationships:\n" + "\n".join(rel_strs))

        if self.paths:
            for i, path in enumerate(self.paths):
                path_parts = []
                for entity, rel in path:
                    if rel:
                        path_parts.append(f"--[{rel.relation_type}]--> {entity.name}")
                    else:
                        path_parts.append(entity.name)
                parts.append(f"Path {i + 1}: {' '.join(path_parts)}")

        return "\n\n".join(parts) if parts else "No results found."

    def to_context(self) -> str:
        """Format as concise context for LLM injection."""
        lines: list[str] = []
        for e in self.entities:
            props = ", ".join(f"{k}={v}" for k, v in e.properties.items())
            line = f"{e.name} ({e.entity_type})"
            if props:
                line += f" [{props}]"
            lines.append(line)

        for r in self.relationships:
            lines.append(f"{r.source_id} -{r.relation_type}-> {r.target_id}")

        return "; ".join(lines)


# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------

class GraphQueryEngine:
    """Translate natural language questions into graph operations."""

    def __init__(self, graph: KnowledgeGraph, llm_provider: Any = None) -> None:
        self._graph = graph
        self._llm = llm_provider

    # -- High-level query --------------------------------------------------

    async def query(self, question: str) -> QueryResult:
        """Answer a natural language question using the knowledge graph.

        Tries pattern matching first, then LLM-based query planning.
        """
        # 1. Try pattern-based query
        result = self._pattern_query(question)
        if result.entities or result.relationships or result.answer:
            return result

        # 2. Try entity-centric search
        result = self._entity_search_query(question)
        if result.entities or result.relationships:
            return result

        # 3. LLM-based query planning
        if self._llm:
            return await self._llm_query(question)

        return QueryResult(answer="No matching information found in the knowledge graph.")

    # -- Pattern-based queries ---------------------------------------------

    def _pattern_query(self, question: str) -> QueryResult:
        """Match common question patterns to graph operations."""
        q = question.lower().strip().rstrip("?")

        # "What is X?" / "Who is X?"
        match = re.match(r"(?:what|who)\s+is\s+(.+)", q)
        if match:
            return self._lookup_entity(match.group(1).strip())

        # "Where does X work/live?"
        match = re.match(r"where\s+(?:does|did)\s+(.+?)\s+(work|live|study)", q)
        if match:
            rel_map = {"work": "works_at", "live": "lives_in", "study": "studies_at"}
            return self._query_relation(match.group(1), rel_map.get(match.group(2), ""))

        # "What does X like/prefer?"
        match = re.match(r"what\s+(?:does|did)\s+(.+?)\s+(?:like|prefer|enjoy)", q)
        if match:
            return self._query_relation(match.group(1), "prefers")

        # "How is X related to Y?" / "What connects X and Y?"
        match = re.match(r"(?:how\s+is|what\s+connects)\s+(.+?)\s+(?:related to|and|to)\s+(.+)", q)
        if match:
            return self._find_connection(match.group(1).strip(), match.group(2).strip())

        # "Tell me about X" / "What do you know about X?"
        match = re.match(r"(?:tell me about|what do you know about)\s+(.+)", q)
        if match:
            return self._full_entity_profile(match.group(1).strip())

        return QueryResult()

    def _lookup_entity(self, name: str) -> QueryResult:
        """Look up an entity and its immediate relationships."""
        entity = self._graph.find_entity(name)
        if not entity:
            results = self._graph.search_entities(name, limit=3)
            if results:
                entity = results[0]

        if not entity:
            return QueryResult()

        rels = self._graph.get_relationships(entity.id)
        answer_parts = [f"{entity.name} is a {entity.entity_type}."]

        for rel in rels:
            target = self._graph.get_entity(rel.target_id)
            source = self._graph.get_entity(rel.source_id)
            if target and rel.source_id == entity.id:
                answer_parts.append(
                    f"{entity.name} {_humanize_relation(rel.relation_type)} {target.name}."
                )
            elif source and rel.target_id == entity.id:
                answer_parts.append(
                    f"{source.name} {_humanize_relation(rel.relation_type)} {entity.name}."
                )

        if entity.properties:
            for k, v in entity.properties.items():
                answer_parts.append(f"{k}: {v}")

        return QueryResult(
            entities=[entity],
            relationships=rels,
            answer=" ".join(answer_parts),
        )

    def _query_relation(self, entity_name: str, relation_type: str) -> QueryResult:
        """Query a specific relation for an entity."""
        entity = self._graph.find_entity(entity_name)
        if not entity:
            return QueryResult()

        rels = self._graph.get_relationships(entity.id, relation_type=relation_type)
        entities: list[Entity] = []
        answers: list[str] = []

        for rel in rels:
            target_id = rel.target_id if rel.source_id == entity.id else rel.source_id
            target = self._graph.get_entity(target_id)
            if target:
                entities.append(target)
                answers.append(
                    f"{entity.name} {_humanize_relation(rel.relation_type)} {target.name}."
                )

        return QueryResult(
            entities=entities,
            relationships=rels,
            answer=" ".join(answers) if answers else f"No {relation_type} found for {entity_name}.",
        )

    def _find_connection(self, name1: str, name2: str) -> QueryResult:
        """Find how two entities are connected."""
        e1 = self._graph.find_entity(name1)
        e2 = self._graph.find_entity(name2)

        if not e1 or not e2:
            return QueryResult(answer=f"Could not find both '{name1}' and '{name2}'.")

        path = self._graph.shortest_path(e1.id, e2.id)
        if not path:
            return QueryResult(answer=f"No connection found between {e1.name} and {e2.name}.")

        path_desc = []
        for entity, rel in path:
            if rel:
                path_desc.append(f"--[{_humanize_relation(rel.relation_type)}]--> {entity.name}")
            else:
                path_desc.append(entity.name)

        return QueryResult(
            paths=[path],
            answer=f"Connection: {' '.join(path_desc)}",
        )

    def _full_entity_profile(self, name: str) -> QueryResult:
        """Build a complete profile for an entity."""
        entity = self._graph.find_entity(name)
        if not entity:
            results = self._graph.search_entities(name, limit=1)
            entity = results[0] if results else None

        if not entity:
            return QueryResult(answer=f"I don't have information about '{name}'.")

        neighbors = self._graph.get_neighbors(entity.id, max_hops=2)
        rels = self._graph.get_relationships(entity.id)

        profile_parts = [f"**{entity.name}** ({entity.entity_type})"]

        if entity.properties:
            for k, v in entity.properties.items():
                profile_parts.append(f"  {k}: {v}")

        if rels:
            profile_parts.append("\nRelationships:")
            for rel in rels:
                other_id = rel.target_id if rel.source_id == entity.id else rel.source_id
                other = self._graph.get_entity(other_id)
                if other:
                    if rel.source_id == entity.id:
                        profile_parts.append(
                            f"  {_humanize_relation(rel.relation_type)} {other.name}"
                        )
                    else:
                        profile_parts.append(
                            f"  {other.name} {_humanize_relation(rel.relation_type)} this"
                        )

        return QueryResult(
            entities=[entity] + [n[0] for n in neighbors],
            relationships=rels,
            answer="\n".join(profile_parts),
        )

    # -- Entity search fallback -------------------------------------------

    def _entity_search_query(self, question: str) -> QueryResult:
        """Fall back to searching for entity names mentioned in the question."""
        # Extract potential entity names (capitalized words)
        words = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", question)
        entities: list[Entity] = []
        rels: list[Relationship] = []

        for word in words:
            found = self._graph.search_entities(word, limit=2)
            entities.extend(found)

        # Also search lowercase keywords
        keywords = re.findall(r"\b\w{3,}\b", question.lower())
        stopwords = {"what", "who", "where", "when", "how", "does", "the", "about", "tell", "know", "that", "this", "with", "from", "have", "has"}
        for kw in keywords:
            if kw not in stopwords:
                found = self._graph.search_entities(kw, limit=1)
                for e in found:
                    if e.id not in {x.id for x in entities}:
                        entities.append(e)

        for entity in entities:
            rels.extend(self._graph.get_relationships(entity.id))

        # Deduplicate
        seen_rel_ids: set[str] = set()
        unique_rels = []
        for r in rels:
            if r.id not in seen_rel_ids:
                seen_rel_ids.add(r.id)
                unique_rels.append(r)

        return QueryResult(entities=entities, relationships=unique_rels)

    # -- Temporal queries --------------------------------------------------

    def query_temporal(
        self,
        entity_name: str,
        at_time: float | None = None,
        from_time: float | None = None,
        to_time: float | None = None,
    ) -> QueryResult:
        """Query relationships valid at a specific time or range."""
        entity = self._graph.find_entity(entity_name)
        if not entity:
            return QueryResult()

        all_rels = self._graph.get_relationships(entity.id)
        filtered: list[Relationship] = []

        for rel in all_rels:
            if at_time and not rel.is_valid_at(at_time):
                continue
            if from_time and rel.valid_until and rel.valid_until < from_time:
                continue
            if to_time and rel.valid_from and rel.valid_from > to_time:
                continue
            filtered.append(rel)

        return QueryResult(
            entities=[entity],
            relationships=filtered,
            answer=f"Found {len(filtered)} relationships for {entity.name} in the given time range.",
        )

    # -- LLM-based query planning -----------------------------------------

    async def _llm_query(self, question: str) -> QueryResult:
        """Use LLM to decompose question into graph operations."""
        if not self._llm:
            return QueryResult()

        # Provide graph summary as context
        stats = self._graph.get_stats()
        entity_names = [e.name for e in self._graph.all_entities()]
        sample = entity_names[:30]

        prompt = f"""Given this knowledge graph with {stats['entities']} entities and {stats['relationships']} relationships:
Known entities (sample): {', '.join(sample)}

Answer this question using ONLY the graph data: {question}

If you can identify entity names from the question, list them. I'll look them up.
Respond with JSON: {{"entity_names": ["..."], "relation_types": ["..."], "operation": "lookup|path|neighbors"}}"""

        try:
            from yoda.core.messages import UserMessage
            messages = [UserMessage(content=prompt).to_provider_format()]
            response = await self._llm.complete(messages)
            content = response.content or ""

            # Parse LLM suggestions and execute
            import json as json_mod
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                plan = json_mod.loads(match.group())
                return self._execute_query_plan(plan)
        except Exception:
            logger.exception("LLM query planning failed")

        return QueryResult(answer="Could not find relevant information.")

    def _execute_query_plan(self, plan: dict[str, Any]) -> QueryResult:
        """Execute a structured query plan from the LLM."""
        entities: list[Entity] = []
        rels: list[Relationship] = []

        for name in plan.get("entity_names", []):
            entity = self._graph.find_entity(name)
            if entity:
                entities.append(entity)
                entity_rels = self._graph.get_relationships(entity.id)
                if plan.get("relation_types"):
                    entity_rels = [
                        r for r in entity_rels
                        if r.relation_type in plan["relation_types"]
                    ]
                rels.extend(entity_rels)

        operation = plan.get("operation", "lookup")
        if operation == "path" and len(entities) >= 2:
            path = self._graph.shortest_path(entities[0].id, entities[1].id)
            if path:
                return QueryResult(entities=entities, paths=[path])

        if operation == "neighbors" and entities:
            neighbors = self._graph.get_neighbors(entities[0].id, max_hops=2)
            neighbor_entities = [n[0] for n in neighbors]
            return QueryResult(entities=entities + neighbor_entities, relationships=rels)

        return QueryResult(entities=entities, relationships=rels)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _humanize_relation(relation_type: str) -> str:
    """Convert snake_case relation to human-readable form."""
    mapping = {
        "works_at": "works at",
        "lives_in": "lives in",
        "studies_at": "studies at",
        "born_in": "was born in",
        "is_a": "is a",
        "part_of": "is part of",
        "located_in": "is located in",
        "prefers": "prefers",
        "dislikes": "dislikes",
        "knows": "knows",
        "uses": "uses",
        "speaks": "speaks",
        "created": "created",
        "related_to": "is related to",
        "has_role": "has the role of",
    }
    return mapping.get(relation_type, relation_type.replace("_", " "))
