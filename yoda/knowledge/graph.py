"""NetworkX-based knowledge graph with SQLite persistence.

Nodes represent entities (people, places, concepts, etc.).
Edges represent typed relationships with metadata and temporal info.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A node in the knowledge graph."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    entity_type: str = "concept"  # person, place, org, concept, event, preference
    properties: dict[str, Any] = field(default_factory=dict)
    source: str = "conversation"  # conversation, extraction, manual
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Relationship:
    """An edge in the knowledge graph."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str = ""
    target_id: str = ""
    relation_type: str = "related_to"
    properties: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    source: str = "conversation"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    valid_from: float | None = None
    valid_until: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relationship:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def is_temporal(self) -> bool:
        return self.valid_from is not None or self.valid_until is not None

    def is_valid_at(self, timestamp: float | None = None) -> bool:
        """Check if the relationship is valid at a given time."""
        ts = timestamp or time.time()
        if self.valid_from and ts < self.valid_from:
            return False
        if self.valid_until and ts > self.valid_until:
            return False
        return True


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """NetworkX-based knowledge graph with SQLite persistence.

    Thread-safe via SQLite WAL mode. The in-memory NetworkX graph is
    the source of truth during runtime; persistence is for durability.
    """

    def __init__(self, persist_path: str | Path = "~/.yoda/kg.db") -> None:
        self._persist_path = Path(persist_path).expanduser()
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._graph = nx.MultiDiGraph()
        self._entity_index: dict[str, str] = {}  # lowercase name -> entity id
        self._db: sqlite3.Connection | None = None

    # -- Lifecycle ---------------------------------------------------------

    async def initialize(self) -> None:
        """Open the database and load the graph into memory."""
        self._db = sqlite3.connect(str(self._persist_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        self._load_from_db()
        logger.info(
            "Knowledge graph loaded: %d entities, %d relationships",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    async def close(self) -> None:
        """Persist and close."""
        if self._db:
            self._db.close()
            self._db = None

    def _create_tables(self) -> None:
        assert self._db
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL DEFAULT 'concept',
                properties TEXT NOT NULL DEFAULT '{}',
                source TEXT NOT NULL DEFAULT 'conversation',
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                aliases TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type);

            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL DEFAULT 'related_to',
                properties TEXT NOT NULL DEFAULT '{}',
                weight REAL NOT NULL DEFAULT 1.0,
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT 'conversation',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                valid_from REAL,
                valid_until REAL,
                FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
            CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relation_type);

            CREATE TABLE IF NOT EXISTS graph_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        self._db.commit()

    def _load_from_db(self) -> None:
        """Load all entities and relationships from SQLite into NetworkX."""
        assert self._db
        self._graph.clear()
        self._entity_index.clear()

        for row in self._db.execute("SELECT * FROM entities"):
            entity = Entity(
                id=row[0], name=row[1], entity_type=row[2],
                properties=json.loads(row[3]), source=row[4],
                confidence=row[5], created_at=row[6], updated_at=row[7],
                access_count=row[8], aliases=json.loads(row[9]),
            )
            self._graph.add_node(entity.id, entity=entity)
            self._entity_index[entity.name.lower()] = entity.id
            for alias in entity.aliases:
                self._entity_index[alias.lower()] = entity.id

        for row in self._db.execute("SELECT * FROM relationships"):
            rel = Relationship(
                id=row[0], source_id=row[1], target_id=row[2],
                relation_type=row[3], properties=json.loads(row[4]),
                weight=row[5], confidence=row[6], source=row[7],
                created_at=row[8], updated_at=row[9],
                valid_from=row[10], valid_until=row[11],
            )
            if self._graph.has_node(rel.source_id) and self._graph.has_node(rel.target_id):
                self._graph.add_edge(
                    rel.source_id, rel.target_id,
                    key=rel.id, relationship=rel,
                )

    # -- Entity CRUD -------------------------------------------------------

    def add_entity(self, entity: Entity) -> Entity:
        """Add or update an entity. Returns the (possibly merged) entity."""
        existing_id = self._entity_index.get(entity.name.lower())
        if existing_id and existing_id != entity.id:
            # Merge into existing
            existing = self.get_entity(existing_id)
            if existing:
                existing.properties.update(entity.properties)
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.updated_at = time.time()
                for alias in entity.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
                self._persist_entity(existing)
                return existing

        self._graph.add_node(entity.id, entity=entity)
        self._entity_index[entity.name.lower()] = entity.id
        for alias in entity.aliases:
            self._entity_index[alias.lower()] = entity.id
        self._persist_entity(entity)
        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID."""
        data = self._graph.nodes.get(entity_id)
        if data:
            return data.get("entity")
        return None

    def find_entity(self, name: str) -> Entity | None:
        """Find entity by name or alias (case-insensitive)."""
        eid = self._entity_index.get(name.lower())
        if eid:
            entity = self.get_entity(eid)
            if entity:
                entity.access_count += 1
                return entity
        return None

    def search_entities(
        self,
        query: str,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Fuzzy search for entities by name substring."""
        query_lower = query.lower()
        results: list[tuple[float, Entity]] = []

        for _, data in self._graph.nodes(data=True):
            entity: Entity = data["entity"]
            if entity_type and entity.entity_type != entity_type:
                continue

            # Score: exact > starts_with > contains > alias match
            name_lower = entity.name.lower()
            score = 0.0
            if name_lower == query_lower:
                score = 1.0
            elif name_lower.startswith(query_lower):
                score = 0.8
            elif query_lower in name_lower:
                score = 0.6
            else:
                for alias in entity.aliases:
                    if query_lower in alias.lower():
                        score = 0.5
                        break

            if score > 0:
                results.append((score * entity.confidence, entity))

        results.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in results[:limit]]

    def remove_entity(self, entity_id: str) -> bool:
        """Remove entity and all its relationships."""
        entity = self.get_entity(entity_id)
        if not entity:
            return False

        self._graph.remove_node(entity_id)
        self._entity_index.pop(entity.name.lower(), None)
        for alias in entity.aliases:
            self._entity_index.pop(alias.lower(), None)

        if self._db:
            self._db.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            self._db.execute(
                "DELETE FROM relationships WHERE source_id = ? OR target_id = ?",
                (entity_id, entity_id),
            )
            self._db.commit()
        return True

    # -- Relationship CRUD -------------------------------------------------

    def add_relationship(self, rel: Relationship) -> Relationship:
        """Add a relationship between two entities."""
        if not self._graph.has_node(rel.source_id):
            raise ValueError(f"Source entity {rel.source_id} not found")
        if not self._graph.has_node(rel.target_id):
            raise ValueError(f"Target entity {rel.target_id} not found")

        # Check for duplicate
        existing = self._find_existing_relationship(
            rel.source_id, rel.target_id, rel.relation_type
        )
        if existing:
            existing.weight = min(existing.weight + 0.1, 2.0)
            existing.confidence = max(existing.confidence, rel.confidence)
            existing.updated_at = time.time()
            existing.properties.update(rel.properties)
            self._persist_relationship(existing)
            return existing

        self._graph.add_edge(
            rel.source_id, rel.target_id,
            key=rel.id, relationship=rel,
        )
        self._persist_relationship(rel)
        return rel

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: str | None = None,
    ) -> list[Relationship]:
        """Get relationships for an entity."""
        results: list[Relationship] = []

        if direction in ("out", "both"):
            for _, _, data in self._graph.out_edges(entity_id, data=True):
                rel: Relationship = data["relationship"]
                if relation_type and rel.relation_type != relation_type:
                    continue
                results.append(rel)

        if direction in ("in", "both"):
            for _, _, data in self._graph.in_edges(entity_id, data=True):
                rel = data["relationship"]
                if relation_type and rel.relation_type != relation_type:
                    continue
                results.append(rel)

        return results

    def remove_relationship(self, rel_id: str) -> bool:
        """Remove a specific relationship by ID."""
        for u, v, key, data in self._graph.edges(keys=True, data=True):
            if key == rel_id:
                self._graph.remove_edge(u, v, key=key)
                if self._db:
                    self._db.execute("DELETE FROM relationships WHERE id = ?", (rel_id,))
                    self._db.commit()
                return True
        return False

    def _find_existing_relationship(
        self, source_id: str, target_id: str, relation_type: str
    ) -> Relationship | None:
        """Find an existing relationship between two entities of the same type."""
        if not self._graph.has_edge(source_id, target_id):
            return None
        for _, data in self._graph[source_id][target_id].items():
            rel: Relationship = data["relationship"]
            if rel.relation_type == relation_type:
                return rel
        return None

    # -- Graph traversal ---------------------------------------------------

    def get_neighbors(
        self,
        entity_id: str,
        max_hops: int = 1,
        relation_types: list[str] | None = None,
    ) -> list[tuple[Entity, Relationship, int]]:
        """Get neighboring entities within N hops.

        Returns: list of (entity, connecting_relationship, hop_distance)
        """
        if not self._graph.has_node(entity_id):
            return []

        visited: set[str] = {entity_id}
        results: list[tuple[Entity, Relationship, int]] = []
        frontier = [entity_id]

        for hop in range(1, max_hops + 1):
            next_frontier: list[str] = []
            for node_id in frontier:
                # Outgoing edges
                for _, target, data in self._graph.out_edges(node_id, data=True):
                    rel: Relationship = data["relationship"]
                    if relation_types and rel.relation_type not in relation_types:
                        continue
                    if target not in visited:
                        visited.add(target)
                        entity = self.get_entity(target)
                        if entity:
                            results.append((entity, rel, hop))
                            next_frontier.append(target)

                # Incoming edges
                for source, _, data in self._graph.in_edges(node_id, data=True):
                    rel = data["relationship"]
                    if relation_types and rel.relation_type not in relation_types:
                        continue
                    if source not in visited:
                        visited.add(source)
                        entity = self.get_entity(source)
                        if entity:
                            results.append((entity, rel, hop))
                            next_frontier.append(source)

            frontier = next_frontier

        return results

    def shortest_path(
        self, source_id: str, target_id: str
    ) -> list[tuple[Entity, Relationship | None]] | None:
        """Find shortest path between two entities.

        Returns: list of (entity, relationship_used) pairs, or None if no path.
        """
        if not self._graph.has_node(source_id) or not self._graph.has_node(target_id):
            return None

        try:
            undirected = self._graph.to_undirected(as_view=True)
            path_ids = nx.shortest_path(undirected, source_id, target_id)
        except nx.NetworkXNoPath:
            return None

        result: list[tuple[Entity, Relationship | None]] = []
        for i, node_id in enumerate(path_ids):
            entity = self.get_entity(node_id)
            if not entity:
                continue
            rel = None
            if i > 0:
                prev_id = path_ids[i - 1]
                # Try both directions
                if self._graph.has_edge(prev_id, node_id):
                    edge_data = next(iter(self._graph[prev_id][node_id].values()))
                    rel = edge_data["relationship"]
                elif self._graph.has_edge(node_id, prev_id):
                    edge_data = next(iter(self._graph[node_id][prev_id].values()))
                    rel = edge_data["relationship"]
            result.append((entity, rel))

        return result

    def get_subgraph(
        self,
        entity_ids: list[str],
        include_connections: bool = True,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract a subgraph around the given entities."""
        node_set = set(entity_ids)
        if include_connections:
            for eid in entity_ids:
                for neighbor in self._graph.neighbors(eid):
                    node_set.add(neighbor)
                for predecessor in self._graph.predecessors(eid):
                    node_set.add(predecessor)

        entities = [self.get_entity(nid) for nid in node_set if self.get_entity(nid)]
        relationships: list[Relationship] = []
        for u, v, data in self._graph.edges(data=True):
            if u in node_set and v in node_set:
                relationships.append(data["relationship"])

        return [e for e in entities if e is not None], relationships

    # -- Statistics --------------------------------------------------------

    @property
    def num_entities(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_relationships(self) -> int:
        return self._graph.number_of_edges()

    def get_stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        type_counts: dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            entity: Entity = data["entity"]
            type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1

        rel_type_counts: dict[str, int] = {}
        for _, _, data in self._graph.edges(data=True):
            rel: Relationship = data["relationship"]
            rel_type_counts[rel.relation_type] = rel_type_counts.get(rel.relation_type, 0) + 1

        connected = 0
        if self._graph.number_of_nodes() > 0:
            undirected = self._graph.to_undirected()
            connected = nx.number_connected_components(undirected)

        return {
            "entities": self._graph.number_of_nodes(),
            "relationships": self._graph.number_of_edges(),
            "entity_types": type_counts,
            "relationship_types": rel_type_counts,
            "connected_components": connected,
        }

    def all_entities(self) -> Iterator[Entity]:
        """Iterate over all entities."""
        for _, data in self._graph.nodes(data=True):
            yield data["entity"]

    def all_relationships(self) -> Iterator[Relationship]:
        """Iterate over all relationships."""
        for _, _, data in self._graph.edges(data=True):
            yield data["relationship"]

    # -- Persistence -------------------------------------------------------

    def _persist_entity(self, entity: Entity) -> None:
        if not self._db:
            return
        self._db.execute(
            """INSERT OR REPLACE INTO entities
               (id, name, entity_type, properties, source, confidence,
                created_at, updated_at, access_count, aliases)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entity.id, entity.name, entity.entity_type,
                json.dumps(entity.properties), entity.source, entity.confidence,
                entity.created_at, entity.updated_at, entity.access_count,
                json.dumps(entity.aliases),
            ),
        )
        self._db.commit()

    def _persist_relationship(self, rel: Relationship) -> None:
        if not self._db:
            return
        self._db.execute(
            """INSERT OR REPLACE INTO relationships
               (id, source_id, target_id, relation_type, properties, weight,
                confidence, source, created_at, updated_at, valid_from, valid_until)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rel.id, rel.source_id, rel.target_id, rel.relation_type,
                json.dumps(rel.properties), rel.weight, rel.confidence,
                rel.source, rel.created_at, rel.updated_at,
                rel.valid_from, rel.valid_until,
            ),
        )
        self._db.commit()

    # -- Export / Import ---------------------------------------------------

    def export_json(self, path: str | Path | None = None) -> dict[str, Any]:
        """Export the entire graph as JSON."""
        data = {
            "entities": [e.to_dict() for e in self.all_entities()],
            "relationships": [r.to_dict() for r in self.all_relationships()],
            "stats": self.get_stats(),
        }
        if path:
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(data, f, indent=2, default=str)
        return data

    def import_json(self, path: str | Path) -> int:
        """Import entities and relationships from JSON. Returns count of items imported."""
        with open(Path(path).expanduser()) as f:
            data = json.load(f)

        count = 0
        for e_data in data.get("entities", []):
            entity = Entity.from_dict(e_data)
            self.add_entity(entity)
            count += 1

        for r_data in data.get("relationships", []):
            rel = Relationship.from_dict(r_data)
            try:
                self.add_relationship(rel)
                count += 1
            except ValueError:
                logger.warning("Skipping relationship %s: missing entity", rel.id)

        return count
