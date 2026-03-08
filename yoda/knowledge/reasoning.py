"""Multi-hop reasoning engine over the knowledge graph.

Supports inference chains, contradiction detection, and confidence propagation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from yoda.knowledge.graph import KnowledgeGraph, Entity, Relationship

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reasoning result types
# ---------------------------------------------------------------------------

@dataclass
class InferenceStep:
    """A single step in a reasoning chain."""

    description: str
    entities_involved: list[str] = field(default_factory=list)
    relationship_used: str = ""
    confidence: float = 1.0


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""

    conclusion: str
    confidence: float
    steps: list[InferenceStep] = field(default_factory=list)
    supporting_facts: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        parts = [f"Conclusion: {self.conclusion} (confidence: {self.confidence:.0%})"]
        if self.steps:
            parts.append("\nReasoning chain:")
            for i, step in enumerate(self.steps, 1):
                parts.append(f"  {i}. {step.description} ({step.confidence:.0%})")
        if self.contradictions:
            parts.append("\nContradictions found:")
            for c in self.contradictions:
                parts.append(f"  ! {c}")
        return "\n".join(parts)


@dataclass
class Contradiction:
    """A detected contradiction between facts."""

    fact_a: str
    fact_b: str
    entity: str
    relation_type: str
    severity: float = 0.5  # 0=minor, 1=critical


# ---------------------------------------------------------------------------
# Inference rules
# ---------------------------------------------------------------------------

# Transitive relations: if A->B and B->C then A->C
_TRANSITIVE_RELATIONS = {"part_of", "located_in", "is_a", "subclass_of"}

# Exclusive relations: entity can only have one value
_EXCLUSIVE_RELATIONS = {"lives_in", "works_at", "born_in", "has_role"}

# Symmetric relations: if A->B then B->A
_SYMMETRIC_RELATIONS = {"knows", "related_to", "married_to", "friends_with"}

# Inverse relations: A->B implies B->A with different type
_INVERSE_RELATIONS = {
    "works_at": "employs",
    "employs": "works_at",
    "teaches": "taught_by",
    "taught_by": "teaches",
    "parent_of": "child_of",
    "child_of": "parent_of",
}


# ---------------------------------------------------------------------------
# Reasoning Engine
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """Multi-hop reasoning over the knowledge graph."""

    def __init__(self, graph: KnowledgeGraph, max_hops: int = 3) -> None:
        self._graph = graph
        self._max_hops = max_hops

    # -- Multi-hop reasoning -----------------------------------------------

    def reason_about(self, entity_name: str, question: str = "") -> ReasoningResult:
        """Perform multi-hop reasoning starting from an entity.

        Traverses the graph up to max_hops, collecting facts and making
        inferences along the way.
        """
        entity = self._graph.find_entity(entity_name)
        if not entity:
            return ReasoningResult(
                conclusion=f"No information found about '{entity_name}'.",
                confidence=0.0,
            )

        steps: list[InferenceStep] = []
        facts: list[str] = []
        visited: set[str] = {entity.id}

        # Direct facts
        direct_rels = self._graph.get_relationships(entity.id)
        for rel in direct_rels:
            other_id = rel.target_id if rel.source_id == entity.id else rel.source_id
            other = self._graph.get_entity(other_id)
            if other:
                fact = f"{entity.name} {rel.relation_type} {other.name}"
                facts.append(fact)
                steps.append(InferenceStep(
                    description=f"Direct fact: {fact}",
                    entities_involved=[entity.name, other.name],
                    relationship_used=rel.relation_type,
                    confidence=rel.confidence,
                ))

        # Multi-hop inference
        frontier = [(entity, 0)]
        while frontier:
            current_entity, current_hop = frontier.pop(0)
            if current_hop >= self._max_hops:
                continue

            neighbors = self._graph.get_neighbors(current_entity.id, max_hops=1)
            for neighbor, rel, _ in neighbors:
                if neighbor.id in visited:
                    continue
                visited.add(neighbor.id)

                # Transitive inference
                inferred = self._apply_transitive(entity, current_entity, neighbor, rel)
                if inferred:
                    steps.append(inferred)
                    facts.append(inferred.description)

                frontier.append((neighbor, current_hop + 1))

        # Check for contradictions
        contradictions = self._detect_contradictions_for(entity)
        contradiction_strs = [
            f"{c.fact_a} CONTRADICTS {c.fact_b}" for c in contradictions
        ]

        # Build conclusion
        if facts:
            conclusion = f"Found {len(facts)} facts about {entity.name}."
        else:
            conclusion = f"No detailed information about {entity.name}."

        confidence = self._compute_chain_confidence(steps)

        return ReasoningResult(
            conclusion=conclusion,
            confidence=confidence,
            steps=steps,
            supporting_facts=facts,
            contradictions=contradiction_strs,
        )

    # -- Inference between two entities ------------------------------------

    def infer_relationship(
        self, entity_a_name: str, entity_b_name: str
    ) -> ReasoningResult:
        """Infer possible relationships between two entities."""
        a = self._graph.find_entity(entity_a_name)
        b = self._graph.find_entity(entity_b_name)

        if not a or not b:
            missing = entity_a_name if not a else entity_b_name
            return ReasoningResult(
                conclusion=f"Entity '{missing}' not found.",
                confidence=0.0,
            )

        steps: list[InferenceStep] = []
        facts: list[str] = []

        # Direct relationship
        direct = self._graph.get_relationships(a.id)
        for rel in direct:
            if rel.target_id == b.id or rel.source_id == b.id:
                fact = f"Direct: {a.name} {rel.relation_type} {b.name}"
                facts.append(fact)
                steps.append(InferenceStep(
                    description=fact,
                    entities_involved=[a.name, b.name],
                    relationship_used=rel.relation_type,
                    confidence=rel.confidence,
                ))

        # Path-based inference
        path = self._graph.shortest_path(a.id, b.id)
        if path and len(path) > 2:
            path_desc = " -> ".join(
                f"{e.name}" + (f" ({r.relation_type})" if r else "")
                for e, r in path
            )
            steps.append(InferenceStep(
                description=f"Connected via path: {path_desc}",
                entities_involved=[e.name for e, _ in path],
                confidence=0.6,
            ))
            facts.append(f"Indirect connection: {path_desc}")

        # Common neighbors
        a_neighbors = {n[0].id for n in self._graph.get_neighbors(a.id, max_hops=1)}
        b_neighbors = {n[0].id for n in self._graph.get_neighbors(b.id, max_hops=1)}
        common = a_neighbors & b_neighbors

        for common_id in common:
            common_entity = self._graph.get_entity(common_id)
            if common_entity:
                steps.append(InferenceStep(
                    description=f"Shared connection: both linked to {common_entity.name}",
                    entities_involved=[a.name, b.name, common_entity.name],
                    confidence=0.5,
                ))

        if facts:
            conclusion = f"Found {len(facts)} connections between {a.name} and {b.name}."
        elif steps:
            conclusion = f"Indirect connections found between {a.name} and {b.name}."
        else:
            conclusion = f"No known relationship between {a.name} and {b.name}."

        return ReasoningResult(
            conclusion=conclusion,
            confidence=self._compute_chain_confidence(steps),
            steps=steps,
            supporting_facts=facts,
        )

    # -- Contradiction detection -------------------------------------------

    def detect_all_contradictions(self) -> list[Contradiction]:
        """Scan the entire graph for contradictions."""
        contradictions: list[Contradiction] = []

        for entity in self._graph.all_entities():
            contradictions.extend(self._detect_contradictions_for(entity))

        return contradictions

    def _detect_contradictions_for(self, entity: Entity) -> list[Contradiction]:
        """Detect contradictions in relationships for a single entity."""
        contradictions: list[Contradiction] = []
        rels = self._graph.get_relationships(entity.id, direction="out")

        # Group by relation type
        by_type: dict[str, list[Relationship]] = {}
        for rel in rels:
            by_type.setdefault(rel.relation_type, []).append(rel)

        # Check exclusive relations
        for rel_type, rel_list in by_type.items():
            if rel_type in _EXCLUSIVE_RELATIONS and len(rel_list) > 1:
                # Filter to currently valid ones
                valid = [r for r in rel_list if r.is_valid_at()]
                if len(valid) > 1:
                    for i in range(len(valid)):
                        for j in range(i + 1, len(valid)):
                            t1 = self._graph.get_entity(valid[i].target_id)
                            t2 = self._graph.get_entity(valid[j].target_id)
                            if t1 and t2 and t1.id != t2.id:
                                contradictions.append(Contradiction(
                                    fact_a=f"{entity.name} {rel_type} {t1.name}",
                                    fact_b=f"{entity.name} {rel_type} {t2.name}",
                                    entity=entity.name,
                                    relation_type=rel_type,
                                    severity=0.8,
                                ))

        return contradictions

    # -- Transitive inference ----------------------------------------------

    def _apply_transitive(
        self,
        origin: Entity,
        intermediate: Entity,
        target: Entity,
        rel: Relationship,
    ) -> InferenceStep | None:
        """Apply transitive inference rules."""
        if rel.relation_type not in _TRANSITIVE_RELATIONS:
            return None

        # Check if origin->intermediate has same relation type
        origin_rels = self._graph.get_relationships(origin.id, direction="out")
        for r in origin_rels:
            if r.target_id == intermediate.id and r.relation_type == rel.relation_type:
                return InferenceStep(
                    description=(
                        f"Inferred: {origin.name} {rel.relation_type} {target.name} "
                        f"(via {intermediate.name})"
                    ),
                    entities_involved=[origin.name, intermediate.name, target.name],
                    relationship_used=rel.relation_type,
                    confidence=r.confidence * rel.confidence * 0.8,
                )

        return None

    # -- Confidence propagation --------------------------------------------

    def _compute_chain_confidence(self, steps: list[InferenceStep]) -> float:
        """Compute overall confidence from a chain of reasoning steps."""
        if not steps:
            return 0.0

        # Geometric mean of step confidences
        product = 1.0
        for step in steps:
            product *= step.confidence

        return product ** (1.0 / len(steps))

    # -- Utility -----------------------------------------------------------

    def get_entity_importance(self, entity_name: str) -> float:
        """Compute importance of an entity based on connectivity and access."""
        entity = self._graph.find_entity(entity_name)
        if not entity:
            return 0.0

        rels = self._graph.get_relationships(entity.id)
        degree = len(rels)
        access = entity.access_count

        # Weighted score: degree (connectivity) + access (usage)
        return min(1.0, (degree * 0.1 + access * 0.05) * entity.confidence)

    def suggest_missing_links(self, entity_name: str) -> list[tuple[str, str, float]]:
        """Suggest potential missing relationships for an entity.

        Returns: list of (target_name, suggested_relation, confidence)
        """
        entity = self._graph.find_entity(entity_name)
        if not entity:
            return []

        suggestions: list[tuple[str, str, float]] = []
        neighbors = self._graph.get_neighbors(entity.id, max_hops=2)

        # Find 2-hop neighbors that might have direct connections
        for neighbor, _, hop in neighbors:
            if hop < 2:
                continue
            # Check if neighbor shares relations with similar entities
            neighbor_rels = self._graph.get_relationships(neighbor.id)
            for rel in neighbor_rels:
                other_id = rel.target_id if rel.source_id == neighbor.id else rel.source_id
                other = self._graph.get_entity(other_id)
                if other and other.entity_type == entity.entity_type:
                    suggestions.append((
                        neighbor.name,
                        rel.relation_type,
                        0.3 * rel.confidence,
                    ))

        # Deduplicate and sort
        seen: set[str] = set()
        unique: list[tuple[str, str, float]] = []
        for name, rel_type, conf in suggestions:
            key = f"{name}:{rel_type}"
            if key not in seen:
                seen.add(key)
                unique.append((name, rel_type, conf))

        unique.sort(key=lambda x: x[2], reverse=True)
        return unique[:10]
