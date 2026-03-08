"""Graph updater — auto-update from conversations, merge duplicates, decay stale relations."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from yoda.knowledge.graph import KnowledgeGraph, Entity, Relationship
from yoda.knowledge.extractor import EntityExtractor, ExtractionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decay parameters
# ---------------------------------------------------------------------------

_DECAY_HALF_LIFE_DAYS = 90.0  # Relationship weight halves every 90 days
_STALE_THRESHOLD = 0.1  # Below this weight, consider removing
_MERGE_SIMILARITY_THRESHOLD = 0.85  # Name similarity for auto-merge


class GraphUpdater:
    """Manages automatic graph updates from conversations and maintenance tasks."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        extractor: EntityExtractor,
    ) -> None:
        self._graph = graph
        self._extractor = extractor

    # -- Auto-update from conversations ------------------------------------

    async def process_message(
        self, user_message: str, assistant_response: str = "", use_llm: bool = False
    ) -> list[str]:
        """Process a conversation turn and update the graph.

        Returns: list of human-readable updates made.
        """
        updates: list[str] = []

        # Extract entities and relationships from user message
        result = await self._extractor.extract(user_message, use_llm=use_llm)

        # Also extract from assistant response if relevant
        if assistant_response:
            resp_result = await self._extractor.extract(assistant_response, use_llm=False)
            result = self._merge_extraction_results(result, resp_result)

        # Apply extraction results to graph
        entity_map: dict[str, Entity] = {}

        for extracted_entity in result.entities:
            # Check for existing entity
            existing = self._graph.find_entity(extracted_entity.name)
            if existing:
                # Update existing
                existing.properties.update(extracted_entity.properties)
                existing.updated_at = time.time()
                existing.confidence = min(1.0, existing.confidence + 0.05)
                entity_map[extracted_entity.name.lower()] = existing
                self._graph.add_entity(existing)
                updates.append(f"Updated entity: {existing.name}")
            else:
                # Add new
                entity = self._graph.add_entity(extracted_entity)
                entity_map[extracted_entity.name.lower()] = entity
                updates.append(f"Added entity: {entity.name} ({entity.entity_type})")

        # Add relationships
        for source_name, target_name, rel_type, properties in result.relationships:
            source = entity_map.get(source_name.lower()) or self._graph.find_entity(source_name)
            target = entity_map.get(target_name.lower()) or self._graph.find_entity(target_name)

            if source and target:
                rel = Relationship(
                    source_id=source.id,
                    target_id=target.id,
                    relation_type=rel_type,
                    properties=properties,
                    source="conversation",
                )
                try:
                    added = self._graph.add_relationship(rel)
                    updates.append(
                        f"Added relationship: {source.name} --[{rel_type}]--> {target.name}"
                    )
                except ValueError as e:
                    logger.warning("Failed to add relationship: %s", e)

        return updates

    # -- Duplicate merging -------------------------------------------------

    def merge_duplicates(self) -> list[str]:
        """Find and merge duplicate entities based on name similarity.

        Returns: list of merge descriptions.
        """
        merges: list[str] = []
        entities = list(self._graph.all_entities())
        merged_ids: set[str] = set()

        for i in range(len(entities)):
            if entities[i].id in merged_ids:
                continue
            for j in range(i + 1, len(entities)):
                if entities[j].id in merged_ids:
                    continue

                sim = _name_similarity(entities[i].name, entities[j].name)
                if sim >= _MERGE_SIMILARITY_THRESHOLD:
                    self._merge_entities(entities[i], entities[j])
                    merged_ids.add(entities[j].id)
                    merges.append(
                        f"Merged '{entities[j].name}' into '{entities[i].name}' "
                        f"(similarity: {sim:.0%})"
                    )

        return merges

    def _merge_entities(self, primary: Entity, secondary: Entity) -> None:
        """Merge secondary entity into primary, preserving all relationships."""
        # Merge properties
        for k, v in secondary.properties.items():
            if k not in primary.properties:
                primary.properties[k] = v

        # Merge aliases
        if secondary.name not in primary.aliases:
            primary.aliases.append(secondary.name)
        for alias in secondary.aliases:
            if alias not in primary.aliases:
                primary.aliases.append(alias)

        # Merge confidence (take max)
        primary.confidence = max(primary.confidence, secondary.confidence)
        primary.access_count += secondary.access_count
        primary.updated_at = time.time()

        # Redirect relationships from secondary to primary
        rels = self._graph.get_relationships(secondary.id)
        for rel in rels:
            new_source = primary.id if rel.source_id == secondary.id else rel.source_id
            new_target = primary.id if rel.target_id == secondary.id else rel.target_id

            if new_source == new_target:
                continue  # Skip self-loops

            new_rel = Relationship(
                source_id=new_source,
                target_id=new_target,
                relation_type=rel.relation_type,
                properties=rel.properties,
                weight=rel.weight,
                confidence=rel.confidence,
                source=rel.source,
                valid_from=rel.valid_from,
                valid_until=rel.valid_until,
            )
            try:
                self._graph.add_relationship(new_rel)
            except ValueError:
                pass

        # Remove secondary
        self._graph.remove_entity(secondary.id)
        # Update primary
        self._graph.add_entity(primary)

    # -- Decay stale relations ---------------------------------------------

    def decay_relationships(self) -> list[str]:
        """Apply temporal decay to relationship weights and remove stale ones.

        Returns: list of decay actions taken.
        """
        actions: list[str] = []
        now = time.time()
        to_remove: list[str] = []

        for rel in self._graph.all_relationships():
            age_days = (now - rel.updated_at) / 86400
            decay_factor = math.exp(-0.693 * age_days / _DECAY_HALF_LIFE_DAYS)
            new_weight = rel.weight * decay_factor

            if new_weight < _STALE_THRESHOLD:
                to_remove.append(rel.id)
                source = self._graph.get_entity(rel.source_id)
                target = self._graph.get_entity(rel.target_id)
                s_name = source.name if source else rel.source_id
                t_name = target.name if target else rel.target_id
                actions.append(
                    f"Removed stale: {s_name} --[{rel.relation_type}]--> {t_name} "
                    f"(weight: {new_weight:.3f})"
                )
            elif abs(new_weight - rel.weight) > 0.01:
                rel.weight = new_weight
                # Re-persist
                self._graph.add_relationship(rel)

        for rel_id in to_remove:
            self._graph.remove_relationship(rel_id)

        return actions

    def reinforce_relationship(self, source_name: str, target_name: str, relation_type: str) -> bool:
        """Reinforce a relationship (boost weight when mentioned again)."""
        source = self._graph.find_entity(source_name)
        target = self._graph.find_entity(target_name)
        if not source or not target:
            return False

        existing = self._graph._find_existing_relationship(
            source.id, target.id, relation_type
        )
        if existing:
            existing.weight = min(existing.weight + 0.2, 2.0)
            existing.updated_at = time.time()
            self._graph.add_relationship(existing)
            return True
        return False

    # -- Orphan cleanup ----------------------------------------------------

    def remove_orphan_entities(self, min_confidence: float = 0.2) -> list[str]:
        """Remove entities with no relationships and low confidence.

        Returns: list of removed entity names.
        """
        removed: list[str] = []
        to_remove: list[str] = []

        for entity in self._graph.all_entities():
            rels = self._graph.get_relationships(entity.id)
            if not rels and entity.confidence < min_confidence and entity.access_count == 0:
                to_remove.append(entity.id)
                removed.append(entity.name)

        for eid in to_remove:
            self._graph.remove_entity(eid)

        return removed

    # -- Full maintenance cycle --------------------------------------------

    def run_maintenance(self) -> dict[str, Any]:
        """Run all maintenance tasks.

        Returns: summary of actions taken.
        """
        decay_actions = self.decay_relationships()
        merge_actions = self.merge_duplicates()
        orphan_actions = self.remove_orphan_entities()

        summary = {
            "decayed": len(decay_actions),
            "merged": len(merge_actions),
            "orphans_removed": len(orphan_actions),
            "details": {
                "decay": decay_actions,
                "merges": merge_actions,
                "orphans": orphan_actions,
            },
        }
        logger.info(
            "Graph maintenance: %d decayed, %d merged, %d orphans removed",
            len(decay_actions), len(merge_actions), len(orphan_actions),
        )
        return summary

    # -- Helpers -----------------------------------------------------------

    def _merge_extraction_results(
        self, a: ExtractionResult, b: ExtractionResult
    ) -> ExtractionResult:
        """Merge two extraction results."""
        merged = ExtractionResult()
        seen_names: set[str] = set()

        for entity in a.entities + b.entities:
            key = entity.name.lower()
            if key not in seen_names:
                seen_names.add(key)
                merged.entities.append(entity)

        seen_rels: set[tuple[str, str, str]] = set()
        for rel in a.relationships + b.relationships:
            key = (rel[0].lower(), rel[1].lower(), rel[2])
            if key not in seen_rels:
                seen_rels.add(key)
                merged.relationships.append(rel)

        return merged


# ---------------------------------------------------------------------------
# String similarity
# ---------------------------------------------------------------------------

def _name_similarity(a: str, b: str) -> float:
    """Compute normalized similarity between two entity names.

    Uses Jaccard similarity on character trigrams.
    """
    if a.lower() == b.lower():
        return 1.0

    a_lower = a.lower().strip()
    b_lower = b.lower().strip()

    # Check if one is a substring of the other
    if a_lower in b_lower or b_lower in a_lower:
        return 0.9

    # Trigram Jaccard
    a_trigrams = {a_lower[i:i+3] for i in range(max(0, len(a_lower) - 2))}
    b_trigrams = {b_lower[i:i+3] for i in range(max(0, len(b_lower) - 2))}

    if not a_trigrams or not b_trigrams:
        return 0.0

    intersection = len(a_trigrams & b_trigrams)
    union = len(a_trigrams | b_trigrams)
    return intersection / union if union > 0 else 0.0
