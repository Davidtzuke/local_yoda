"""LLM-powered entity and relationship extraction from text.

Combines regex-based NER patterns with LLM extraction for high-quality
entity/relationship graphs from conversations.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from yoda.knowledge.graph import Entity, Relationship

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex-based NER patterns (fast, no LLM needed)
# ---------------------------------------------------------------------------

# Patterns for extracting entities and relations from conversation text
_PERSON_PATTERNS = [
    r"(?:my (?:friend|brother|sister|mom|dad|mother|father|wife|husband|partner|boss|colleague|coworker|manager|teacher|doctor)(?:'s name is| is| named)?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    r"(?:I'm|I am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    r"(?:my name is|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:is my|told me|said|asked|works at|lives in)",
]

_RELATION_PATTERNS = [
    (r"(?:my\s+(\w+))\s+(?:is|named)\s+([A-Z][a-z]+)", "has_{0}"),
    (r"I\s+(?:work|am working)\s+(?:at|for)\s+(.+?)(?:\s+as\s+(?:a|an)\s+(.+?))?[.\s]", "works_at"),
    (r"I\s+(?:live|am living|reside)\s+(?:in|at)\s+(.+?)(?:[.,\s]|$)", "lives_in"),
    (r"I\s+(?:study|studied|am studying)\s+(?:at|in)\s+(.+?)(?:[.,\s]|$)", "studies_at"),
    (r"I\s+(?:like|love|enjoy|prefer)\s+(.+?)(?:[.,\s]|$)", "prefers"),
    (r"I\s+(?:hate|dislike|don't like|avoid)\s+(.+?)(?:[.,\s]|$)", "dislikes"),
    (r"I\s+(?:use|am using)\s+(.+?)(?:\s+for\s+(.+?))?(?:[.,\s]|$)", "uses"),
    (r"I\s+(?:speak|know)\s+(\w+(?:\s+and\s+\w+)*?)(?:\s+(?:language|languages))?(?:[.,\s]|$)", "speaks"),
    (r"I\s+(?:was born|grew up)\s+(?:in|at)\s+(.+?)(?:[.,\s]|$)", "born_in"),
    (r"(\w+)\s+is\s+(?:a|an)\s+(.+?)(?:[.,\s]|$)", "is_a"),
]

_LOCATION_PATTERNS = [
    r"(?:in|at|from|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
]

_ORG_PATTERNS = [
    r"(?:at|for|from|with)\s+([A-Z][a-z]*(?:\s+[A-Z][a-z]*)+(?:\s+(?:Inc|LLC|Corp|Ltd|Co)\.?)?)",
    r"(?:at|for|from|with)\s+(Google|Apple|Microsoft|Amazon|Meta|OpenAI|Anthropic|Netflix|Tesla|SpaceX)",
]


class ExtractionResult:
    """Container for extraction results."""

    def __init__(self) -> None:
        self.entities: list[Entity] = []
        self.relationships: list[tuple[str, str, str, dict[str, Any]]] = []
        # (source_name, target_name, relation_type, properties)

    def add_entity(self, name: str, entity_type: str = "concept", **kwargs: Any) -> Entity:
        # Deduplicate
        for e in self.entities:
            if e.name.lower() == name.lower():
                e.properties.update(kwargs.get("properties", {}))
                return e
        entity = Entity(name=name, entity_type=entity_type, **kwargs)
        self.entities.append(entity)
        return entity

    def add_relationship(
        self, source_name: str, target_name: str,
        relation_type: str, properties: dict[str, Any] | None = None,
    ) -> None:
        self.relationships.append((
            source_name, target_name, relation_type, properties or {}
        ))


class EntityExtractor:
    """Extract entities and relationships from text using patterns and LLM."""

    def __init__(self, llm_provider: Any = None) -> None:
        self._llm = llm_provider

    # -- Pattern-based extraction ------------------------------------------

    def extract_patterns(self, text: str) -> ExtractionResult:
        """Fast regex-based extraction — no LLM needed."""
        result = ExtractionResult()

        # Extract people
        for pattern in _PERSON_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1).strip()
                if len(name) > 1 and not name.lower() in {"the", "and", "but", "for"}:
                    result.add_entity(name, "person")

        # Extract relations
        for pattern, rel_type_template in _RELATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = [g.strip() for g in match.groups() if g]
                if not groups:
                    continue

                if "{0}" in rel_type_template and len(groups) >= 1:
                    rel_type = rel_type_template.format(groups[0].lower())
                    if len(groups) >= 2:
                        target_name = groups[1]
                    else:
                        continue
                else:
                    rel_type = rel_type_template
                    target_name = groups[0]

                # Determine entity type from relation
                entity_type = _infer_entity_type(rel_type, target_name)
                result.add_entity(target_name, entity_type)
                result.add_entity("user", "person")
                result.add_relationship("user", target_name, rel_type)

                # Secondary targets (e.g., "at Google as an engineer")
                if len(groups) >= 2 and rel_type == "works_at":
                    role = groups[1]
                    result.add_entity(role, "role")
                    result.add_relationship("user", role, "has_role")

        # Extract locations
        for pattern in _LOCATION_PATTERNS:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if len(name) > 2:
                    result.add_entity(name, "place")

        # Extract organizations
        for pattern in _ORG_PATTERNS:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if len(name) > 2:
                    result.add_entity(name, "organization")

        return result

    # -- LLM-powered extraction --------------------------------------------

    async def extract_with_llm(self, text: str) -> ExtractionResult:
        """Use LLM for high-quality extraction with coreference resolution."""
        if not self._llm:
            return self.extract_patterns(text)

        prompt = _build_extraction_prompt(text)

        try:
            from yoda.core.messages import SystemMessage, UserMessage

            messages = [
                SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT).to_provider_format(),
                UserMessage(content=prompt).to_provider_format(),
            ]
            response = await self._llm.complete(messages)
            content = response.content or ""
            return self._parse_llm_response(content, text)
        except Exception:
            logger.exception("LLM extraction failed, falling back to patterns")
            return self.extract_patterns(text)

    # -- Combined extraction -----------------------------------------------

    async def extract(self, text: str, use_llm: bool = True) -> ExtractionResult:
        """Extract entities and relations using both patterns and LLM.

        Pattern results are always included. LLM results are merged in
        if available, with LLM having higher confidence.
        """
        pattern_result = self.extract_patterns(text)

        if use_llm and self._llm:
            llm_result = await self.extract_with_llm(text)
            return self._merge_results(pattern_result, llm_result)

        return pattern_result

    # -- Coreference resolution (simple) -----------------------------------

    def resolve_coreferences(
        self, text: str, known_entities: list[Entity]
    ) -> dict[str, str]:
        """Simple pronoun/reference resolution.

        Returns: mapping of pronoun/reference -> entity name
        """
        resolutions: dict[str, str] = {}
        # Build mention context
        last_person: str | None = None
        last_place: str | None = None
        last_org: str | None = None

        for entity in known_entities:
            if entity.entity_type == "person":
                last_person = entity.name
            elif entity.entity_type == "place":
                last_place = entity.name
            elif entity.entity_type == "organization":
                last_org = entity.name

        # Resolve pronouns
        pronoun_map = {
            "he": last_person, "him": last_person, "his": last_person,
            "she": last_person, "her": last_person, "hers": last_person,
            "they": last_person, "them": last_person, "their": last_person,
            "there": last_place, "that place": last_place,
            "the company": last_org, "the organization": last_org,
        }

        for pronoun, entity_name in pronoun_map.items():
            if entity_name and re.search(rf"\b{pronoun}\b", text, re.IGNORECASE):
                resolutions[pronoun] = entity_name

        return resolutions

    # -- Internal helpers --------------------------------------------------

    def _parse_llm_response(self, content: str, original_text: str) -> ExtractionResult:
        """Parse structured JSON from LLM response."""
        result = ExtractionResult()

        # Try to extract JSON from the response
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return self.extract_patterns(original_text)

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return self.extract_patterns(original_text)

        for e in data.get("entities", []):
            result.add_entity(
                name=e.get("name", ""),
                entity_type=e.get("type", "concept"),
                properties=e.get("properties", {}),
                confidence=0.9,
                source="llm_extraction",
            )

        for r in data.get("relationships", []):
            result.add_relationship(
                source_name=r.get("source", ""),
                target_name=r.get("target", ""),
                relation_type=r.get("type", "related_to"),
                properties=r.get("properties", {}),
            )

        return result

    def _merge_results(
        self, pattern: ExtractionResult, llm: ExtractionResult
    ) -> ExtractionResult:
        """Merge pattern and LLM results, deduplicating entities."""
        merged = ExtractionResult()

        # Add all LLM entities (higher confidence)
        for entity in llm.entities:
            entity.confidence = max(entity.confidence, 0.85)
            merged.entities.append(entity)

        # Add pattern entities not already found
        llm_names = {e.name.lower() for e in llm.entities}
        for entity in pattern.entities:
            if entity.name.lower() not in llm_names:
                entity.confidence = 0.7
                merged.entities.append(entity)

        # Combine relationships
        seen_rels: set[tuple[str, str, str]] = set()
        for rel in llm.relationships:
            key = (rel[0].lower(), rel[1].lower(), rel[2])
            if key not in seen_rels:
                seen_rels.add(key)
                merged.relationships.append(rel)

        for rel in pattern.relationships:
            key = (rel[0].lower(), rel[1].lower(), rel[2])
            if key not in seen_rels:
                seen_rels.add(key)
                merged.relationships.append(rel)

        return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_entity_type(relation_type: str, name: str) -> str:
    """Infer entity type from the relationship type."""
    type_map = {
        "works_at": "organization",
        "studies_at": "organization",
        "lives_in": "place",
        "born_in": "place",
        "speaks": "language",
        "uses": "tool",
        "prefers": "preference",
        "dislikes": "preference",
        "is_a": "concept",
        "has_role": "role",
    }
    for prefix, etype in type_map.items():
        if relation_type.startswith(prefix):
            return etype
    return "concept"


_EXTRACTION_SYSTEM_PROMPT = """You are an entity and relationship extraction engine.
Given text, extract entities (people, places, organizations, concepts, preferences)
and relationships between them. Output ONLY valid JSON with this schema:

{
  "entities": [
    {"name": "...", "type": "person|place|organization|concept|event|preference", "properties": {}}
  ],
  "relationships": [
    {"source": "...", "target": "...", "type": "...", "properties": {}}
  ]
}

Relationship types should be lowercase_snake_case verbs like: works_at, lives_in,
knows, prefers, dislikes, created, uses, is_a, part_of, located_in, etc.

Be thorough but only extract clearly stated facts. Resolve coreferences (pronouns
to named entities). Always include "user" as an entity for first-person statements."""


def _build_extraction_prompt(text: str) -> str:
    return f"""Extract all entities and relationships from this text:

---
{text}
---

Return ONLY the JSON object, no other text."""
