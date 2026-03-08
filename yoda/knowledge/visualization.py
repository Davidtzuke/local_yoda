"""Knowledge graph visualization — D3.js JSON export and Mermaid diagram generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from yoda.knowledge.graph import KnowledgeGraph, Entity, Relationship


# ---------------------------------------------------------------------------
# Color scheme for entity types
# ---------------------------------------------------------------------------

_TYPE_COLORS = {
    "person": "#4CAF50",
    "place": "#2196F3",
    "organization": "#FF9800",
    "concept": "#9C27B0",
    "event": "#F44336",
    "preference": "#E91E63",
    "tool": "#00BCD4",
    "language": "#795548",
    "role": "#607D8B",
}

_TYPE_SHAPES = {
    "person": "circle",
    "place": "diamond",
    "organization": "square",
    "concept": "triangle",
    "event": "star",
    "preference": "hexagon",
}


class GraphVisualizer:
    """Generate visualization exports from the knowledge graph."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    # -- D3.js force-directed graph JSON -----------------------------------

    def to_d3_json(
        self,
        entity_ids: list[str] | None = None,
        max_nodes: int = 200,
    ) -> dict[str, Any]:
        """Export graph as D3.js force-directed graph JSON.

        Format:
        {
            "nodes": [{"id": ..., "name": ..., "type": ..., "color": ..., ...}],
            "links": [{"source": ..., "target": ..., "type": ..., ...}]
        }
        """
        nodes: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        node_ids: set[str] = set()

        if entity_ids:
            entities_iter = (
                self._graph.get_entity(eid)
                for eid in entity_ids
                if self._graph.get_entity(eid)
            )
        else:
            entities_iter = self._graph.all_entities()

        for entity in entities_iter:
            if entity is None:
                continue
            if len(nodes) >= max_nodes:
                break
            nodes.append(self._entity_to_d3_node(entity))
            node_ids.add(entity.id)

        for rel in self._graph.all_relationships():
            if rel.source_id in node_ids and rel.target_id in node_ids:
                links.append(self._relationship_to_d3_link(rel))

        return {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "total_entities": self._graph.num_entities,
                "total_relationships": self._graph.num_relationships,
                "exported_nodes": len(nodes),
                "exported_links": len(links),
            },
        }

    def export_d3_json(
        self,
        path: str | Path,
        entity_ids: list[str] | None = None,
        max_nodes: int = 200,
    ) -> Path:
        """Export D3.js JSON to a file."""
        data = self.to_d3_json(entity_ids=entity_ids, max_nodes=max_nodes)
        out_path = Path(path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return out_path

    def _entity_to_d3_node(self, entity: Entity) -> dict[str, Any]:
        """Convert entity to D3.js node dict."""
        return {
            "id": entity.id,
            "name": entity.name,
            "type": entity.entity_type,
            "color": _TYPE_COLORS.get(entity.entity_type, "#999"),
            "shape": _TYPE_SHAPES.get(entity.entity_type, "circle"),
            "size": max(5, min(30, 5 + entity.access_count * 2)),
            "confidence": entity.confidence,
            "properties": entity.properties,
        }

    def _relationship_to_d3_link(self, rel: Relationship) -> dict[str, Any]:
        """Convert relationship to D3.js link dict."""
        return {
            "source": rel.source_id,
            "target": rel.target_id,
            "type": rel.relation_type,
            "label": rel.relation_type.replace("_", " "),
            "weight": rel.weight,
            "confidence": rel.confidence,
            "temporal": rel.is_temporal,
        }

    # -- Mermaid diagram generation ----------------------------------------

    def to_mermaid(
        self,
        entity_ids: list[str] | None = None,
        max_nodes: int = 50,
        direction: str = "LR",
    ) -> str:
        """Generate a Mermaid flowchart diagram.

        Args:
            entity_ids: Specific entities to include (None = all)
            max_nodes: Maximum nodes to include
            direction: Graph direction (LR, TB, RL, BT)

        Returns: Mermaid diagram string
        """
        lines = [f"graph {direction}"]
        node_ids: set[str] = set()
        node_map: dict[str, str] = {}  # entity_id -> mermaid_safe_id

        # Collect entities
        if entity_ids:
            entities = [
                self._graph.get_entity(eid)
                for eid in entity_ids
                if self._graph.get_entity(eid)
            ]
        else:
            entities = list(self._graph.all_entities())

        # Limit nodes
        entities = entities[:max_nodes]

        # Add nodes
        for entity in entities:
            if entity is None:
                continue
            safe_id = _mermaid_safe_id(entity.id)
            node_map[entity.id] = safe_id
            node_ids.add(entity.id)

            shape = _mermaid_node_shape(entity.entity_type)
            label = _mermaid_escape(entity.name)
            lines.append(f"    {safe_id}{shape[0]}\"{label}\"{shape[1]}")

        # Add edges
        for rel in self._graph.all_relationships():
            if rel.source_id in node_ids and rel.target_id in node_ids:
                source = node_map[rel.source_id]
                target = node_map[rel.target_id]
                label = _mermaid_escape(rel.relation_type.replace("_", " "))
                lines.append(f"    {source} -->|\"{label}\"| {target}")

        # Add styles
        lines.append("")
        for entity_type, color in _TYPE_COLORS.items():
            typed_nodes = [
                node_map[e.id] for e in entities
                if e is not None and e.entity_type == entity_type and e.id in node_map
            ]
            if typed_nodes:
                style_class = f"class_{entity_type}"
                lines.append(f"    classDef {style_class} fill:{color},color:#fff")
                lines.append(f"    class {','.join(typed_nodes)} {style_class}")

        return "\n".join(lines)

    def export_mermaid(
        self,
        path: str | Path,
        entity_ids: list[str] | None = None,
        max_nodes: int = 50,
        direction: str = "LR",
    ) -> Path:
        """Export Mermaid diagram to a file."""
        mermaid = self.to_mermaid(
            entity_ids=entity_ids, max_nodes=max_nodes, direction=direction
        )
        out_path = Path(path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(mermaid)
        return out_path

    # -- Summary text visualization ----------------------------------------

    def to_ascii(self, entity_id: str, max_depth: int = 2) -> str:
        """Generate ASCII tree visualization centered on an entity."""
        entity = self._graph.get_entity(entity_id)
        if not entity:
            return f"Entity {entity_id} not found."

        lines: list[str] = []
        self._ascii_tree(entity, lines, visited=set(), depth=0, max_depth=max_depth, prefix="")
        return "\n".join(lines)

    def _ascii_tree(
        self,
        entity: Entity,
        lines: list[str],
        visited: set[str],
        depth: int,
        max_depth: int,
        prefix: str,
    ) -> None:
        """Recursively build ASCII tree."""
        if entity.id in visited or depth > max_depth:
            return

        visited.add(entity.id)
        icon = _type_icon(entity.entity_type)
        lines.append(f"{prefix}{icon} {entity.name} ({entity.entity_type})")

        rels = self._graph.get_relationships(entity.id, direction="out")
        for i, rel in enumerate(rels):
            target = self._graph.get_entity(rel.target_id)
            if not target or target.id in visited:
                continue

            is_last = i == len(rels) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")

            lines.append(f"{prefix}{connector}[{rel.relation_type}]")
            self._ascii_tree(target, lines, visited, depth + 1, max_depth, child_prefix)

    # -- HTML export with embedded D3.js -----------------------------------

    def export_html(
        self,
        path: str | Path,
        title: str = "Yoda Knowledge Graph",
        entity_ids: list[str] | None = None,
        max_nodes: int = 200,
    ) -> Path:
        """Export interactive HTML visualization with embedded D3.js."""
        data = self.to_d3_json(entity_ids=entity_ids, max_nodes=max_nodes)
        html = _D3_HTML_TEMPLATE.replace("{{TITLE}}", title).replace(
            "{{GRAPH_DATA}}", json.dumps(data, default=str)
        )
        out_path = Path(path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(html)
        return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mermaid_safe_id(entity_id: str) -> str:
    """Convert entity ID to Mermaid-safe identifier."""
    return "n_" + entity_id.replace("-", "_")


def _mermaid_escape(text: str) -> str:
    """Escape text for Mermaid labels."""
    return text.replace('"', "'").replace("\n", " ")


def _mermaid_node_shape(entity_type: str) -> tuple[str, str]:
    """Return Mermaid shape delimiters for entity type."""
    shapes = {
        "person": ("([", "])"),     # stadium
        "place": ("{{", "}}"),      # hexagon
        "organization": ("[", "]"),  # rectangle
        "concept": ("(", ")"),       # rounded
        "event": (">", "]"),         # asymmetric
        "preference": ("([", "])"),  # stadium
    }
    return shapes.get(entity_type, ("(", ")"))


def _type_icon(entity_type: str) -> str:
    """Return a simple ASCII icon for entity type."""
    icons = {
        "person": "[P]",
        "place": "[L]",
        "organization": "[O]",
        "concept": "[C]",
        "event": "[E]",
        "preference": "[*]",
        "tool": "[T]",
        "language": "[#]",
        "role": "[R]",
    }
    return icons.get(entity_type, "[?]")


# ---------------------------------------------------------------------------
# D3.js HTML template
# ---------------------------------------------------------------------------

_D3_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{TITLE}}</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  body { margin: 0; font-family: sans-serif; background: #1a1a2e; color: #eee; }
  svg { width: 100vw; height: 100vh; }
  .link { stroke: #555; stroke-opacity: 0.6; }
  .link-label { font-size: 9px; fill: #aaa; }
  .node-label { font-size: 11px; fill: #eee; text-anchor: middle; }
  .tooltip { position: absolute; background: #16213e; border: 1px solid #0f3460;
             padding: 8px 12px; border-radius: 4px; font-size: 12px; pointer-events: none; }
  h1 { position: fixed; top: 10px; left: 20px; font-size: 18px; opacity: 0.7; }
</style>
</head>
<body>
<h1>{{TITLE}}</h1>
<svg></svg>
<script>
const data = {{GRAPH_DATA}};
const width = window.innerWidth, height = window.innerHeight;
const svg = d3.select("svg").attr("viewBox", [0, 0, width, height]);

const simulation = d3.forceSimulation(data.nodes)
  .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
  .force("charge", d3.forceManyBody().strength(-200))
  .force("center", d3.forceCenter(width / 2, height / 2));

const link = svg.append("g").selectAll("line")
  .data(data.links).join("line")
  .attr("class", "link")
  .attr("stroke-width", d => Math.max(1, d.weight * 2));

const linkLabel = svg.append("g").selectAll("text")
  .data(data.links).join("text")
  .attr("class", "link-label")
  .text(d => d.label);

const node = svg.append("g").selectAll("circle")
  .data(data.nodes).join("circle")
  .attr("r", d => d.size || 8)
  .attr("fill", d => d.color)
  .call(d3.drag()
    .on("start", (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on("end", (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

const label = svg.append("g").selectAll("text")
  .data(data.nodes).join("text")
  .attr("class", "node-label")
  .attr("dy", d => -(d.size || 8) - 4)
  .text(d => d.name);

simulation.on("tick", () => {
  link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  linkLabel.attr("x", d => (d.source.x + d.target.x) / 2)
           .attr("y", d => (d.source.y + d.target.y) / 2);
  node.attr("cx", d => d.x).attr("cy", d => d.y);
  label.attr("x", d => d.x).attr("y", d => d.y);
});
</script>
</body>
</html>"""
