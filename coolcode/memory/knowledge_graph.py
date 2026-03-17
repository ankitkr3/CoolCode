"""Knowledge graph with PageRank and community detection for insight ranking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """A knowledge graph that tracks relationships between code entities,
    insights, and agent learnings. Uses PageRank to surface the most
    influential/connected insights and community detection to cluster
    related knowledge.
    """

    def __init__(self, persist_path: str | None = None):
        self._graph = nx.DiGraph()
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self._persist_path.read_text())
        self._graph = nx.node_link_graph(data)

    def save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._graph)
        self._persist_path.write_text(json.dumps(data))

    def add_entity(self, entity_id: str, entity_type: str, **attrs: Any) -> None:
        """Add a node (file, function, class, insight, pattern, etc.)."""
        self._graph.add_node(entity_id, type=entity_type, **attrs)

    def add_relationship(
        self, source: str, target: str, relation: str, weight: float = 1.0, **attrs: Any
    ) -> None:
        """Add a directed edge (calls, imports, depends_on, related_to, etc.)."""
        self._graph.add_edge(source, target, relation=relation, weight=weight, **attrs)

    def get_pagerank(self, top_k: int = 20) -> list[tuple[str, float]]:
        """Return top-k entities ranked by PageRank (most influential first)."""
        if len(self._graph) == 0:
            return []
        pr = nx.pagerank(self._graph, weight="weight")
        ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def detect_communities(self) -> list[set[str]]:
        """Detect communities using greedy modularity on the undirected projection."""
        if len(self._graph) == 0:
            return []
        undirected = self._graph.to_undirected()
        from networkx.algorithms.community import greedy_modularity_communities

        return [set(c) for c in greedy_modularity_communities(undirected)]

    def get_neighbors(self, entity_id: str, depth: int = 1) -> dict[str, Any]:
        """Get the local neighborhood of an entity up to a given depth."""
        if entity_id not in self._graph:
            return {}
        visited: set[str] = set()
        frontier = {entity_id}
        subgraph_nodes: set[str] = set()
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                subgraph_nodes.add(node)
                next_frontier.update(self._graph.successors(node))
                next_frontier.update(self._graph.predecessors(node))
            frontier = next_frontier - visited
        subgraph_nodes.update(frontier)
        sub = self._graph.subgraph(subgraph_nodes)
        return {
            "nodes": [
                {"id": n, **self._graph.nodes[n]} for n in sub.nodes
            ],
            "edges": [
                {"source": u, "target": v, **d} for u, v, d in sub.edges(data=True)
            ],
        }

    def query_by_type(self, entity_type: str) -> list[dict[str, Any]]:
        """Get all entities of a specific type."""
        return [
            {"id": n, **self._graph.nodes[n]}
            for n in self._graph.nodes
            if self._graph.nodes[n].get("type") == entity_type
        ]

    @property
    def stats(self) -> dict[str, int]:
        return {"nodes": len(self._graph.nodes), "edges": len(self._graph.edges)}
