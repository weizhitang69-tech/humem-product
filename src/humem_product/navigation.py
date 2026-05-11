from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable


SEMANTIC_INDEX_MODES = frozenset({"auto", "exact", "ann"})


@dataclass(frozen=True, slots=True)
class SemanticNavigationConfig:
    mode: str = "auto"
    min_items: int = 256
    m: int = 8
    ef_construction: int = 64
    ef_search: int = 32
    seed: int = 13

    def __post_init__(self) -> None:
        if self.mode not in SEMANTIC_INDEX_MODES:
            choices = ", ".join(sorted(SEMANTIC_INDEX_MODES))
            raise ValueError(f"semantic_index must be one of: {choices}")
        if self.min_items < 1:
            raise ValueError("semantic_index_min_items must be at least 1")
        if self.m < 1:
            raise ValueError("semantic_index_m must be at least 1")
        if self.ef_construction < self.m:
            raise ValueError("semantic_index_ef_construction must be >= semantic_index_m")
        if self.ef_search < 1:
            raise ValueError("semantic_index_ef_search must be at least 1")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "mode": self.mode,
            "min_items": self.min_items,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class NavigationHit:
    chunk_id: str
    score: float


@dataclass(slots=True)
class _Node:
    item_id: str
    vector: list[float]
    level: int
    neighbors: dict[int, set[str]] = field(default_factory=dict)


class SemanticNavigationIndex:
    """Small HNSW-like navigation graph for chunk embeddings.

    The graph is intentionally runtime-only. It is a candidate recall structure
    over source chunks, not a durable HuMem memory relation graph.
    """

    def __init__(self, config: SemanticNavigationConfig | None = None) -> None:
        self.config = config or SemanticNavigationConfig()
        self._rng = random.Random(self.config.seed)
        self._nodes: dict[str, _Node] = {}
        self._entry_id: str | None = None
        self._max_level = 0
        self._dimension: int | None = None
        self.last_visited = 0

    @property
    def item_count(self) -> int:
        return len(self._nodes)

    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def entry_id(self) -> str | None:
        return self._entry_id

    @property
    def max_level(self) -> int:
        return self._max_level

    def build(self, items: Iterable[tuple[str, list[float]]]) -> None:
        self._rng = random.Random(self.config.seed)
        self._nodes.clear()
        self._entry_id = None
        self._max_level = 0
        self._dimension = None
        self.last_visited = 0

        for item_id, vector in items:
            normalized = normalize_vector(vector)
            if not normalized:
                continue
            if self._dimension is None:
                self._dimension = len(normalized)
            if len(normalized) != self._dimension:
                continue
            self._insert(item_id, normalized)

        self.last_visited = 0

    def add_items(self, items: Iterable[tuple[str, list[float]]]) -> int:
        added = 0
        for item_id, vector in items:
            if item_id in self._nodes:
                continue
            normalized = normalize_vector(vector)
            if not normalized:
                continue
            if self._dimension is None:
                self._dimension = len(normalized)
            if len(normalized) != self._dimension:
                continue
            self._insert(item_id, normalized)
            added += 1
        self.last_visited = 0
        return added

    def snapshot(self) -> dict[str, object]:
        nodes = [
            {
                "item_id": node.item_id,
                "vector": list(node.vector),
                "level": node.level,
            }
            for node in sorted(self._nodes.values(), key=lambda item: item.item_id)
        ]
        edges: list[dict[str, object]] = []
        for node in sorted(self._nodes.values(), key=lambda item: item.item_id):
            for level, neighbors in sorted(node.neighbors.items()):
                for neighbor_id in sorted(neighbors):
                    edges.append(
                        {
                            "source": node.item_id,
                            "target": neighbor_id,
                            "level": level,
                        }
                    )
        return {
            "config": self.config.to_dict(),
            "entry_id": self._entry_id,
            "max_level": self._max_level,
            "dimension": self._dimension,
            "nodes": nodes,
            "edges": edges,
        }

    @classmethod
    def from_snapshot(
        cls,
        payload: dict[str, object],
        *,
        config: SemanticNavigationConfig | None = None,
    ) -> "SemanticNavigationIndex":
        if config is None:
            raw_config = payload.get("config", {})
            config = (
                SemanticNavigationConfig(**raw_config)
                if isinstance(raw_config, dict)
                else SemanticNavigationConfig()
            )
        index = cls(config)
        dimension = payload.get("dimension")
        index._dimension = int(dimension) if isinstance(dimension, int | float) else None
        entry_id = payload.get("entry_id")
        index._entry_id = str(entry_id) if entry_id else None
        max_level = payload.get("max_level")
        index._max_level = int(max_level) if isinstance(max_level, int | float) else 0

        for raw_node in payload.get("nodes", []):
            if not isinstance(raw_node, dict):
                continue
            item_id = str(raw_node.get("item_id") or "")
            vector = raw_node.get("vector")
            if not item_id or not isinstance(vector, list):
                continue
            normalized = normalize_vector(vector)
            if not normalized:
                continue
            if index._dimension is None:
                index._dimension = len(normalized)
            if len(normalized) != index._dimension:
                continue
            level = int(raw_node.get("level", 0))
            index._nodes[item_id] = _Node(item_id=item_id, vector=normalized, level=level)

        for raw_edge in payload.get("edges", []):
            if not isinstance(raw_edge, dict):
                continue
            source = str(raw_edge.get("source") or "")
            target = str(raw_edge.get("target") or "")
            level = int(raw_edge.get("level", 0))
            if source not in index._nodes or target not in index._nodes:
                continue
            index._nodes[source].neighbors.setdefault(level, set()).add(target)

        if index._entry_id not in index._nodes:
            index._entry_id = next(iter(index._nodes), None)
        if index._nodes and index._max_level < max(node.level for node in index._nodes.values()):
            index._max_level = max(node.level for node in index._nodes.values())
        return index

    def search(self, query: list[float], *, limit: int) -> list[NavigationHit]:
        self.last_visited = 0
        if limit <= 0 or self._entry_id is None or self._dimension is None:
            return []

        normalized_query = normalize_vector(query)
        if len(normalized_query) != self._dimension:
            return []

        entry_id = self._entry_id
        for level in range(self._max_level, 0, -1):
            entry_id = self._greedy_search(normalized_query, entry_id, level)

        ef = max(self.config.ef_search, limit)
        hits = self._search_layer(normalized_query, entry_id, ef=ef, level=0)
        return [
            NavigationHit(chunk_id=item_id, score=score)
            for score, item_id in hits[:limit]
        ]

    def _insert(self, item_id: str, vector: list[float]) -> None:
        if item_id in self._nodes:
            return

        level = self._random_level()
        self._nodes[item_id] = _Node(item_id=item_id, vector=vector, level=level)

        if self._entry_id is None:
            self._entry_id = item_id
            self._max_level = level
            return

        entry_id = self._entry_id
        for current_level in range(self._max_level, level, -1):
            entry_id = self._greedy_search(vector, entry_id, current_level)

        for current_level in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(
                vector,
                entry_id,
                ef=self.config.ef_construction,
                level=current_level,
            )
            neighbor_ids = [
                candidate_id
                for _score, candidate_id in candidates
                if candidate_id != item_id
            ][: self.config.m]
            for neighbor_id in neighbor_ids:
                self._connect(item_id, neighbor_id, current_level)
            if candidates:
                entry_id = candidates[0][1]

        if level > self._max_level:
            self._max_level = level
            self._entry_id = item_id

    def _random_level(self) -> int:
        level = 0
        probability = 1.0 / max(self.config.m, 2)
        while level < 32 and self._rng.random() < probability:
            level += 1
        return level

    def _greedy_search(self, query: list[float], entry_id: str, level: int) -> str:
        current_id = entry_id
        current_score = self._score(query, self._nodes[current_id].vector)
        improved = True

        while improved:
            improved = False
            for neighbor_id in sorted(self._nodes[current_id].neighbors.get(level, ())):
                neighbor_score = self._score(query, self._nodes[neighbor_id].vector)
                if neighbor_score > current_score:
                    current_id = neighbor_id
                    current_score = neighbor_score
                    improved = True
        return current_id

    def _search_layer(
        self,
        query: list[float],
        entry_id: str,
        *,
        ef: int,
        level: int,
    ) -> list[tuple[float, str]]:
        visited = {entry_id}
        candidates = {entry_id}
        best: dict[str, float] = {
            entry_id: self._score(query, self._nodes[entry_id].vector),
        }

        while candidates:
            current_id = max(
                candidates,
                key=lambda candidate_id: (best.get(candidate_id, -1.0), candidate_id),
            )
            candidates.remove(current_id)
            current_score = best.get(current_id, self._score(query, self._nodes[current_id].vector))
            if len(best) >= ef and current_score < min(best.values()):
                break

            for neighbor_id in sorted(self._nodes[current_id].neighbors.get(level, ())):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                score = self._score(query, self._nodes[neighbor_id].vector)
                worst_score = min(best.values()) if best else -1.0
                if len(best) < ef or score > worst_score:
                    candidates.add(neighbor_id)
                    best[neighbor_id] = score
                    if len(best) > ef:
                        worst_id = min(best, key=lambda item_id: (best[item_id], item_id))
                        best.pop(worst_id, None)

        self.last_visited += len(visited)
        return sorted(
            ((score, item_id) for item_id, score in best.items()),
            key=lambda item: item[0],
            reverse=True,
        )

    def _connect(self, left_id: str, right_id: str, level: int) -> None:
        if left_id == right_id:
            return
        self._nodes[left_id].neighbors.setdefault(level, set()).add(right_id)
        self._nodes[right_id].neighbors.setdefault(level, set()).add(left_id)
        self._prune_neighbors(left_id, level)
        self._prune_neighbors(right_id, level)

    def _prune_neighbors(self, item_id: str, level: int) -> None:
        neighbors = self._nodes[item_id].neighbors.get(level)
        if not neighbors or len(neighbors) <= self.config.m:
            return
        owner = self._nodes[item_id]
        ranked = sorted(
            neighbors,
            key=lambda neighbor_id: self._score(owner.vector, self._nodes[neighbor_id].vector),
            reverse=True,
        )
        self._nodes[item_id].neighbors[level] = set(ranked[: self.config.m])

    @staticmethod
    def _score(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        return sum(a * b for a, b in zip(left, right))


def exact_navigation_hits(
    items: Iterable[tuple[str, list[float]]],
    query: list[float],
    *,
    limit: int,
) -> list[NavigationHit]:
    if limit <= 0:
        return []
    hits = [
        NavigationHit(chunk_id=item_id, score=cosine_similarity(query, vector))
        for item_id, vector in items
    ]
    hits.sort(key=lambda item: item.score, reverse=True)
    return hits[:limit]


def normalize_vector(vector: list[float]) -> list[float]:
    if not vector:
        return []
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return [0.0 for _value in values]
    return [value / norm for value in values]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
