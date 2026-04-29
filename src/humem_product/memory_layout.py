from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .embeddings import cosine_similarity
from .memory_space import MemorySpace
from .models import MemoryFragment


@dataclass(frozen=True, slots=True)
class LayoutResult:
    layout_model: str
    has_embedding_layout: bool
    node_count: int
    semantic_edge_count: int
    relation_edge_count: int
    embedding_scope: str = "none"
    layout_updated_at: str | None = None


def apply_memory_layout(
    space: MemorySpace,
    *,
    fragment_embeddings: dict[str, list[float]] | None = None,
    embedding_scope: str = "none",
    iterations: int = 120,
    semantic_neighbors: int = 4,
) -> LayoutResult:
    updated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    fragments = list(space.fragments.values())
    if not fragments:
        return LayoutResult(
            layout_model="empty",
            has_embedding_layout=False,
            node_count=0,
            semantic_edge_count=0,
            relation_edge_count=0,
            embedding_scope=embedding_scope,
            layout_updated_at=updated_at,
        )

    for fragment in fragments:
        space._refresh_fragment_depth(fragment)

    edges: dict[tuple[str, str], float] = {}
    relation_edge_count = 0
    for relation in space.relations.values():
        if relation.source_id not in space.fragments or relation.target_id not in space.fragments:
            continue
        key = _edge_key(relation.source_id, relation.target_id)
        edges[key] = max(edges.get(key, 0.0), 0.20 + relation.weight * 0.55)
        relation_edge_count += 1

    semantic_edge_count = 0
    usable_embeddings = {
        fragment_id: vector
        for fragment_id, vector in (fragment_embeddings or {}).items()
        if fragment_id in space.fragments and vector
    }
    if usable_embeddings:
        semantic_edge_count = _add_semantic_edges(
            edges,
            usable_embeddings,
            neighbors=semantic_neighbors,
        )

    if not edges:
        layout_model = "hash-fallback"
    elif usable_embeddings:
        layout_model = "embedding-force"
    else:
        layout_model = "relation-force"

    _run_force_layout(space, fragments, edges, iterations=iterations)

    for fragment in fragments:
        fragment.metadata["layout_model"] = layout_model
        fragment.metadata["layout_has_embedding"] = bool(usable_embeddings)
        fragment.metadata["layout_embedding_scope"] = embedding_scope if usable_embeddings else "none"
        fragment.metadata["layout_semantic_edges"] = semantic_edge_count
        fragment.metadata["layout_relation_edges"] = relation_edge_count
        fragment.metadata["layout_updated_at"] = updated_at

    return LayoutResult(
        layout_model=layout_model,
        has_embedding_layout=bool(usable_embeddings),
        node_count=len(fragments),
        semantic_edge_count=semantic_edge_count,
        relation_edge_count=relation_edge_count,
        embedding_scope=embedding_scope if usable_embeddings else "none",
        layout_updated_at=updated_at,
    )


def _add_semantic_edges(
    edges: dict[tuple[str, str], float],
    embeddings: dict[str, list[float]],
    *,
    neighbors: int,
) -> int:
    ids = sorted(embeddings)
    added = 0
    for source_id in ids:
        scored: list[tuple[float, str]] = []
        source_vector = embeddings[source_id]
        for target_id in ids:
            if source_id == target_id:
                continue
            similarity = cosine_similarity(source_vector, embeddings[target_id])
            if similarity <= 0.15:
                continue
            scored.append((similarity, target_id))

        for similarity, target_id in sorted(scored, reverse=True)[:neighbors]:
            key = _edge_key(source_id, target_id)
            weight = 0.18 + min(similarity, 1.0) * 0.72
            if weight > edges.get(key, 0.0):
                edges[key] = weight
                added += 1
    return added


def _run_force_layout(
    space: MemorySpace,
    fragments: list[MemoryFragment],
    edges: dict[tuple[str, str], float],
    *,
    iterations: int,
) -> None:
    if len(fragments) == 1:
        fragment = fragments[0]
        fragment.x = 0.0
        fragment.y = 0.0
        fragment.z = space._depth_to_height(fragment.depth)
        return

    positions = {fragment.fragment_id: [fragment.x * 2.2, fragment.y * 2.2] for fragment in fragments}
    by_id = {fragment.fragment_id: fragment for fragment in fragments}
    node_count = len(fragments)
    target_radius = max(1.4, math.sqrt(node_count) * 0.32)
    edge_items = list(edges.items())

    for step in range(max(1, iterations)):
        forces = {fragment.fragment_id: [0.0, 0.0] for fragment in fragments}

        for left_index in range(node_count):
            left = fragments[left_index]
            left_pos = positions[left.fragment_id]
            for right_index in range(left_index + 1, node_count):
                right = fragments[right_index]
                right_pos = positions[right.fragment_id]
                dx = left_pos[0] - right_pos[0]
                dy = left_pos[1] - right_pos[1]
                distance_sq = max(dx * dx + dy * dy, 0.04)
                distance = math.sqrt(distance_sq)
                force = 0.035 / distance_sq
                fx = dx / distance * force
                fy = dy / distance * force
                forces[left.fragment_id][0] += fx
                forces[left.fragment_id][1] += fy
                forces[right.fragment_id][0] -= fx
                forces[right.fragment_id][1] -= fy

        for (source_id, target_id), weight in edge_items:
            source = by_id[source_id]
            target = by_id[target_id]
            source_pos = positions[source_id]
            target_pos = positions[target_id]
            dx = target_pos[0] - source_pos[0]
            dy = target_pos[1] - source_pos[1]
            distance = max(math.sqrt(dx * dx + dy * dy), 0.01)
            layer_gap = abs(source.depth - target.depth)
            desired = 0.38 + layer_gap * 0.18 + (1.0 - weight) * 0.75
            force = (distance - desired) * weight * 0.018
            fx = dx / distance * force
            fy = dy / distance * force
            forces[source_id][0] += fx
            forces[source_id][1] += fy
            forces[target_id][0] -= fx
            forces[target_id][1] -= fy

        cooling = 1.0 - step / max(iterations, 1)
        step_size = 0.38 * cooling + 0.035
        for fragment in fragments:
            pos = positions[fragment.fragment_id]
            force = forces[fragment.fragment_id]
            pos[0] += max(-0.18, min(force[0] * step_size, 0.18))
            pos[1] += max(-0.18, min(force[1] * step_size, 0.18))
            radius = math.sqrt(pos[0] * pos[0] + pos[1] * pos[1])
            if radius > target_radius:
                scale = target_radius / radius
                pos[0] *= scale
                pos[1] *= scale

    max_abs = max(max(abs(pos[0]), abs(pos[1])) for pos in positions.values()) or 1.0
    for fragment in fragments:
        pos = positions[fragment.fragment_id]
        fragment.x = pos[0] / max_abs
        fragment.y = pos[1] / max_abs
        fragment.z = space._depth_to_height(fragment.depth)


def _edge_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)
