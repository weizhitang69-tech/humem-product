from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from .models import MemoryFragment, MemoryRelation, RetrievalHit
from .parser import normalize_text, parse_sentence


class MemorySpace:
    def __init__(
        self,
        total_layers: int = 8,
        top_layer_quota: int = 50,
        bottom_layer_quota: int = 10,
        sealed_bottom_layers: int = 2,
    ) -> None:
        self.total_layers = total_layers
        self.top_layer_quota = top_layer_quota
        self.bottom_layer_quota = bottom_layer_quota
        self.sealed_bottom_layers = sealed_bottom_layers

        self.fragments: dict[str, MemoryFragment] = {}
        self.fragment_index: dict[tuple[str, str], str] = {}
        self.relations: dict[str, MemoryRelation] = {}
        self.out_edges: dict[str, set[str]] = defaultdict(set)
        self.in_edges: dict[str, set[str]] = defaultdict(set)

    def ingest_sentence(self, text: str) -> list[str]:
        parsed_fragments, parsed_relations = parse_sentence(text)
        fragment_ids: list[str] = []

        for parsed in parsed_fragments:
            key = (parsed.normalized_text, parsed.kind)
            existing_id = self.fragment_index.get(key)
            if existing_id is None:
                layer = self._initial_layer(parsed.salience, parsed.kind)
                x, y = self._semantic_xy(parsed.normalized_text, parsed.kind)
                fragment = MemoryFragment(
                    fragment_id=str(uuid4()),
                    text=parsed.text,
                    normalized_text=parsed.normalized_text,
                    kind=parsed.kind,
                    layer=layer,
                    x=x,
                    y=y,
                    z=self._layer_to_height(layer),
                    activation=parsed.salience,
                    strength=parsed.salience,
                    ease=0.48 + parsed.salience * 0.3,
                    reinforcements=1,
                    source_count=1,
                    metadata=dict(parsed.metadata),
                )
                self.fragments[fragment.fragment_id] = fragment
                self.fragment_index[key] = fragment.fragment_id
                existing_id = fragment.fragment_id
            else:
                fragment = self.fragments[existing_id]
                fragment.source_count += 1
                self.reinforce(existing_id, amount=0.28, reason="repeat_ingest")

            fragment_ids.append(existing_id)

        for parsed_relation in parsed_relations:
            source_id = self.fragment_index.get(parsed_relation.source_key)
            target_id = self.fragment_index.get(parsed_relation.target_key)
            if source_id is None or target_id is None or source_id == target_id:
                continue
            self._upsert_relation(
                source_id=source_id,
                target_id=target_id,
                relation_type=parsed_relation.relation_type,
                weight=parsed_relation.weight,
                metadata=dict(parsed_relation.metadata),
            )

        self._rebuild_cross_layer_flags()
        return fragment_ids

    def retrieve(self, query: str, limit: int = 120) -> list[RetrievalHit]:
        query_terms = {
            fragment.normalized_text
            for fragment in parse_sentence(query)[0]
            if fragment.kind != "clause"
        }
        if not query_terms:
            query_terms = {normalize_text(query)}

        per_layer_matches: dict[int, list[tuple[float, str]]] = defaultdict(list)
        for fragment in self.fragments.values():
            score = self._match_score(fragment, query_terms)
            if score <= 0:
                continue
            per_layer_matches[fragment.layer].append((score, fragment.fragment_id))

        selected_ids: set[str] = set()
        direct_hits: dict[str, tuple[float, str | None]] = {}

        sealed_start = self.total_layers - self.sealed_bottom_layers
        for layer in range(self.total_layers):
            candidates = sorted(per_layer_matches.get(layer, []), reverse=True)
            quota = self._layer_quota(layer)

            if layer >= sealed_start:
                quota = 0

            for score, fragment_id in candidates[:quota]:
                selected_ids.add(fragment_id)
                direct_hits[fragment_id] = (score, None)

        expansion_hits = self._expand_connected_retrieval(selected_ids, per_layer_matches)
        for fragment_id, payload in expansion_hits.items():
            selected_ids.add(fragment_id)
            direct_hits.setdefault(fragment_id, payload)

        ranked = sorted(
            (
                RetrievalHit(
                    fragment_id=fragment.fragment_id,
                    text=fragment.text,
                    kind=fragment.kind,
                    layer=fragment.layer,
                    score=score,
                    activation=fragment.activation,
                    strength=fragment.strength,
                    via_relation=via_relation,
                )
                for fragment_id, (score, via_relation) in direct_hits.items()
                if (fragment := self.fragments.get(fragment_id)) is not None
            ),
            key=lambda item: (item.score, -item.layer, item.strength),
            reverse=True,
        )

        for hit in ranked[:limit]:
            self.fragments[hit.fragment_id].retrievals += 1
            self.reinforce(hit.fragment_id, amount=0.08, reason="retrieval")

        return ranked[:limit]

    def reinforce(
        self,
        fragment_id: str,
        amount: float = 0.4,
        reason: str = "manual",
        _visited: set[str] | None = None,
        _depth: int = 0,
    ) -> None:
        if fragment_id not in self.fragments:
            return

        visited = _visited or set()
        if fragment_id in visited or _depth > 2:
            return
        visited.add(fragment_id)

        fragment = self.fragments[fragment_id]
        fragment.activation = min(fragment.activation + amount, 3.0)
        fragment.strength = min(fragment.strength + amount * 0.8, 4.0)
        fragment.ease = min(fragment.ease + amount * 0.12, 0.98)
        fragment.reinforcements += 1
        old_layer = fragment.layer
        fragment.layer = self._move_up(fragment.layer, amount)
        fragment.z = self._layer_to_height(fragment.layer)
        fragment.metadata["last_reinforcement_reason"] = reason

        if old_layer != fragment.layer:
            self._rebuild_cross_layer_flags()

        for relation_id in self.out_edges.get(fragment_id, set()) | self.in_edges.get(fragment_id, set()):
            relation = self.relations[relation_id]
            neighbor_id = relation.target_id if relation.source_id == fragment_id else relation.source_id
            if neighbor_id in visited:
                continue

            neighbor = self.fragments[neighbor_id]
            layer_gap = abs(neighbor.layer - fragment.layer)
            propagated = amount * relation.weight * (0.68 ** layer_gap) * (0.66 ** (_depth + 1))
            if propagated < 0.05:
                continue
            self.reinforce(
                neighbor_id,
                amount=propagated,
                reason=f"linked_from:{fragment_id}",
                _visited=visited,
                _depth=_depth + 1,
            )

    def forget(self, step: float = 0.14) -> None:
        any_layer_change = False
        for fragment in self.fragments.values():
            fragment.activation = max(fragment.activation - step, 0.0)
            fragment.strength = max(fragment.strength - step * 0.65, 0.0)
            fragment.ease = max(fragment.ease - step * 0.08, 0.05)
            fragment.forgettings += 1

            old_layer = fragment.layer
            if fragment.activation < 0.22 and fragment.layer < self.total_layers - 1:
                fragment.layer = min(fragment.layer + 1, self.total_layers - 1)
            elif fragment.activation > 1.15:
                fragment.layer = self._move_up(fragment.layer, step * 0.5)

            if old_layer != fragment.layer:
                fragment.z = self._layer_to_height(fragment.layer)
                any_layer_change = True

        if any_layer_change:
            self._rebuild_cross_layer_flags()

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": {
                "total_layers": self.total_layers,
                "top_layer_quota": self.top_layer_quota,
                "bottom_layer_quota": self.bottom_layer_quota,
                "sealed_bottom_layers": self.sealed_bottom_layers,
            },
            "fragments": [asdict(fragment) for fragment in self.fragments.values()],
            "relations": [asdict(relation) for relation in self.relations.values()],
        }

    def _initial_layer(self, salience: float, kind: str) -> int:
        kind_bias = {
            "clause": 0,
            "action": -1,
            "concept": 0,
            "number": 1,
            "term": 1,
        }.get(kind, 1)
        raw = round((1.0 - salience) * (self.total_layers - 1)) + kind_bias
        return max(0, min(self.total_layers - 1, raw))

    def _layer_quota(self, layer: int) -> int:
        if self.total_layers <= 1:
            return self.top_layer_quota
        ratio = layer / (self.total_layers - 1)
        quota = self.top_layer_quota - (self.top_layer_quota - self.bottom_layer_quota) * ratio
        return max(self.bottom_layer_quota, int(round(quota)))

    def _move_up(self, layer: int, amount: float) -> int:
        if amount <= 0:
            return layer
        shift = max(1, int(math.ceil(amount * 2.4)))
        return max(0, layer - shift)

    def _semantic_xy(self, normalized_text: str, kind: str) -> tuple[float, float]:
        digest = hashlib.sha256(f"{kind}:{normalized_text}".encode("utf-8")).digest()
        x_raw = int.from_bytes(digest[:8], "big") / 2**64
        y_raw = int.from_bytes(digest[8:16], "big") / 2**64
        return (x_raw * 2.0 - 1.0, y_raw * 2.0 - 1.0)

    def _layer_to_height(self, layer: int) -> float:
        if self.total_layers == 1:
            return 1.0
        return 1.0 - layer / (self.total_layers - 1)

    def _match_score(self, fragment: MemoryFragment, query_terms: set[str]) -> float:
        if fragment.normalized_text in query_terms:
            direct = 1.8
        elif any(term and term in fragment.normalized_text for term in query_terms):
            direct = 1.15
        else:
            return 0.0

        layer_bonus = 1.0 + (self.total_layers - fragment.layer) / self.total_layers
        return direct * layer_bonus + fragment.activation * 0.25 + fragment.strength * 0.2

    def _expand_connected_retrieval(
        self,
        selected_ids: set[str],
        per_layer_matches: dict[int, list[tuple[float, str]]],
    ) -> dict[str, tuple[float, str | None]]:
        expansion_hits: dict[str, tuple[float, str | None]] = {}
        selected_layers = {self.fragments[fragment_id].layer for fragment_id in selected_ids}
        highest_selected_layer = min(selected_layers) if selected_layers else 0

        candidate_scores: dict[str, float] = {}
        candidate_relations: dict[str, str] = {}

        for source_id in selected_ids:
            for relation_id in self.out_edges.get(source_id, set()) | self.in_edges.get(source_id, set()):
                relation = self.relations[relation_id]
                target_id = relation.target_id if relation.source_id == source_id else relation.source_id
                if target_id in selected_ids:
                    continue
                target = self.fragments[target_id]
                if target.layer < highest_selected_layer:
                    continue

                source = self.fragments[source_id]
                source_score = self._match_score(source, {source.normalized_text})
                score = source_score * relation.weight * (0.72 ** abs(source.layer - target.layer))
                if score > candidate_scores.get(target_id, 0.0):
                    candidate_scores[target_id] = score
                    candidate_relations[target_id] = relation.relation_type

        sealed_start = self.total_layers - self.sealed_bottom_layers
        for target_id, score in candidate_scores.items():
            target = self.fragments[target_id]
            if target.layer < sealed_start:
                continue

            peers = sorted(
                (
                    item for item in per_layer_matches.get(target.layer, [])
                    if item[1] == target_id
                ),
                reverse=True,
            )
            direct_match_bonus = peers[0][0] if peers else 0.0
            expansion_hits[target_id] = (
                score + direct_match_bonus,
                candidate_relations[target_id],
            )

        return expansion_hits

    def _upsert_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        identity = f"{source_id}|{target_id}|{relation_type}"
        relation_id = hashlib.sha1(identity.encode("utf-8")).hexdigest()
        relation = self.relations.get(relation_id)
        if relation is None:
            relation = MemoryRelation(
                relation_id=relation_id,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                weight=weight,
                metadata=metadata or {},
            )
            self.relations[relation_id] = relation
            self.out_edges[source_id].add(relation_id)
            self.in_edges[target_id].add(relation_id)
        else:
            relation.weight = min(max(relation.weight, weight), 1.0)
            if metadata:
                relation.metadata.update(metadata)

    def _rebuild_cross_layer_flags(self) -> None:
        for relation in self.relations.values():
            source_layer = self.fragments[relation.source_id].layer
            target_layer = self.fragments[relation.target_id].layer
            relation.cross_layer = source_layer != target_layer
