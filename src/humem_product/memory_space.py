from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from .models import MemoryFragment, MemoryRelation, RetrievalHit
from .parser import normalize_text, parse_sentence


DEFAULT_LAYER_ACCESS_WEIGHTS = [1.0, 0.88, 0.74, 0.60, 0.46, 0.33, 0.22, 0.14]


@dataclass(slots=True)
class MemoryDynamicsConfig:
    layer_access_weights: list[float] = field(default_factory=lambda: list(DEFAULT_LAYER_ACCESS_WEIGHTS))
    base_depth_offset: float = 0.72
    activation_depth_lift: float = 0.18
    strength_depth_lift: float = 0.08
    retrieval_depth_lift: float = 0.05
    relation_depth_decay: float = 0.76
    relation_access_floor: float = 0.55
    forgetting_model: str = "ebbinghaus"
    retention_floor: float = 0.05
    base_forgetting_rate: float = 0.34
    strength_retention_lift: float = 0.28
    retrieval_retention_lift: float = 0.08
    feedback_retention_lift: float = 0.12
    negative_feedback_forgetting_boost: float = 0.35

    def __post_init__(self) -> None:
        if self.forgetting_model not in {"ebbinghaus", "linear"}:
            raise ValueError("forgetting_model must be one of: ebbinghaus, linear")
        if not 0.0 <= self.retention_floor < 1.0:
            raise ValueError("retention_floor must be >= 0 and < 1")
        if self.base_forgetting_rate < 0:
            raise ValueError("base_forgetting_rate must be non-negative")


class MemorySpace:
    def __init__(
        self,
        total_layers: int = 8,
        top_layer_quota: int = 50,
        bottom_layer_quota: int = 10,
        sealed_bottom_layers: int = 2,
        dynamics: MemoryDynamicsConfig | dict[str, Any] | None = None,
    ) -> None:
        self.total_layers = total_layers
        self.top_layer_quota = top_layer_quota
        self.bottom_layer_quota = bottom_layer_quota
        self.sealed_bottom_layers = sealed_bottom_layers
        self.dynamics = _coerce_dynamics(dynamics)

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
                depth = self._state_to_depth(layer, parsed.salience, parsed.salience, 0)
                fragment = MemoryFragment(
                    fragment_id=str(uuid4()),
                    text=parsed.text,
                    normalized_text=parsed.normalized_text,
                    kind=parsed.kind,
                    layer=layer,
                    x=x,
                    y=y,
                    z=self._depth_to_height(depth),
                    depth=depth,
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

    def retrieve(
        self,
        query: str,
        limit: int = 120,
        *,
        mutate: bool = True,
        reinforcement_amount: float = 0.08,
    ) -> list[RetrievalHit]:
        query_terms = {
            fragment.normalized_text
            for fragment in parse_sentence(query)[0]
            if fragment.kind != "clause"
        }
        if not query_terms:
            query_terms = {normalize_text(query)}

        per_layer_matches: dict[int, list[tuple[float, str]]] = defaultdict(list)
        raw_keyword_scores: dict[str, float] = {}
        for fragment in self.fragments.values():
            self.refresh_fragment_state(fragment)
            raw_score = self._raw_match_score(fragment, query_terms)
            score = raw_score * self.accessibility_weight(fragment.depth)
            if score <= 0:
                continue
            per_layer_matches[fragment.layer].append((score, fragment.fragment_id))
            raw_keyword_scores[fragment.fragment_id] = raw_score

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
                    depth=fragment.depth,
                    score=score,
                    activation=fragment.activation,
                    strength=fragment.strength,
                    accessibility=self.accessibility_weight(fragment.depth),
                    raw_keyword_score=raw_keyword_scores.get(fragment_id),
                    relation_bonus=score if via_relation else 0.0,
                    via_relation=via_relation,
                )
                for fragment_id, (score, via_relation) in direct_hits.items()
                if (fragment := self.fragments.get(fragment_id)) is not None
            ),
            key=lambda item: (item.score, -item.layer, item.strength),
            reverse=True,
        )

        if mutate:
            for hit in ranked[:limit]:
                self.fragments[hit.fragment_id].retrievals += 1
                if reinforcement_amount > 0:
                    self.reinforce(hit.fragment_id, amount=reinforcement_amount, reason="retrieval")

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
        self.refresh_fragment_state(fragment)
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

    def suppress(
        self,
        fragment_id: str,
        *,
        amount: float = 0.32,
        reason: str = "negative_feedback",
        demote_layers: int = 1,
    ) -> None:
        if fragment_id not in self.fragments:
            return

        fragment = self.fragments[fragment_id]
        fragment.activation = max(fragment.activation - amount, 0.0)
        fragment.strength = max(fragment.strength - amount * 0.9, 0.0)
        fragment.ease = max(fragment.ease - amount * 0.14, 0.05)
        fragment.forgettings += 1

        old_layer = fragment.layer
        if demote_layers > 0 and fragment.layer < self.total_layers - 1:
            fragment.layer = min(fragment.layer + demote_layers, self.total_layers - 1)
        self.refresh_fragment_state(fragment)
        fragment.metadata["last_suppression_reason"] = reason

        if old_layer != fragment.layer:
            self._rebuild_cross_layer_flags()

    def forget(self, step: float = 0.14) -> None:
        if self.dynamics.forgetting_model == "linear":
            self._forget_linear(step=step)
            return

        any_layer_change = False
        for fragment in self.fragments.values():
            old_elapsed = max(fragment.forgettings, 0)
            new_elapsed = old_elapsed + 1
            previous_retention = self._ebbinghaus_retention(fragment, step=step, elapsed_cycles=old_elapsed)
            current_retention = self._ebbinghaus_retention(fragment, step=step, elapsed_cycles=new_elapsed)
            cycle_retention = current_retention / previous_retention if previous_retention > 0 else current_retention
            cycle_retention = max(0.0, min(cycle_retention, 1.0))
            is_anchor = _is_consolidation_anchor(fragment)

            if is_anchor:
                cycle_retention = max(cycle_retention, 0.94)

            fragment.activation = max(fragment.activation * cycle_retention, 0.0)
            strength_retention = 1.0 - (1.0 - cycle_retention) * 0.38
            ease_retention = 1.0 - (1.0 - cycle_retention) * 0.22
            if is_anchor:
                strength_retention = max(strength_retention, 0.98)
                ease_retention = max(ease_retention, 0.985)
            fragment.strength = max(fragment.strength * strength_retention, 0.0)
            fragment.ease = max(fragment.ease * ease_retention, 0.05)
            fragment.forgettings += 1

            old_layer = fragment.layer
            support_signal = self._fragment_support_signal(fragment)
            if (
                is_anchor
                and current_retention < 0.12
                and fragment.activation < 0.08
                and fragment.layer < self.total_layers - 1
            ):
                fragment.layer = min(fragment.layer + 1, self.total_layers - 1)
            elif (
                not is_anchor
                and current_retention < 0.24
                and fragment.activation < 0.22
                and fragment.layer < self.total_layers - 1
            ):
                fragment.layer = min(fragment.layer + 1, self.total_layers - 1)
            elif (
                current_retention > 0.72
                and fragment.activation > 0.72
                and support_signal >= 3.0
            ):
                fragment.layer = self._move_up(fragment.layer, step * 0.5)
            elif fragment.activation > 1.15:
                fragment.layer = self._move_up(fragment.layer, step * 0.5)

            if old_layer != fragment.layer:
                any_layer_change = True
            self.refresh_fragment_state(fragment)

        if any_layer_change:
            self._rebuild_cross_layer_flags()

    def retention_for_fragment(self, fragment: MemoryFragment, *, step: float = 0.14) -> float:
        if self.dynamics.forgetting_model == "linear":
            return max(self.dynamics.retention_floor, 1.0 - max(fragment.forgettings, 0) * max(step, 0.0))
        return self._ebbinghaus_retention(fragment, step=step, elapsed_cycles=max(fragment.forgettings, 0))

    def _forget_linear(self, *, step: float) -> None:
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
                any_layer_change = True
            self.refresh_fragment_state(fragment)

        if any_layer_change:
            self._rebuild_cross_layer_flags()

    def _ebbinghaus_retention(
        self,
        fragment: MemoryFragment,
        *,
        step: float,
        elapsed_cycles: int,
    ) -> float:
        if elapsed_cycles <= 0:
            return 1.0
        rate_scale = max(step, 0.0) / 0.14 if step > 0 else 0.0
        rate = self.dynamics.base_forgetting_rate * rate_scale
        negative_feedback = _feedback_count(fragment, "negative")
        if negative_feedback:
            rate *= 1.0 + min(negative_feedback, 4) * self.dynamics.negative_feedback_forgetting_boost
        if _is_consolidation_anchor(fragment):
            rate *= 0.42
        stability = self._fragment_stability(fragment)
        value = math.exp(-rate * elapsed_cycles / max(stability, 0.1))
        return max(self.dynamics.retention_floor, min(value, 1.0))

    def _fragment_stability(self, fragment: MemoryFragment) -> float:
        relation_count = len(self.out_edges.get(fragment.fragment_id, set()) | self.in_edges.get(fragment.fragment_id, set()))
        positive_feedback = _feedback_count(fragment, "positive")
        source_support = min(max(fragment.source_count - 1, 0) * 0.06, 0.24)
        relation_support = min(relation_count * 0.04, 0.24)
        reinforcement_support = math.log1p(max(fragment.reinforcements - 1, 0)) * 0.05
        stability = (
            1.0
            + max(fragment.strength, 0.0) * self.dynamics.strength_retention_lift
            + math.log1p(max(fragment.retrievals, 0)) * self.dynamics.retrieval_retention_lift
            + positive_feedback * self.dynamics.feedback_retention_lift
            + source_support
            + relation_support
            + reinforcement_support
        )
        if _is_consolidation_anchor(fragment):
            stability += 1.2
        return stability

    def _fragment_support_signal(self, fragment: MemoryFragment) -> float:
        relation_count = len(self.out_edges.get(fragment.fragment_id, set()) | self.in_edges.get(fragment.fragment_id, set()))
        return (
            math.log1p(max(fragment.retrievals, 0))
            + math.log1p(max(fragment.reinforcements, 0))
            + _feedback_count(fragment, "positive")
            + min(max(fragment.source_count - 1, 0), 3) * 0.25
            + min(relation_count, 3) * 0.2
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": {
                "total_layers": self.total_layers,
                "top_layer_quota": self.top_layer_quota,
                "bottom_layer_quota": self.bottom_layer_quota,
                "sealed_bottom_layers": self.sealed_bottom_layers,
                "dynamics": asdict(self.dynamics),
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

    def _depth_to_height(self, depth: float) -> float:
        if self.total_layers == 1:
            return 1.0
        clamped = max(0.0, min(float(depth), self.total_layers - 1))
        return 1.0 - clamped / (self.total_layers - 1)

    def _state_to_depth(
        self,
        layer: int,
        activation: float,
        strength: float,
        retrievals: int,
    ) -> float:
        layer = max(0, min(self.total_layers - 1, int(layer)))
        if self.total_layers == 1:
            return 0.0
        lift = (
            activation * self.dynamics.activation_depth_lift
            + strength * self.dynamics.strength_depth_lift
            + math.log1p(max(retrievals, 0)) * self.dynamics.retrieval_depth_lift
        )
        depth = layer + self.dynamics.base_depth_offset - lift
        lower = float(layer)
        upper = min(float(layer) + 0.97, float(self.total_layers - 1))
        return max(lower, min(depth, upper))

    def refresh_fragment_state(self, fragment: MemoryFragment) -> None:
        fragment.layer = max(0, min(self.total_layers - 1, int(fragment.layer)))
        fragment.depth = self._state_to_depth(
            fragment.layer,
            fragment.activation,
            fragment.strength,
            fragment.retrievals,
        )
        fragment.z = self._depth_to_height(fragment.depth)

    _refresh_fragment_depth = refresh_fragment_state

    def accessibility_weight(self, depth: float) -> float:
        if self.total_layers <= 1:
            return 1.0

        bucket = max(0, min(self.total_layers - 1, int(math.floor(depth))))
        fraction = max(0.0, min(float(depth) - bucket, 0.999))

        def bucket_weight(index: int) -> float:
            weights = self.dynamics.layer_access_weights or DEFAULT_LAYER_ACCESS_WEIGHTS
            if index < len(weights):
                return weights[index]
            if self.total_layers <= 1:
                return weights[-1]
            ratio = index / (self.total_layers - 1)
            return max(0.08, 1.0 - ratio * 0.88)

        current = bucket_weight(bucket)
        next_weight = bucket_weight(min(bucket + 1, self.total_layers - 1))
        return current + (next_weight - current) * fraction

    def _raw_match_score(self, fragment: MemoryFragment, query_terms: set[str]) -> float:
        if fragment.normalized_text in query_terms:
            direct = 1.8
        elif any(term and term in fragment.normalized_text for term in query_terms):
            direct = 1.15
        else:
            return 0.0

        return direct + fragment.activation * 0.25 + fragment.strength * 0.2

    def _match_score(self, fragment: MemoryFragment, query_terms: set[str]) -> float:
        return self._raw_match_score(fragment, query_terms) * self.accessibility_weight(fragment.depth)

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
                depth_gap = abs(source.depth - target.depth)
                target_access = self.accessibility_weight(target.depth)
                score = (
                    source_score
                    * relation.weight
                    * (self.dynamics.relation_depth_decay ** depth_gap)
                    * (self.dynamics.relation_access_floor + target_access * (1.0 - self.dynamics.relation_access_floor))
                )
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


def _coerce_dynamics(value: MemoryDynamicsConfig | dict[str, Any] | None) -> MemoryDynamicsConfig:
    if value is None:
        return MemoryDynamicsConfig()
    if isinstance(value, MemoryDynamicsConfig):
        return value
    allowed = set(MemoryDynamicsConfig.__dataclass_fields__)
    return MemoryDynamicsConfig(**{key: val for key, val in value.items() if key in allowed})


def _feedback_count(fragment: MemoryFragment, key: str) -> int:
    feedback = fragment.metadata.get("feedback")
    if not isinstance(feedback, dict):
        return 0
    value = feedback.get(key, 0)
    return int(value) if isinstance(value, int | float) else 0


def _is_consolidation_anchor(fragment: MemoryFragment) -> bool:
    consolidation = fragment.metadata.get("consolidation")
    return isinstance(consolidation, dict) and bool(consolidation.get("anchor"))
