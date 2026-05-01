from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from .models import MemoryFragment


@dataclass(frozen=True, slots=True)
class ConsolidationCandidate:
    group_key: str
    title: str
    anchor_text: str
    theme_terms: list[str]
    support_fragment_ids: list[str]
    score: float


@dataclass(slots=True)
class MemoryConsolidationResult:
    created_anchor_ids: list[str] = field(default_factory=list)
    reinforced_anchor_ids: list[str] = field(default_factory=list)
    support_relations: int = 0
    candidates: list[ConsolidationCandidate] = field(default_factory=list)
    skipped_groups: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


def build_consolidation_candidates(
    fragments: Iterable[MemoryFragment],
    *,
    total_layers: int,
    scope: str = "document",
    max_anchors: int = 8,
    keywords_per_anchor: int = 5,
    min_support: int = 3,
) -> tuple[list[ConsolidationCandidate], int]:
    if scope not in {"document", "chunk", "global"}:
        raise ValueError("scope must be one of: document, chunk, global")
    if max_anchors < 1:
        raise ValueError("max_anchors must be at least 1")
    if keywords_per_anchor < 2:
        raise ValueError("keywords_per_anchor must be at least 2")
    if min_support < 2:
        raise ValueError("min_support must be at least 2")

    grouped: dict[str, list[MemoryFragment]] = defaultdict(list)
    titles: dict[str, str] = {}
    skipped = 0

    for fragment in fragments:
        if _is_anchor(fragment) or fragment.kind == "clause":
            continue
        if fragment.kind not in {"action", "concept", "number", "term"}:
            continue
        key, title = _group_identity(fragment, scope)
        if key is None:
            skipped += 1
            continue
        grouped[key].append(fragment)
        titles.setdefault(key, title)

    candidates: list[ConsolidationCandidate] = []
    for key, items in grouped.items():
        ranked = _rank_support_fragments(items, total_layers=total_layers)
        unique = _unique_by_normalized_text(ranked)
        if len(unique) < min_support:
            skipped += 1
            continue
        selected = unique[:keywords_per_anchor]
        theme_terms = [fragment.text for fragment in selected]
        title = titles.get(key, key)
        candidates.append(
            ConsolidationCandidate(
                group_key=key,
                title=title,
                anchor_text=_build_anchor_text(title, theme_terms),
                theme_terms=theme_terms,
                support_fragment_ids=[fragment.fragment_id for fragment in selected],
                score=sum(_fragment_value(fragment, total_layers=total_layers) for fragment in selected),
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:max_anchors], skipped


def _rank_support_fragments(
    fragments: Iterable[MemoryFragment],
    *,
    total_layers: int,
) -> list[MemoryFragment]:
    return sorted(
        fragments,
        key=lambda fragment: (
            _fragment_value(fragment, total_layers=total_layers),
            fragment.source_count,
            fragment.text.lower(),
        ),
        reverse=True,
    )


def _fragment_value(fragment: MemoryFragment, *, total_layers: int) -> float:
    if total_layers <= 1:
        layer_access = 1.0
    else:
        layer_access = 1.0 - min(max(fragment.layer, 0), total_layers - 1) / (total_layers - 1)
    feedback = fragment.metadata.get("feedback")
    positive = int(feedback.get("positive", 0)) if isinstance(feedback, dict) else 0
    negative = int(feedback.get("negative", 0)) if isinstance(feedback, dict) else 0
    return (
        fragment.activation
        + fragment.strength * 0.9
        + math.log1p(fragment.retrievals) * 0.45
        + fragment.source_count * 0.16
        + layer_access * 0.35
        + positive * 0.25
        - negative * 0.35
    )


def _unique_by_normalized_text(fragments: Iterable[MemoryFragment]) -> list[MemoryFragment]:
    selected: list[MemoryFragment] = []
    seen: set[str] = set()
    for fragment in fragments:
        if fragment.normalized_text in seen:
            continue
        seen.add(fragment.normalized_text)
        selected.append(fragment)
    return selected


def _is_anchor(fragment: MemoryFragment) -> bool:
    consolidation = fragment.metadata.get("consolidation")
    return isinstance(consolidation, dict) and bool(consolidation.get("anchor"))


def _group_identity(fragment: MemoryFragment, scope: str) -> tuple[str | None, str]:
    if scope == "global":
        return "global", "Global memory"

    source = _best_source(fragment.metadata.get("sources"))
    if source:
        if scope == "chunk":
            chunk_id = str(source.get("chunk_id", ""))
            if chunk_id:
                return f"chunk:{chunk_id}", str(source.get("title") or chunk_id)
        document_id = str(source.get("document_id", ""))
        if document_id:
            return f"document:{document_id}", str(source.get("title") or document_id)

    memories = fragment.metadata.get("memories")
    if isinstance(memories, list) and memories:
        first = memories[0]
        if isinstance(first, dict):
            source_name = str(first.get("source") or "memory")
            return f"memory:{source_name}", source_name

    return None, ""


def _best_source(value: Any) -> dict[str, Any] | None:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return first
    return None


def _build_anchor_text(title: str, theme_terms: list[str]) -> str:
    terms = ", ".join(theme_terms)
    return f"Consolidated memory: {title} centers on {terms}."
