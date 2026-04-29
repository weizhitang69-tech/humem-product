from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryFragment:
    fragment_id: str
    text: str
    normalized_text: str
    kind: str
    layer: int
    x: float
    y: float
    z: float
    depth: float = 0.0
    activation: float = 0.0
    strength: float = 0.0
    ease: float = 0.5
    retrievals: int = 0
    reinforcements: int = 0
    forgettings: int = 0
    source_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryRelation:
    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    weight: float
    cross_layer: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalHit:
    fragment_id: str
    text: str
    kind: str
    layer: int
    depth: float
    score: float
    activation: float
    strength: float
    accessibility: float
    raw_keyword_score: float | None = None
    relation_bonus: float = 0.0
    via_relation: str | None = None


@dataclass(slots=True)
class ParsedFragment:
    text: str
    normalized_text: str
    kind: str
    salience: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedRelation:
    source_key: tuple[str, str]
    target_key: tuple[str, str]
    relation_type: str
    weight: float
    metadata: dict[str, Any] = field(default_factory=dict)
