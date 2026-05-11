from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .embeddings import EmbeddingProvider, cosine_similarity, make_embedding_provider
from .llm import LLMProvider, make_llm_provider
from .navigation import (
    NavigationHit,
    SemanticNavigationConfig,
    SemanticNavigationIndex,
    exact_navigation_hits,
)
from .parser import normalize_text


EVENT_STORE_VERSION = 4
EVENT_SCHEMA_REVISION = 2
DEFAULT_COLLECTION_ID = "default"
ACTIVE_STATUS = "active"
DELETED_STATUS = "deleted"
FILTER_OPERATORS = frozenset({"eq", "ne", "in", "nin", "contains", "gte", "lte", "exists"})
SUBTAG_ROLES = frozenset(
    {
        "when",
        "who",
        "action",
        "object",
        "place",
        "state",
        "cause",
        "outcome",
        "intent",
        "from_where",
        "to_where",
        "with_who",
    }
)


@dataclass(slots=True)
class CollectionSchema:
    allowed_roles: list[str] = field(default_factory=lambda: sorted(SUBTAG_ROLES))
    required_roles: list[str] = field(default_factory=list)
    embedding_model: str | None = None
    metadata_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryCollection:
    collection_id: str
    name: str
    schema: CollectionSchema = field(default_factory=CollectionSchema)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    status: str = ACTIVE_STATUS


@dataclass(slots=True)
class EventFilter:
    collections: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    event: dict[str, dict[str, Any]] = field(default_factory=dict)
    subtag: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    recall_max_difficulty: float | None = None
    include_deleted: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_constraints(self) -> bool:
        return bool(
            self.collections
            or self.roles
            or self.event
            or self.subtag
            or self.metadata
            or self.recall_max_difficulty is not None
            or self.include_deleted
        )

    @property
    def has_complex_constraints(self) -> bool:
        return bool(self.event or self.subtag or self.metadata or self.recall_max_difficulty is not None)


@dataclass(slots=True)
class EventMaintenanceResult:
    deleted: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    compacted: int = 0
    backed_up: str | None = None
    ann_dirty: bool = False
    ann_rebuilt: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddingItem:
    item_id: str
    event_id: str
    subtag_id: str | None
    collection_id: str
    item_type: str
    role: str
    text: str
    embedding: list[float]
    embedding_model: str | None
    position: int = 0


@dataclass(slots=True)
class EventSemanticMatch:
    embedding_score: float = 0.0
    matched_subtags: list[MemorySubTag] = field(default_factory=list)
    item_hits: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class MemorySubTag:
    subtag_id: str
    role: str
    value: str
    position: int
    embedding_text: str
    confidence: float = 1.0
    embedding: list[float] | None = None
    embedding_model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RecallState:
    retrievals: int = 0
    reinforcements: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    last_recalled_at: str | None = None
    recall_difficulty: float = 0.0
    stability: float = 1.0


@dataclass(slots=True)
class MemoryEvent:
    event_id: str
    main_label: str
    captured_at: str
    subtags: list[MemorySubTag]
    compressed_trace: str
    source: dict[str, Any]
    recall_state: RecallState = field(default_factory=RecallState)
    metadata: dict[str, Any] = field(default_factory=dict)
    main_embedding: list[float] | None = None
    embedding_model: str | None = None
    x: float = 0.0
    y: float = 0.0
    z: float = 1.0
    collection_id: str = DEFAULT_COLLECTION_ID
    status: str = ACTIVE_STATUS
    deleted_at: str | None = None
    updated_at: str | None = None
    version: int = 1


@dataclass(slots=True)
class RetrievalTargetSlot:
    role: str
    position: int | None = None


@dataclass(slots=True)
class RetrievalPlan:
    key_question: str
    retrieval_terms: list[str]
    target_roles: list[str] = field(default_factory=list)
    target_slots: list[RetrievalTargetSlot] = field(default_factory=list)
    time_hint: str | None = None
    recall_precision: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventEvidence:
    event_id: str
    main_label: str
    captured_at: str
    compressed_trace: str
    score: float
    keyword_score: float
    embedding_score: float
    recall_difficulty: float
    matched_subtags: list[MemorySubTag]
    matched_slots: list[dict[str, Any]] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    collection_id: str = DEFAULT_COLLECTION_ID
    source: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventAnswer:
    query: str
    answer: str
    evidence: list[EventEvidence]
    retrieval_plan: RetrievalPlan
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventFeedbackResult:
    positive: list[str] = field(default_factory=list)
    negative: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class EventRAG:
    """Event-centric temporal memory RAG.

    EventRAG stores compact structured event memories. LLM extraction creates a
    main label, ordered subtags, and a compressed trace. Embeddings are generated
    only for the main label and subtag embedding text.
    """

    def __init__(
        self,
        *,
        llm_provider: str | LLMProvider | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str = "https://api.openai.com/v1",
        embedding_provider: str | EmbeddingProvider | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: str | None = None,
        embedding_dimensions: int | None = None,
        require_providers: bool = True,
        forgetting_rate: float = 0.18,
        base_recall_threshold: float = 0.18,
        semantic_index: str = "auto",
        semantic_index_min_items: int = 256,
        semantic_index_m: int = 8,
        semantic_index_ef_construction: int = 64,
        semantic_index_ef_search: int = 32,
        semantic_index_seed: int = 13,
    ) -> None:
        self.llm_provider = make_llm_provider(
            llm_provider,
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url,
        )
        self.embedding_provider = make_embedding_provider(
            embedding_provider,
            model=embedding_model,
            api_key=embedding_api_key,
            dimensions=embedding_dimensions,
        )
        if require_providers:
            if self.llm_provider is None:
                raise ValueError("EventRAG requires an llm_provider")
            if self.embedding_provider is None:
                raise ValueError("EventRAG requires an embedding_provider")

        self.forgetting_rate = forgetting_rate
        self.base_recall_threshold = base_recall_threshold
        self.semantic_navigation_config = SemanticNavigationConfig(
            mode=semantic_index,
            min_items=semantic_index_min_items,
            m=semantic_index_m,
            ef_construction=semantic_index_ef_construction,
            ef_search=semantic_index_ef_search,
            seed=semantic_index_seed,
        )
        self._semantic_navigation_indexes: dict[str, SemanticNavigationIndex] = {}
        self._semantic_navigation_signatures: dict[str, tuple[tuple[str, str | None, int, str], ...]] = {}
        self._semantic_navigation_persisted: set[str] = set()
        self._semantic_navigation_tombstones: set[str] = set()
        self._semantic_navigation_build_count = 0
        self._last_semantic_navigation_strategy = "none"
        self._last_semantic_navigation_visited = 0
        self.events: dict[str, MemoryEvent] = {}
        self.collections: dict[str, MemoryCollection] = {
            DEFAULT_COLLECTION_ID: MemoryCollection(
                collection_id=DEFAULT_COLLECTION_ID,
                name=DEFAULT_COLLECTION_ID,
                schema=CollectionSchema(
                    embedding_model=getattr(self.embedding_provider, "model", None),
                ),
            )
        }
        self.events_log: list[dict[str, Any]] = []

    def ensure_collection(
        self,
        collection: str | None = None,
        *,
        schema: CollectionSchema | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        collection_id = _normalize_collection_id(collection)
        if collection_id in self.collections:
            return collection_id
        now = _utc_now()
        self.collections[collection_id] = MemoryCollection(
            collection_id=collection_id,
            name=collection_id,
            schema=_coerce_collection_schema(schema),
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self._log("collection_create", {"collection_id": collection_id})
        return collection_id

    def create_collection(
        self,
        name: str,
        *,
        collection_id: str | None = None,
        schema: CollectionSchema | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCollection:
        resolved_id = _normalize_collection_id(collection_id or name)
        now = _utc_now()
        existing = self.collections.get(resolved_id)
        if existing is not None:
            existing.name = name or existing.name
            if schema is not None:
                existing.schema = _coerce_collection_schema(schema)
            if metadata:
                existing.metadata.update(metadata)
            existing.updated_at = now
            existing.status = ACTIVE_STATUS
            return existing
        collection = MemoryCollection(
            collection_id=resolved_id,
            name=name,
            schema=_coerce_collection_schema(schema),
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self.collections[resolved_id] = collection
        self._log("collection_create", {"collection_id": resolved_id, "name": name})
        return collection

    def update_collection(
        self,
        collection_id: str,
        *,
        name: str | None = None,
        schema: CollectionSchema | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str | None = None,
    ) -> MemoryCollection:
        resolved_id = _normalize_collection_id(collection_id)
        if resolved_id not in self.collections:
            raise KeyError(f"unknown collection: {resolved_id}")
        collection = self.collections[resolved_id]
        if name is not None:
            collection.name = name
        if schema is not None:
            collection.schema = _coerce_collection_schema(schema)
        if metadata:
            collection.metadata.update(metadata)
        if status is not None:
            collection.status = status
        collection.updated_at = _utc_now()
        self._log("collection_update", {"collection_id": resolved_id})
        return collection

    def list_collections(self) -> list[MemoryCollection]:
        return sorted(self.collections.values(), key=lambda item: item.collection_id)

    def remember(
        self,
        text: str,
        *,
        source: str = "memory",
        metadata: dict[str, Any] | None = None,
        captured_at: str | datetime | None = None,
        collection: str = DEFAULT_COLLECTION_ID,
    ) -> list[str]:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        self._require_llm()
        self._require_embedding()

        captured = _coerce_datetime(captured_at)
        collection_id = self.ensure_collection(collection)
        events = self._extract_events(
            text.strip(),
            captured_at=_to_iso(captured),
            source=source,
            metadata=metadata or {},
            collection_id=collection_id,
        )
        self._embed_events(events)
        for event in events:
            self.events[event.event_id] = event
        self._add_events_to_semantic_navigation_indexes(events)
        self.refresh_layout()
        self._log(
            "remember",
            {
                "event_ids": [event.event_id for event in events],
                "source": source,
                "collection_id": collection_id,
            },
        )
        return [event.event_id for event in events]

    def add_memory(
        self,
        text: str,
        *,
        source: str = "memory",
        metadata: dict[str, Any] | None = None,
        captured_at: str | datetime | None = None,
        collection: str = DEFAULT_COLLECTION_ID,
    ) -> list[str]:
        return self.remember(
            text,
            source=source,
            metadata=metadata,
            captured_at=captured_at,
            collection=collection,
        )

    def add_document(
        self,
        text: str,
        *,
        document_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        captured_at: str | datetime | None = None,
        collection: str = DEFAULT_COLLECTION_ID,
    ) -> list[str]:
        source_metadata = dict(metadata or {})
        if document_id:
            source_metadata["document_id"] = document_id
        if title:
            source_metadata["title"] = title
        return self.remember(
            text,
            source=title or document_id or "document",
            metadata=source_metadata,
            captured_at=captured_at,
            collection=collection,
        )

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 8,
        mutate: bool = True,
        now: str | datetime | None = None,
        collection: str | None = None,
        filter: EventFilter | dict[str, Any] | str | None = None,
    ) -> list[EventEvidence]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        self._require_llm()
        self._require_embedding()

        plan = self.plan_query(query)
        evidence = self.retrieve_with_plan(
            plan,
            limit=limit,
            mutate=mutate,
            now=now,
            collection=collection,
            filter=filter,
        )
        return evidence

    def retrieve_with_plan(
        self,
        plan: RetrievalPlan,
        *,
        limit: int = 8,
        mutate: bool = True,
        now: str | datetime | None = None,
        collection: str | None = None,
        filter: EventFilter | dict[str, Any] | str | None = None,
    ) -> list[EventEvidence]:
        self._require_embedding()
        retrieval_terms = _unique_non_empty(plan.retrieval_terms or [plan.key_question])
        if not retrieval_terms:
            retrieval_terms = [plan.key_question]
        query_vectors = self.embedding_provider.embed_texts(retrieval_terms)  # type: ignore[union-attr]
        now_dt = _coerce_datetime(now)
        event_filter = _coerce_event_filter(filter, collection=collection)

        candidates: list[EventEvidence] = []
        normalized_terms = [normalize_text(term) for term in retrieval_terms if normalize_text(term)]
        target_slots = _target_slots_from_plan(plan)
        if not plan.target_slots and target_slots:
            plan.target_slots = list(target_slots)
        target_roles = _target_roles_from_slots(target_slots, plan.target_roles)
        if event_filter.roles:
            target_roles = target_roles | {role for role in event_filter.roles if role in SUBTAG_ROLES}
        keyword_event_ids = self._keyword_candidate_event_ids(normalized_terms, target_roles, event_filter=event_filter)
        semantic_matches = self._semantic_event_candidates(
            query_vectors,
            target_roles=target_roles,
            limit=max(limit * 8, 24),
            event_filter=event_filter,
        )

        if self._last_semantic_navigation_strategy == "exact":
            candidate_ids = {
                event_id
                for event_id, event in self.events.items()
                if self._event_visible(event, event_filter=event_filter)
            }
        else:
            candidate_ids = set(keyword_event_ids) | set(semantic_matches)

        for event_id in candidate_ids:
            event = self.events.get(event_id)
            if event is None:
                continue
            if not self._event_visible(event, event_filter=event_filter):
                continue
            evidence = self._score_event(
                event,
                normalized_terms=normalized_terms,
                query_vectors=query_vectors,
                target_roles=target_roles,
                target_slots=target_slots,
                recall_precision=plan.recall_precision,
                now=now_dt,
                semantic_match=semantic_matches.get(event.event_id),
                event_filter=event_filter,
            )
            if evidence is not None:
                candidates.append(evidence)

        candidates.sort(key=lambda item: item.score, reverse=True)
        selected = candidates[: max(limit, 0)]
        if mutate:
            recalled_at = _to_iso(now_dt)
            for item in selected:
                event = self.events[item.event_id]
                event.recall_state.retrievals += 1
                event.recall_state.last_recalled_at = recalled_at
                self.refresh_recall_state(event, now=now_dt)
            if selected:
                self._log("retrieve", {"event_ids": [item.event_id for item in selected]})
        return selected

    def plan_query(self, query: str) -> RetrievalPlan:
        self._require_llm()
        payload = self._generate_json_with_retry(
            schema_name="retrieval_plan",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the user question into a compact memory retrieval plan. "
                        "Return JSON with key_question, retrieval_terms, target_roles, target_slots, "
                        "time_hint, and recall_precision from 0 to 1. "
                        "target_slots is an array of role/position objects such as "
                        "{\"role\":\"place\",\"position\":3}; omit position when unknown. "
                        "The same role can appear multiple times in an event; position is the "
                        "event-local stage number for before/after logic, not a global event id."
                    ),
                },
                {"role": "user", "content": query},
            ],
            validator=_coerce_retrieval_plan,
        )
        return payload

    def answer(
        self,
        query: str,
        *,
        limit: int = 6,
        now: str | datetime | None = None,
        collection: str | None = None,
        filter: EventFilter | dict[str, Any] | str | None = None,
    ) -> EventAnswer:
        plan = self.plan_query(query)
        evidence = self.retrieve_with_plan(
            plan,
            limit=limit,
            mutate=True,
            now=now,
            collection=collection,
            filter=filter,
        )
        context = [_evidence_context(item) for item in evidence]
        answer_text = self.llm_provider.complete(  # type: ignore[union-attr]
            [
                {
                    "role": "system",
                    "content": (
                        "Answer using only the structured memory evidence. "
                        "Prefer matched_slots for the direct answer, then use stages with the "
                        "same position to recover event logic. "
                        "Do not rely on raw source text."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"query": query, "retrieval_plan": asdict(plan), "evidence": context},
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        if not answer_text:
            answer_text = _fallback_answer(query, evidence)
        diagnostics = {
            "event_count": len(self.events),
            "evidence_count": len(evidence),
            "embedded_items": self.embedded_item_count(),
            "context_uses_raw_source": False,
            "store_version": EVENT_STORE_VERSION,
        }
        return EventAnswer(
            query=query,
            answer=answer_text,
            evidence=evidence,
            retrieval_plan=plan,
            diagnostics=diagnostics,
        )

    def apply_feedback(
        self,
        *,
        positive_event_ids: Iterable[str] | None = None,
        negative_event_ids: Iterable[str] | None = None,
        reason: str = "user_feedback",
        now: str | datetime | None = None,
    ) -> EventFeedbackResult:
        now_dt = _coerce_datetime(now)
        positive = _unique_non_empty(positive_event_ids or [])
        positive_set = set(positive)
        negative = [item for item in _unique_non_empty(negative_event_ids or []) if item not in positive_set]
        result = EventFeedbackResult()

        for event_id in positive:
            event = self.events.get(event_id)
            if event is None:
                result.ignored.append(event_id)
                continue
            event.recall_state.positive_feedback += 1
            event.recall_state.reinforcements += 1
            event.metadata["last_feedback_reason"] = reason
            self.refresh_recall_state(event, now=now_dt)
            result.positive.append(event_id)

        for event_id in negative:
            event = self.events.get(event_id)
            if event is None:
                result.ignored.append(event_id)
                continue
            event.recall_state.negative_feedback += 1
            event.metadata["last_feedback_reason"] = reason
            self.refresh_recall_state(event, now=now_dt)
            result.negative.append(event_id)

        result.diagnostics = {
            "event_count": len(self.events),
            "positive": len(result.positive),
            "negative": len(result.negative),
            "ignored": len(result.ignored),
        }
        self._log(
            "feedback",
            {
                "positive": result.positive,
                "negative": result.negative,
                "ignored": result.ignored,
                "reason": reason,
            },
        )
        return result

    def delete_event(
        self,
        event_id: str,
        *,
        soft: bool = True,
        now: str | datetime | None = None,
    ) -> EventMaintenanceResult:
        event = self.events.get(event_id)
        result = EventMaintenanceResult()
        if event is None:
            result.diagnostics = {"ignored": [event_id], "reason": "missing_event"}
            return result
        now_iso = _to_iso(_coerce_datetime(now))
        if soft:
            event.status = DELETED_STATUS
            event.deleted_at = now_iso
            event.updated_at = now_iso
            event.version += 1
            self._mark_event_items_tombstoned(event)
            result.ann_dirty = bool(self._semantic_navigation_indexes)
        else:
            self._mark_event_items_tombstoned(event)
            self.events.pop(event_id, None)
            result.ann_dirty = bool(self._semantic_navigation_indexes)
        result.deleted.append(event_id)
        self.refresh_layout()
        self._log("delete", {"event_id": event_id, "soft": soft})
        return result

    def replace_event(
        self,
        event_id: str,
        text: str,
        *,
        source: str = "memory",
        metadata: dict[str, Any] | None = None,
        captured_at: str | datetime | None = None,
        collection: str | None = None,
    ) -> list[str]:
        previous = self.events.get(event_id)
        if previous is None:
            raise KeyError(f"unknown event: {event_id}")
        self.delete_event(event_id, soft=True, now=captured_at)
        replacement_metadata = dict(metadata or {})
        replacement_metadata.setdefault("replaces_event_id", event_id)
        replacement_collection = collection or previous.collection_id
        event_ids = self.remember(
            text,
            source=source,
            metadata=replacement_metadata,
            captured_at=captured_at,
            collection=replacement_collection,
        )
        for replacement_id in event_ids:
            replacement = self.events.get(replacement_id)
            if replacement is not None:
                replacement.metadata.update(replacement_metadata)
        self._log("replace", {"event_id": event_id, "replacement_event_ids": event_ids})
        return event_ids

    def patch_event_metadata(
        self,
        event_id: str,
        metadata: dict[str, Any],
        *,
        now: str | datetime | None = None,
    ) -> EventMaintenanceResult:
        event = self.events.get(event_id)
        result = EventMaintenanceResult()
        if event is None:
            result.diagnostics = {"ignored": [event_id], "reason": "missing_event"}
            return result
        event.metadata.update(metadata)
        event.updated_at = _to_iso(_coerce_datetime(now))
        event.version += 1
        result.updated.append(event_id)
        self._log("metadata_patch", {"event_id": event_id, "keys": sorted(metadata)})
        return result

    def compact(
        self,
        *,
        purge_deleted: bool = False,
        older_than_days: int | None = None,
        now: str | datetime | None = None,
    ) -> EventMaintenanceResult:
        result = EventMaintenanceResult()
        now_dt = _coerce_datetime(now)
        purge_ids: list[str] = []
        if purge_deleted:
            for event_id, event in self.events.items():
                if event.status != DELETED_STATUS:
                    continue
                if older_than_days is not None and event.deleted_at:
                    age_days = (now_dt - _parse_iso(event.deleted_at)).total_seconds() / 86400.0
                    if age_days < older_than_days:
                        continue
                purge_ids.append(event_id)
            for event_id in purge_ids:
                self.events.pop(event_id, None)
            result.deleted.extend(purge_ids)
            result.compacted = len(purge_ids)

        had_indexes = bool(self._semantic_navigation_indexes)
        index_names = list(self._semantic_navigation_indexes)
        if had_indexes:
            self._semantic_navigation_indexes.clear()
            self._semantic_navigation_signatures.clear()
            self._semantic_navigation_persisted.clear()
            self._semantic_navigation_tombstones.clear()
            if any(name.startswith("collection:") for name in index_names):
                collections = {
                    name.split(":", 2)[1]
                    for name in index_names
                    if name.startswith("collection:")
                }
                for collection_id in sorted(collections):
                    self.build_semantic_indexes(force=True, collection=collection_id)
            else:
                self.build_semantic_indexes(force=True)
            result.ann_rebuilt = True
        else:
            self._semantic_navigation_tombstones.clear()

        self.refresh_layout()
        self._log(
            "compact",
            {
                "purge_deleted": purge_deleted,
                "purged": purge_ids,
                "ann_rebuilt": result.ann_rebuilt,
            },
        )
        return result

    def refresh_recall_state(
        self,
        event: MemoryEvent,
        *,
        now: str | datetime | None = None,
    ) -> float:
        now_dt = _coerce_datetime(now)
        captured = _parse_iso(event.captured_at)
        age_days = max((now_dt - captured).total_seconds() / 86400.0, 0.0)
        state = event.recall_state
        stability = (
            1.0
            + state.reinforcements * 0.32
            + math.log1p(max(state.retrievals, 0)) * 0.22
            + state.positive_feedback * 0.45
            - min(state.negative_feedback, 5) * 0.12
        )
        stability = max(stability, 0.25)
        retention = math.exp(-self.forgetting_rate * age_days / stability)
        difficulty = 1.0 - retention
        difficulty += min(state.negative_feedback, 4) * 0.08
        difficulty -= min(state.positive_feedback, 4) * 0.05
        state.stability = stability
        state.recall_difficulty = max(0.0, min(difficulty, 0.98))
        return state.recall_difficulty

    def refresh_layout(self) -> None:
        active_events = [event for event in self.events.values() if event.status == ACTIVE_STATUS]
        if not active_events:
            return

        captured_times = [_parse_iso(event.captured_at).timestamp() for event in active_events]
        min_time = min(captured_times)
        max_time = max(captured_times)
        span = max(max_time - min_time, 1.0)

        for event in active_events:
            event_time = _parse_iso(event.captured_at).timestamp()
            event.z = (event_time - min_time) / span if len(active_events) > 1 else 1.0
            event.x, event.y = _embedding_xy(event.main_embedding, event.main_label)
            self.refresh_recall_state(event)

    def embedded_item_count(self) -> int:
        active_events = [event for event in self.events.values() if event.status == ACTIVE_STATUS]
        total = sum(1 for event in active_events if event.main_embedding)
        total += sum(1 for event in active_events for subtag in event.subtags if subtag.embedding)
        return total

    def stats(self) -> dict[str, Any]:
        active_events = [event for event in self.events.values() if event.status == ACTIVE_STATUS]
        deleted_events = [event for event in self.events.values() if event.status == DELETED_STATUS]
        difficulties = [event.recall_state.recall_difficulty for event in active_events]
        collection_counts: dict[str, int] = {}
        for event in active_events:
            collection_counts[event.collection_id] = collection_counts.get(event.collection_id, 0) + 1
        return {
            "version": EVENT_STORE_VERSION,
            "schema_revision": EVENT_SCHEMA_REVISION,
            "events": len(active_events),
            "total_events": len(self.events),
            "deleted_events": len(deleted_events),
            "collections": len(self.collections),
            "collection_event_counts": collection_counts,
            "subtags": sum(len(event.subtags) for event in active_events),
            "embedded_items": self.embedded_item_count(),
            "mean_recall_difficulty": sum(difficulties) / len(difficulties) if difficulties else 0.0,
            "oldest_captured_at": min((event.captured_at for event in active_events), default=None),
            "newest_captured_at": max((event.captured_at for event in active_events), default=None),
            "semantic_navigation": self.semantic_navigation_stats(),
        }

    def set_semantic_index(
        self,
        mode: str,
        *,
        min_items: int | None = None,
        m: int | None = None,
        ef_construction: int | None = None,
        ef_search: int | None = None,
        seed: int | None = None,
    ) -> None:
        config = self.semantic_navigation_config
        updated = SemanticNavigationConfig(
            mode=mode,
            min_items=config.min_items if min_items is None else min_items,
            m=config.m if m is None else m,
            ef_construction=config.ef_construction if ef_construction is None else ef_construction,
            ef_search=config.ef_search if ef_search is None else ef_search,
            seed=config.seed if seed is None else seed,
        )
        if updated == self.semantic_navigation_config:
            return
        self.semantic_navigation_config = updated
        self._invalidate_semantic_navigation_indexes()

    def build_semantic_indexes(
        self,
        *,
        roles: Iterable[str] | None = None,
        force: bool = False,
        collection: str | None = None,
    ) -> dict[str, Any]:
        collection_id = _normalize_collection_id(collection) if collection else None
        items = self._embedding_items(collections=[collection_id] if collection_id else None)
        role_names = {
            item.role
            for item in items
            if item.item_type == "subtag" and item.role in SUBTAG_ROLES
        }
        if roles is not None:
            requested = {role for role in roles if role in SUBTAG_ROLES}
            role_names = role_names & requested
        prefix = f"collection:{collection_id}:" if collection_id else ""
        index_names = [f"{prefix}global", *[f"{prefix}role:{role}" for role in sorted(role_names)]]
        built: list[str] = []
        reused: list[str] = []
        for index_name in index_names:
            index_items = self._items_for_index(index_name, items)
            if not index_items:
                continue
            signature = self._semantic_navigation_signature(index_items)
            if (
                not force
                and index_name in self._semantic_navigation_indexes
                and self._semantic_navigation_signatures.get(index_name) == signature
            ):
                reused.append(index_name)
                continue
            index = SemanticNavigationIndex(self.semantic_navigation_config)
            index.build((item.item_id, item.embedding) for item in index_items)
            self._semantic_navigation_indexes[index_name] = index
            self._semantic_navigation_signatures[index_name] = signature
            self._semantic_navigation_build_count += 1
            self._semantic_navigation_persisted.discard(index_name)
            built.append(index_name)
        return {
            "index_names": index_names,
            "built": built,
            "reused": reused,
            "item_count": len(items),
            "collection": collection_id,
            "build_count": self._semantic_navigation_build_count,
        }

    def semantic_navigation_stats(self) -> dict[str, Any]:
        items = self._embedding_items()
        return {
            "mode": self.semantic_navigation_config.mode,
            "min_items": self.semantic_navigation_config.min_items,
            "embedded_items": len(items),
            "index_built": bool(self._semantic_navigation_indexes),
            "index_names": sorted(self._semantic_navigation_indexes),
            "persisted_index_names": sorted(self._semantic_navigation_persisted),
            "tombstoned_items": len(self._semantic_navigation_tombstones),
            "index_count": len(self._semantic_navigation_indexes),
            "build_count": self._semantic_navigation_build_count,
            "last_strategy": self._last_semantic_navigation_strategy,
            "last_visited": self._last_semantic_navigation_visited,
        }

    def semantic_index_snapshots(self) -> list[dict[str, Any]]:
        items = self._embedding_items()
        snapshots: list[dict[str, Any]] = []
        for index_name, index in sorted(self._semantic_navigation_indexes.items()):
            index_items = self._items_for_index(index_name, items)
            signature = self._semantic_navigation_signature(index_items)
            if self._semantic_navigation_signatures.get(index_name) != signature:
                continue
            snapshots.append(
                {
                    "index_name": index_name,
                    "embedding_model": self._index_embedding_model(index_items),
                    "item_count": len(index_items),
                    "signature": [list(item) for item in signature],
                    "tombstones": sorted(
                        item_id
                        for item_id in self._semantic_navigation_tombstones
                        if _tombstone_belongs_to_index(index_name, item_id)
                    ),
                    "snapshot": index.snapshot(),
                }
            )
        return snapshots

    def restore_semantic_index_snapshots(self, snapshots: Iterable[dict[str, Any]]) -> int:
        items = self._embedding_items()
        restored = 0
        for payload in snapshots:
            index_name = str(payload.get("index_name") or "")
            if not index_name:
                continue
            index_items = self._items_for_index(index_name, items)
            signature = self._semantic_navigation_signature(index_items)
            raw_signature = payload.get("signature")
            stored_signature = tuple(
                (str(item[0]), item[1] if item[1] is None else str(item[1]), int(item[2]), str(item[3]))
                for item in raw_signature
                if isinstance(item, list | tuple) and len(item) == 4
            ) if isinstance(raw_signature, list) else ()
            if signature != stored_signature:
                continue
            snapshot = payload.get("snapshot")
            if not isinstance(snapshot, dict):
                continue
            tombstones = {
                str(item)
                for item in payload.get("tombstones", [])
                if isinstance(item, str)
            }
            index = SemanticNavigationIndex.from_snapshot(
                snapshot,
                config=self.semantic_navigation_config,
            )
            if index.item_count < len(index_items):
                continue
            self._semantic_navigation_indexes[index_name] = index
            self._semantic_navigation_signatures[index_name] = signature
            self._semantic_navigation_persisted.add(index_name)
            self._semantic_navigation_tombstones.update(tombstones)
            restored += 1
        return restored

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": EVENT_STORE_VERSION,
            "kind": "event_memory",
            "event_rag_config": {
                "schema_revision": EVENT_SCHEMA_REVISION,
                "forgetting_rate": self.forgetting_rate,
                "base_recall_threshold": self.base_recall_threshold,
                "llm_model": getattr(self.llm_provider, "model", None),
                "embedding_model": getattr(self.embedding_provider, "model", None),
                "semantic_navigation": self.semantic_navigation_config.to_dict(),
            },
            "collections": [_collection_to_dict(collection) for collection in self.list_collections()],
            "events": [_event_to_dict(event) for event in self.events.values()],
            "events_log": list(self.events_log),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        llm_provider: str | LLMProvider | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str = "https://api.openai.com/v1",
        embedding_provider: str | EmbeddingProvider | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: str | None = None,
        require_providers: bool = False,
    ) -> "EventRAG":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_snapshot(
            payload,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            require_providers=require_providers,
        )

    @classmethod
    def from_snapshot(
        cls,
        payload: dict[str, Any],
        *,
        llm_provider: str | LLMProvider | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str = "https://api.openai.com/v1",
        embedding_provider: str | EmbeddingProvider | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: str | None = None,
        require_providers: bool = False,
    ) -> "EventRAG":
        if int(payload.get("version", 0)) != EVENT_STORE_VERSION:
            raise ValueError("EventRAG can only load v4 event-memory stores")
        config = payload.get("event_rag_config", {})
        if not isinstance(config, dict):
            config = {}
        semantic_navigation = config.get("semantic_navigation", {})
        if not isinstance(semantic_navigation, dict):
            semantic_navigation = {}
        instance = cls(
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            require_providers=require_providers,
            forgetting_rate=float(config.get("forgetting_rate", 0.18)),
            base_recall_threshold=float(config.get("base_recall_threshold", 0.18)),
            semantic_index=semantic_navigation.get("mode", "auto"),
            semantic_index_min_items=semantic_navigation.get("min_items", 256),
            semantic_index_m=semantic_navigation.get("m", 8),
            semantic_index_ef_construction=semantic_navigation.get("ef_construction", 64),
            semantic_index_ef_search=semantic_navigation.get("ef_search", 32),
            semantic_index_seed=semantic_navigation.get("seed", 13),
        )
        instance.events = {
            str(item["event_id"]): _event_from_dict(item)
            for item in payload.get("events", [])
            if isinstance(item, dict) and item.get("event_id")
        }
        collections = {
            collection.collection_id: collection
            for collection in (
                _collection_from_dict(item)
                for item in payload.get("collections", [])
                if isinstance(item, dict)
            )
        }
        if DEFAULT_COLLECTION_ID not in collections:
            collections[DEFAULT_COLLECTION_ID] = instance.collections[DEFAULT_COLLECTION_ID]
        for event in instance.events.values():
            if event.collection_id not in collections:
                collections[event.collection_id] = MemoryCollection(
                    collection_id=event.collection_id,
                    name=event.collection_id,
                    schema=CollectionSchema(
                        embedding_model=getattr(instance.embedding_provider, "model", None),
                    ),
                )
        instance.collections = collections
        instance.events_log = [
            dict(item) for item in payload.get("events_log", []) if isinstance(item, dict)
        ]
        instance.refresh_layout()
        return instance

    def _semantic_event_candidates(
        self,
        query_vectors: list[list[float]],
        *,
        target_roles: set[str],
        limit: int,
        event_filter: EventFilter,
    ) -> dict[str, EventSemanticMatch]:
        collection_ids = event_filter.collections or None
        items = self._embedding_items(collections=collection_ids)
        if not items or not query_vectors:
            self._last_semantic_navigation_strategy = "none"
            self._last_semantic_navigation_visited = 0
            return {}

        config = self.semantic_navigation_config
        force_exact = event_filter.has_complex_constraints
        if force_exact or config.mode == "exact" or (config.mode == "auto" and len(items) < config.min_items):
            self._last_semantic_navigation_strategy = "exact"
            self._last_semantic_navigation_visited = len(items) * len(query_vectors)
            role_items = self._role_filtered_items(items, target_roles)
            return self._exact_semantic_matches(
                [item for item in role_items if self._event_visible(self.events[item.event_id], event_filter=event_filter)],
                query_vectors,
                limit=limit,
            )

        matches = self._ann_semantic_matches(
            items,
            query_vectors,
            target_roles=target_roles,
            limit=limit,
            event_filter=event_filter,
        )
        if matches:
            self._last_semantic_navigation_strategy = "ann"
            return matches

        self._last_semantic_navigation_strategy = "ann_fallback_exact"
        self._last_semantic_navigation_visited = len(items) * len(query_vectors)
        return self._exact_semantic_matches(
            [
                item
                for item in self._role_filtered_items(items, target_roles)
                if self._event_visible(self.events[item.event_id], event_filter=event_filter)
            ],
            query_vectors,
            limit=limit,
        )

    def _ann_semantic_matches(
        self,
        items: list[EmbeddingItem],
        query_vectors: list[list[float]],
        *,
        target_roles: set[str],
        limit: int,
        event_filter: EventFilter,
    ) -> dict[str, EventSemanticMatch]:
        index_prefix = ""
        if len(event_filter.collections) == 1:
            index_prefix = f"collection:{event_filter.collections[0]}:"
        index_names = (
            [f"{index_prefix}role:{role}" for role in sorted(target_roles)]
            if target_roles
            else [f"{index_prefix}global"]
        )
        item_by_id = {
            item.item_id: item
            for item in items
            if self._event_visible(self.events[item.event_id], event_filter=event_filter)
        }
        hits: list[NavigationHit] = []
        visited = 0
        for index_name in index_names:
            index_items = self._items_for_index(index_name, items)
            if not index_items:
                continue
            index = self._ensure_semantic_navigation_index(index_name, index_items)
            for query_vector in query_vectors:
                query_hits = index.search(query_vector, limit=limit)
                hits.extend(query_hits)
                visited += index.last_visited

        if not hits and target_roles:
            fallback_name = f"{index_prefix}global"
            index_items = self._items_for_index(fallback_name, items)
            if index_items:
                index = self._ensure_semantic_navigation_index(fallback_name, index_items)
                for query_vector in query_vectors:
                    query_hits = index.search(query_vector, limit=limit)
                    hits.extend(query_hits)
                    visited += index.last_visited

        self._last_semantic_navigation_visited = visited
        return self._aggregate_item_hits(hits, item_by_id, limit=limit)

    def _exact_semantic_matches(
        self,
        items: list[EmbeddingItem],
        query_vectors: list[list[float]],
        *,
        limit: int,
    ) -> dict[str, EventSemanticMatch]:
        item_by_id = {item.item_id: item for item in items}
        hits: list[NavigationHit] = []
        for query_vector in query_vectors:
            hits.extend(
                exact_navigation_hits(
                    ((item.item_id, item.embedding) for item in items),
                    query_vector,
                    limit=limit,
                )
            )
        return self._aggregate_item_hits(hits, item_by_id, limit=limit)

    def _aggregate_item_hits(
        self,
        hits: list[NavigationHit],
        item_by_id: dict[str, EmbeddingItem],
        *,
        limit: int,
    ) -> dict[str, EventSemanticMatch]:
        by_event: dict[str, EventSemanticMatch] = {}
        sorted_hits = sorted(hits, key=lambda item: item.score, reverse=True)[: max(limit * 3, limit)]
        seen_item_ids: set[str] = set()
        for hit in sorted_hits:
            if hit.chunk_id in seen_item_ids:
                continue
            seen_item_ids.add(hit.chunk_id)
            item = item_by_id.get(hit.chunk_id)
            if item is None:
                continue
            match = by_event.setdefault(item.event_id, EventSemanticMatch())
            match.embedding_score = max(match.embedding_score, max(0.0, min(hit.score, 1.0)))
            event = self.events.get(item.event_id)
            if event and item.subtag_id:
                subtag = next((candidate for candidate in event.subtags if candidate.subtag_id == item.subtag_id), None)
                if subtag and all(existing.subtag_id != subtag.subtag_id for existing in match.matched_subtags):
                    match.matched_subtags.append(subtag)
                    match.matched_subtags.sort(key=lambda tag: tag.position)
            match.item_hits.append(
                {
                    "item_id": item.item_id,
                    "role": item.role,
                    "item_type": item.item_type,
                    "position": item.position,
                    "score": hit.score,
                }
            )
        return by_event

    def _keyword_candidate_event_ids(
        self,
        normalized_terms: list[str],
        target_roles: set[str],
        *,
        event_filter: EventFilter,
    ) -> set[str]:
        event_ids: set[str] = set()
        for event in self.events.values():
            if not self._event_visible(event, event_filter=event_filter):
                continue
            keyword_score, _exactness, _subtags = _keyword_match(event, normalized_terms, target_roles)
            if keyword_score > 0:
                event_ids.add(event.event_id)
        return event_ids

    def _ensure_semantic_navigation_index(
        self,
        index_name: str,
        items: list[EmbeddingItem],
    ) -> SemanticNavigationIndex:
        signature = self._semantic_navigation_signature(items)
        if (
            index_name in self._semantic_navigation_indexes
            and self._semantic_navigation_signatures.get(index_name) == signature
        ):
            return self._semantic_navigation_indexes[index_name]

        existing = self._semantic_navigation_indexes.get(index_name)
        existing_signature = self._semantic_navigation_signatures.get(index_name)
        if existing is not None and existing_signature is not None:
            old = set(existing_signature)
            new = set(signature)
            if old.issubset(new):
                old_item_ids = {item[0] for item in old}
                missing_items = [item for item in items if item.item_id not in old_item_ids]
                added = existing.add_items((item.item_id, item.embedding) for item in missing_items)
                if added == len(missing_items):
                    self._semantic_navigation_signatures[index_name] = signature
                    self._semantic_navigation_persisted.discard(index_name)
                    return existing
            elif new.issubset(old):
                self._semantic_navigation_signatures[index_name] = signature
                self._semantic_navigation_persisted.discard(index_name)
                return existing

        index = SemanticNavigationIndex(self.semantic_navigation_config)
        index.build((item.item_id, item.embedding) for item in items)
        self._semantic_navigation_indexes[index_name] = index
        self._semantic_navigation_signatures[index_name] = signature
        self._semantic_navigation_persisted.discard(index_name)
        self._semantic_navigation_build_count += 1
        return index

    def _embedding_items(
        self,
        *,
        collections: Iterable[str] | None = None,
        include_deleted: bool = False,
    ) -> list[EmbeddingItem]:
        collection_set = {_normalize_collection_id(item) for item in collections or []}
        items: list[EmbeddingItem] = []
        for event in self.events.values():
            if not include_deleted and event.status != ACTIVE_STATUS:
                continue
            if collection_set and event.collection_id not in collection_set:
                continue
            if event.main_embedding:
                items.append(
                    EmbeddingItem(
                        item_id=f"{event.event_id}:main",
                        event_id=event.event_id,
                        subtag_id=None,
                        collection_id=event.collection_id,
                        item_type="main_label",
                        role="main_label",
                        text=event.main_label,
                        embedding=event.main_embedding,
                        embedding_model=event.embedding_model,
                        position=0,
                    )
                )
            for subtag in event.subtags:
                if not subtag.embedding:
                    continue
                items.append(
                    EmbeddingItem(
                        item_id=f"{event.event_id}:{subtag.subtag_id}",
                        event_id=event.event_id,
                        subtag_id=subtag.subtag_id,
                        collection_id=event.collection_id,
                        item_type="subtag",
                        role=subtag.role,
                        text=subtag.embedding_text,
                        embedding=subtag.embedding,
                        embedding_model=subtag.embedding_model,
                        position=subtag.position,
                    )
                )
        return items

    def _role_filtered_items(
        self,
        items: list[EmbeddingItem],
        target_roles: set[str],
    ) -> list[EmbeddingItem]:
        if not target_roles:
            return items
        role_items = [item for item in items if item.role in target_roles]
        return role_items or items

    def _items_for_index(self, index_name: str, items: list[EmbeddingItem]) -> list[EmbeddingItem]:
        if index_name.startswith("collection:"):
            _prefix, collection_id, rest = index_name.split(":", 2)
            collection_items = [item for item in items if item.collection_id == collection_id]
            return self._items_for_index(rest, collection_items)
        if index_name == "global":
            return items
        if index_name.startswith("role:"):
            role = index_name.split(":", 1)[1]
            return [item for item in items if item.role == role]
        return []

    def _semantic_navigation_signature(
        self,
        items: list[EmbeddingItem],
    ) -> tuple[tuple[str, str | None, int, str], ...]:
        return tuple(
            sorted(
                (
                    item.item_id,
                    item.embedding_model,
                    len(item.embedding),
                    _embedding_checksum(item.embedding),
                )
                for item in items
            )
        )

    def _index_embedding_model(self, items: list[EmbeddingItem]) -> str | None:
        models = {item.embedding_model for item in items if item.embedding_model}
        return sorted(models)[0] if len(models) == 1 else None

    def _invalidate_semantic_navigation_indexes(self) -> None:
        self._semantic_navigation_indexes.clear()
        self._semantic_navigation_signatures.clear()
        self._semantic_navigation_persisted.clear()
        self._semantic_navigation_tombstones.clear()
        self._last_semantic_navigation_strategy = "none"
        self._last_semantic_navigation_visited = 0

    def _add_events_to_semantic_navigation_indexes(self, events: list[MemoryEvent]) -> None:
        if not self._semantic_navigation_indexes:
            return
        new_items = self._embedding_items_for_events(events)
        if not new_items:
            return
        for index_name, index in list(self._semantic_navigation_indexes.items()):
            index_items = self._items_for_index(index_name, new_items)
            if not index_items:
                continue
            added = index.add_items((item.item_id, item.embedding) for item in index_items)
            if added != len(index_items):
                self._semantic_navigation_indexes.pop(index_name, None)
                self._semantic_navigation_signatures.pop(index_name, None)
                self._semantic_navigation_persisted.discard(index_name)
                continue
            all_items = self._items_for_index(index_name, self._embedding_items())
            self._semantic_navigation_signatures[index_name] = self._semantic_navigation_signature(all_items)
            self._semantic_navigation_persisted.discard(index_name)

    def _embedding_items_for_events(self, events: list[MemoryEvent]) -> list[EmbeddingItem]:
        original_events = self.events
        try:
            self.events = {event.event_id: event for event in events if event.status == ACTIVE_STATUS}
            return self._embedding_items(include_deleted=False)
        finally:
            self.events = original_events

    def _mark_event_items_tombstoned(self, event: MemoryEvent) -> None:
        item_ids = {f"{event.event_id}:main"}
        item_ids.update(f"{event.event_id}:{subtag.subtag_id}" for subtag in event.subtags)
        self._semantic_navigation_tombstones.update(item_ids)
        for index_name in list(self._semantic_navigation_indexes):
            index_items = self._items_for_index(index_name, self._embedding_items())
            self._semantic_navigation_signatures[index_name] = self._semantic_navigation_signature(index_items)
            self._semantic_navigation_persisted.discard(index_name)

    def _event_visible(self, event: MemoryEvent, *, event_filter: EventFilter) -> bool:
        return _event_matches_filter(self, event, event_filter)

    def _score_event(
        self,
        event: MemoryEvent,
        *,
        normalized_terms: list[str],
        query_vectors: list[list[float]],
        target_roles: set[str],
        target_slots: list[RetrievalTargetSlot],
        recall_precision: float,
        now: datetime,
        semantic_match: EventSemanticMatch | None = None,
        event_filter: EventFilter | None = None,
    ) -> EventEvidence | None:
        if event_filter is not None and not self._event_visible(event, event_filter=event_filter):
            return None
        difficulty = self.refresh_recall_state(event, now=now)
        keyword_score, exactness, matched_subtags = _keyword_match(event, normalized_terms, target_roles)
        if semantic_match is None:
            embedding_score, embedding_matches = _embedding_match(event, query_vectors)
            item_hits: list[dict[str, Any]] = []
        else:
            embedding_score = semantic_match.embedding_score
            embedding_matches = semantic_match.matched_subtags
            item_hits = semantic_match.item_hits
        matched_subtags = _rank_matched_subtags(
            event,
            keyword_matches=matched_subtags,
            embedding_matches=embedding_matches,
            target_slots=target_slots,
            target_roles=target_roles,
        )

        slot_bonus = _slot_match_bonus(matched_subtags, target_slots, target_roles)
        score = keyword_score * 0.46 + embedding_score * 0.44 + exactness * 0.10 + slot_bonus
        score = max(0.0, min(score, 1.5))

        precision = max(0.0, min(float(recall_precision), 1.0))
        threshold = self.base_recall_threshold + difficulty * 0.48 - precision * 0.16
        threshold = max(0.05, min(threshold, 0.78))
        if score < threshold and (exactness < 0.65 or score < threshold * 0.72):
            return None

        return EventEvidence(
            event_id=event.event_id,
            main_label=event.main_label,
            captured_at=event.captured_at,
            compressed_trace=event.compressed_trace,
            score=score,
            keyword_score=keyword_score,
            embedding_score=embedding_score,
            recall_difficulty=difficulty,
            matched_subtags=matched_subtags,
            matched_slots=_matched_slots_payload(target_slots, matched_subtags),
            stages=_stages_payload(event.subtags),
            collection_id=event.collection_id,
            source=dict(event.source),
            diagnostics={
                "exactness": exactness,
                "threshold": threshold,
                "target_roles": sorted(target_roles),
                "target_slots": [_target_slot_to_dict(slot) for slot in target_slots],
                "slot_bonus": slot_bonus,
                "semantic_item_hits": item_hits,
            },
        )

    def _extract_events(
        self,
        text: str,
        *,
        captured_at: str,
        source: str,
        metadata: dict[str, Any],
        collection_id: str,
    ) -> list[MemoryEvent]:
        collection = self.collections[collection_id]
        allowed_roles = set(collection.schema.allowed_roles or sorted(SUBTAG_ROLES))
        required_roles = set(collection.schema.required_roles or [])
        return self._generate_json_with_retry(
            schema_name="event_extraction",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract durable user memory as event chunks. Return JSON with an events array. "
                        "Each event must have main_label, compressed_trace, and ordered subtags. "
                        "Each subtag must have role, value, position, embedding_text, confidence. "
                        f"Allowed roles: {', '.join(sorted(allowed_roles))}. "
                        "Use position to preserve before/after logic and causal state changes."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"captured_at": captured_at, "text": text},
                        ensure_ascii=False,
                    ),
                },
            ],
            validator=lambda payload: _coerce_event_extraction(
                payload,
                text=text,
                captured_at=captured_at,
                source=source,
                metadata=metadata,
                collection_id=collection_id,
                allowed_roles=allowed_roles,
                required_roles=required_roles,
            ),
        )

    def _embed_events(self, events: list[MemoryEvent]) -> None:
        texts: list[str] = []
        owners: list[MemoryEvent | MemorySubTag] = []
        for event in events:
            texts.append(event.main_label)
            owners.append(event)
            for subtag in event.subtags:
                texts.append(subtag.embedding_text)
                owners.append(subtag)

        embeddings = self.embedding_provider.embed_texts(texts)  # type: ignore[union-attr]
        model = self.embedding_provider.model  # type: ignore[union-attr]
        for owner, embedding in zip(owners, embeddings):
            if isinstance(owner, MemoryEvent):
                owner.main_embedding = embedding
                owner.embedding_model = model
            else:
                owner.embedding = embedding
                owner.embedding_model = model

    def _generate_json_with_retry(
        self,
        *,
        schema_name: str,
        messages: list[dict[str, str]],
        validator: Any,
    ) -> Any:
        last_error: Exception | None = None
        current_messages = list(messages)
        for attempt in range(2):
            payload = self.llm_provider.generate_json(current_messages, schema_name=schema_name)  # type: ignore[union-attr]
            try:
                return validator(payload)
            except ValueError as exc:
                last_error = exc
                if attempt == 0:
                    current_messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                f"The previous {schema_name} JSON failed validation: {exc}. "
                                "Return corrected JSON only."
                            ),
                        },
                    ]
        raise ValueError(f"{schema_name} JSON failed validation: {last_error}")

    def _require_llm(self) -> None:
        if self.llm_provider is None:
            raise ValueError("EventRAG requires an llm_provider for this operation")

    def _require_embedding(self) -> None:
        if self.embedding_provider is None:
            raise ValueError("EventRAG requires an embedding_provider for this operation")

    def _log(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events_log.append(
            {
                "log_id": str(uuid4()),
                "event_type": event_type,
                "payload": payload,
                "created_at": _utc_now(),
            }
        )


def _coerce_event_extraction(
    payload: dict[str, Any],
    *,
    text: str,
    captured_at: str,
    source: str,
    metadata: dict[str, Any],
    collection_id: str = DEFAULT_COLLECTION_ID,
    allowed_roles: set[str] | None = None,
    required_roles: set[str] | None = None,
) -> list[MemoryEvent]:
    items = payload.get("events")
    if not isinstance(items, list) or not items:
        raise ValueError("events must be a non-empty array")

    events: list[MemoryEvent] = []
    allowed = allowed_roles or set(SUBTAG_ROLES)
    required = required_roles or set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each event must be an object")
        main_label = _required_text(item, "main_label")
        compressed_trace = _required_text(item, "compressed_trace")
        event_id = str(item.get("event_id") or uuid4())
        raw_subtags = item.get("subtags")
        if not isinstance(raw_subtags, list):
            raise ValueError("event subtags must be an array")
        subtags: list[MemorySubTag] = []
        for index, raw in enumerate(raw_subtags):
            if not isinstance(raw, dict):
                raise ValueError("each subtag must be an object")
            role = str(raw.get("role", "state")).strip().lower()
            if role not in SUBTAG_ROLES:
                role = "state"
            if role not in allowed:
                raise ValueError(f"role {role!r} is not allowed in collection {collection_id!r}")
            value = _required_text(raw, "value")
            position = int(raw.get("position", index + 1))
            embedding_text = str(raw.get("embedding_text") or f"{role}: {value}").strip()
            confidence = max(0.0, min(float(raw.get("confidence", 1.0)), 1.0))
            subtags.append(
                MemorySubTag(
                    subtag_id=str(raw.get("subtag_id") or f"{event_id}:{index}"),
                    role=role,
                    value=value,
                    position=position,
                    embedding_text=embedding_text,
                    confidence=confidence,
                    metadata=dict(raw.get("metadata", {})) if isinstance(raw.get("metadata"), dict) else {},
                )
            )
        if not subtags:
            if "state" not in allowed:
                raise ValueError(f"role 'state' is not allowed in collection {collection_id!r}")
            subtags.append(
                MemorySubTag(
                    subtag_id=f"{event_id}:0",
                    role="state",
                    value=main_label,
                    position=1,
                    embedding_text=f"state: {main_label}",
                )
            )
        roles = {subtag.role for subtag in subtags}
        missing_roles = sorted(required - roles)
        if missing_roles:
            raise ValueError(f"event missing required roles: {', '.join(missing_roles)}")
        subtags.sort(key=lambda subtag: (subtag.position, subtag.role, subtag.value))
        events.append(
            MemoryEvent(
                event_id=event_id,
                main_label=main_label,
                captured_at=captured_at,
                subtags=subtags,
                compressed_trace=compressed_trace,
                source={"type": source, "text": text, "metadata": dict(metadata)},
                metadata=dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {},
                collection_id=collection_id,
                updated_at=captured_at,
            )
        )
    return events


def _coerce_retrieval_plan(payload: dict[str, Any]) -> RetrievalPlan:
    key_question = _required_text(payload, "key_question")
    terms = payload.get("retrieval_terms")
    if not isinstance(terms, list):
        raise ValueError("retrieval_terms must be an array")
    retrieval_terms = _unique_non_empty(str(term) for term in terms)
    if not retrieval_terms:
        retrieval_terms = [key_question]

    roles = payload.get("target_roles", [])
    target_roles = [
        str(role).strip().lower()
        for role in roles
        if str(role).strip().lower() in SUBTAG_ROLES
    ] if isinstance(roles, list) else []
    target_slots = _coerce_target_slots(payload.get("target_slots", []))
    target_roles = _unique_non_empty([*target_roles, *(slot.role for slot in target_slots)])
    precision = max(0.0, min(float(payload.get("recall_precision", 0.5)), 1.0))
    time_hint = payload.get("time_hint")
    return RetrievalPlan(
        key_question=key_question,
        retrieval_terms=retrieval_terms,
        target_roles=target_roles,
        target_slots=target_slots,
        time_hint=str(time_hint) if time_hint else None,
        recall_precision=precision,
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
    )


def _coerce_target_slots(value: Any) -> list[RetrievalTargetSlot]:
    if not isinstance(value, list):
        return []
    slots: list[RetrievalTargetSlot] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role not in SUBTAG_ROLES:
            continue
        raw_position = item.get("position")
        position: int | None
        if raw_position in (None, ""):
            position = None
        else:
            try:
                position = int(raw_position)
            except (TypeError, ValueError):
                position = None
        slot = RetrievalTargetSlot(role=role, position=position)
        if slot not in slots:
            slots.append(slot)
    return slots


def _target_slots_from_plan(plan: RetrievalPlan) -> list[RetrievalTargetSlot]:
    slots = [
        slot
        for slot in plan.target_slots
        if slot.role in SUBTAG_ROLES
    ]
    if slots:
        return slots
    return [RetrievalTargetSlot(role=role) for role in _unique_non_empty(plan.target_roles) if role in SUBTAG_ROLES]


def _target_roles_from_slots(
    target_slots: list[RetrievalTargetSlot],
    target_roles: Iterable[str],
) -> set[str]:
    roles = {slot.role for slot in target_slots if slot.role in SUBTAG_ROLES}
    roles.update(role for role in target_roles if role in SUBTAG_ROLES)
    return roles


def _target_slot_to_dict(slot: RetrievalTargetSlot) -> dict[str, Any]:
    return {"role": slot.role, "position": slot.position}


def _subtag_matches_slot(subtag: MemorySubTag, slot: RetrievalTargetSlot) -> bool:
    if subtag.role != slot.role:
        return False
    return slot.position is None or subtag.position == slot.position


def _subtag_matches_exact_position(subtag: MemorySubTag, slot: RetrievalTargetSlot) -> bool:
    return slot.position is not None and subtag.role == slot.role and subtag.position == slot.position


def _rank_matched_subtags(
    event: MemoryEvent,
    *,
    keyword_matches: list[MemorySubTag],
    embedding_matches: list[MemorySubTag],
    target_slots: list[RetrievalTargetSlot],
    target_roles: set[str],
    limit: int = 8,
) -> list[MemorySubTag]:
    exact_slot_matches = [
        subtag
        for subtag in event.subtags
        if any(_subtag_matches_exact_position(subtag, slot) for slot in target_slots)
    ]
    matched_candidates = _unique_subtags([*keyword_matches, *embedding_matches])
    role_matches = [
        subtag
        for subtag in matched_candidates
        if subtag.role in target_roles and subtag not in exact_slot_matches
    ]
    semantic_matches = [
        subtag
        for subtag in matched_candidates
        if subtag not in exact_slot_matches and subtag not in role_matches
    ]
    context_positions = {subtag.position for subtag in exact_slot_matches}
    stage_context = [
        subtag
        for subtag in event.subtags
        if subtag.position in context_positions
        and subtag not in exact_slot_matches
        and subtag not in role_matches
        and subtag not in semantic_matches
    ]
    ranked = _unique_subtags([*exact_slot_matches, *role_matches, *semantic_matches, *stage_context])
    ranked.sort(key=lambda subtag: _subtag_sort_key(subtag, target_slots, target_roles))
    return ranked[:limit]


def _subtag_sort_key(
    subtag: MemorySubTag,
    target_slots: list[RetrievalTargetSlot],
    target_roles: set[str],
) -> tuple[int, int, str, str]:
    if any(_subtag_matches_exact_position(subtag, slot) for slot in target_slots):
        bucket = 0
    elif subtag.role in target_roles:
        bucket = 1
    elif any(_subtag_matches_slot(subtag, slot) for slot in target_slots):
        bucket = 2
    else:
        bucket = 3
    return (bucket, subtag.position, subtag.role, subtag.value)


def _unique_subtags(subtags: Iterable[MemorySubTag]) -> list[MemorySubTag]:
    result: list[MemorySubTag] = []
    seen: set[str] = set()
    for subtag in subtags:
        if subtag.subtag_id in seen:
            continue
        seen.add(subtag.subtag_id)
        result.append(subtag)
    return result


def _slot_match_bonus(
    matched_subtags: list[MemorySubTag],
    target_slots: list[RetrievalTargetSlot],
    target_roles: set[str],
) -> float:
    if target_slots and any(
        any(_subtag_matches_exact_position(subtag, slot) for slot in target_slots)
        for subtag in matched_subtags
    ):
        return 0.14
    if target_slots and any(
        any(_subtag_matches_slot(subtag, slot) for slot in target_slots)
        for subtag in matched_subtags
    ):
        return 0.08
    if target_roles and any(subtag.role in target_roles for subtag in matched_subtags):
        return 0.06
    return 0.0


def _matched_slots_payload(
    target_slots: list[RetrievalTargetSlot],
    matched_subtags: list[MemorySubTag],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for slot in target_slots:
        matches = [subtag for subtag in matched_subtags if _subtag_matches_slot(subtag, slot)]
        payload.append(
            {
                "role": slot.role,
                "position": slot.position,
                "matches": [_subtag_context(subtag) for subtag in matches],
            }
        )
    return payload


def _stages_payload(subtags: list[MemorySubTag]) -> list[dict[str, Any]]:
    stages: dict[int, list[dict[str, Any]]] = {}
    for subtag in sorted(subtags, key=lambda item: (item.position, item.role, item.value)):
        stages.setdefault(subtag.position, []).append(_subtag_context(subtag))
    return [
        {"position": position, "subtags": items}
        for position, items in sorted(stages.items())
    ]


def _subtag_context(subtag: MemorySubTag) -> dict[str, Any]:
    return {
        "subtag_id": subtag.subtag_id,
        "role": subtag.role,
        "value": subtag.value,
        "position": subtag.position,
        "confidence": subtag.confidence,
    }


def _keyword_match(
    event: MemoryEvent,
    normalized_terms: list[str],
    target_roles: set[str],
) -> tuple[float, float, list[MemorySubTag]]:
    if not normalized_terms:
        return 0.0, 0.0, []

    keyword_score = 0.0
    exactness = 0.0
    matched: dict[str, tuple[float, MemorySubTag]] = {}
    label = normalize_text(event.main_label)
    trace = normalize_text(event.compressed_trace)
    for term in normalized_terms:
        if not term:
            continue
        if term == label:
            keyword_score += 0.42
            exactness = max(exactness, 1.0)
        elif term in label or label in term:
            keyword_score += 0.25
            exactness = max(exactness, 0.68)
        elif term in trace:
            keyword_score += 0.12
            exactness = max(exactness, 0.42)

        for subtag in event.subtags:
            if target_roles and subtag.role not in target_roles:
                continue
            value = normalize_text(subtag.value)
            embedding_text = normalize_text(subtag.embedding_text)
            score = 0.0
            if term == value or term == embedding_text:
                score = 0.30
                exactness = max(exactness, 0.92)
            elif term in value or value in term or term in embedding_text:
                score = 0.18
                exactness = max(exactness, 0.62)
            if score and target_roles and subtag.role in target_roles:
                score += 0.05
            if score:
                previous = matched.get(subtag.subtag_id)
                if previous is None or score > previous[0]:
                    matched[subtag.subtag_id] = (score, subtag)
                keyword_score += score

    keyword_score = min(keyword_score, 1.0)
    matched_subtags = [item[1] for item in sorted(matched.values(), key=lambda item: item[0], reverse=True)]
    matched_subtags.sort(key=lambda subtag: subtag.position)
    return keyword_score, exactness, matched_subtags[:6]


def _embedding_match(
    event: MemoryEvent,
    query_vectors: list[list[float]],
) -> tuple[float, list[MemorySubTag]]:
    if not query_vectors:
        return 0.0, []
    best = 0.0
    subtag_scores: list[tuple[float, MemorySubTag]] = []
    for query_vector in query_vectors:
        if event.main_embedding:
            best = max(best, cosine_similarity(query_vector, event.main_embedding))
        for subtag in event.subtags:
            if not subtag.embedding:
                continue
            score = cosine_similarity(query_vector, subtag.embedding)
            if score > best:
                best = score
            if score > 0.20:
                subtag_scores.append((score, subtag))
    unique: dict[str, tuple[float, MemorySubTag]] = {}
    for score, subtag in subtag_scores:
        previous = unique.get(subtag.subtag_id)
        if previous is None or score > previous[0]:
            unique[subtag.subtag_id] = (score, subtag)
    matched = [item[1] for item in sorted(unique.values(), key=lambda item: item[0], reverse=True)]
    matched.sort(key=lambda subtag: subtag.position)
    return max(0.0, min(best, 1.0)), matched[:6]


def _evidence_context(evidence: EventEvidence) -> dict[str, Any]:
    return {
        "event_id": evidence.event_id,
        "main_label": evidence.main_label,
        "captured_at": evidence.captured_at,
        "compressed_trace": evidence.compressed_trace,
        "recall_difficulty": evidence.recall_difficulty,
        "score": evidence.score,
        "matched_slots": evidence.matched_slots,
        "stages": evidence.stages,
        "matched_subtags": [
            _subtag_context(subtag)
            for subtag in evidence.matched_subtags
        ],
    }


def _fallback_answer(query: str, evidence: list[EventEvidence]) -> str:
    if not evidence:
        return f"No event memory matched the query: {query}"
    top = evidence[0]
    subtags = ", ".join(f"{item.role}:{item.value}" for item in top.matched_subtags[:4])
    return f"{top.main_label}: {top.compressed_trace}" + (f" ({subtags})" if subtags else "")


def _event_to_dict(event: MemoryEvent) -> dict[str, Any]:
    payload = asdict(event)
    payload["recall_state"] = asdict(event.recall_state)
    payload["subtags"] = [asdict(subtag) for subtag in event.subtags]
    return payload


def _event_from_dict(item: dict[str, Any]) -> MemoryEvent:
    subtags = [
        MemorySubTag(**subtag)
        for subtag in item.get("subtags", [])
        if isinstance(subtag, dict)
    ]
    recall_state = item.get("recall_state", {})
    if not isinstance(recall_state, dict):
        recall_state = {}
    return MemoryEvent(
        event_id=str(item["event_id"]),
        main_label=str(item["main_label"]),
        captured_at=str(item["captured_at"]),
        subtags=subtags,
        compressed_trace=str(item.get("compressed_trace", "")),
        source=dict(item.get("source", {})) if isinstance(item.get("source"), dict) else {},
        recall_state=RecallState(**{key: value for key, value in recall_state.items() if key in RecallState.__dataclass_fields__}),
        metadata=dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {},
        main_embedding=item.get("main_embedding") if isinstance(item.get("main_embedding"), list) else None,
        embedding_model=item.get("embedding_model"),
        x=float(item.get("x", 0.0)),
        y=float(item.get("y", 0.0)),
        z=float(item.get("z", 1.0)),
        collection_id=_normalize_collection_id(item.get("collection_id")),
        status=str(item.get("status") or ACTIVE_STATUS),
        deleted_at=item.get("deleted_at"),
        updated_at=item.get("updated_at"),
        version=int(item.get("version", 1)),
    )


def _collection_to_dict(collection: MemoryCollection) -> dict[str, Any]:
    return {
        "collection_id": collection.collection_id,
        "name": collection.name,
        "schema": asdict(collection.schema),
        "metadata": dict(collection.metadata),
        "created_at": collection.created_at,
        "updated_at": collection.updated_at,
        "status": collection.status,
    }


def _collection_from_dict(item: dict[str, Any]) -> MemoryCollection:
    return MemoryCollection(
        collection_id=_normalize_collection_id(item.get("collection_id")),
        name=str(item.get("name") or item.get("collection_id") or DEFAULT_COLLECTION_ID),
        schema=_coerce_collection_schema(item.get("schema")),
        metadata=dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {},
        created_at=str(item.get("created_at") or _utc_now()),
        updated_at=str(item.get("updated_at") or _utc_now()),
        status=str(item.get("status") or ACTIVE_STATUS),
    )


def _coerce_collection_schema(value: CollectionSchema | dict[str, Any] | None) -> CollectionSchema:
    if isinstance(value, CollectionSchema):
        allowed = [role for role in value.allowed_roles if role in SUBTAG_ROLES]
        required = [role for role in value.required_roles if role in SUBTAG_ROLES]
        return CollectionSchema(
            allowed_roles=allowed or sorted(SUBTAG_ROLES),
            required_roles=required,
            embedding_model=value.embedding_model,
            metadata_fields=dict(value.metadata_fields),
        )
    if not isinstance(value, dict):
        return CollectionSchema()
    allowed_raw = value.get("allowed_roles", sorted(SUBTAG_ROLES))
    required_raw = value.get("required_roles", [])
    allowed = [
        str(role).strip().lower()
        for role in allowed_raw
        if str(role).strip().lower() in SUBTAG_ROLES
    ] if isinstance(allowed_raw, list) else sorted(SUBTAG_ROLES)
    required = [
        str(role).strip().lower()
        for role in required_raw
        if str(role).strip().lower() in SUBTAG_ROLES
    ] if isinstance(required_raw, list) else []
    return CollectionSchema(
        allowed_roles=allowed or sorted(SUBTAG_ROLES),
        required_roles=[role for role in required if role in (allowed or SUBTAG_ROLES)],
        embedding_model=str(value["embedding_model"]) if value.get("embedding_model") else None,
        metadata_fields=dict(value.get("metadata_fields", {})) if isinstance(value.get("metadata_fields"), dict) else {},
    )


def _coerce_event_filter(
    value: EventFilter | dict[str, Any] | str | None,
    *,
    collection: str | None = None,
) -> EventFilter:
    if isinstance(value, EventFilter):
        payload = dict(value.raw)
        result = EventFilter(
            collections=list(value.collections),
            roles=list(value.roles),
            event=dict(value.event),
            subtag=dict(value.subtag),
            metadata=dict(value.metadata),
            recall_max_difficulty=value.recall_max_difficulty,
            include_deleted=value.include_deleted,
            raw=payload,
        )
    else:
        if isinstance(value, str) and value.strip():
            payload = json.loads(value)
        elif isinstance(value, dict):
            payload = value
        else:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("event filter must be a JSON object")
        result = EventFilter(raw=dict(payload))
        raw_collections: list[Any] = []
        if "collection" in payload:
            raw_collections.append(payload["collection"])
        if isinstance(payload.get("collections"), list):
            raw_collections.extend(payload["collections"])
        result.collections = _unique_non_empty(_normalize_collection_id(item) for item in raw_collections)
        roles = payload.get("roles", [])
        result.roles = [
            str(role).strip().lower()
            for role in roles
            if str(role).strip().lower() in SUBTAG_ROLES
        ] if isinstance(roles, list) else []
        lifecycle = payload.get("lifecycle", {})
        if isinstance(lifecycle, dict):
            result.include_deleted = bool(lifecycle.get("include_deleted", False))
        recall = payload.get("recall", {})
        if isinstance(recall, dict) and recall.get("max_difficulty") is not None:
            result.recall_max_difficulty = float(recall["max_difficulty"])
        result.event.update(_coerce_condition_map(payload.get("event", {})))
        result.subtag.update(_coerce_condition_map(payload.get("subtag", {})))
        result.metadata.update(_coerce_condition_map(payload.get("metadata", {})))
        where = payload.get("where", {})
        if isinstance(where, dict):
            for key, condition in where.items():
                key_text = str(key)
                if key_text.startswith("event."):
                    result.event[key_text.split(".", 1)[1]] = _coerce_condition(condition)
                elif key_text.startswith("subtag."):
                    result.subtag[key_text.split(".", 1)[1]] = _coerce_condition(condition)
                elif key_text.startswith("metadata."):
                    result.metadata[key_text.split(".", 1)[1]] = _coerce_condition(condition)
                elif key_text == "recall.max_difficulty":
                    result.recall_max_difficulty = float(_condition_value(condition))
                elif key_text in {"captured_at", "main_label", "event_id", "status", "collection_id"}:
                    result.event[key_text] = _coerce_condition(condition)

    if collection is not None:
        result.collections = [_normalize_collection_id(collection)]
    return result


def _coerce_condition_map(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _coerce_condition(condition) for key, condition in value.items()}


def _coerce_condition(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and any(str(key) in FILTER_OPERATORS for key in value):
        return {str(key): item for key, item in value.items() if str(key) in FILTER_OPERATORS}
    return {"eq": value}


def _condition_value(value: Any) -> Any:
    condition = _coerce_condition(value)
    for key in ("eq", "lte", "gte"):
        if key in condition:
            return condition[key]
    return next(iter(condition.values()), None)


def _event_matches_filter(rag: EventRAG, event: MemoryEvent, event_filter: EventFilter) -> bool:
    if event.status == DELETED_STATUS and not event_filter.include_deleted:
        return False
    if event_filter.collections and event.collection_id not in event_filter.collections:
        return False
    if event_filter.roles and not any(subtag.role in event_filter.roles for subtag in event.subtags):
        return False
    event_values = {
        "event_id": event.event_id,
        "main_label": event.main_label,
        "captured_at": event.captured_at,
        "collection_id": event.collection_id,
        "status": event.status,
        "deleted_at": event.deleted_at,
        "updated_at": event.updated_at,
        "version": event.version,
    }
    for key, condition in event_filter.event.items():
        if not _match_condition(event_values.get(key), condition):
            return False
    for key, condition in event_filter.metadata.items():
        if not _match_condition(_metadata_lookup(event, key), condition):
            return False
    if event_filter.subtag:
        if not any(
            all(_match_condition(_subtag_value(subtag, key), condition) for key, condition in event_filter.subtag.items())
            for subtag in event.subtags
        ):
            return False
    if event_filter.recall_max_difficulty is not None:
        rag.refresh_recall_state(event)
        if event.recall_state.recall_difficulty > event_filter.recall_max_difficulty:
            return False
    return True


def _match_condition(actual: Any, condition: dict[str, Any]) -> bool:
    for operator, expected in condition.items():
        if operator == "exists":
            exists = actual is not None
            if bool(expected) != exists:
                return False
        elif operator == "eq" and actual != expected:
            return False
        elif operator == "ne" and actual == expected:
            return False
        elif operator == "in":
            if not isinstance(expected, list) or actual not in expected:
                return False
        elif operator == "nin":
            if isinstance(expected, list) and actual in expected:
                return False
        elif operator == "contains":
            if actual is None or str(expected).lower() not in str(actual).lower():
                return False
        elif operator == "gte":
            if actual is None or str(actual) < str(expected):
                return False
        elif operator == "lte":
            if actual is None or str(actual) > str(expected):
                return False
    return True


def _metadata_lookup(event: MemoryEvent, dotted_key: str) -> Any:
    value = _nested_lookup(event.metadata, dotted_key)
    if value is not None:
        return value
    source_metadata = event.source.get("metadata", {}) if isinstance(event.source, dict) else {}
    if isinstance(source_metadata, dict):
        return _nested_lookup(source_metadata, dotted_key)
    return None


def _nested_lookup(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _subtag_value(subtag: MemorySubTag, key: str) -> Any:
    return {
        "subtag_id": subtag.subtag_id,
        "role": subtag.role,
        "value": subtag.value,
        "position": subtag.position,
        "embedding_text": subtag.embedding_text,
        "confidence": subtag.confidence,
    }.get(key)


def _normalize_collection_id(value: Any) -> str:
    text = str(value or DEFAULT_COLLECTION_ID).strip()
    return text or DEFAULT_COLLECTION_ID


def _tombstone_belongs_to_index(index_name: str, item_id: str) -> bool:
    if not index_name.startswith("collection:"):
        return True
    return True


def _embedding_xy(vector: list[float] | None, fallback_text: str) -> tuple[float, float]:
    if vector and len(vector) >= 2:
        norm = math.sqrt(sum(float(value) * float(value) for value in vector)) or 1.0
        return (
            max(-1.0, min(float(vector[0]) / norm, 1.0)),
            max(-1.0, min(float(vector[1]) / norm, 1.0)),
        )
    digest = sum((index + 1) * ord(ch) for index, ch in enumerate(fallback_text))
    angle = (digest % 6283) / 1000.0
    return math.cos(angle) * 0.72, math.sin(angle) * 0.72


def _embedding_checksum(vector: list[float]) -> str:
    value = sum((index + 1) * abs(float(item)) for index, item in enumerate(vector))
    return f"{value:.12f}"


def _required_text(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return re.sub(r"\s+", " ", value.strip())


def _unique_non_empty(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _coerce_datetime(value: str | datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc).replace(microsecond=0)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(microsecond=0)
    return _parse_iso(value)


def _parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def _to_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _utc_now() -> str:
    return _to_iso(datetime.now(timezone.utc))
