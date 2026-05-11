from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .consolidation import MemoryConsolidationResult, build_consolidation_candidates
from .embeddings import EmbeddingProvider, make_embedding_provider
from .memory_layout import LayoutResult, apply_memory_layout
from .memory_space import MemoryDynamicsConfig, MemorySpace
from .models import MemoryFragment, MemoryRelation, RetrievalHit
from .navigation import (
    NavigationHit,
    SemanticNavigationConfig,
    SemanticNavigationIndex,
    exact_navigation_hits,
)
from .parser import normalize_text
from .policy import RetrievalProfile, make_retrieval_profile


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u3002\uff01\uff1f\uff1b;])\s+|\n+")


@dataclass(slots=True)
class SourceDocument:
    document_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SourceChunk:
    chunk_id: str
    document_id: str
    text: str
    ordinal: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    embedding_model: str | None = None


@dataclass(slots=True)
class MemoryEvidence:
    fragment_id: str
    text: str
    kind: str
    layer: int
    depth: float
    score: float
    activation: float
    strength: float
    accessibility: float
    via_relation: str | None
    document_id: str | None = None
    chunk_id: str | None = None
    title: str | None = None
    chunk_text: str | None = None
    memory_score: float | None = None
    embedding_score: float | None = None
    raw_keyword_score: float | None = None
    raw_embedding_score: float | None = None
    relation_bonus: float = 0.0
    final_score: float | None = None
    spatial_score: float | None = None
    layout_score: float | None = None


@dataclass(slots=True)
class RAGAnswer:
    query: str
    answer: str
    evidence: list[MemoryEvidence]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeedbackResult:
    query: str | None
    positive: list[str]
    negative: list[str]
    ignored: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class LayeredMemoryRAG:
    """Product-facing layered memory RAG module.

    The module keeps HuMem's rule-based layered memory mechanics, then adds the
    operational pieces an application needs: document ingestion, source
    tracking, evidence objects, extractive answer composition, decay/reinforce
    hooks, and JSON persistence.
    """

    def __init__(
        self,
        *,
        total_layers: int = 8,
        top_layer_quota: int = 50,
        bottom_layer_quota: int = 10,
        sealed_bottom_layers: int = 2,
        embedding_provider: str | EmbeddingProvider | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: str | None = None,
        embedding_dimensions: int | None = None,
        retrieval_profile: str | RetrievalProfile | dict[str, Any] | None = None,
        memory_weight: float | None = None,
        embedding_weight: float | None = None,
        semantic_index: str = "auto",
        semantic_index_min_items: int = 256,
        semantic_index_m: int = 8,
        semantic_index_ef_construction: int = 64,
        semantic_index_ef_search: int = 32,
        semantic_index_seed: int = 13,
        dynamics: MemoryDynamicsConfig | dict[str, Any] | None = None,
    ) -> None:
        self.space = MemorySpace(
            total_layers=total_layers,
            top_layer_quota=top_layer_quota,
            bottom_layer_quota=bottom_layer_quota,
            sealed_bottom_layers=sealed_bottom_layers,
            dynamics=dynamics,
        )
        self.documents: dict[str, SourceDocument] = {}
        self.chunks: dict[str, SourceChunk] = {}
        self.embedding_provider = make_embedding_provider(
            embedding_provider,
            model=embedding_model,
            api_key=embedding_api_key,
            dimensions=embedding_dimensions,
        )
        self.retrieval_profile = make_retrieval_profile(
            retrieval_profile,
            memory_weight=memory_weight,
            embedding_weight=embedding_weight,
        )
        self.memory_weight = self.retrieval_profile.memory_weight
        self.embedding_weight = self.retrieval_profile.embedding_weight
        self.semantic_navigation_config = SemanticNavigationConfig(
            mode=semantic_index,
            min_items=semantic_index_min_items,
            m=semantic_index_m,
            ef_construction=semantic_index_ef_construction,
            ef_search=semantic_index_ef_search,
            seed=semantic_index_seed,
        )
        self._semantic_navigation_index: SemanticNavigationIndex | None = None
        self._semantic_navigation_signature: tuple[tuple[str, str | None, int], ...] | None = None
        self._semantic_navigation_build_count = 0
        self._last_semantic_navigation_strategy = "none"
        self._last_semantic_navigation_visited = 0

    @property
    def total_layers(self) -> int:
        return self.space.total_layers

    def add_document(
        self,
        text: str,
        *,
        document_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 700,
        chunk_overlap: int = 80,
        cool_down_cycles: int = 1,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        if chunk_size < 80:
            raise ValueError("chunk_size must be at least 80 characters")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

        doc_id = document_id or str(uuid4())
        doc_title = title or doc_id
        document = SourceDocument(
            document_id=doc_id,
            title=doc_title,
            text=text,
            metadata=dict(metadata or {}),
        )
        self.documents[doc_id] = document

        chunk_texts = _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings = self._embed_chunks(chunk_texts)

        for ordinal, chunk_text in enumerate(chunk_texts):
            chunk_id = f"{doc_id}:{ordinal}"
            chunk = SourceChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                text=chunk_text,
                ordinal=ordinal,
                metadata={"title": doc_title},
                embedding=embeddings[ordinal] if embeddings else None,
                embedding_model=self.embedding_provider.model if embeddings and self.embedding_provider else None,
            )
            self.chunks[chunk_id] = chunk
            fragment_ids = self.space.ingest_sentence(chunk_text)
            self._attach_source(fragment_ids, document=document, chunk=chunk)

        for _ in range(cool_down_cycles):
            self.space.forget()

        self._invalidate_semantic_navigation_index()
        return doc_id

    def add_memory(
        self,
        text: str,
        *,
        source: str = "memory",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        fragment_ids = self.space.ingest_sentence(text)
        for fragment_id in fragment_ids:
            fragment = self.space.fragments[fragment_id]
            memories = _metadata_list(fragment.metadata, "memories")
            memories.append({"source": source, "text": text, "metadata": dict(metadata or {})})
        return fragment_ids

    def retrieve(self, query: str, *, limit: int = 8) -> list[MemoryEvidence]:
        hits = self.space.retrieve(
            query,
            limit=max(limit * 4, limit),
            mutate=self.retrieval_profile.reinforce_on_read,
            reinforcement_amount=self.retrieval_profile.read_reinforcement,
        )
        memory_evidence = [self._hit_to_evidence(hit) for hit in hits]
        semantic_evidence = self._semantic_evidence(query, limit=max(limit * 4, limit))
        evidence = _merge_hybrid_evidence(
            memory_evidence,
            semantic_evidence,
            memory_weight=self.memory_weight,
            embedding_weight=self.embedding_weight,
        )
        return evidence[:limit]

    def answer(self, query: str, *, limit: int = 6) -> RAGAnswer:
        evidence = self.retrieve(query, limit=limit)
        answer_text = _compose_answer(query, evidence)
        diagnostics = {
            "fragment_count": len(self.space.fragments),
            "relation_count": len(self.space.relations),
            "layer_histogram": self.layer_histogram(),
            "evidence_count": len(evidence),
            "embedding_enabled": self.embedding_provider is not None,
            "embedded_chunks": sum(1 for chunk in self.chunks.values() if chunk.embedding),
            "memory_weight": self.memory_weight,
            "embedding_weight": self.embedding_weight,
            "retrieval_profile": self.retrieval_profile.name,
            "reinforce_on_read": self.retrieval_profile.reinforce_on_read,
            "accessibility_weighted": True,
            "forgetting_model": self.space.dynamics.forgetting_model,
            "semantic_navigation": self.semantic_navigation_stats(),
        }
        return RAGAnswer(
            query=query,
            answer=answer_text,
            evidence=evidence,
            diagnostics=diagnostics,
        )

    def reinforce(self, fragment_id: str, *, amount: float = 0.6, reason: str = "user_feedback") -> None:
        self.space.reinforce(fragment_id, amount=amount, reason=reason)

    def apply_feedback(
        self,
        *,
        query: str | None = None,
        positive_fragment_ids: Iterable[str] | None = None,
        negative_fragment_ids: Iterable[str] | None = None,
        reason: str = "user_feedback",
        positive_amount: float | None = None,
        negative_amount: float | None = None,
    ) -> FeedbackResult:
        positive_amount = self.retrieval_profile.feedback_positive_amount if positive_amount is None else positive_amount
        negative_amount = self.retrieval_profile.feedback_negative_amount if negative_amount is None else negative_amount
        if positive_amount < 0 or negative_amount < 0:
            raise ValueError("feedback amounts must be non-negative")

        positive_ids = _unique_strings(positive_fragment_ids or [])
        positive_id_set = set(positive_ids)
        negative_ids = [
            fragment_id
            for fragment_id in _unique_strings(negative_fragment_ids or [])
            if fragment_id not in positive_id_set
        ]

        applied_positive: list[str] = []
        applied_negative: list[str] = []
        ignored: list[str] = []

        for fragment_id in positive_ids:
            fragment = self.space.fragments.get(fragment_id)
            if fragment is None:
                ignored.append(fragment_id)
                continue
            self.space.reinforce(fragment_id, amount=positive_amount, reason=reason)
            _record_feedback_metadata(fragment.metadata, query=query, reason=reason, sentiment="positive")
            applied_positive.append(fragment_id)

        for fragment_id in negative_ids:
            fragment = self.space.fragments.get(fragment_id)
            if fragment is None:
                ignored.append(fragment_id)
                continue
            self.space.suppress(
                fragment_id,
                amount=negative_amount,
                reason=reason,
                demote_layers=self.retrieval_profile.feedback_demote_layers,
            )
            _record_feedback_metadata(fragment.metadata, query=query, reason=reason, sentiment="negative")
            applied_negative.append(fragment_id)

        return FeedbackResult(
            query=query,
            positive=applied_positive,
            negative=applied_negative,
            ignored=ignored,
            diagnostics={
                "retrieval_profile": self.retrieval_profile.name,
                "positive_amount": positive_amount,
                "negative_amount": negative_amount,
                "feedback_demote_layers": self.retrieval_profile.feedback_demote_layers,
                "fragment_count": len(self.space.fragments),
                "layer_histogram": self.layer_histogram(),
            },
        )

    def decay(self, *, step: float = 0.14, cycles: int = 1) -> None:
        for _ in range(cycles):
            self.space.forget(step=step)

    def embed_missing_chunks(self, *, batch_size: int = 32) -> int:
        if self.embedding_provider is None:
            raise ValueError("embedding_provider is required to embed chunks")

        missing = [
            chunk
            for chunk in self.chunks.values()
            if not chunk.embedding or chunk.embedding_model != self.embedding_provider.model
        ]
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset : offset + batch_size]
            embeddings = self.embedding_provider.embed_texts([chunk.text for chunk in batch])
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding
                chunk.embedding_model = self.embedding_provider.model
        if missing:
            self._invalidate_semantic_navigation_index()
        return len(missing)

    def consolidate(
        self,
        *,
        scope: str = "document",
        max_anchors: int = 8,
        keywords_per_anchor: int = 5,
        min_support: int = 3,
        anchor_layer: int = 0,
        relation_weight: float = 0.88,
    ) -> MemoryConsolidationResult:
        candidates, skipped_groups = build_consolidation_candidates(
            self.space.fragments.values(),
            total_layers=self.total_layers,
            scope=scope,
            max_anchors=max_anchors,
            keywords_per_anchor=keywords_per_anchor,
            min_support=min_support,
        )

        result = MemoryConsolidationResult(
            candidates=candidates,
            skipped_groups=skipped_groups,
            diagnostics={
                "scope": scope,
                "max_anchors": max_anchors,
                "keywords_per_anchor": keywords_per_anchor,
                "min_support": min_support,
                "fragment_count": len(self.space.fragments),
            },
        )

        for candidate in candidates:
            anchor_key = (normalize_text(candidate.anchor_text), "clause")
            existing_anchor_id = self._existing_consolidation_anchor(scope, candidate.group_key)
            existed = existing_anchor_id is not None or anchor_key in self.space.fragment_index
            if existing_anchor_id is not None:
                anchor_id = existing_anchor_id
            else:
                fragment_ids = self.space.ingest_sentence(candidate.anchor_text)
                anchor_id = self.space.fragment_index.get(anchor_key, fragment_ids[0] if fragment_ids else None)
            if anchor_id is None:
                continue

            anchor = self.space.fragments[anchor_id]
            anchor.layer = max(0, min(anchor_layer, self.total_layers - 1))
            anchor.metadata["consolidation"] = {
                "anchor": True,
                "scope": scope,
                "group_key": candidate.group_key,
                "title": candidate.title,
                "theme_terms": candidate.theme_terms,
                "support_fragment_ids": candidate.support_fragment_ids,
                "score": candidate.score,
            }
            self.space.refresh_fragment_state(anchor)

            if existed:
                self.space.reinforce(anchor_id, amount=0.18, reason="consolidation_refresh")
                result.reinforced_anchor_ids.append(anchor_id)
            else:
                self.space.reinforce(anchor_id, amount=0.24, reason="consolidation_anchor")
                result.created_anchor_ids.append(anchor_id)

            for support_id in candidate.support_fragment_ids:
                if support_id == anchor_id or support_id not in self.space.fragments:
                    continue
                self.space._upsert_relation(
                    source_id=anchor_id,
                    target_id=support_id,
                    relation_type="consolidates",
                    weight=relation_weight,
                    metadata={"scope": scope, "group_key": candidate.group_key},
                )
                result.support_relations += 1

        self.space._rebuild_cross_layer_flags()
        result.diagnostics.update(
            {
                "created_anchors": len(result.created_anchor_ids),
                "reinforced_anchors": len(result.reinforced_anchor_ids),
                "support_relations": result.support_relations,
                "layer_histogram": self.layer_histogram(),
            }
        )
        return result

    def _existing_consolidation_anchor(self, scope: str, group_key: str) -> str | None:
        for fragment in self.space.fragments.values():
            consolidation = fragment.metadata.get("consolidation")
            if not isinstance(consolidation, dict) or not consolidation.get("anchor"):
                continue
            if consolidation.get("scope") == scope and consolidation.get("group_key") == group_key:
                return fragment.fragment_id
        return None

    def layout_memory_space(
        self,
        *,
        use_embeddings: bool = True,
        embed_fragments: bool = False,
        embedding_scope: str | None = None,
        iterations: int = 120,
        semantic_neighbors: int = 4,
    ) -> LayoutResult:
        if embedding_scope is None:
            embedding_scope = "fragment" if embed_fragments else ("chunk" if use_embeddings else "none")
        if embedding_scope not in {"chunk", "fragment", "none"}:
            raise ValueError("embedding_scope must be one of: chunk, fragment, none")

        fragment_embeddings: dict[str, list[float]] = {}
        if use_embeddings and embedding_scope != "none":
            if embedding_scope == "fragment":
                fragment_embeddings = self._fragment_embeddings(embed_missing=embed_fragments)
            else:
                fragment_embeddings = self._existing_fragment_embeddings()

        return apply_memory_layout(
            self.space,
            fragment_embeddings=fragment_embeddings,
            embedding_scope=embedding_scope,
            iterations=iterations,
            semantic_neighbors=semantic_neighbors,
        )

    def layer_histogram(self) -> list[int]:
        histogram = [0] * self.space.total_layers
        for fragment in self.space.fragments.values():
            if 0 <= fragment.layer < len(histogram):
                histogram[fragment.layer] += 1
        return histogram

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
        self.semantic_navigation_config = SemanticNavigationConfig(
            mode=mode,
            min_items=config.min_items if min_items is None else min_items,
            m=config.m if m is None else m,
            ef_construction=config.ef_construction if ef_construction is None else ef_construction,
            ef_search=config.ef_search if ef_search is None else ef_search,
            seed=config.seed if seed is None else seed,
        )
        self._invalidate_semantic_navigation_index()

    def semantic_navigation_stats(self) -> dict[str, Any]:
        embedded_chunks = sum(1 for chunk in self.chunks.values() if chunk.embedding)
        index_items = self._semantic_navigation_index.item_count if self._semantic_navigation_index else 0
        return {
            "mode": self.semantic_navigation_config.mode,
            "config": self.semantic_navigation_config.to_dict(),
            "available": self.embedding_provider is not None and embedded_chunks > 0,
            "embedded_chunks": embedded_chunks,
            "index_built": self._semantic_navigation_index is not None,
            "index_items": index_items,
            "build_count": self._semantic_navigation_build_count,
            "last_strategy": self._last_semantic_navigation_strategy,
            "last_visited": self._last_semantic_navigation_visited,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": 3,
            "rag_config": self.rag_config(),
            "memory_space": self.space.snapshot(),
            "documents": [asdict(document) for document in self.documents.values()],
            "chunks": [asdict(chunk) for chunk in self.chunks.values()],
        }

    def rag_config(self) -> dict[str, Any]:
        return {
            "retrieval_profile": self.retrieval_profile.to_dict(),
            "memory_weight": self.memory_weight,
            "embedding_weight": self.embedding_weight,
            "semantic_navigation": self.semantic_navigation_config.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LayeredMemoryRAG":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        space_payload = payload["memory_space"]
        config = space_payload["config"]
        rag_config = payload.get("rag_config", {})
        semantic_navigation = rag_config.get("semantic_navigation", {})
        if not isinstance(semantic_navigation, dict):
            semantic_navigation = {}
        instance = cls(
            **config,
            retrieval_profile=rag_config.get("retrieval_profile"),
            memory_weight=rag_config.get("memory_weight"),
            embedding_weight=rag_config.get("embedding_weight"),
            semantic_index=semantic_navigation.get("mode", "auto"),
            semantic_index_min_items=semantic_navigation.get("min_items", 256),
            semantic_index_m=semantic_navigation.get("m", 8),
            semantic_index_ef_construction=semantic_navigation.get("ef_construction", 64),
            semantic_index_ef_search=semantic_navigation.get("ef_search", 32),
            semantic_index_seed=semantic_navigation.get("seed", 13),
        )
        instance.documents = {
            item["document_id"]: SourceDocument(**item)
            for item in payload.get("documents", [])
        }
        instance.chunks = {
            item["chunk_id"]: SourceChunk(**item)
            for item in payload.get("chunks", [])
        }
        instance._restore_space(space_payload)
        instance._invalidate_semantic_navigation_index()
        return instance

    @classmethod
    def load_with_embeddings(
        cls,
        path: str | Path,
        *,
        embedding_provider: str | EmbeddingProvider,
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: str | None = None,
        embedding_dimensions: int | None = None,
        retrieval_profile: str | RetrievalProfile | dict[str, Any] | None = None,
        memory_weight: float | None = None,
        embedding_weight: float | None = None,
        semantic_index: str | None = None,
    ) -> "LayeredMemoryRAG":
        instance = cls.load(path)
        instance.embedding_provider = make_embedding_provider(
            embedding_provider,
            model=embedding_model,
            api_key=embedding_api_key,
            dimensions=embedding_dimensions,
        )
        instance._invalidate_semantic_navigation_index()
        if retrieval_profile is not None or memory_weight is not None or embedding_weight is not None:
            instance.retrieval_profile = make_retrieval_profile(
                retrieval_profile or instance.retrieval_profile,
                memory_weight=memory_weight,
                embedding_weight=embedding_weight,
            )
            instance.memory_weight = instance.retrieval_profile.memory_weight
            instance.embedding_weight = instance.retrieval_profile.embedding_weight
        if semantic_index is not None:
            instance.set_semantic_index(semantic_index)
        return instance

    def _attach_source(
        self,
        fragment_ids: Iterable[str],
        *,
        document: SourceDocument,
        chunk: SourceChunk,
    ) -> None:
        for fragment_id in fragment_ids:
            fragment = self.space.fragments[fragment_id]
            sources = _metadata_list(fragment.metadata, "sources")
            source = {
                "document_id": document.document_id,
                "chunk_id": chunk.chunk_id,
                "title": document.title,
            }
            if source not in sources:
                sources.append(source)

    def _hit_to_evidence(self, hit: RetrievalHit) -> MemoryEvidence:
        fragment = self.space.fragments[hit.fragment_id]
        source = _best_source(fragment.metadata.get("sources"))
        document_id = source.get("document_id") if source else None
        chunk_id = source.get("chunk_id") if source else None
        title = source.get("title") if source else None
        chunk_text = self.chunks[chunk_id].text if chunk_id in self.chunks else None

        return MemoryEvidence(
            fragment_id=hit.fragment_id,
            text=hit.text,
            kind=hit.kind,
            layer=hit.layer,
            depth=hit.depth,
            score=hit.score,
            activation=hit.activation,
            strength=hit.strength,
            accessibility=hit.accessibility,
            via_relation=hit.via_relation,
            document_id=document_id,
            chunk_id=chunk_id,
            title=title,
            chunk_text=chunk_text,
            memory_score=hit.score,
            raw_keyword_score=hit.raw_keyword_score,
            relation_bonus=hit.relation_bonus,
            final_score=hit.score,
            spatial_score=hit.accessibility,
            layout_score=hit.accessibility,
        )

    def _embed_chunks(self, chunk_texts: list[str]) -> list[list[float]]:
        if self.embedding_provider is None:
            return []
        return self.embedding_provider.embed_texts(chunk_texts)

    def _fragment_embeddings(self, *, embed_missing: bool) -> dict[str, list[float]]:
        embeddings: dict[str, list[float]] = {}
        missing: list[MemoryFragment] = []
        provider_model = self.embedding_provider.model if self.embedding_provider else None

        for fragment in self.space.fragments.values():
            cached = fragment.metadata.get("embedding")
            cached_model = fragment.metadata.get("embedding_model")
            if isinstance(cached, list) and cached and (provider_model is None or cached_model == provider_model):
                embeddings[fragment.fragment_id] = [float(value) for value in cached]
            else:
                missing.append(fragment)

        if missing and embed_missing:
            if self.embedding_provider is None:
                raise ValueError("embedding_provider is required to embed fragments")
            embedded = self.embedding_provider.embed_texts([fragment.text for fragment in missing])
            for fragment, vector in zip(missing, embedded):
                fragment.metadata["embedding"] = vector
                fragment.metadata["embedding_model"] = self.embedding_provider.model
                embeddings[fragment.fragment_id] = vector

        return embeddings

    def _existing_fragment_embeddings(self) -> dict[str, list[float]]:
        embeddings: dict[str, list[float]] = {}
        for fragment in self.space.fragments.values():
            source = _best_source(fragment.metadata.get("sources"))
            chunk_id = source.get("chunk_id") if source else None
            chunk = self.chunks.get(chunk_id or "")
            if chunk and chunk.embedding:
                embeddings[fragment.fragment_id] = chunk.embedding
        return embeddings

    def _semantic_evidence(self, query: str, *, limit: int) -> list[MemoryEvidence]:
        if self.embedding_provider is None:
            self._last_semantic_navigation_strategy = "none"
            self._last_semantic_navigation_visited = 0
            return []
        embedded_chunks = [chunk for chunk in self.chunks.values() if chunk.embedding]
        if not embedded_chunks:
            self._last_semantic_navigation_strategy = "none"
            self._last_semantic_navigation_visited = 0
            return []

        query_embedding = self.embedding_provider.embed_query(query)
        hits = self._semantic_chunk_hits(query_embedding, embedded_chunks, limit=limit)

        evidence: list[MemoryEvidence] = []
        for hit in hits:
            chunk = self.chunks.get(hit.chunk_id)
            if chunk is None:
                continue
            fragment = self._representative_fragment_for_chunk(chunk.chunk_id)
            if fragment is None:
                continue
            document = self.documents.get(chunk.document_id)
            evidence.append(
                MemoryEvidence(
                    fragment_id=fragment.fragment_id,
                    text=fragment.text,
                    kind=fragment.kind,
                    layer=fragment.layer,
                    depth=fragment.depth,
                    score=hit.score,
                    activation=fragment.activation,
                    strength=fragment.strength,
                    accessibility=self.space.accessibility_weight(fragment.depth),
                    via_relation="embedding",
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    title=document.title if document else chunk.metadata.get("title"),
                    chunk_text=chunk.text,
                    embedding_score=hit.score,
                    raw_embedding_score=hit.score,
                    final_score=hit.score,
                    spatial_score=self.space.accessibility_weight(fragment.depth),
                    layout_score=self.space.accessibility_weight(fragment.depth),
                )
            )
        return evidence

    def _semantic_chunk_hits(
        self,
        query_embedding: list[float],
        embedded_chunks: list[SourceChunk],
        *,
        limit: int,
    ) -> list[NavigationHit]:
        items = [
            (chunk.chunk_id, chunk.embedding or [])
            for chunk in embedded_chunks
            if chunk.embedding
        ]
        if not items:
            self._last_semantic_navigation_strategy = "none"
            self._last_semantic_navigation_visited = 0
            return []

        config = self.semantic_navigation_config
        if config.mode == "exact" or (config.mode == "auto" and len(items) < config.min_items):
            self._last_semantic_navigation_strategy = "exact"
            self._last_semantic_navigation_visited = len(items)
            return exact_navigation_hits(items, query_embedding, limit=limit)

        signature = self._semantic_navigation_items_signature(embedded_chunks)
        if self._semantic_navigation_index is None or self._semantic_navigation_signature != signature:
            index = SemanticNavigationIndex(config)
            index.build(items)
            self._semantic_navigation_index = index
            self._semantic_navigation_signature = signature
            self._semantic_navigation_build_count += 1

        hits = self._semantic_navigation_index.search(query_embedding, limit=limit)
        self._last_semantic_navigation_visited = self._semantic_navigation_index.last_visited
        if hits:
            self._last_semantic_navigation_strategy = "ann"
            return hits

        self._last_semantic_navigation_strategy = "ann_fallback_exact"
        self._last_semantic_navigation_visited = len(items)
        return exact_navigation_hits(items, query_embedding, limit=limit)

    def _semantic_navigation_items_signature(
        self,
        embedded_chunks: Iterable[SourceChunk],
    ) -> tuple[tuple[str, str | None, int], ...]:
        return tuple(
            (chunk.chunk_id, chunk.embedding_model, len(chunk.embedding or []))
            for chunk in embedded_chunks
            if chunk.embedding
        )

    def _invalidate_semantic_navigation_index(self) -> None:
        self._semantic_navigation_index = None
        self._semantic_navigation_signature = None
        self._last_semantic_navigation_strategy = "none"
        self._last_semantic_navigation_visited = 0

    def _representative_fragment_for_chunk(self, chunk_id: str) -> MemoryFragment | None:
        candidates: list[MemoryFragment] = []
        for fragment in self.space.fragments.values():
            for source in fragment.metadata.get("sources", []):
                if isinstance(source, dict) and source.get("chunk_id") == chunk_id:
                    candidates.append(fragment)
                    break
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda item: (item.kind == "clause", -item.layer, item.strength, item.activation),
            reverse=True,
        )[0]

    def _restore_space(self, payload: dict[str, Any]) -> None:
        self.space.fragments.clear()
        self.space.fragment_index.clear()
        self.space.relations.clear()
        self.space.out_edges = defaultdict(set)
        self.space.in_edges = defaultdict(set)

        for item in payload.get("fragments", []):
            if "depth" not in item:
                item["depth"] = float(item.get("layer", 0))
            fragment = MemoryFragment(**item)
            self.space._refresh_fragment_depth(fragment)
            self.space.fragments[fragment.fragment_id] = fragment
            self.space.fragment_index[(fragment.normalized_text, fragment.kind)] = fragment.fragment_id

        for item in payload.get("relations", []):
            relation = MemoryRelation(**item)
            self.space.relations[relation.relation_id] = relation
            self.space.out_edges[relation.source_id].add(relation.relation_id)
            self.space.in_edges[relation.target_id].add(relation.relation_id)

        self.space._rebuild_cross_layer_flags()


def _metadata_list(metadata: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = metadata.get(key)
    if not isinstance(value, list):
        value = []
        metadata[key] = value
    return value


def _unique_strings(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _record_feedback_metadata(
    metadata: dict[str, Any],
    *,
    query: str | None,
    reason: str,
    sentiment: str,
) -> None:
    feedback = metadata.get("feedback")
    if not isinstance(feedback, dict):
        feedback = {}
        metadata["feedback"] = feedback
    feedback[sentiment] = int(feedback.get(sentiment, 0)) + 1
    feedback["last_reason"] = reason
    if query:
        feedback["last_query"] = query

    events = feedback.get("events")
    if not isinstance(events, list):
        events = []
        feedback["events"] = events
    events.append({"sentiment": sentiment, "reason": reason, "query": query})


def _best_source(value: Any) -> dict[str, str] | None:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return first
    return None


def _chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if not sentences:
        sentences = [text.strip()]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= chunk_size:
            current = sentence
        else:
            chunks.extend(_slice_long_text(sentence, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
            current = ""

    if current:
        chunks.append(current)

    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = [chunks[0]]
    for index in range(1, len(chunks)):
        prefix = chunks[index - 1][-chunk_overlap:].strip()
        overlapped.append(f"{prefix} {chunks[index]}".strip())
    return overlapped


def _slice_long_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        pieces.append(text[start : start + chunk_size].strip())
        start += step
    return [piece for piece in pieces if piece]


def _dedupe_evidence(evidence: list[MemoryEvidence]) -> list[MemoryEvidence]:
    seen: set[tuple[str | None, str]] = set()
    deduped: list[MemoryEvidence] = []
    for item in evidence:
        key = (item.chunk_id, item.text.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _merge_hybrid_evidence(
    memory_evidence: list[MemoryEvidence],
    semantic_evidence: list[MemoryEvidence],
    *,
    memory_weight: float,
    embedding_weight: float,
) -> list[MemoryEvidence]:
    merged: dict[tuple[str | None, str], MemoryEvidence] = {}
    max_memory = max((item.score for item in memory_evidence), default=1.0) or 1.0

    for item in memory_evidence:
        key = (item.chunk_id, item.text.lower())
        normalized_memory = item.score / max_memory
        item.memory_score = item.score
        item.score = memory_weight * normalized_memory
        item.final_score = item.score
        merged[key] = item

    for item in semantic_evidence:
        key = (item.chunk_id, item.text.lower())
        normalized_embedding = max(0.0, min(1.0, (item.embedding_score or item.score)))
        accessible_embedding = normalized_embedding * item.accessibility
        if key in merged:
            existing = merged[key]
            existing.embedding_score = item.embedding_score
            existing.raw_embedding_score = item.raw_embedding_score
            existing.spatial_score = item.accessibility
            existing.layout_score = item.layout_score
            existing.score = existing.score + embedding_weight * accessible_embedding
            existing.final_score = existing.score
            if existing.chunk_text is None:
                existing.chunk_text = item.chunk_text
            if existing.via_relation is None and item.via_relation:
                existing.via_relation = item.via_relation
        else:
            item.score = embedding_weight * accessible_embedding
            item.final_score = item.score
            merged[key] = item

    return sorted(_dedupe_evidence(list(merged.values())), key=lambda item: item.score, reverse=True)


def _compose_answer(query: str, evidence: list[MemoryEvidence]) -> str:
    if not evidence:
        return f"No memory matched the query: {query}"

    clause_evidence = [item for item in evidence if item.kind == "clause" and item.chunk_text]
    if clause_evidence:
        selected = clause_evidence[:3]
        return " ".join(item.chunk_text or item.text for item in selected)

    chunk_texts: list[str] = []
    for item in evidence:
        text = item.chunk_text or item.text
        if text not in chunk_texts:
            chunk_texts.append(text)
        if len(chunk_texts) >= 3:
            break
    return " ".join(chunk_texts)
