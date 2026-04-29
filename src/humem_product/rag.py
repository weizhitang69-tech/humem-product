from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .embeddings import EmbeddingProvider, cosine_similarity, make_embedding_provider
from .memory_layout import LayoutResult, apply_memory_layout
from .memory_space import MemorySpace
from .models import MemoryFragment, MemoryRelation, RetrievalHit


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
    spatial_score: float | None = None
    layout_score: float | None = None


@dataclass(slots=True)
class RAGAnswer:
    query: str
    answer: str
    evidence: list[MemoryEvidence]
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
        memory_weight: float = 0.65,
        embedding_weight: float = 0.35,
    ) -> None:
        self.space = MemorySpace(
            total_layers=total_layers,
            top_layer_quota=top_layer_quota,
            bottom_layer_quota=bottom_layer_quota,
            sealed_bottom_layers=sealed_bottom_layers,
        )
        self.documents: dict[str, SourceDocument] = {}
        self.chunks: dict[str, SourceChunk] = {}
        self.embedding_provider = make_embedding_provider(
            embedding_provider,
            model=embedding_model,
            api_key=embedding_api_key,
            dimensions=embedding_dimensions,
        )
        self.memory_weight = memory_weight
        self.embedding_weight = embedding_weight

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
        hits = self.space.retrieve(query, limit=max(limit * 4, limit))
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
            "accessibility_weighted": True,
        }
        return RAGAnswer(
            query=query,
            answer=answer_text,
            evidence=evidence,
            diagnostics=diagnostics,
        )

    def reinforce(self, fragment_id: str, *, amount: float = 0.6, reason: str = "user_feedback") -> None:
        self.space.reinforce(fragment_id, amount=amount, reason=reason)

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
        return len(missing)

    def layout_memory_space(
        self,
        *,
        use_embeddings: bool = True,
        embed_fragments: bool = False,
        iterations: int = 120,
        semantic_neighbors: int = 4,
    ) -> LayoutResult:
        fragment_embeddings: dict[str, list[float]] = {}
        if use_embeddings:
            if embed_fragments:
                if self.embedding_provider is None:
                    raise ValueError("embedding_provider is required to embed fragments")
                fragments = list(self.space.fragments.values())
                embeddings = self.embedding_provider.embed_texts([fragment.text for fragment in fragments])
                fragment_embeddings = {
                    fragment.fragment_id: embedding
                    for fragment, embedding in zip(fragments, embeddings)
                }
            else:
                fragment_embeddings = self._existing_fragment_embeddings()

        return apply_memory_layout(
            self.space,
            fragment_embeddings=fragment_embeddings,
            iterations=iterations,
            semantic_neighbors=semantic_neighbors,
        )

    def layer_histogram(self) -> list[int]:
        histogram = [0] * self.space.total_layers
        for fragment in self.space.fragments.values():
            if 0 <= fragment.layer < len(histogram):
                histogram[fragment.layer] += 1
        return histogram

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": 2,
            "memory_space": self.space.snapshot(),
            "documents": [asdict(document) for document in self.documents.values()],
            "chunks": [asdict(chunk) for chunk in self.chunks.values()],
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
        instance = cls(**config)
        instance.documents = {
            item["document_id"]: SourceDocument(**item)
            for item in payload.get("documents", [])
        }
        instance.chunks = {
            item["chunk_id"]: SourceChunk(**item)
            for item in payload.get("chunks", [])
        }
        instance._restore_space(space_payload)
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
        memory_weight: float = 0.65,
        embedding_weight: float = 0.35,
    ) -> "LayeredMemoryRAG":
        instance = cls.load(path)
        instance.embedding_provider = make_embedding_provider(
            embedding_provider,
            model=embedding_model,
            api_key=embedding_api_key,
            dimensions=embedding_dimensions,
        )
        instance.memory_weight = memory_weight
        instance.embedding_weight = embedding_weight
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
            spatial_score=hit.accessibility,
            layout_score=hit.accessibility,
        )

    def _embed_chunks(self, chunk_texts: list[str]) -> list[list[float]]:
        if self.embedding_provider is None:
            return []
        return self.embedding_provider.embed_texts(chunk_texts)

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
            return []
        embedded_chunks = [chunk for chunk in self.chunks.values() if chunk.embedding]
        if not embedded_chunks:
            return []

        query_embedding = self.embedding_provider.embed_query(query)
        scored = sorted(
            (
                (cosine_similarity(query_embedding, chunk.embedding or []), chunk)
                for chunk in embedded_chunks
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        evidence: list[MemoryEvidence] = []
        for similarity, chunk in scored[:limit]:
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
                    score=similarity,
                    activation=fragment.activation,
                    strength=fragment.strength,
                    accessibility=self.space.accessibility_weight(fragment.depth),
                    via_relation="embedding",
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    title=document.title if document else chunk.metadata.get("title"),
                    chunk_text=chunk.text,
                    embedding_score=similarity,
                    spatial_score=self.space.accessibility_weight(fragment.depth),
                    layout_score=self.space.accessibility_weight(fragment.depth),
                )
            )
        return evidence

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
        merged[key] = item

    for item in semantic_evidence:
        key = (item.chunk_id, item.text.lower())
        normalized_embedding = max(0.0, min(1.0, (item.embedding_score or item.score)))
        accessible_embedding = normalized_embedding * item.accessibility
        if key in merged:
            existing = merged[key]
            existing.embedding_score = item.embedding_score
            existing.spatial_score = item.accessibility
            existing.layout_score = item.layout_score
            existing.score = existing.score + embedding_weight * accessible_embedding
            if existing.chunk_text is None:
                existing.chunk_text = item.chunk_text
            if existing.via_relation is None and item.via_relation:
                existing.via_relation = item.via_relation
        else:
            item.score = embedding_weight * accessible_embedding
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
