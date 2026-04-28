from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

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


@dataclass(slots=True)
class MemoryEvidence:
    fragment_id: str
    text: str
    kind: str
    layer: int
    score: float
    activation: float
    strength: float
    via_relation: str | None
    document_id: str | None = None
    chunk_id: str | None = None
    title: str | None = None
    chunk_text: str | None = None


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
    ) -> None:
        self.space = MemorySpace(
            total_layers=total_layers,
            top_layer_quota=top_layer_quota,
            bottom_layer_quota=bottom_layer_quota,
            sealed_bottom_layers=sealed_bottom_layers,
        )
        self.documents: dict[str, SourceDocument] = {}
        self.chunks: dict[str, SourceChunk] = {}

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

        for ordinal, chunk_text in enumerate(
            _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunk_id = f"{doc_id}:{ordinal}"
            chunk = SourceChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                text=chunk_text,
                ordinal=ordinal,
                metadata={"title": doc_title},
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
        evidence = [self._hit_to_evidence(hit) for hit in hits]
        return _dedupe_evidence(evidence)[:limit]

    def answer(self, query: str, *, limit: int = 6) -> RAGAnswer:
        evidence = self.retrieve(query, limit=limit)
        answer_text = _compose_answer(query, evidence)
        diagnostics = {
            "fragment_count": len(self.space.fragments),
            "relation_count": len(self.space.relations),
            "layer_histogram": self.layer_histogram(),
            "evidence_count": len(evidence),
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

    def layer_histogram(self) -> list[int]:
        histogram = [0] * self.space.total_layers
        for fragment in self.space.fragments.values():
            if 0 <= fragment.layer < len(histogram):
                histogram[fragment.layer] += 1
        return histogram

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": 1,
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
            score=hit.score,
            activation=hit.activation,
            strength=hit.strength,
            via_relation=hit.via_relation,
            document_id=document_id,
            chunk_id=chunk_id,
            title=title,
            chunk_text=chunk_text,
        )

    def _restore_space(self, payload: dict[str, Any]) -> None:
        self.space.fragments.clear()
        self.space.fragment_index.clear()
        self.space.relations.clear()
        self.space.out_edges = defaultdict(set)
        self.space.in_edges = defaultdict(set)

        for item in payload.get("fragments", []):
            fragment = MemoryFragment(**item)
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
