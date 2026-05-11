# 0012: Persisted EventRAG HNSW index

Date: 2026-05-11

Status: Accepted

## Context

EventRAG v4 stores compact event memories and embeddings for main labels and
structured subtags. Exact embedding scans are a useful correctness baseline, but
large stores need approximate semantic candidate recall. The event model also
has directional roles such as `from_where` and `to_where`, so ANN recall must
remain role-aware rather than replacing structured reasoning.

## Decision

Persist HNSW-style indexes for v4 SQLite stores:

- index `embedding_items`, not raw source text;
- build a global index and role-specific indexes for subtag roles;
- restore persisted indexes only when their item signature still matches the
  current embeddings;
- use HNSW as a candidate layer before role, direction, position, keyword, and
  recall-difficulty scoring;
- keep JSON stores lightweight and do not embed large ANN graphs in JSON;
- leave PLL/PPL graph path indexing out of this round.

## Consequences

EventRAG can avoid full embedding scans on larger SQLite stores while preserving
semantic directionality through subtags. The ANN graph is still a rebuildable
cache: event records, subtags, and embedding items remain the source of truth.
