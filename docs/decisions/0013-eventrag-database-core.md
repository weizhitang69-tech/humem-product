# 0013: EventRAG SQLite database core

Date: 2026-05-11

Status: Accepted

## Context

EventRAG v4 already stores event chunks, structured subtags, embeddings, and
persisted HNSW snapshots. The next step is to behave more like a lightweight
embedded vector memory database without introducing a separate service or new
runtime dependency.

## Decision

Upgrade SQLite v4 stores with database-core capabilities:

- collections/namespaces with per-collection role schemas;
- standardized AND-only event filter DSL;
- incremental HNSW insertion for new `embedding_items`;
- soft delete, replace, metadata patch, compaction, and backup;
- WAL and busy timeout defaults for concurrent local readers;
- ANN tombstones so deleted items are hidden immediately and cleaned on compact.

The source of truth remains events, subtags, and embedding items. HNSW snapshots
remain rebuildable caches.

## Consequences

EventRAG can now be embedded as a small local vector memory database. SQLite is
the primary target for database-level behavior; JSON stores remain lightweight
and portable. Sharding and a full HTTP query service remain future work.
