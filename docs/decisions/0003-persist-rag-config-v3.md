# 0003: Persist RAG configuration with store version 3

Date: 2026-05-01

Status: Accepted

## Context

JSON and SQLite stores preserved the memory graph, documents, chunks, embeddings,
layout, and dynamics. They did not preserve the RAG-level retrieval policy, so a
store could be reloaded with different scoring behavior by accident.

## Decision

Move the store version to `3` and persist a `rag_config` payload containing the
active retrieval profile and hybrid scoring weights. SQLite `store_meta` now
stores both memory-space config and RAG config while still accepting the older
flat config shape.

## Consequences

Experiments become reproducible across save/load and JSON-to-SQLite migration.
Older stores remain loadable because missing `rag_config` falls back to the
balanced profile.

