# 0011: Event-centric temporal memory rewrite

Date: 2026-05-11

Status: Accepted

## Context

The previous HuMem Product runtime stored memory as lightweight fragments and
relations. Its 3D coordinates were useful for demos, but the horizontal
placement was derived from hashes or a force layout, and full source chunks were
still the primary embedding target. That made the space visually expressive but
not grounded enough in the semantic structure of the memory itself.

## Decision

Add `EventRAG` as the new main memory path:

- write-time LLM extraction turns text into event chunks;
- each event has a `main_label`, wall-clock `captured_at`, ordered structured
  subtags, and a compressed trace;
- embeddings are generated only for the main label and subtag embedding text,
  not for the full original source;
- query-time LLM planning creates retrieval terms, target roles, time hints, and
  recall precision;
- retrieval combines keyword search, subtag/main-label embedding search, and a
  time-based recall-difficulty gate;
- old memories are not deleted, but the Ebbinghaus-style difficulty threshold
  makes them require more precise prompts;
- the v4 store persists events, subtags, compressed traces, embedding items,
  source records, and event logs;
- the viewer maps vertical position to captured time and horizontal position to
  an embedding-derived semantic plane.

`LayeredMemoryRAG` remains available as a legacy path and as a source for v3 to
v4 migration.

## Consequences

HuMem's product center moves from fragment-level layered RAG to structured
event memory. The system now sends compact structured evidence to the answer
LLM, preserving memory clues while reducing prompt context. The previous
biomimetic 3D interpretation is superseded for v4 stores by a time axis plus a
semantic embedding plane.
