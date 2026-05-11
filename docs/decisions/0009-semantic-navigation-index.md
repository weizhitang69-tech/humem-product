# 0009: Runtime semantic navigation index

Date: 2026-05-06

Status: Accepted

## Context

HuMem Product already supports hybrid retrieval by exact cosine scanning over
embedded source chunks. That baseline is simple and correct, but repeated
queries over a large embedded store pay the cost of scanning every chunk. The
project also explores graph-shaped memory, so KANN/HNSW-style navigation is a
natural candidate, but it must not be confused with HuMem's durable memory
relations.

## Decision

Add a pure Python, dependency-free `SemanticNavigationIndex` as a runtime-only
candidate recall structure:

- nodes are `SourceChunk.embedding` vectors, not memory fragments;
- the index uses deterministic HNSW-like levels and cosine similarity;
- `semantic_index="exact"` preserves full cosine scan as the correctness
  baseline;
- `semantic_index="auto"` keeps exact scan for small stores and lazily builds
  ANN only after the embedded chunk threshold;
- `semantic_index="ann"` forces the navigation graph for experiments and
  benchmarks;
- ANN output is only a candidate list, then HuMem's existing memory weight,
  embedding weight, accessibility, and relation-aware evidence merge still
  decide final ranking;
- the graph is invalidated after document ingestion, chunk embedding, and store
  load, then rebuilt on demand.

The index is not persisted to JSON or SQLite, does not change store version, and
does not create HuMem `relations`. Only the lightweight semantic navigation
policy is stored with the existing RAG config.

## Consequences

Large, long-lived embedded stores can avoid repeated full cosine scans once the
runtime index is built, while small stores and CLI one-off usage keep the exact
path by default. The architecture stays aligned with HuMem's core idea: the
memory graph remains the source of memory dynamics, and KANN/HNSW is only a
navigation layer for semantic chunk recall.

The local evaluation report now includes a semantic navigation probe comparing
exact top-k results with ANN top-k overlap, query effort, and timing. It
deliberately avoids claiming ANN speedups on small fixtures.
