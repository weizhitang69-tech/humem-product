# 0007: Deterministic memory consolidation cycle

Date: 2026-05-01

Status: Accepted

## Context

HuMem Product had write, retrieve, reinforce, decay, feedback, and layout. It
still lacked an offline "review" process that turns repeated or valuable details
into sparse upper-layer anchors. Without that, the system could recall memory
but could not reorganize itself into more useful long-term structure.

## Decision

Add a deterministic consolidation cycle:

- rank non-clause fragments by activation, strength, retrievals, source count,
  layer accessibility, and feedback;
- group candidates by document, chunk, or global scope;
- create or refresh an upper-layer `Consolidated memory` anchor;
- connect the anchor to supporting fragments with `consolidates` relations;
- expose the cycle through Python, CLI, visualization metadata, tests, and the
  local evaluation report.

The implementation deliberately avoids LLM calls. It is a stable research
baseline that can later be replaced or augmented by embedding clustering or
model-generated summaries.

## Consequences

HuMem now has a full memory lifecycle: ingestion, retrieval, reinforcement,
feedback, decay, layout, and consolidation. Consolidation gives the project a
distinct value proposition beyond ordinary RAG: the memory graph can create new
high-level anchors from usage and structure while preserving source evidence.

