# 0002: Feedback as memory-state mutation

Date: 2026-05-01

Status: Accepted

## Context

The original module reinforced memories when they were read, but it had no
first-class way to apply user or evaluator feedback. A product memory layer needs
to learn from "useful" and "not useful" evidence, not only from repeated reads.

## Decision

Add `LayeredMemoryRAG.apply_feedback()` and `MemorySpace.suppress()`:

- positive feedback reinforces selected fragments;
- negative feedback lowers activation, strength, and ease;
- negative feedback can demote fragments into deeper layers;
- feedback metadata is stored on fragments for later inspection.

## Consequences

HuMem can now support explicit curation loops, thumbs-up/down UX, evaluation
harnesses, and future learning-to-rerank experiments. Feedback changes are still
deterministic and local, so they remain easy to test.

