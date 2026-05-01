# 0001: Retrieval profiles for repeatable experiments

Date: 2026-05-01

Status: Accepted

## Context

HuMem Product already combined layered memory scores with optional embedding
similarity, but the behavior was controlled by loose numeric parameters. That
made experiments hard to name, compare, persist, or explain.

## Decision

Add `RetrievalProfile` in `src/humem_product/policy.py` with named profiles:

- `balanced`
- `conservative`
- `semantic`
- `exploratory`
- `archival`

Each profile owns memory/embedding weights, read-time reinforcement behavior,
and default feedback strengths. `LayeredMemoryRAG` now resolves a profile during
construction while preserving explicit weight overrides.

## Consequences

Researchers can run the same memory store under different retrieval assumptions
without editing internals. The default profile keeps previous balanced behavior,
while `archival` gives a non-mutating read mode for audits and evaluation.

