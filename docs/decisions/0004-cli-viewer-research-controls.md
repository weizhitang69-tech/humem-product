# 0004: Expose research controls through CLI and viewer metadata

Date: 2026-05-01

Status: Accepted

## Context

HuMem already had CLI ingestion, retrieval, decay, layout, migration, and a 3D
viewer. The new profile and feedback capabilities needed to be usable without
writing Python glue code.

## Decision

Expose retrieval profiles on `ingest` and `ask`, add a `feedback` CLI command,
and include retrieval profile metadata in graph data used by the viewer.

## Consequences

The command line can now run profile comparison and feedback experiments against
the same store. The viewer can show which profile produced the current graph
metadata, reducing ambiguity during demos.

