# 0006: Add feedback probe to local evaluation reports

Date: 2026-05-01

Status: Accepted

## Context

Unit tests can prove that feedback APIs change fragment state, but the existing
local evaluation report only visualized forgetting, repeated recall, and linked
recall. It did not show whether curation feedback was moving memory state in the
intended direction.

## Decision

Extend `scripts/evaluate_memory_system.py` with a negative-feedback probe and
write its activation/layer movement into `reports/memory_system_report.md` and
`reports/memory_system_metrics.json`.

## Consequences

The evaluation report now covers the full memory dynamics loop: decay,
reinforcement, relation-based recall, and explicit feedback. This gives future
research changes a small but useful regression signal.

