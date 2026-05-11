# 0010: Ebbinghaus-style forgetting dynamics

Date: 2026-05-06

Status: Accepted

## Context

HuMem Product previously used a simple linear forgetting cycle: each `forget()`
call subtracted fixed amounts from activation, strength, and ease, then moved
weak memories down a layer. That was easy to reason about, but it did not
capture the nonlinear shape of memory retention or the way repeated recall,
feedback, and structural support should slow forgetting.

## Decision

Use a cycle-based Ebbinghaus-style retention curve as the default natural
forgetting model:

- `forgetting_model="ebbinghaus"` is the default in `MemoryDynamicsConfig`;
- `forgetting_model="linear"` remains available as a compatibility baseline;
- retention uses `exp(-rate * elapsed_cycles / stability)`;
- stability is raised by strength, retrievals, positive feedback,
  reinforcements, repeated sources, and relation support;
- negative feedback increases the effective forgetting rate;
- consolidation anchors decay gently and resist ordinary layer sinking;
- HuMem layer movement remains a state decision, not a direct output of the
  curve.

The model remains cycle-based rather than wall-clock based. Existing CLI calls
such as `decay --cycles 3 --step 0.14` still work, with `step` scaling the
forgetting rate in Ebbinghaus mode.

## Consequences

One-time details now fade with a nonlinear retention curve and sink when their
retention and activation are low. Recalled, reinforced, positively reviewed, or
structurally supported memories retain more activation under the same number of
cycles. Negative feedback still has immediate suppressive behavior and now also
accelerates later forgetting.

No store version bump is required. The new dynamics fields are persisted inside
the existing memory-space config for both JSON and SQLite stores, and old stores
load with the new defaults.
