# 0008: Biomimetic memory-space viewer

Date: 2026-05-01

Status: Accepted

## Context

The original viewer showed a useful 3D node-link graph, but the experience still
felt like a generic network visualization. HuMem's product value is not only
retrieval: it models layered accessibility, reinforcement, decay, relation
paths, and consolidation anchors. The interface should make those ideas visible.

## Decision

Redesign the viewer as a biomimetic memory-space surface:

- layer planes become translucent memory laminae;
- links are presented as synapses with relation-colored pulse particles;
- consolidated memories render as anchor bodies with halos;
- the top panel shows memory vitals such as activation, strength, access, and
  anchor count;
- the lower insight panel summarizes upper recall, deep trace, anchor density,
  and mean access;
- view modes focus the organism, anchors, or deep traces;
- node details include an activation ring, memory state, consolidation terms,
  source/chunk information, and synaptic neighbors.

The data API remains the same `/api/graph` shape with the existing additions for
consolidation metadata. This keeps the viewer a static asset served by the
current local visualization server.

## Consequences

The viewer now expresses HuMem's research model rather than merely displaying
it. It is still dependency-light and bundled with the package, but gives users a
clearer sense of which memories are surface anchors, which are submerged traces,
and how recall paths move through the graph.

