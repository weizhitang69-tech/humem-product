from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humem_product import LayeredMemoryRAG  # noqa: E402


@dataclass(slots=True)
class ForgetPoint:
    cycle: int
    activation: float
    strength: float
    ease: float
    layer: int


@dataclass(slots=True)
class RecallPoint:
    attempt: int
    top_score: float
    target_activation: float
    target_strength: float
    target_layer: int
    evidence_count: int


@dataclass(slots=True)
class LinkedRecallResult:
    anchor_query: str
    target_text: str
    target_layer: int
    found: bool
    via_relation: str | None
    evidence: list[dict[str, object]]


@dataclass(slots=True)
class FeedbackProbeResult:
    query: str
    suppressed_text: str
    before_activation: float
    after_activation: float
    before_strength: float
    after_strength: float
    before_layer: int
    after_layer: int
    feedback_negative: int


@dataclass(slots=True)
class ConsolidationProbeResult:
    created_anchors: int
    reinforced_anchors: int
    support_relations: int
    anchor_layer: int | None
    anchor_terms: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local HuMem Product evaluation visuals")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    forget_curve = run_forgetting_curve()
    recall_curve = run_recall_reinforcement()
    linked_recall = run_linked_recall_probe()
    feedback_probe = run_feedback_probe()
    consolidation_probe = run_consolidation_probe()

    write_forgetting_svg(args.output / "forgetting_curve.svg", forget_curve)
    write_recall_svg(args.output / "recall_reinforcement.svg", recall_curve)
    write_linked_recall_svg(args.output / "linked_recall.svg", linked_recall)
    write_report(
        args.output / "memory_system_report.md",
        forget_curve=forget_curve,
        recall_curve=recall_curve,
        linked_recall=linked_recall,
        feedback_probe=feedback_probe,
        consolidation_probe=consolidation_probe,
    )
    write_json(
        args.output / "memory_system_metrics.json",
        forget_curve=forget_curve,
        recall_curve=recall_curve,
        linked_recall=linked_recall,
        feedback_probe=feedback_probe,
        consolidation_probe=consolidation_probe,
    )

    print(f"report: {args.output / 'memory_system_report.md'}")
    print(f"forgetting curve: {args.output / 'forgetting_curve.svg'}")
    print(f"recall reinforcement: {args.output / 'recall_reinforcement.svg'}")
    print(f"linked recall: {args.output / 'linked_recall.svg'}")


def run_forgetting_curve() -> list[ForgetPoint]:
    rag = LayeredMemoryRAG()
    rag.add_document(
        "The release checklist is important for Project Atlas. "
        "A temporary vault code 45123789 appeared once on the receipt. "
        "The team rarely needs that one-time number again.",
        document_id="forgetting-demo",
        title="Forgetting Demo",
        cool_down_cycles=0,
    )

    target_id = find_fragment_id(rag, lambda text, kind: text == "45123789" and kind == "number")
    points: list[ForgetPoint] = []

    for cycle in range(13):
        fragment = rag.space.fragments[target_id]
        points.append(
            ForgetPoint(
                cycle=cycle,
                activation=fragment.activation,
                strength=fragment.strength,
                ease=fragment.ease,
                layer=fragment.layer,
            )
        )
        rag.decay(step=0.14, cycles=1)

    return points


def run_recall_reinforcement() -> list[RecallPoint]:
    rag = LayeredMemoryRAG()
    rag.add_document(
        "Alice keeps the robot launch checklist in the blue notebook. "
        "The launch checklist helps the team prepare the robot deployment.",
        document_id="recall-demo",
        title="Recall Demo",
        cool_down_cycles=2,
    )

    target_id = find_fragment_id(rag, lambda text, kind: text == "checklist" and kind == "term")
    points: list[RecallPoint] = []

    for attempt in range(1, 9):
        hits = rag.space.retrieve("robot launch checklist", limit=12)
        target = rag.space.fragments[target_id]
        points.append(
            RecallPoint(
                attempt=attempt,
                top_score=hits[0].score if hits else 0.0,
                target_activation=target.activation,
                target_strength=target.strength,
                target_layer=target.layer,
                evidence_count=len(hits),
            )
        )

    return points


def run_linked_recall_probe() -> LinkedRecallResult:
    rag = LayeredMemoryRAG(total_layers=5, sealed_bottom_layers=2)
    rag.add_document(
        "topmemory links bottommemory.",
        document_id="linked-demo",
        title="Linked Recall Demo",
        cool_down_cycles=0,
    )

    top_id = find_fragment_id(rag, lambda text, _kind: text == "topmemory")
    bottom_id = find_fragment_id(rag, lambda text, _kind: text == "bottommemory")

    rag.space.fragments[top_id].layer = 0
    rag.space.fragments[top_id].z = rag.space._layer_to_height(0)
    rag.space.fragments[bottom_id].layer = 4
    rag.space.fragments[bottom_id].z = rag.space._layer_to_height(4)
    rag.space._rebuild_cross_layer_flags()

    target_layer_before_retrieve = rag.space.fragments[bottom_id].layer
    evidence = rag.retrieve("topmemory", limit=10)
    bottom_hit = next((item for item in evidence if item.fragment_id == bottom_id), None)

    return LinkedRecallResult(
        anchor_query="topmemory",
        target_text="bottommemory",
        target_layer=target_layer_before_retrieve,
        found=bottom_hit is not None,
        via_relation=bottom_hit.via_relation if bottom_hit else None,
        evidence=[
            {
                "text": item.text,
                "kind": item.kind,
                "layer": item.layer,
                "score": round(item.score, 4),
                "via_relation": item.via_relation,
            }
            for item in evidence
        ],
    )


def run_feedback_probe() -> FeedbackProbeResult:
    rag = LayeredMemoryRAG(retrieval_profile="conservative")
    rag.add_document(
        "The correct incident response runbook is in vault seven. "
        "The outdated incident response runbook is in the old binder.",
        document_id="feedback-demo",
        title="Feedback Demo",
        cool_down_cycles=0,
    )

    suppressed_id = find_fragment_id(rag, lambda text, _kind: text == "outdated")
    suppressed = rag.space.fragments[suppressed_id]
    before_activation = suppressed.activation
    before_strength = suppressed.strength
    before_layer = suppressed.layer

    rag.apply_feedback(
        query="incident response runbook",
        negative_fragment_ids=[suppressed_id],
        reason="evaluation_probe",
    )

    return FeedbackProbeResult(
        query="incident response runbook",
        suppressed_text=suppressed.text,
        before_activation=before_activation,
        after_activation=suppressed.activation,
        before_strength=before_strength,
        after_strength=suppressed.strength,
        before_layer=before_layer,
        after_layer=suppressed.layer,
        feedback_negative=int(suppressed.metadata.get("feedback", {}).get("negative", 0)),
    )


def run_consolidation_probe() -> ConsolidationProbeResult:
    rag = LayeredMemoryRAG()
    rag.add_document(
        "Alice keeps the robot launch checklist in the blue notebook. "
        "The robot launch checklist helps deployment readiness. "
        "Alice reviews deployment readiness before launch.",
        document_id="consolidation-demo",
        title="Consolidation Demo",
        cool_down_cycles=0,
    )

    result = rag.consolidate(min_support=3, keywords_per_anchor=4)
    anchor = (
        rag.space.fragments[result.created_anchor_ids[0]]
        if result.created_anchor_ids
        else None
    )
    consolidation = anchor.metadata.get("consolidation", {}) if anchor else {}
    return ConsolidationProbeResult(
        created_anchors=len(result.created_anchor_ids),
        reinforced_anchors=len(result.reinforced_anchor_ids),
        support_relations=result.support_relations,
        anchor_layer=anchor.layer if anchor else None,
        anchor_terms=list(consolidation.get("theme_terms", [])) if isinstance(consolidation, dict) else [],
    )


def find_fragment_id(
    rag: LayeredMemoryRAG,
    predicate: Callable[[str, str], bool],
) -> str:
    for fragment_id, fragment in rag.space.fragments.items():
        if predicate(fragment.normalized_text, fragment.kind):
            return fragment_id
    raise RuntimeError("target fragment not found")


def write_forgetting_svg(path: Path, points: list[ForgetPoint]) -> None:
    width, height = 980, 520
    margin = 70
    chart_w = width - margin * 2
    chart_h = height - margin * 2
    max_y = max(max(p.activation, p.strength, p.ease) for p in points) * 1.15
    max_y = max(max_y, 1.0)
    max_x = max(p.cycle for p in points)

    def xy(cycle: float, value: float) -> tuple[float, float]:
        x = margin + (cycle / max_x) * chart_w
        y = height - margin - (value / max_y) * chart_h
        return x, y

    def line(values: list[float], color: str) -> str:
        coords = [xy(points[index].cycle, value) for index, value in enumerate(values)]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        circles = "\n".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
            for x, y in coords
        )
        return f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="3"/>\n{circles}'

    layer_bars = []
    for point in points:
        x, _ = xy(point.cycle, 0)
        bar_h = 18 + point.layer * 18
        layer_bars.append(
            f'<rect x="{x - 12:.1f}" y="{height - margin - bar_h:.1f}" width="24" height="{bar_h:.1f}" fill="#CBD5E1" opacity="0.9"/>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#F8FAFC"/>
  <text x="38" y="42" font-family="Arial" font-size="26" font-weight="700" fill="#0F172A">Forgetting Curve: one-time code 45123789</text>
  <text x="38" y="68" font-family="Arial" font-size="14" fill="#475569">Activation and strength decay over cycles; layer bars grow as the fragment sinks deeper.</text>
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#334155" stroke-width="1.5"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#334155" stroke-width="1.5"/>
  {"".join(layer_bars)}
  {line([p.activation for p in points], "#2563EB")}
  {line([p.strength for p in points], "#0F766E")}
  {line([p.ease for p in points], "#F59E0B")}
  <text x="{margin}" y="{height - 28}" font-family="Arial" font-size="13" fill="#475569">forget cycles</text>
  <text x="24" y="{margin + 12}" font-family="Arial" font-size="13" fill="#475569">value</text>
  <rect x="650" y="26" width="250" height="86" rx="10" fill="#FFFFFF" stroke="#CBD5E1"/>
  <circle cx="674" cy="52" r="5" fill="#2563EB"/><text x="690" y="57" font-family="Arial" font-size="13" fill="#0F172A">activation</text>
  <circle cx="674" cy="76" r="5" fill="#0F766E"/><text x="690" y="81" font-family="Arial" font-size="13" fill="#0F172A">strength</text>
  <circle cx="674" cy="100" r="5" fill="#F59E0B"/><text x="690" y="105" font-family="Arial" font-size="13" fill="#0F172A">ease</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_recall_svg(path: Path, points: list[RecallPoint]) -> None:
    width, height = 980, 520
    margin = 70
    chart_w = width - margin * 2
    chart_h = height - margin * 2
    max_y = max(max(p.top_score, p.target_activation, p.target_strength) for p in points) * 1.15
    max_y = max(max_y, 1.0)
    max_x = max(p.attempt for p in points)

    def xy(attempt: float, value: float) -> tuple[float, float]:
        x = margin + ((attempt - 1) / max(max_x - 1, 1)) * chart_w
        y = height - margin - (value / max_y) * chart_h
        return x, y

    def line(values: list[float], color: str) -> str:
        coords = [xy(points[index].attempt, value) for index, value in enumerate(values)]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        circles = "\n".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
            for x, y in coords
        )
        return f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="3"/>\n{circles}'

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#F8FAFC"/>
  <text x="38" y="42" font-family="Arial" font-size="26" font-weight="700" fill="#0F172A">Recall Reinforcement: robot launch checklist</text>
  <text x="38" y="68" font-family="Arial" font-size="14" fill="#475569">Repeated retrieval raises raw memory score and reinforces the target memory into upper layers.</text>
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#334155" stroke-width="1.5"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#334155" stroke-width="1.5"/>
  {line([p.top_score for p in points], "#7C3AED")}
  {line([p.target_activation for p in points], "#2563EB")}
  {line([p.target_strength for p in points], "#0F766E")}
  <text x="{margin}" y="{height - 28}" font-family="Arial" font-size="13" fill="#475569">query attempts</text>
  <text x="24" y="{margin + 12}" font-family="Arial" font-size="13" fill="#475569">value</text>
  <rect x="632" y="26" width="282" height="86" rx="10" fill="#FFFFFF" stroke="#CBD5E1"/>
  <circle cx="656" cy="52" r="5" fill="#7C3AED"/><text x="672" y="57" font-family="Arial" font-size="13" fill="#0F172A">raw retrieval score</text>
  <circle cx="656" cy="76" r="5" fill="#2563EB"/><text x="672" y="81" font-family="Arial" font-size="13" fill="#0F172A">target activation</text>
  <circle cx="656" cy="100" r="5" fill="#0F766E"/><text x="672" y="105" font-family="Arial" font-size="13" fill="#0F172A">target strength</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_linked_recall_svg(path: Path, result: LinkedRecallResult) -> None:
    status_color = "#0F766E" if result.found else "#DC2626"
    status = "FOUND" if result.found else "NOT FOUND"
    via = result.via_relation or "none"
    evidence_rows = "\n".join(
        f'<text x="78" y="{282 + i * 30}" font-family="Arial" font-size="14" fill="#0F172A">{i + 1}. {escape_xml(str(item["text"]))} | layer={item["layer"]} | score={item["score"]} | via={item["via_relation"]}</text>'
        for i, item in enumerate(result.evidence[:6])
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="520" viewBox="0 0 980 520">
  <rect width="100%" height="100%" fill="#F8FAFC"/>
  <text x="38" y="42" font-family="Arial" font-size="26" font-weight="700" fill="#0F172A">Linked Recall Probe</text>
  <text x="38" y="68" font-family="Arial" font-size="14" fill="#475569">A sealed bottom-layer detail is retrieved through a relation from an upper-layer anchor.</text>
  <rect x="74" y="126" width="230" height="96" rx="14" fill="#ECFDF5" stroke="#0F766E"/>
  <text x="106" y="164" font-family="Arial" font-size="18" font-weight="700" fill="#0F172A">Query Anchor</text>
  <text x="106" y="194" font-family="Arial" font-size="15" fill="#334155">{escape_xml(result.anchor_query)}</text>
  <path d="M 304 174 C 402 96, 518 96, 616 174" stroke="#2563EB" stroke-width="3" fill="none" marker-end="url(#arrow)"/>
  <rect x="616" y="126" width="260" height="96" rx="14" fill="#EFF6FF" stroke="#2563EB"/>
  <text x="650" y="164" font-family="Arial" font-size="18" font-weight="700" fill="#0F172A">Sealed Detail</text>
  <text x="650" y="194" font-family="Arial" font-size="15" fill="#334155">{escape_xml(result.target_text)} | layer {result.target_layer}</text>
  <rect x="376" y="158" width="170" height="42" rx="21" fill="{status_color}"/>
  <text x="425" y="184" font-family="Arial" font-size="15" font-weight="700" fill="#FFFFFF">{status}</text>
  <text x="390" y="226" font-family="Arial" font-size="14" fill="#475569">via relation: {escape_xml(via)}</text>
  <text x="74" y="258" font-family="Arial" font-size="18" font-weight="700" fill="#0F172A">Top Evidence</text>
  {evidence_rows}
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563EB"/>
    </marker>
  </defs>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_report(
    path: Path,
    *,
    forget_curve: list[ForgetPoint],
    recall_curve: list[RecallPoint],
    linked_recall: LinkedRecallResult,
    feedback_probe: FeedbackProbeResult,
    consolidation_probe: ConsolidationProbeResult,
) -> None:
    start_forget = forget_curve[0]
    end_forget = forget_curve[-1]
    start_recall = recall_curve[0]
    end_recall = recall_curve[-1]
    report = f"""# HuMem Product Local Evaluation Report

Generated by `scripts/evaluate_memory_system.py`.

## Visuals

![Forgetting curve](forgetting_curve.svg)

![Recall reinforcement](recall_reinforcement.svg)

![Linked recall](linked_recall.svg)

## Summary

| Probe | Result |
| --- | --- |
| Forgetting activation | {start_forget.activation:.3f} -> {end_forget.activation:.3f} |
| Forgetting strength | {start_forget.strength:.3f} -> {end_forget.strength:.3f} |
| Forgetting layer | {start_forget.layer} -> {end_forget.layer} |
| Recall top score | {start_recall.top_score:.3f} -> {end_recall.top_score:.3f} |
| Recall target activation | {start_recall.target_activation:.3f} -> {end_recall.target_activation:.3f} |
| Recall target strength | {start_recall.target_strength:.3f} -> {end_recall.target_strength:.3f} |
| Linked bottom detail found | {linked_recall.found} |
| Linked recall relation | {linked_recall.via_relation} |
| Feedback suppressed activation | {feedback_probe.before_activation:.3f} -> {feedback_probe.after_activation:.3f} |
| Feedback suppressed layer | {feedback_probe.before_layer} -> {feedback_probe.after_layer} |
| Consolidation anchors created | {consolidation_probe.created_anchors} |
| Consolidation support relations | {consolidation_probe.support_relations} |
| Consolidation anchor layer | {consolidation_probe.anchor_layer} |

## How To Read This

- The forgetting curve should trend downward for activation/strength while the layer moves deeper.
- The recall reinforcement curve should trend upward or stay high as repeated reads strengthen useful memory.
- The linked recall probe should show a bottom-layer detail surfacing through a relation, proving sealed details are not simply lost.
- The feedback probe should weaken or demote the suppressed memory after negative feedback.
- The consolidation probe should create an upper-layer anchor connected to several support fragments.
"""
    path.write_text(report, encoding="utf-8")


def write_json(
    path: Path,
    *,
    forget_curve: list[ForgetPoint],
    recall_curve: list[RecallPoint],
    linked_recall: LinkedRecallResult,
    feedback_probe: FeedbackProbeResult,
    consolidation_probe: ConsolidationProbeResult,
) -> None:
    payload = {
        "forget_curve": [asdict(point) for point in forget_curve],
        "recall_curve": [asdict(point) for point in recall_curve],
        "linked_recall": {
            "anchor_query": linked_recall.anchor_query,
            "target_text": linked_recall.target_text,
            "target_layer": linked_recall.target_layer,
            "found": linked_recall.found,
            "via_relation": linked_recall.via_relation,
            "evidence": linked_recall.evidence,
        },
        "feedback_probe": asdict(feedback_probe),
        "consolidation_probe": asdict(consolidation_probe),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
