from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_structured_memory_fidelity import (  # noqa: E402
    DEFAULT_SEED,
    build_health_event,
    build_loan_event,
    build_meeting_event,
    build_project_event,
    build_purchase_event,
    build_travel_event,
)


DEFAULT_SAMPLES = 2000
DEFAULT_LENGTHS = [80, 160, 320, 640, 1000, 1600, 2400]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate context-token savings from structured memory evidence."
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--lengths",
        type=str,
        default=",".join(str(item) for item in DEFAULT_LENGTHS),
        help="Comma-separated target paragraph lengths in characters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "structured_context_savings",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    lengths = [int(item.strip()) for item in args.lengths.split(",") if item.strip()]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = make_token_counter()
    rows = build_rows(
        sample_count=args.samples,
        lengths=lengths,
        rng=rng,
        count_tokens=tokenizer["count"],
    )
    metrics = summarize(rows, tokenizer_name=tokenizer["name"], seed=args.seed)

    dataset_path = output_dir / "dataset.jsonl"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    chart_path = output_dir / "token_savings_by_paragraph_length.svg"

    write_jsonl(dataset_path, rows)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_report(metrics, chart_path.name), encoding="utf-8")
    chart_path.write_text(render_svg(metrics), encoding="utf-8")

    print(f"dataset: {dataset_path}")
    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    print(f"chart: {chart_path}")


def build_rows(
    *,
    sample_count: int,
    lengths: list[int],
    rng: random.Random,
    count_tokens,
) -> list[dict[str, Any]]:
    builders = [
        build_loan_event,
        build_travel_event,
        build_project_event,
        build_health_event,
        build_purchase_event,
        build_meeting_event,
    ]
    rows: list[dict[str, Any]] = []
    for index in range(sample_count):
        builder = builders[index % len(builders)]
        event = builder(rng)
        task = rng.choice(event.tasks)
        target_length = lengths[index % len(lengths)]
        paragraph = expand_paragraph(
            event.original_text,
            event_type=event.event_type,
            target_chars=target_length,
            rng=rng,
        )
        raw_context = render_raw_context(task.query, paragraph)
        structured_json_context = render_structured_json_context(
            task.query,
            event.main_label,
            event.compressed_trace,
            event.subtags,
        )
        compact_context = render_compact_context(
            task.query,
            event.main_label,
            event.subtags,
        )
        answer_only_context = render_answer_slot_context(task.query, task.target_role, task.expected_value)

        raw_tokens = count_tokens(raw_context)
        structured_json_tokens = count_tokens(structured_json_context)
        compact_tokens = count_tokens(compact_context)
        answer_only_tokens = count_tokens(answer_only_context)

        rows.append(
            {
                "id": f"context-savings-{index + 1:04d}",
                "event_type": event.event_type,
                "target_paragraph_chars": target_length,
                "actual_paragraph_chars": len(paragraph),
                "query": task.query,
                "expected_value": task.expected_value,
                "raw_context": raw_context,
                "structured_json_context": structured_json_context,
                "compact_structured_context": compact_context,
                "answer_slot_context": answer_only_context,
                "tokens": {
                    "raw_text": raw_tokens,
                    "structured_json": structured_json_tokens,
                    "compact_structured": compact_tokens,
                    "answer_slot": answer_only_tokens,
                },
                "savings_vs_raw": {
                    "structured_json": savings(raw_tokens, structured_json_tokens),
                    "compact_structured": savings(raw_tokens, compact_tokens),
                    "answer_slot": savings(raw_tokens, answer_only_tokens),
                },
                "subtags": event.subtags,
                "original_text": event.original_text,
                "expanded_paragraph": paragraph,
            }
        )
    return rows


def render_raw_context(query: str, paragraph: str) -> str:
    return f"问题：{query}\n请只根据以下原文回答。\n原文：{paragraph}"


def render_structured_json_context(
    query: str,
    main_label: str,
    compressed_trace: str,
    subtags: list[dict[str, Any]],
) -> str:
    evidence = {
        "main_label": main_label,
        "compressed_trace": compressed_trace,
        "subtags": [
            {
                "role": item["role"],
                "position": item["position"],
                "value": item["value"],
                "confidence": item.get("confidence", 1.0),
            }
            for item in subtags
        ],
    }
    return (
        "问题："
        + query
        + "\n请只根据以下结构化记忆回答。\n结构化记忆："
        + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    )


def render_compact_context(query: str, main_label: str, subtags: list[dict[str, Any]]) -> str:
    parts = [
        f"{item['role']}@{item['position']}={item['value']}"
        for item in subtags
    ]
    return f"问题：{query}\n记忆：main={main_label}; " + "; ".join(parts)


def render_answer_slot_context(query: str, role: str, expected_value: str) -> str:
    return f"问题：{query}\n命中槽位：{role}={expected_value}"


def expand_paragraph(
    original: str,
    *,
    event_type: str,
    target_chars: int,
    rng: random.Random,
) -> str:
    sentences = [original]
    while len("".join(sentences)) < target_chars:
        sentences.append(make_filler_sentence(event_type, rng))
    paragraph = "".join(sentences)
    if len(paragraph) <= target_chars:
        return paragraph
    return paragraph[:target_chars].rstrip("，。；、") + "。"


def make_filler_sentence(event_type: str, rng: random.Random) -> str:
    shared = [
        "这条记录还包含了一些当时的背景描述，但这些背景不会改变核心事实。",
        "用户当时语气比较犹豫，并且补充了若干和主要问题关系不大的细节。",
        "系统需要保留可审计原文，不过回答时通常只需要其中几个关键槽位。",
        "后续对话里可能会追问原因、地点、负责人或下一步动作。",
    ]
    by_type = {
        "loan": [
            "对话里还提到预算、工资到账时间和不同银行的活动页面。",
            "用户顺手比较了几个额度入口，但没有给出新的申请结果。",
        ],
        "travel": [
            "行程备注里还写了天气、交通方式和备用集合点。",
            "用户提到车票截图已经保存，但截图内容不是本轮问题重点。",
        ],
        "project": [
            "会议记录里还有指标截图、风险备注和下游依赖清单。",
            "团队讨论了一些备选方案，但最终只保留了一个明确下一步。",
        ],
        "health": [
            "记录里还包含排队时间、缴费方式和医生的常规提醒。",
            "用户描述了若干轻微症状，但关键事实仍是看诊原因和医嘱。",
        ],
        "purchase": [
            "购物备注里还包含优惠券、配送时间和售后政策。",
            "用户比较了几个店铺评价，不过没有改变最终处理方式。",
        ],
        "meeting": [
            "会议纪要里还保留了参会人补充意见和几个待确认问题。",
            "记录中出现了多个上下文备注，但负责人和动作是主要事实。",
        ],
    }
    return rng.choice(shared + by_type.get(event_type, []))


def make_token_counter() -> dict[str, Any]:
    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("cl100k_base")
        return {
            "name": "tiktoken:cl100k_base",
            "count": lambda text: len(encoding.encode(text)),
        }
    except Exception:
        return {
            "name": "heuristic:cjk1_latin4",
            "count": heuristic_token_count,
        }


LATIN_RE = re.compile(r"[A-Za-z0-9_]+")


def heuristic_token_count(text: str) -> int:
    count = 0
    consumed = [False] * len(text)
    for match in LATIN_RE.finditer(text):
        token_count = max(1, math.ceil(len(match.group(0)) / 4))
        count += token_count
        for index in range(match.start(), match.end()):
            consumed[index] = True
    for index, char in enumerate(text):
        if consumed[index] or char.isspace():
            continue
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF:
            count += 1
        elif char in "，。！？；：、,.!?;:()[]{}<>/=+-|@":
            count += 1
        else:
            count += 1
    return count


def savings(raw_tokens: int, compressed_tokens: int) -> float:
    if raw_tokens <= 0:
        return 0.0
    return round(1.0 - compressed_tokens / raw_tokens, 4)


def summarize(rows: list[dict[str, Any]], *, tokenizer_name: str, seed: int) -> dict[str, Any]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["target_paragraph_chars"])].append(row)

    by_length: list[dict[str, Any]] = []
    for length, items in sorted(groups.items()):
        averages = {
            key: avg(item["tokens"][key] for item in items)
            for key in ("raw_text", "structured_json", "compact_structured", "answer_slot")
        }
        by_length.append(
            {
                "target_paragraph_chars": length,
                "samples": len(items),
                "avg_actual_paragraph_chars": round(avg(item["actual_paragraph_chars"] for item in items), 1),
                "avg_tokens": {key: round(value, 2) for key, value in averages.items()},
                "avg_savings_vs_raw": {
                    key: round(avg(item["savings_vs_raw"][key] for item in items), 4)
                    for key in ("structured_json", "compact_structured", "answer_slot")
                },
            }
        )

    overall = {
        "avg_tokens": {
            key: round(avg(row["tokens"][key] for row in rows), 2)
            for key in ("raw_text", "structured_json", "compact_structured", "answer_slot")
        },
        "avg_savings_vs_raw": {
            key: round(avg(row["savings_vs_raw"][key] for row in rows), 4)
            for key in ("structured_json", "compact_structured", "answer_slot")
        },
    }
    return {
        "sample_count": len(rows),
        "seed": seed,
        "tokenizer": tokenizer_name,
        "overall": overall,
        "by_length": by_length,
    }


def avg(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def render_report(metrics: dict[str, Any], chart_name: str) -> str:
    lines = [
        "# Structured Context Savings Evaluation",
        "",
        f"- Samples: {metrics['sample_count']}",
        f"- Seed: {metrics['seed']}",
        f"- Tokenizer: `{metrics['tokenizer']}`",
        f"- Chart: ![{chart_name}]({chart_name})",
        "",
        "## Overall",
        "",
        "| Context | Avg Tokens | Avg Savings vs Raw |",
        "| --- | ---: | ---: |",
    ]
    overall = metrics["overall"]
    for key, label in [
        ("raw_text", "Raw original text"),
        ("structured_json", "Structured JSON"),
        ("compact_structured", "Compact structured"),
        ("answer_slot", "Answer slot only"),
    ]:
        savings_text = "-"
        if key != "raw_text":
            savings_text = f"{overall['avg_savings_vs_raw'][key]:.2%}"
        lines.append(f"| {label} | {overall['avg_tokens'][key]:.2f} | {savings_text} |")

    lines.extend(
        [
            "",
            "## By Paragraph Length",
            "",
            "| Target Chars | Raw | Structured JSON | Compact | Answer Slot | Compact Savings |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics["by_length"]:
        tokens = row["avg_tokens"]
        savings_row = row["avg_savings_vs_raw"]
        lines.append(
            f"| {row['target_paragraph_chars']} | {tokens['raw_text']:.2f} | "
            f"{tokens['structured_json']:.2f} | {tokens['compact_structured']:.2f} | "
            f"{tokens['answer_slot']:.2f} | {savings_row['compact_structured']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Raw context includes the user query plus the whole paragraph.",
            "- Structured JSON includes main label, compressed trace, and role/position/value subtags.",
            "- Compact structured is closer to the minimal evidence format a prompt would feed to the answer model.",
            "- Answer slot only is an optimistic lower bound after retrieval and reranking already selected the exact slot.",
        ]
    )
    return "\n".join(lines)


def render_svg(metrics: dict[str, Any]) -> str:
    width = 980
    height = 560
    margin_left = 82
    margin_right = 34
    margin_top = 36
    margin_bottom = 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    rows = metrics["by_length"]
    x_values = [row["target_paragraph_chars"] for row in rows]
    series = {
        "Raw original text": ("raw_text", "#1f4e79"),
        "Structured JSON": ("structured_json", "#7a5195"),
        "Compact structured": ("compact_structured", "#2f855a"),
        "Answer slot only": ("answer_slot", "#d97706"),
    }
    max_y = max(row["avg_tokens"]["raw_text"] for row in rows) * 1.08
    min_x = min(x_values)
    max_x = max(x_values)

    def sx(value: float) -> float:
        if max_x == min_x:
            return margin_left + plot_w / 2
        return margin_left + (value - min_x) / (max_x - min_x) * plot_w

    def sy(value: float) -> float:
        return margin_top + plot_h - value / max_y * plot_h

    y_ticks = nice_ticks(max_y, count=6)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="34" y="28" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#172033">Structured Memory Context Savings</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#243042" stroke-width="1.2"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#243042" stroke-width="1.2"/>',
    ]
    for tick in y_ticks:
        y = sy(tick)
        elements.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        elements.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{int(tick)}</text>')
    for x in x_values:
        px = sx(x)
        elements.append(f'<line x1="{px:.2f}" y1="{margin_top + plot_h}" x2="{px:.2f}" y2="{margin_top + plot_h + 5}" stroke="#243042" stroke-width="1"/>')
        elements.append(f'<text x="{px:.2f}" y="{margin_top + plot_h + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{x}</text>')

    for label, (key, color) in series.items():
        points = [
            f"{sx(row['target_paragraph_chars']):.2f},{sy(row['avg_tokens'][key]):.2f}"
            for row in rows
        ]
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for row in rows:
            x = sx(row["target_paragraph_chars"])
            y = sy(row["avg_tokens"][key])
            elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')

    elements.append(f'<text x="{margin_left + plot_w / 2}" y="{height - 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#374151">Target paragraph length (characters)</text>')
    elements.append(f'<text x="18" y="{margin_top + plot_h / 2}" transform="rotate(-90 18 {margin_top + plot_h / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#374151">Average tokens</text>')

    legend_x = margin_left + 28
    legend_y = margin_top + 18
    for index, (label, (_key, color)) in enumerate(series.items()):
        y = legend_y + index * 22
        elements.append(f'<rect x="{legend_x}" y="{y - 10}" width="14" height="4" fill="{color}"/>')
        elements.append(f'<text x="{legend_x + 22}" y="{y - 5}" font-family="Arial, sans-serif" font-size="13" fill="#1f2937">{label}</text>')

    elements.append("</svg>")
    return "\n".join(elements)


def nice_ticks(max_y: float, *, count: int) -> list[float]:
    if max_y <= 0:
        return [0]
    step = max_y / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(step))
    normalized = step / magnitude
    if normalized <= 1:
        nice_step = magnitude
    elif normalized <= 2:
        nice_step = 2 * magnitude
    elif normalized <= 5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude
    top = math.ceil(max_y / nice_step) * nice_step
    return [nice_step * index for index in range(int(top / nice_step) + 1)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
