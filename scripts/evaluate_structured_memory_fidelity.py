from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_SAMPLES = 2000
DEFAULT_SEED = 20260511


@dataclass(frozen=True, slots=True)
class QATask:
    query: str
    target_role: str
    target_position: int
    expected_value: str
    answer_prefix: str = ""


@dataclass(frozen=True, slots=True)
class SyntheticEvent:
    event_type: str
    original_text: str
    main_label: str
    compressed_trace: str
    subtags: list[dict[str, Any]]
    tasks: list[QATask]
    truth: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate structured-memory fidelity fixtures."
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "structured_memory_fidelity",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = build_dataset(sample_count=args.samples, rng=rng)
    predictions, metrics = evaluate_dataset(examples, seed=args.seed)

    dataset_path = output_dir / "dataset.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"

    write_jsonl(dataset_path, examples)
    write_jsonl(predictions_path, predictions)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(render_report(metrics), encoding="utf-8")

    print(f"dataset: {dataset_path}")
    print(f"predictions: {predictions_path}")
    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")


def build_dataset(*, sample_count: int, rng: random.Random) -> list[dict[str, Any]]:
    builders: list[Callable[[random.Random], SyntheticEvent]] = [
        build_loan_event,
        build_travel_event,
        build_project_event,
        build_health_event,
        build_purchase_event,
        build_meeting_event,
    ]
    examples: list[dict[str, Any]] = []
    for index in range(sample_count):
        event = builders[index % len(builders)](rng)
        task = rng.choice(event.tasks)
        noisy_subtags, noise_profile = add_extraction_noise(event.subtags, task, rng)
        examples.append(
            {
                "id": f"structured-fidelity-{index + 1:04d}",
                "event_type": event.event_type,
                "original_text": event.original_text,
                "main_label": event.main_label,
                "compressed_trace": event.compressed_trace,
                "subtags": event.subtags,
                "noisy_subtags": noisy_subtags,
                "noise_profile": noise_profile,
                "query": task.query,
                "target_role": task.target_role,
                "target_position": task.target_position,
                "expected_value": task.expected_value,
                "expected_answer": format_answer(task.answer_prefix, task.expected_value),
                "truth": event.truth,
            }
        )
    return examples


def evaluate_dataset(examples: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    modes = {
        "full_role_position": predict_full_role_position,
        "no_position": predict_no_position,
        "value_bag": predict_value_bag,
        "noisy_full_role_position": predict_noisy_full_role_position,
    }
    predictions: list[dict[str, Any]] = []
    counters: dict[str, Counter[str]] = {mode: Counter() for mode in modes}
    by_event: dict[str, Counter[str]] = defaultdict(Counter)
    by_role: dict[str, Counter[str]] = defaultdict(Counter)
    error_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for example in examples:
        for mode, predictor in modes.items():
            predicted_value, error_type = predictor(example)
            correct = normalize_value(predicted_value) == normalize_value(example["expected_value"])
            if correct:
                error_type = "correct"
            elif error_type == "correct":
                error_type = "wrong_value"
            counters[mode]["total"] += 1
            counters[mode]["correct"] += int(correct)
            if not correct:
                counters[mode][error_type] += 1

            event_key = f"{mode}|{example['event_type']}"
            role_key = f"{mode}|{example['target_role']}"
            by_event[event_key]["total"] += 1
            by_event[event_key]["correct"] += int(correct)
            by_role[role_key]["total"] += 1
            by_role[role_key]["correct"] += int(correct)

            row = {
                "id": example["id"],
                "mode": mode,
                "event_type": example["event_type"],
                "query": example["query"],
                "target_role": example["target_role"],
                "target_position": example["target_position"],
                "expected_value": example["expected_value"],
                "expected_answer": example["expected_answer"],
                "predicted_value": predicted_value,
                "predicted_answer": format_answer(answer_prefix_for(example), predicted_value),
                "correct": correct,
                "error_type": error_type,
            }
            predictions.append(row)
            if not correct and len(error_examples[mode]) < 8:
                error_examples[mode].append(
                    {
                        "id": example["id"],
                        "event_type": example["event_type"],
                        "query": example["query"],
                        "original_text": example["original_text"],
                        "expected_value": example["expected_value"],
                        "predicted_value": predicted_value,
                        "error_type": error_type,
                    }
                )

    metrics = {
        "sample_count": len(examples),
        "seed": seed,
        "modes": {
            mode: summarize_counter(counter)
            for mode, counter in counters.items()
        },
        "accuracy_by_event_type": summarize_grouped_counters(by_event),
        "accuracy_by_target_role": summarize_grouped_counters(by_role),
        "error_examples": error_examples,
        "interpretation": {
            "full_role_position": "Uses role and position. This estimates whether subtags preserve enough information.",
            "no_position": "Uses role only. This estimates damage from dropping event order/stage.",
            "value_bag": "Ignores role and position. This estimates a weak prompt that treats subtags as loose text.",
            "noisy_full_role_position": "Uses role and position after simulated extraction noise.",
        },
    }
    return predictions, metrics


def predict_full_role_position(example: dict[str, Any]) -> tuple[str, str]:
    return pick_subtag_value(
        example["subtags"],
        role=example["target_role"],
        position=example["target_position"],
    )


def predict_noisy_full_role_position(example: dict[str, Any]) -> tuple[str, str]:
    return pick_subtag_value(
        example["noisy_subtags"],
        role=example["target_role"],
        position=example["target_position"],
    )


def predict_no_position(example: dict[str, Any]) -> tuple[str, str]:
    return pick_subtag_value(
        example["subtags"],
        role=example["target_role"],
        position=None,
    )


def predict_value_bag(example: dict[str, Any]) -> tuple[str, str]:
    query = example["query"]
    role_priority = infer_roles_from_query(query)
    values = [
        (subtag["role"], subtag["position"], subtag["value"])
        for subtag in example["subtags"]
    ]
    for role in role_priority:
        for subtag_role, _position, value in values:
            if subtag_role == role:
                if role == example["target_role"]:
                    return value, "missing_position"
                return value, "wrong_role"
    return values[0][2] if values else "", "missing_target"


def pick_subtag_value(
    subtags: list[dict[str, Any]],
    *,
    role: str,
    position: int | None,
) -> tuple[str, str]:
    role_matches = [item for item in subtags if item.get("role") == role]
    if not role_matches:
        return "", "missing_target"
    if position is not None:
        position_matches = [item for item in role_matches if int(item.get("position", -1)) == position]
        if position_matches:
            return str(position_matches[0].get("value", "")), "correct"
        return str(role_matches[0].get("value", "")), "wrong_position"
    return str(role_matches[0].get("value", "")), "missing_position"


def add_extraction_noise(
    subtags: list[dict[str, Any]],
    task: QATask,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    noisy = [dict(item) for item in subtags]
    target_indexes = [
        index
        for index, item in enumerate(noisy)
        if item["role"] == task.target_role and item["position"] == task.target_position
    ]
    noise_profile = {"kind": "none", "target_affected": False}

    roll = rng.random()
    if roll < 0.05 and target_indexes:
        removed = noisy.pop(target_indexes[0])
        noise_profile = {"kind": "drop_target", "target_affected": True, "removed": removed}
    elif roll < 0.11 and target_indexes:
        index = target_indexes[0]
        old_value = noisy[index]["value"]
        noisy[index]["value"] = wrong_value_for_role(task.target_role, old_value, rng)
        noisy[index]["embedding_text"] = f"{role_label(task.target_role)}: {noisy[index]['value']}"
        noise_profile = {
            "kind": "corrupt_target_value",
            "target_affected": True,
            "old_value": old_value,
            "new_value": noisy[index]["value"],
        }
    elif roll < 0.15 and target_indexes:
        index = target_indexes[0]
        old_role = noisy[index]["role"]
        noisy[index]["role"] = rng.choice([role for role in ALL_ROLES if role != old_role])
        noisy[index]["embedding_text"] = f"{role_label(noisy[index]['role'])}: {noisy[index]['value']}"
        noise_profile = {
            "kind": "corrupt_target_role",
            "target_affected": True,
            "old_role": old_role,
            "new_role": noisy[index]["role"],
        }
    elif roll < 0.21 and noisy:
        index = rng.randrange(len(noisy))
        noisy[index]["position"] = 1 if int(noisy[index]["position"]) != 1 else 2
        noise_profile = {
            "kind": "position_noise",
            "target_affected": index in target_indexes,
        }
    return noisy, noise_profile


def build_loan_event(rng: random.Random) -> SyntheticEvent:
    person = rng.choice(NAMES)
    time = rng.choice(["上个月", "三月初", "周一晚上", "去年十二月"])
    item = rng.choice(["电脑", "相机", "课程费用", "搬家押金"])
    loan_type = rng.choice(["消费贷", "短期贷款", "分期额度"])
    cause = rng.choice(["实习 offer 还没确定", "收入证明还没开好", "银行卡流水不够稳定", "担心利率临时上调"])
    state = rng.choice(["offer 已经下来了", "收入证明已经补齐", "流水已经稳定三个月", "利率重新降下来了"])
    intent = rng.choice(["想重新看看额度", "准备重新提交申请", "想先比较两家银行", "打算等工资到账再申请"])
    original = f"{person}{time}本来想申请一笔{loan_type}买{item}，但因为{cause}，所以暂时没申请。现在{state}，{intent}。"
    subtags = [
        subtag("who", person, 1),
        subtag("when", time, 1),
        subtag("action", f"想申请{loan_type}买{item}", 1),
        subtag("object", item, 1),
        subtag("cause", cause, 1),
        subtag("outcome", "暂时没申请", 1),
        subtag("when", "现在", 2),
        subtag("state", state, 2),
        subtag("intent", intent, 2),
    ]
    tasks = [
        QATask("为什么没申请贷款？", "cause", 1, cause, "因为"),
        QATask("之前最后做了什么决定？", "outcome", 1, "暂时没申请"),
        QATask("现在想做什么？", "intent", 2, intent),
        QATask("要买什么？", "object", 1, item),
    ]
    return event("loan", original, f"申请{loan_type}买{item}", subtags, tasks)


def build_travel_event(rng: random.Random) -> SyntheticEvent:
    person = rng.choice(NAMES)
    time = rng.choice(["昨天早上", "五一假期", "周五下午", "四月二十号"])
    start = rng.choice(CITIES)
    end = rng.choice([city for city in CITIES if city != start])
    companion = rng.choice(NAMES)
    purpose = rng.choice(["出差", "看展", "参加面试", "探亲"])
    hotel = rng.choice(["云杉酒店", "河岸民宿", "北站公寓", "星河青旅"])
    original = f"{person}{time}从{start}出发去{end}{purpose}，和{companion}一起，晚上住在{hotel}。"
    subtags = [
        subtag("who", person, 1),
        subtag("when", time, 1),
        subtag("from_where", start, 1),
        subtag("to_where", end, 2),
        subtag("action", purpose, 2),
        subtag("with_who", companion, 2),
        subtag("place", hotel, 3),
    ]
    tasks = [
        QATask("从哪里出发？", "from_where", 1, start),
        QATask("到哪里去？", "to_where", 2, end),
        QATask("和谁一起？", "with_who", 2, companion),
        QATask("晚上住在哪里？", "place", 3, hotel),
    ]
    return event("travel", original, f"从{start}去{end}{purpose}", subtags, tasks)


def build_project_event(rng: random.Random) -> SyntheticEvent:
    person = rng.choice(NAMES)
    time = rng.choice(["今天午后", "周三评审会", "版本冻结前", "客户演示后"])
    project = rng.choice(["记忆系统", "支付后台", "数据看板", "客服助手"])
    decision = rng.choice(["先保留结构化标签", "暂停灰度发布", "把召回阈值调高", "优先修复过滤器"])
    cause = rng.choice(["线上误召回变多", "客户追问原因链", "指标波动太大", "审计需要可解释证据"])
    next_action = rng.choice(["明天补一组回归测试", "今晚写 ADR", "本周整理错误样本", "下次会前做压测"])
    original = f"{person}{time}讨论{project}时决定{decision}，原因是{cause}，下一步是{next_action}。"
    subtags = [
        subtag("who", person, 1),
        subtag("when", time, 1),
        subtag("object", project, 1),
        subtag("action", decision, 1),
        subtag("cause", cause, 1),
        subtag("intent", next_action, 2),
    ]
    tasks = [
        QATask("讨论的是哪个项目？", "object", 1, project),
        QATask("做了什么决定？", "action", 1, decision),
        QATask("为什么这么决定？", "cause", 1, cause, "因为"),
        QATask("下一步要做什么？", "intent", 2, next_action),
    ]
    return event("project", original, f"{project}决策", subtags, tasks)


def build_health_event(rng: random.Random) -> SyntheticEvent:
    person = rng.choice(NAMES)
    time = rng.choice(["上周六", "昨天下午", "三月体检后", "今天早上"])
    symptom = rng.choice(["胃痛反复", "睡眠很浅", "膝盖酸胀", "嗓子发炎"])
    hospital = rng.choice(["南山门诊", "仁和医院", "校医院", "社区诊所"])
    advice = rng.choice(["先观察三天", "饭后吃药", "减少跑步量", "复查血常规"])
    medicine_place = rng.choice(["书桌左边抽屉", "厨房白色药盒", "背包夹层", "床头柜第二层"])
    original = f"{person}{time}因为{symptom}去{hospital}看诊，医生建议{advice}，药放在{medicine_place}。"
    subtags = [
        subtag("who", person, 1),
        subtag("when", time, 1),
        subtag("cause", symptom, 1),
        subtag("place", hospital, 1),
        subtag("action", advice, 2),
        subtag("place", medicine_place, 3),
    ]
    tasks = [
        QATask("为什么去看诊？", "cause", 1, symptom, "因为"),
        QATask("去哪里看诊？", "place", 1, hospital),
        QATask("医生建议什么？", "action", 2, advice),
        QATask("药放在哪里？", "place", 3, medicine_place),
    ]
    return event("health", original, f"{symptom}看诊", subtags, tasks)


def build_purchase_event(rng: random.Random) -> SyntheticEvent:
    person = rng.choice(NAMES)
    time = rng.choice(["上周", "昨天晚上", "双十一前", "周日"])
    product = rng.choice(["降噪耳机", "人体工学椅", "移动硬盘", "显示器"])
    platform = rng.choice(["京东", "淘宝", "苹果官网", "拼多多"])
    cause = rng.choice(["旧设备坏了", "价格降到预算内", "工作需要双屏", "朋友推荐这一款"])
    outcome = rng.choice(["先加入购物车", "已经下单", "取消了订单", "等优惠券再买"])
    original = f"{person}{time}在{platform}看了{product}，原因是{cause}，最后{outcome}。"
    subtags = [
        subtag("who", person, 1),
        subtag("when", time, 1),
        subtag("place", platform, 1),
        subtag("object", product, 1),
        subtag("cause", cause, 1),
        subtag("outcome", outcome, 2),
    ]
    tasks = [
        QATask("看了什么东西？", "object", 1, product),
        QATask("在哪个平台看的？", "place", 1, platform),
        QATask("为什么想买？", "cause", 1, cause, "因为"),
        QATask("最后怎么处理？", "outcome", 2, outcome),
    ]
    return event("purchase", original, f"考虑购买{product}", subtags, tasks)


def build_meeting_event(rng: random.Random) -> SyntheticEvent:
    person = rng.choice(NAMES)
    time = rng.choice(["今天十点", "周二例会", "月底复盘时", "上线前一天"])
    place = rng.choice(["三号会议室", "飞书会议", "客户办公室", "白板区"])
    topic = rng.choice(["续费风险", "模型成本", "权限审批", "发布节奏"])
    action = rng.choice(["记录了三个待办", "要求先做小流量验证", "决定延后上线", "让产品补充说明"])
    owner = rng.choice(NAMES)
    original = f"{person}{time}在{place}开会讨论{topic}，会上{action}，负责人是{owner}。"
    subtags = [
        subtag("who", person, 1),
        subtag("when", time, 1),
        subtag("place", place, 1),
        subtag("object", topic, 1),
        subtag("action", action, 2),
        subtag("who", owner, 2),
    ]
    tasks = [
        QATask("在哪里开会？", "place", 1, place),
        QATask("讨论什么主题？", "object", 1, topic),
        QATask("会上做了什么？", "action", 2, action),
        QATask("负责人是谁？", "who", 2, owner),
    ]
    return event("meeting", original, f"会议讨论{topic}", subtags, tasks)


def event(
    event_type: str,
    original: str,
    main_label: str,
    subtags: list[dict[str, Any]],
    tasks: list[QATask],
) -> SyntheticEvent:
    truth = {
        f"{item['role']}@{item['position']}": item["value"]
        for item in subtags
    }
    compressed_trace = "；".join(
        f"{role_label(item['role'])}:{item['value']}"
        for item in subtags
        if item["role"] in {"cause", "action", "outcome", "intent", "from_where", "to_where", "place"}
    )
    return SyntheticEvent(
        event_type=event_type,
        original_text=original,
        main_label=main_label,
        compressed_trace=compressed_trace,
        subtags=subtags,
        tasks=tasks,
        truth=truth,
    )


def subtag(role: str, value: str, position: int, confidence: float = 0.94) -> dict[str, Any]:
    return {
        "role": role,
        "value": value,
        "position": position,
        "embedding_text": f"{role_label(role)}: {value}",
        "confidence": confidence,
    }


def infer_roles_from_query(query: str) -> list[str]:
    if "为什么" in query or "原因" in query:
        return ["cause", "intent", "outcome", "action"]
    if "从哪里" in query or "出发" in query:
        return ["place", "from_where", "to_where"]
    if "到哪里" in query or "去哪里" in query:
        return ["place", "to_where", "from_where"]
    if "谁" in query:
        return ["who", "with_who"]
    if "哪里" in query or "在哪" in query:
        return ["place", "from_where", "to_where"]
    if "什么" in query and ("做" in query or "建议" in query):
        return ["action", "intent", "outcome"]
    if "什么" in query:
        return ["object", "action", "intent"]
    if "最后" in query:
        return ["outcome", "action", "intent"]
    return ["action", "cause", "object", "place", "who"]


def wrong_value_for_role(role: str, old_value: str, rng: random.Random) -> str:
    pools = {
        "who": NAMES,
        "with_who": NAMES,
        "from_where": CITIES,
        "to_where": CITIES,
        "place": PLACES,
        "object": OBJECTS,
        "cause": CAUSES,
        "action": ACTIONS,
        "intent": INTENTS,
        "outcome": OUTCOMES,
        "state": STATES,
        "when": TIMES,
    }
    candidates = [item for item in pools.get(role, OBJECTS) if item != old_value]
    return rng.choice(candidates or ["未知"])


def role_label(role: str) -> str:
    return {
        "when": "时间",
        "who": "人物",
        "action": "动作",
        "object": "对象",
        "place": "地点",
        "state": "状态",
        "cause": "原因",
        "outcome": "结果",
        "intent": "意图",
        "from_where": "出发地",
        "to_where": "目的地",
        "with_who": "同行人",
    }.get(role, role)


def answer_prefix_for(example: dict[str, Any]) -> str:
    return "因为" if example["target_role"] == "cause" else ""


def format_answer(prefix: str, value: str) -> str:
    if not value:
        return ""
    return f"{prefix}{value}" if prefix else value


def normalize_value(value: str) -> str:
    return "".join(str(value).strip().lower().split())


def summarize_counter(counter: Counter[str]) -> dict[str, Any]:
    total = int(counter["total"])
    correct = int(counter["correct"])
    errors = {
        key: int(value)
        for key, value in sorted(counter.items())
        if key not in {"total", "correct"}
    }
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "errors": errors,
    }


def summarize_grouped_counters(counters: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, counter in sorted(counters.items()):
        total = int(counter["total"])
        correct = int(counter["correct"])
        result[key] = {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
        }
    return result


def render_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Structured Memory Fidelity Evaluation",
        "",
        f"- Samples: {metrics['sample_count']}",
        f"- Seed: {metrics['seed']}",
        "",
        "## Accuracy By Simulation Mode",
        "",
        "| Mode | Accuracy | Correct / Total | Main Error Counts |",
        "| --- | ---: | ---: | --- |",
    ]
    for mode, payload in metrics["modes"].items():
        errors = ", ".join(
            f"{key}={value}"
            for key, value in payload["errors"].items()
            if key != "correct"
        )
        lines.append(
            f"| `{mode}` | {payload['accuracy']:.2%} | "
            f"{payload['correct']} / {payload['total']} | {errors or '-'} |"
        )

    lines.extend(
        [
            "",
            "## What This Means",
            "",
            "- `full_role_position` estimates whether the structured subtags themselves preserve the facts needed to answer.",
            "- `no_position` shows the loss from removing event order/stage when the same role appears more than once.",
            "- `value_bag` approximates a weak prompt where the model treats subtags as loose text and may confuse roles.",
            "- `noisy_full_role_position` estimates the combined risk of structured recall plus extraction errors.",
            "",
            "## Error Examples",
            "",
        ]
    )
    for mode, examples in metrics["error_examples"].items():
        lines.append(f"### {mode}")
        if not examples:
            lines.append("")
            lines.append("No errors found.")
            lines.append("")
            continue
        lines.append("")
        for item in examples[:5]:
            lines.append(
                f"- `{item['id']}` {item['query']} expected `{item['expected_value']}`, "
                f"predicted `{item['predicted_value']}` ({item['error_type']})."
            )
        lines.append("")

    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


NAMES = ["小林", "Maya", "阿哲", "陈晨", "李然", "周宁", "王珂", "苏晴"]
CITIES = ["北京", "上海", "深圳", "广州", "杭州", "成都", "南京", "厦门"]
TIMES = ["上个月", "三月初", "周一晚上", "昨天早上", "今天十点", "上周六"]
PLACES = [
    "云杉酒店",
    "河岸民宿",
    "南山门诊",
    "仁和医院",
    "三号会议室",
    "飞书会议",
    "京东",
    "淘宝",
]
OBJECTS = ["电脑", "相机", "课程费用", "搬家押金", "记忆系统", "支付后台", "降噪耳机", "显示器"]
CAUSES = ["实习 offer 还没确定", "收入证明还没开好", "线上误召回变多", "旧设备坏了", "价格降到预算内"]
ACTIONS = ["暂时没申请", "先保留结构化标签", "饭后吃药", "已经下单", "决定延后上线"]
INTENTS = ["想重新看看额度", "准备重新提交申请", "明天补一组回归测试", "本周整理错误样本"]
OUTCOMES = ["暂时没申请", "先加入购物车", "已经下单", "取消了订单", "等优惠券再买"]
STATES = ["offer 已经下来了", "收入证明已经补齐", "流水已经稳定三个月", "利率重新降下来了"]
ALL_ROLES = [
    "when",
    "who",
    "action",
    "object",
    "place",
    "state",
    "cause",
    "outcome",
    "intent",
    "from_where",
    "to_where",
    "with_who",
]


if __name__ == "__main__":
    main()
