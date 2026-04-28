from __future__ import annotations

import re
from collections.abc import Iterable

from .models import ParsedFragment, ParsedRelation

CLAUSE_SPLIT_RE = re.compile(r"[，,。！？；;]")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]{1,4}")

STOPWORDS = {
    "的",
    "了",
    "呢",
    "吗",
    "啊",
    "呀",
    "和",
    "与",
    "及",
    "在",
    "是",
    "有",
    "把",
    "将",
    "向",
    "对",
    "给",
    "被",
}

VERB_HINTS = {
    "是",
    "有",
    "喜欢",
    "讨厌",
    "爱",
    "告诉",
    "记得",
    "忘记",
    "看见",
    "看到",
    "认识",
    "帮助",
    "攻击",
    "提升",
    "下降",
    "移动",
    "连接",
    "强化",
    "提起",
    "检索",
    "召回",
    "学习",
    "训练",
    "保存",
    "更新",
    "击败",
    "创造",
    "设计",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def split_clauses(text: str) -> list[str]:
    clauses = [part.strip() for part in CLAUSE_SPLIT_RE.split(text) if part.strip()]
    return clauses or [text.strip()]


def tokenize_clause(clause: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(clause) if token.strip()]


def infer_kind(token: str) -> str:
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return "number"
    if any(ch.isascii() and ch.isalpha() for ch in token):
        return "term"
    if token in VERB_HINTS:
        return "action"
    if len(token) <= 1 or token in STOPWORDS:
        return "marker"
    return "concept"


def estimate_salience(token: str, kind: str) -> float:
    base = {
        "clause": 0.72,
        "action": 0.82,
        "concept": 0.65,
        "number": 0.6,
        "term": 0.58,
        "marker": 0.25,
    }.get(kind, 0.5)
    length_bonus = min(len(token) * 0.03, 0.18)
    return min(base + length_bonus, 0.95)


def _append_if_new(
    fragments: dict[tuple[str, str], ParsedFragment],
    fragment: ParsedFragment,
) -> None:
    key = (fragment.normalized_text, fragment.kind)
    if key not in fragments:
        fragments[key] = fragment


def _build_clause_relations(
    clause_key: tuple[str, str],
    token_keys: list[tuple[str, str]],
) -> Iterable[ParsedRelation]:
    for token_key in token_keys:
        yield ParsedRelation(
            source_key=clause_key,
            target_key=token_key,
            relation_type="contains",
            weight=0.78,
        )

    for index in range(len(token_keys) - 1):
        yield ParsedRelation(
            source_key=token_keys[index],
            target_key=token_keys[index + 1],
            relation_type="sequence",
            weight=0.46,
        )

    for left in range(len(token_keys)):
        for right in range(left + 1, len(token_keys)):
            yield ParsedRelation(
                source_key=token_keys[left],
                target_key=token_keys[right],
                relation_type="co_clause",
                weight=0.22,
            )


def _infer_grammar_relations(
    clause: str,
    token_keys: list[tuple[str, str]],
) -> Iterable[ParsedRelation]:
    if not token_keys:
        return []

    relations: list[ParsedRelation] = []
    tokens = [text for text, _kind in token_keys]

    for index, (_token_text, kind) in enumerate(token_keys):
        if kind != "action":
            continue

        if index > 0:
            relations.append(
                ParsedRelation(
                    source_key=token_keys[index - 1],
                    target_key=token_keys[index],
                    relation_type="subject_of",
                    weight=0.74,
                )
            )

        if index + 1 < len(token_keys):
            relations.append(
                ParsedRelation(
                    source_key=token_keys[index + 1],
                    target_key=token_keys[index],
                    relation_type="object_of",
                    weight=0.74,
                )
            )

        window_left = max(index - 1, 0)
        window_right = min(index + 2, len(token_keys))
        for related in range(window_left, window_right):
            if related == index:
                continue
            relations.append(
                ParsedRelation(
                    source_key=token_keys[related],
                    target_key=token_keys[index],
                    relation_type="action_of",
                    weight=0.56,
                )
            )

    if "被" in clause:
        passive_index = tokens.index("被") if "被" in tokens else None
        if passive_index is not None and passive_index > 0 and passive_index + 1 < len(token_keys):
            relations.append(
                ParsedRelation(
                    source_key=token_keys[passive_index - 1],
                    target_key=token_keys[passive_index + 1],
                    relation_type="passive_of",
                    weight=0.86,
                )
            )

    return relations


def parse_sentence(text: str) -> tuple[list[ParsedFragment], list[ParsedRelation]]:
    fragments: dict[tuple[str, str], ParsedFragment] = {}
    relations: list[ParsedRelation] = []

    for clause_index, clause in enumerate(split_clauses(text)):
        clause_fragment = ParsedFragment(
            text=clause,
            normalized_text=normalize_text(clause),
            kind="clause",
            salience=estimate_salience(clause, "clause"),
            metadata={"clause_index": clause_index},
        )
        _append_if_new(fragments, clause_fragment)
        clause_key = (clause_fragment.normalized_text, clause_fragment.kind)

        token_keys: list[tuple[str, str]] = []
        for token_index, token in enumerate(tokenize_clause(clause)):
            kind = infer_kind(token)
            if kind == "marker":
                continue

            parsed_fragment = ParsedFragment(
                text=token,
                normalized_text=normalize_text(token),
                kind=kind,
                salience=estimate_salience(token, kind),
                metadata={"clause_index": clause_index, "token_index": token_index},
            )
            _append_if_new(fragments, parsed_fragment)
            token_keys.append((parsed_fragment.normalized_text, parsed_fragment.kind))

        relations.extend(_build_clause_relations(clause_key, token_keys))
        relations.extend(_infer_grammar_relations(clause, token_keys))

    return list(fragments.values()), relations
