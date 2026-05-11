from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humem_product import (  # noqa: E402
    CollectionSchema,
    EventMemoryDB,
    EventRAG,
    LayeredMemoryRAG,
    RetrievalPlan,
    RetrievalTargetSlot,
)
from humem_product.storage import load_event_rag, migrate_v3_to_event_rag, save_event_rag  # noqa: E402
from humem_product.cli import _event_answer_to_dict  # noqa: E402
from humem_product.visualization import build_graph_data  # noqa: E402


class FakeEventLLMProvider:
    model = "fake-event-llm"

    def generate_json(self, messages: list[dict[str, str]], *, schema_name: str) -> dict[str, Any]:
        if schema_name == "event_extraction":
            return {
                "events": [
                    {
                        "main_label": "申请消费贷款买电脑",
                        "compressed_trace": "用户上个月因实习 offer 未确定暂缓申请消费贷买电脑，现在 offer 已确定，想重新看额度。",
                        "subtags": [
                            {
                                "role": "when",
                                "value": "上个月",
                                "position": 1,
                                "embedding_text": "时间: 上个月",
                                "confidence": 0.94,
                            },
                            {
                                "role": "who",
                                "value": "我",
                                "position": 1,
                                "embedding_text": "主语: 我",
                                "confidence": 0.93,
                            },
                            {
                                "role": "cause",
                                "value": "实习 offer 还没确定",
                                "position": 1,
                                "embedding_text": "原因: offer没确定",
                                "confidence": 0.98,
                            },
                            {
                                "role": "action",
                                "value": "暂时没申请消费贷款",
                                "position": 1,
                                "embedding_text": "动作: 没申请贷款",
                                "confidence": 0.97,
                            },
                            {
                                "role": "when",
                                "value": "现在",
                                "position": 2,
                                "embedding_text": "时间: 现在",
                                "confidence": 0.92,
                            },
                            {
                                "role": "state",
                                "value": "offer 已经下来了",
                                "position": 2,
                                "embedding_text": "状态: offer下来了",
                                "confidence": 0.98,
                            },
                            {
                                "role": "intent",
                                "value": "重新看看额度",
                                "position": 2,
                                "embedding_text": "意图: 看贷款额度",
                                "confidence": 0.94,
                            },
                        ],
                    }
                ]
            }
        if schema_name == "retrieval_plan":
            user_payload = messages[-1]["content"]
            if "为什么" in user_payload or "没申请" in user_payload:
                return {
                    "key_question": "为什么没申请贷款",
                    "retrieval_terms": ["贷款", "申请贷款", "消费贷款", "没申请贷款", "offer没确定"],
                    "target_roles": ["cause", "action", "state"],
                    "time_hint": "之前",
                    "recall_precision": 0.86,
                }
            return {
                "key_question": "贷款额度",
                "retrieval_terms": ["额度"],
                "target_roles": ["intent"],
                "time_hint": None,
                "recall_precision": 0.1,
            }
        raise AssertionError(schema_name)

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = json.loads(messages[-1]["content"])
        evidence = payload.get("evidence", [])
        if not evidence:
            return "没有召回相关事件记忆。"
        subtags = evidence[0].get("matched_subtags", [])
        cause = next((item["value"] for item in subtags if item["role"] == "cause"), None)
        if cause:
            return f"你之前没申请贷款，是因为{cause}。"
        return evidence[0]["compressed_trace"]


class RecordingEmbeddingProvider:
    model = "recording-embedding"

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.inputs.extend(texts)
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _embed(self, text: str) -> list[float]:
        normalized = text.lower()
        loan = 1.0 if any(term in normalized for term in ("贷款", "消费贷", "申请", "loan")) else 0.0
        offer = 1.0 if "offer" in normalized else 0.0
        quota = 1.0 if "额度" in normalized else 0.0
        if loan or offer or quota:
            return [loan, offer, quota]
        return [0.05, 0.02, 0.01]


class DirectionalEventLLMProvider:
    model = "directional-event-llm"

    def generate_json(self, messages: list[dict[str, str]], *, schema_name: str) -> dict[str, Any]:
        if schema_name == "event_extraction":
            return {
                "events": [
                    {
                        "main_label": "从北京到上海出差",
                        "compressed_trace": "用户从北京出发，到上海出差，并且和小王一起。",
                        "subtags": [
                            {
                                "role": "from_where",
                                "value": "北京",
                                "position": 1,
                                "embedding_text": "地点: 北京",
                                "confidence": 0.98,
                            },
                            {
                                "role": "to_where",
                                "value": "上海",
                                "position": 2,
                                "embedding_text": "地点: 上海",
                                "confidence": 0.98,
                            },
                            {
                                "role": "with_who",
                                "value": "小王",
                                "position": 2,
                                "embedding_text": "同行人: 小王",
                                "confidence": 0.95,
                            },
                            {
                                "role": "action",
                                "value": "出差",
                                "position": 2,
                                "embedding_text": "动作: 出差",
                                "confidence": 0.92,
                            },
                        ],
                    }
                ]
            }
        if schema_name == "retrieval_plan":
            query = messages[-1]["content"]
            if "从哪里" in query:
                return {
                    "key_question": "从哪里出发",
                    "retrieval_terms": ["地点"],
                    "target_roles": ["from_where"],
                    "time_hint": None,
                    "recall_precision": 0.9,
                }
            return {
                "key_question": "到哪里去",
                "retrieval_terms": ["地点"],
                "target_roles": ["to_where"],
                "time_hint": None,
                "recall_precision": 0.9,
            }
        raise AssertionError(schema_name)

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = json.loads(messages[-1]["content"])
        subtags = payload["evidence"][0]["matched_subtags"] if payload.get("evidence") else []
        if not subtags:
            return "没有相关地点记忆。"
        return subtags[0]["value"]


class DirectionalEmbeddingProvider:
    model = "directional-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        if any(term in text for term in ("地点", "北京", "上海")):
            return [1.0, 0.0, 0.0]
        if "小王" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class MultiPlaceEventLLMProvider:
    model = "multi-place-event-llm"

    def generate_json(self, messages: list[dict[str, str]], *, schema_name: str) -> dict[str, Any]:
        if schema_name == "event_extraction":
            return {
                "events": [
                    {
                        "main_label": "北京上海出差入住云杉酒店",
                        "compressed_trace": "用户昨天从北京出发，到上海出差，晚上入住云杉酒店。",
                        "subtags": [
                            {
                                "role": "when",
                                "value": "昨天",
                                "position": 1,
                                "embedding_text": "时间: 昨天",
                                "confidence": 0.96,
                            },
                            {
                                "role": "place",
                                "value": "北京",
                                "position": 1,
                                "embedding_text": "地点: 北京 出发",
                                "confidence": 0.98,
                            },
                            {
                                "role": "action",
                                "value": "出发",
                                "position": 1,
                                "embedding_text": "动作: 出发",
                                "confidence": 0.96,
                            },
                            {
                                "role": "place",
                                "value": "上海",
                                "position": 2,
                                "embedding_text": "地点: 上海 出差",
                                "confidence": 0.98,
                            },
                            {
                                "role": "action",
                                "value": "出差",
                                "position": 2,
                                "embedding_text": "动作: 出差",
                                "confidence": 0.96,
                            },
                            {
                                "role": "place",
                                "value": "云杉酒店",
                                "position": 3,
                                "embedding_text": "地点: 云杉酒店 入住 住宿 酒店",
                                "confidence": 0.99,
                            },
                            {
                                "role": "action",
                                "value": "入住",
                                "position": 3,
                                "embedding_text": "动作: 入住",
                                "confidence": 0.95,
                            },
                        ],
                    }
                ]
            }
        if schema_name == "retrieval_plan":
            query = messages[-1]["content"]
            if "住" in query or "酒店" in query:
                return {
                    "key_question": "住在哪里",
                    "retrieval_terms": ["住宿", "酒店"],
                    "target_roles": ["place"],
                    "target_slots": [{"role": "place", "position": 3}],
                    "time_hint": None,
                    "recall_precision": 0.92,
                }
            if "出发" in query or "从哪里" in query:
                return {
                    "key_question": "从哪里出发",
                    "retrieval_terms": ["出发", "地点"],
                    "target_roles": ["place"],
                    "target_slots": [{"role": "place", "position": 1}],
                    "time_hint": None,
                    "recall_precision": 0.92,
                }
            return {
                "key_question": "到哪里出差",
                "retrieval_terms": ["出差", "地点"],
                "target_roles": ["place"],
                "target_slots": [{"role": "place", "position": 2}],
                "time_hint": None,
                "recall_precision": 0.92,
            }

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = json.loads(messages[-1]["content"])
        evidence = payload.get("evidence", [])
        if not evidence:
            return "没有相关地点记忆。"
        slots = evidence[0].get("matched_slots", [])
        for slot in slots:
            if slot.get("matches"):
                return slot["matches"][0]["value"]
        return evidence[0]["matched_subtags"][0]["value"]


class MultiPlaceEmbeddingProvider:
    model = "multi-place-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        place = 1.0 if any(term in text for term in ("地点", "北京", "上海", "云杉", "酒店", "住宿")) else 0.0
        depart = 1.0 if "出发" in text or "北京" in text else 0.0
        lodge = 1.0 if any(term in text for term in ("住", "住宿", "酒店", "入住", "云杉")) else 0.0
        trip = 1.0 if "出差" in text or "上海" in text else 0.0
        if place or depart or lodge or trip:
            return [place, depart, lodge, trip]
        return [0.01, 0.01, 0.01, 0.01]


class RetryRoleLLMProvider:
    model = "retry-role-llm"

    def __init__(self) -> None:
        self.extraction_calls = 0

    def generate_json(self, messages: list[dict[str, str]], *, schema_name: str) -> dict[str, Any]:
        if schema_name == "event_extraction":
            self.extraction_calls += 1
            role = "object" if self.extraction_calls == 1 else "cause"
            return {
                "events": [
                    {
                        "main_label": "schema retry event",
                        "compressed_trace": "schema retry compressed trace",
                        "subtags": [
                            {
                                "role": role,
                                "value": "offer pending",
                                "position": 1,
                                "embedding_text": f"{role}: offer pending",
                                "confidence": 0.9,
                            }
                        ],
                    }
                ]
            }
        if schema_name == "retrieval_plan":
            return {
                "key_question": "why",
                "retrieval_terms": ["offer pending"],
                "target_roles": ["cause"],
                "time_hint": None,
                "recall_precision": 1.0,
            }
        raise AssertionError(schema_name)

    def complete(self, messages: list[dict[str, str]]) -> str:
        return "schema retry answer"


class EventRAGTests(unittest.TestCase):
    def test_loan_event_answer_uses_ordered_cause_subtag(self) -> None:
        rag = _build_rag()
        rag.remember(
            "我上个月本来打算申请一笔消费贷买电脑，但是后来因为实习 offer 还没确定，所以暂时没申请。现在 offer 已经下来了，我想重新看看额度。",
            captured_at="2026-05-11T00:00:00+00:00",
        )

        answer = rag.answer("我之前为什么没申请贷款？", limit=3, now="2026-05-11T00:00:00+00:00")

        self.assertIn("offer", answer.answer)
        self.assertIn("没确定", answer.answer)
        self.assertEqual(answer.diagnostics["context_uses_raw_source"], False)
        self.assertEqual(answer.evidence[0].main_label, "申请消费贷款买电脑")
        self.assertIn("cause", {subtag.role for subtag in answer.evidence[0].matched_subtags})

    def test_embedding_inputs_exclude_full_source_text(self) -> None:
        embedding = RecordingEmbeddingProvider()
        rag = EventRAG(llm_provider=FakeEventLLMProvider(), embedding_provider=embedding)
        source_text = "我上个月本来打算申请一笔消费贷买电脑，但是后来因为实习 offer 还没确定，所以暂时没申请。现在 offer 已经下来了，我想重新看看额度。"

        rag.remember(source_text)

        self.assertNotIn(source_text, embedding.inputs)
        self.assertIn("申请消费贷款买电脑", embedding.inputs)
        self.assertIn("原因: offer没确定", embedding.inputs)
        self.assertIn("动作: 没申请贷款", embedding.inputs)
        self.assertTrue(all("暂时没申请。现在" not in item for item in embedding.inputs))

    def test_old_memory_requires_more_precise_recall_terms(self) -> None:
        rag = _build_rag()
        rag.remember(
            "我上个月本来打算申请一笔消费贷买电脑，但是后来因为实习 offer 还没确定，所以暂时没申请。现在 offer 已经下来了，我想重新看看额度。",
            captured_at="2025-01-01T00:00:00+00:00",
        )

        broad = RetrievalPlan(
            key_question="贷款额度",
            retrieval_terms=["额度"],
            target_roles=["intent"],
            recall_precision=0.0,
        )
        precise = RetrievalPlan(
            key_question="为什么没申请贷款",
            retrieval_terms=["申请消费贷款", "offer没确定", "没申请贷款"],
            target_roles=["cause", "action"],
            recall_precision=1.0,
        )

        broad_hits = rag.retrieve_with_plan(broad, mutate=False, now="2026-05-11T00:00:00+00:00")
        precise_hits = rag.retrieve_with_plan(precise, mutate=False, now="2026-05-11T00:00:00+00:00")

        self.assertEqual(broad_hits, [])
        self.assertEqual(len(precise_hits), 1)
        self.assertGreater(precise_hits[0].recall_difficulty, 0.9)

    def test_positive_feedback_reduces_recall_difficulty(self) -> None:
        rag = _build_rag()
        event_id = rag.remember(
            "我上个月本来打算申请一笔消费贷买电脑，但是后来因为实习 offer 还没确定，所以暂时没申请。现在 offer 已经下来了，我想重新看看额度。",
            captured_at="2026-04-01T00:00:00+00:00",
        )[0]
        event = rag.events[event_id]
        before = rag.refresh_recall_state(event, now="2026-05-11T00:00:00+00:00")

        rag.apply_feedback(
            positive_event_ids=[event_id],
            reason="unit_test",
            now="2026-05-11T00:00:00+00:00",
        )
        after = rag.refresh_recall_state(event, now="2026-05-11T00:00:00+00:00")

        self.assertLess(after, before)

    def test_json_and_sqlite_round_trip_preserve_event_memory(self) -> None:
        rag = _build_rag()
        rag.remember("我上个月本来打算申请一笔消费贷买电脑，但是后来因为实习 offer 还没确定，所以暂时没申请。现在 offer 已经下来了，我想重新看看额度。")

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "memory-events.json"
            sqlite_path = Path(temp_dir) / "memory-events.db"
            save_event_rag(json_path, rag)
            save_event_rag(sqlite_path, rag)

            from_json = load_event_rag(
                json_path,
                llm_provider=FakeEventLLMProvider(),
                embedding_provider=RecordingEmbeddingProvider(),
                require_providers=True,
            )
            from_sqlite = load_event_rag(
                sqlite_path,
                llm_provider=FakeEventLLMProvider(),
                embedding_provider=RecordingEmbeddingProvider(),
                require_providers=True,
            )

        self.assertEqual(len(from_json.events), 1)
        self.assertEqual(len(from_sqlite.events), 1)
        self.assertEqual(from_json.stats()["version"], 4)
        self.assertEqual(from_sqlite.stats()["embedded_items"], from_json.stats()["embedded_items"])

    def test_v3_migration_reextracts_chunks_into_event_store(self) -> None:
        legacy = LayeredMemoryRAG()
        legacy.add_document(
            "我上个月本来打算申请一笔消费贷买电脑，但是后来因为实习 offer 还没确定，所以暂时没申请。现在 offer 已经下来了，我想重新看看额度。",
            document_id="loan",
            title="Loan Memory",
            cool_down_cycles=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "legacy.json"
            target = Path(temp_dir) / "events.json"
            legacy.save(source)
            migrated = migrate_v3_to_event_rag(
                source,
                target,
                llm_provider=FakeEventLLMProvider(),
                embedding_provider=RecordingEmbeddingProvider(),
            )
            payload = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(payload["version"], 4)
        self.assertEqual(len(migrated.events), 1)
        event = next(iter(migrated.events.values()))
        self.assertEqual(event.source["metadata"]["legacy_store_version"], 3)

    def test_event_visualization_uses_time_axis_and_embedding_plane(self) -> None:
        rag = _build_rag()
        rag.remember("first loan event", captured_at="2026-04-01T00:00:00+00:00")
        rag.remember("second loan event", captured_at="2026-05-01T00:00:00+00:00")

        graph = build_graph_data(rag)

        self.assertEqual(graph["meta"]["storeVersion"], 4)
        self.assertEqual(graph["meta"]["timeAxis"], "captured_at")
        self.assertEqual(graph["meta"]["layoutModel"], "time-embedding-plane")
        event_nodes = [node for node in graph["nodes"] if node["kind"] == "event"]
        self.assertEqual(len(event_nodes), 2)
        self.assertGreater(max(node["z"] for node in event_nodes), min(node["z"] for node in event_nodes))

    def test_cli_stats_reads_v4_store_without_providers(self) -> None:
        rag = _build_rag()
        rag.remember("loan event")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.json"
            save_event_rag(path, rag)
            result = subprocess.run(
                [sys.executable, "-m", "humem_product.cli", "stats", "--store", str(path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(payload["version"], 4)
        self.assertEqual(payload["events"], 1)

    def test_event_rag_requires_providers_for_runtime_operations(self) -> None:
        with self.assertRaises(ValueError):
            EventRAG()

    def test_event_ann_matches_exact_core_recall(self) -> None:
        exact = _build_rag(semantic_index="exact")
        ann = _build_rag(semantic_index="ann", semantic_index_min_items=1)
        for index in range(5):
            text = f"loan event {index}"
            exact.remember(text)
            ann.remember(text)
        plan = RetrievalPlan(
            key_question="为什么没申请贷款",
            retrieval_terms=["申请消费贷款", "offer没确定", "没申请贷款"],
            target_roles=["cause", "action"],
            recall_precision=1.0,
        )

        exact_hits = exact.retrieve_with_plan(plan, mutate=False)
        ann_hits = ann.retrieve_with_plan(plan, mutate=False)

        self.assertGreater(len(ann_hits), 0)
        self.assertEqual(ann.semantic_navigation_stats()["last_strategy"], "ann")
        self.assertEqual(exact_hits[0].main_label, ann_hits[0].main_label)

    def test_role_aware_ann_preserves_from_to_direction(self) -> None:
        rag = _build_directional_rag()
        rag.remember("我从北京到上海出差，和小王一起。")

        from_answer = rag.answer("我从哪里出发？")
        to_answer = rag.answer("我到哪里去？")

        self.assertEqual(from_answer.answer, "北京")
        self.assertEqual(to_answer.answer, "上海")
        self.assertEqual(from_answer.evidence[0].matched_subtags[0].role, "from_where")
        self.assertEqual(to_answer.evidence[0].matched_subtags[0].role, "to_where")
        self.assertEqual(rag.semantic_navigation_stats()["last_strategy"], "ann")

    def test_slot_aware_retrieval_disambiguates_repeated_place_role(self) -> None:
        rag = _build_multi_place_rag(semantic_index="ann")
        rag.remember("我昨天从北京出发去上海出差，晚上住在云杉酒店。")

        lodging = rag.answer("我住在哪里？")
        origin = rag.answer("我从哪里出发？")

        self.assertEqual(lodging.answer, "云杉酒店")
        self.assertEqual(origin.answer, "北京")
        self.assertEqual(lodging.retrieval_plan.target_slots[0].role, "place")
        self.assertEqual(lodging.retrieval_plan.target_slots[0].position, 3)
        self.assertEqual(lodging.evidence[0].matched_subtags[0].value, "云杉酒店")
        self.assertEqual(origin.evidence[0].matched_subtags[0].value, "北京")
        self.assertEqual(lodging.evidence[0].matched_slots[0]["matches"][0]["value"], "云杉酒店")
        self.assertIn(3, {stage["position"] for stage in lodging.evidence[0].stages})
        self.assertEqual(rag.semantic_navigation_stats()["last_strategy"], "ann")

    def test_slot_aware_semantic_recall_matches_exact_and_ann(self) -> None:
        exact = _build_multi_place_rag(semantic_index="exact")
        ann = _build_multi_place_rag(semantic_index="ann")
        exact.remember("我昨天从北京出发去上海出差，晚上住在云杉酒店。")
        ann.remember("我昨天从北京出发去上海出差，晚上住在云杉酒店。")
        plan = RetrievalPlan(
            key_question="住宿地点",
            retrieval_terms=["住宿", "酒店"],
            target_roles=["place"],
            target_slots=[RetrievalTargetSlot(role="place", position=3)],
            recall_precision=1.0,
        )

        exact_hits = exact.retrieve_with_plan(plan, mutate=False)
        ann_hits = ann.retrieve_with_plan(plan, mutate=False)

        self.assertEqual(exact_hits[0].matched_subtags[0].value, "云杉酒店")
        self.assertEqual(ann_hits[0].matched_subtags[0].value, "云杉酒店")
        self.assertEqual(ann.semantic_navigation_stats()["last_strategy"], "ann")

    def test_cli_event_answer_dict_includes_slot_and_stage_fields(self) -> None:
        rag = _build_multi_place_rag(semantic_index="ann")
        rag.remember("我昨天从北京出发去上海出差，晚上住在云杉酒店。")
        answer = rag.answer("我住在哪里？")

        payload = _event_answer_to_dict(answer)

        self.assertEqual(payload["retrieval_plan"]["target_roles"], ["place"])
        self.assertEqual(payload["retrieval_plan"]["target_slots"], [{"role": "place", "position": 3}])
        self.assertIn("matched_subtags", payload["evidence"][0])
        self.assertEqual(payload["evidence"][0]["matched_slots"][0]["matches"][0]["value"], "云杉酒店")
        self.assertEqual(payload["evidence"][0]["stages"][2]["position"], 3)

    def test_sqlite_persists_and_restores_event_ann_index(self) -> None:
        rag = _build_rag(semantic_index="ann", semantic_index_min_items=1)
        rag.remember("loan event")
        build_result = rag.build_semantic_indexes(force=True)
        self.assertIn("global", build_result["built"])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.db"
            save_event_rag(path, rag)
            restored = load_event_rag(
                path,
                llm_provider=FakeEventLLMProvider(),
                embedding_provider=RecordingEmbeddingProvider(),
                require_providers=True,
            )
            stats = restored.semantic_navigation_stats()
            plan = RetrievalPlan(
                key_question="为什么没申请贷款",
                retrieval_terms=["申请消费贷款", "offer没确定", "没申请贷款"],
                target_roles=["cause", "action"],
                recall_precision=1.0,
            )
            hits = restored.retrieve_with_plan(plan, mutate=False)
            after = restored.semantic_navigation_stats()

        self.assertIn("global", stats["persisted_index_names"])
        self.assertEqual(stats["build_count"], 0)
        self.assertGreater(len(hits), 0)
        self.assertEqual(after["last_strategy"], "ann")
        self.assertEqual(after["build_count"], 0)

    def test_event_ann_incrementally_adds_new_memory(self) -> None:
        rag = _build_rag(semantic_index="ann", semantic_index_min_items=1)
        rag.remember("loan event")
        rag.build_semantic_indexes(force=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.db"
            save_event_rag(path, rag)
            restored = load_event_rag(
                path,
                llm_provider=FakeEventLLMProvider(),
                embedding_provider=RecordingEmbeddingProvider(),
                require_providers=True,
            )
            self.assertIn("global", restored.semantic_navigation_stats()["persisted_index_names"])
            restored.remember("new loan event")
            self.assertEqual(restored.semantic_navigation_stats()["persisted_index_names"], [])
            build_count_after_remember = restored.semantic_navigation_stats()["build_count"]
            plan = RetrievalPlan(
                key_question="为什么没申请贷款",
                retrieval_terms=["申请消费贷款", "offer没确定", "没申请贷款"],
                target_roles=["cause", "action"],
                recall_precision=1.0,
            )
            restored.retrieve_with_plan(plan, mutate=False)

        self.assertEqual(restored.semantic_navigation_stats()["build_count"], build_count_after_remember)

    def test_event_auto_uses_exact_below_ann_threshold(self) -> None:
        rag = _build_rag(semantic_index="auto", semantic_index_min_items=999)
        rag.remember("loan event")
        plan = RetrievalPlan(
            key_question="为什么没申请贷款",
            retrieval_terms=["申请消费贷款"],
            target_roles=["cause"],
            recall_precision=1.0,
        )

        rag.retrieve_with_plan(plan, mutate=False)

        self.assertEqual(rag.semantic_navigation_stats()["last_strategy"], "exact")

    def test_collection_filter_isolates_event_recall(self) -> None:
        rag = _build_rag(semantic_index="exact")
        finance_ids = rag.remember("same loan memory", collection="finance", metadata={"tenant": "a"})
        rag.remember("same loan memory", collection="travel", metadata={"tenant": "b"})
        plan = RetrievalPlan(
            key_question="why loan",
            retrieval_terms=["loan", "offer"],
            target_roles=["cause"],
            recall_precision=1.0,
        )

        hits = rag.retrieve_with_plan(plan, mutate=False, collection="finance")

        self.assertGreater(len(hits), 0)
        self.assertEqual({hit.collection_id for hit in hits}, {"finance"})
        self.assertEqual({hit.event_id for hit in hits}, set(finance_ids))

    def test_collection_schema_retries_invalid_roles(self) -> None:
        llm = RetryRoleLLMProvider()
        rag = EventRAG(llm_provider=llm, embedding_provider=RecordingEmbeddingProvider())
        rag.create_collection(
            "causes",
            collection_id="causes",
            schema=CollectionSchema(allowed_roles=["cause"], required_roles=["cause"]),
        )

        event_ids = rag.remember("schema controlled memory", collection="causes")

        self.assertEqual(llm.extraction_calls, 2)
        self.assertEqual(rag.events[event_ids[0]].subtags[0].role, "cause")
        self.assertEqual(rag.events[event_ids[0]].collection_id, "causes")

    def test_filter_dsl_matches_metadata_time_role_and_recall(self) -> None:
        rag = _build_rag(semantic_index="exact")
        kept = rag.remember(
            "loan memory one",
            metadata={"user_id": "u1"},
            captured_at="2026-05-10T00:00:00+00:00",
        )
        rag.remember(
            "loan memory two",
            metadata={"user_id": "u2"},
            captured_at="2025-01-01T00:00:00+00:00",
        )
        plan = RetrievalPlan(
            key_question="why loan",
            retrieval_terms=["loan", "offer"],
            target_roles=["cause"],
            recall_precision=1.0,
        )

        hits = rag.retrieve_with_plan(
            plan,
            mutate=False,
            now="2026-05-11T00:00:00+00:00",
            filter={
                "roles": ["cause"],
                "where": {
                    "metadata.user_id": {"eq": "u1"},
                    "event.captured_at": {"gte": "2026-01-01T00:00:00+00:00"},
                },
                "recall": {"max_difficulty": 0.8},
            },
        )

        self.assertEqual({hit.event_id for hit in hits}, set(kept))

    def test_soft_delete_hides_event_and_compact_purges_it(self) -> None:
        rag = _build_rag(semantic_index="ann", semantic_index_min_items=1)
        event_ids = rag.remember("loan event")
        rag.build_semantic_indexes(force=True)

        delete_result = rag.delete_event(event_ids[0])
        hits = rag.retrieve_with_plan(
            RetrievalPlan(
                key_question="why loan",
                retrieval_terms=["loan", "offer"],
                target_roles=["cause"],
                recall_precision=1.0,
            ),
            mutate=False,
        )
        compact_result = rag.compact(purge_deleted=True)

        self.assertEqual(delete_result.deleted, event_ids)
        self.assertTrue(delete_result.ann_dirty)
        self.assertEqual(hits, [])
        self.assertEqual(compact_result.compacted, 1)
        self.assertEqual(rag.stats()["total_events"], 0)

    def test_replace_event_hides_old_and_preserves_replacement_metadata(self) -> None:
        rag = _build_rag()
        old_ids = rag.remember("old loan event", collection="finance")

        new_ids = rag.replace_event(old_ids[0], "new loan event")

        self.assertEqual(rag.events[old_ids[0]].status, "deleted")
        self.assertEqual(rag.events[new_ids[0]].metadata["replaces_event_id"], old_ids[0])
        self.assertEqual(rag.events[new_ids[0]].collection_id, "finance")

    def test_backup_restore_and_concurrent_read_loads(self) -> None:
        rag = _build_rag(semantic_index="ann", semantic_index_min_items=1)
        rag.remember("loan event", collection="finance")
        rag.build_semantic_indexes(force=True, collection="finance")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.db"
            backup_path = Path(temp_dir) / "backup.db"
            save_event_rag(path, rag)
            db = EventMemoryDB(path)

            result = db.backup(backup_path)
            restored = load_event_rag(backup_path)
            errors: list[Exception] = []

            def read_stats() -> None:
                try:
                    self.assertEqual(db.load().stats()["events"], 1)
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

            threads = [threading.Thread(target=read_stats) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(result.backed_up, str(backup_path))
        self.assertEqual(restored.stats()["events"], 1)
        self.assertEqual(errors, [])

    def test_cli_index_persists_event_ann_status(self) -> None:
        rag = _build_rag(semantic_index="ann", semantic_index_min_items=1)
        rag.remember("loan event")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.db"
            save_event_rag(path, rag)
            index_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "humem_product.cli",
                    "index",
                    "--store",
                    str(path),
                    "--semantic-index",
                    "ann",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            stats_result = subprocess.run(
                [sys.executable, "-m", "humem_product.cli", "stats", "--store", str(path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            index_payload = json.loads(index_result.stdout)
            stats_payload = json.loads(stats_result.stdout)

        self.assertIn("global", index_payload["result"]["built"])
        self.assertIn("global", stats_payload["semantic_navigation"]["persisted_index_names"])

    def test_cli_collection_delete_compact_backup_flow(self) -> None:
        rag = _build_rag(semantic_index="ann", semantic_index_min_items=1)
        event_ids = rag.remember("loan event", collection="finance")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.db"
            backup_path = Path(temp_dir) / "backup.db"
            save_event_rag(path, rag)

            create_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "humem_product.cli",
                    "collection",
                    "create",
                    "--store",
                    str(path),
                    "cli",
                    "--schema-json",
                    json.dumps({"allowed_roles": ["cause"], "required_roles": []}),
                    "--json",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "humem_product.cli",
                    "index",
                    "--store",
                    str(path),
                    "--collection",
                    "finance",
                    "--json",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "humem_product.cli",
                    "delete",
                    "--store",
                    str(path),
                    "--event-id",
                    event_ids[0],
                    "--json",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
                capture_output=True,
                text=True,
                check=True,
            )
            compact_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "humem_product.cli",
                    "compact",
                    "--store",
                    str(path),
                    "--purge-deleted",
                    "--json",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
                capture_output=True,
                text=True,
                check=True,
            )
            backup_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "humem_product.cli",
                    "backup",
                    "--store",
                    str(path),
                    "--to",
                    str(backup_path),
                    "--json",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
                capture_output=True,
                text=True,
                check=True,
            )

        self.assertEqual(json.loads(create_result.stdout)["collection"]["collection_id"], "cli")
        self.assertEqual(json.loads(compact_result.stdout)["compacted"], 1)
        self.assertEqual(json.loads(backup_result.stdout)["backed_up"], str(backup_path))


def _build_rag(
    *,
    semantic_index: str = "auto",
    semantic_index_min_items: int = 256,
) -> EventRAG:
    return EventRAG(
        llm_provider=FakeEventLLMProvider(),
        embedding_provider=RecordingEmbeddingProvider(),
        semantic_index=semantic_index,
        semantic_index_min_items=semantic_index_min_items,
    )


def _build_directional_rag() -> EventRAG:
    return EventRAG(
        llm_provider=DirectionalEventLLMProvider(),
        embedding_provider=DirectionalEmbeddingProvider(),
        semantic_index="ann",
        semantic_index_min_items=1,
    )


def _build_multi_place_rag(*, semantic_index: str = "ann") -> EventRAG:
    return EventRAG(
        llm_provider=MultiPlaceEventLLMProvider(),
        embedding_provider=MultiPlaceEmbeddingProvider(),
        semantic_index=semantic_index,
        semantic_index_min_items=1,
    )


if __name__ == "__main__":
    unittest.main()
