from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humem_product import LayeredMemoryRAG  # noqa: E402
from humem_product.memory_layout import apply_memory_layout  # noqa: E402
from humem_product.navigation import exact_navigation_hits  # noqa: E402
from humem_product.storage import load_rag, migrate_json_to_sqlite, save_rag  # noqa: E402
from humem_product.visualization import build_graph_data  # noqa: E402


class FakeEmbeddingProvider:
    model = "fake-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        launch_terms = ("robot", "launch", "checklist", "deployment", "plan")
        runbook_terms = ("incident", "response", "runbook")
        if any(term in lowered for term in launch_terms):
            return [1.0, 0.0, 0.0]
        if any(term in lowered for term in runbook_terms):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class LayeredMemoryRAGTests(unittest.TestCase):
    def test_document_ingestion_returns_answer_with_sources(self) -> None:
        rag = LayeredMemoryRAG()
        doc_id = rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook. "
            "The temporary launch code 45123789 appeared once on the receipt.",
            document_id="doc-1",
            title="Launch Notes",
            cool_down_cycles=0,
        )

        answer = rag.answer("robot launch checklist", limit=5)

        self.assertEqual(doc_id, "doc-1")
        self.assertGreater(len(answer.evidence), 0)
        self.assertIn("Launch Notes", {item.title for item in answer.evidence})
        self.assertIn("robot", answer.answer.lower())
        self.assertEqual(answer.diagnostics["fragment_count"], len(rag.space.fragments))

    def test_store_round_trip_preserves_memory(self) -> None:
        rag = LayeredMemoryRAG()
        rag.add_document(
            "Maya stores the incident response runbook in vault seven.",
            document_id="ops",
            title="Ops Runbook",
            cool_down_cycles=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            rag.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 3)
            self.assertEqual(payload["rag_config"]["retrieval_profile"]["name"], "balanced")
            self.assertIn("dynamics", payload["memory_space"]["config"])

            restored = LayeredMemoryRAG.load(path)
            answer = restored.answer("incident response", limit=3)

        self.assertEqual(len(restored.documents), 1)
        self.assertGreater(len(answer.evidence), 0)
        self.assertIn("runbook", answer.answer.lower())

    def test_retrieval_profile_is_persisted(self) -> None:
        rag = LayeredMemoryRAG(retrieval_profile="semantic")
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook.",
            document_id="launch",
            title="Launch Notes",
            cool_down_cycles=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            rag.save(path)
            restored = LayeredMemoryRAG.load(path)

        self.assertEqual(restored.retrieval_profile.name, "semantic")
        self.assertEqual(restored.memory_weight, 0.45)
        self.assertEqual(restored.embedding_weight, 0.55)

    def test_archival_profile_does_not_mutate_on_read(self) -> None:
        rag = LayeredMemoryRAG(retrieval_profile="archival")
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook.",
            document_id="launch",
            title="Launch Notes",
            cool_down_cycles=0,
        )
        target = next(
            fragment
            for fragment in rag.space.fragments.values()
            if fragment.normalized_text == "checklist"
        )
        before = (target.retrievals, target.activation, target.strength, target.layer)

        rag.answer("robot launch checklist", limit=3)

        after = (target.retrievals, target.activation, target.strength, target.layer)
        self.assertEqual(after, before)

    def test_feedback_reinforces_and_suppresses_fragments(self) -> None:
        rag = LayeredMemoryRAG()
        rag.add_document(
            "alpha memory marker. beta memory marker.",
            document_id="feedback",
            title="Feedback",
            cool_down_cycles=0,
        )
        alpha = next(
            fragment
            for fragment in rag.space.fragments.values()
            if fragment.normalized_text == "alpha"
        )
        beta = next(
            fragment
            for fragment in rag.space.fragments.values()
            if fragment.normalized_text == "beta"
        )
        before_alpha = (alpha.activation, alpha.strength)
        before_beta = (beta.activation, beta.strength, beta.layer)

        result = rag.apply_feedback(
            query="memory marker",
            positive_fragment_ids=[alpha.fragment_id],
            negative_fragment_ids=[beta.fragment_id, "missing"],
            reason="unit_test",
        )

        self.assertEqual(result.positive, [alpha.fragment_id])
        self.assertEqual(result.negative, [beta.fragment_id])
        self.assertEqual(result.ignored, ["missing"])
        self.assertGreater(alpha.activation, before_alpha[0])
        self.assertGreater(alpha.strength, before_alpha[1])
        self.assertLess(beta.activation, before_beta[0])
        self.assertLess(beta.strength, before_beta[1])
        self.assertGreaterEqual(beta.layer, before_beta[2])
        self.assertEqual(alpha.metadata["feedback"]["positive"], 1)
        self.assertEqual(beta.metadata["feedback"]["negative"], 1)

    def test_consolidation_creates_upper_layer_anchor(self) -> None:
        rag = LayeredMemoryRAG()
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook. "
            "The robot launch checklist helps deployment readiness. "
            "Alice reviews deployment readiness before launch.",
            document_id="launch",
            title="Launch Notes",
            cool_down_cycles=0,
        )

        result = rag.consolidate(min_support=3, keywords_per_anchor=4)

        self.assertEqual(len(result.created_anchor_ids), 1)
        self.assertGreaterEqual(result.support_relations, 3)
        anchor = rag.space.fragments[result.created_anchor_ids[0]]
        self.assertEqual(anchor.layer, 0)
        self.assertTrue(anchor.metadata["consolidation"]["anchor"])
        self.assertIn("launch", {term.lower() for term in anchor.metadata["consolidation"]["theme_terms"]})
        self.assertTrue(
            any(relation.relation_type == "consolidates" for relation in rag.space.relations.values())
        )

        refreshed = rag.consolidate(min_support=3, keywords_per_anchor=4)
        self.assertEqual(refreshed.created_anchor_ids, [])
        self.assertEqual(refreshed.reinforced_anchor_ids, [anchor.fragment_id])

    def test_optional_embeddings_enable_semantic_recall(self) -> None:
        rag = LayeredMemoryRAG(embedding_provider=FakeEmbeddingProvider())
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook.",
            document_id="launch",
            title="Launch Notes",
            cool_down_cycles=0,
        )

        answer = rag.answer("Where is the robotics deployment plan stored?", limit=3)

        self.assertGreater(len(answer.evidence), 0)
        self.assertIn("blue notebook", answer.answer.lower())
        self.assertTrue(any(item.embedding_score is not None for item in answer.evidence))
        self.assertTrue(any(item.raw_embedding_score is not None for item in answer.evidence))
        stored_chunk = next(iter(rag.chunks.values()))
        self.assertEqual(stored_chunk.embedding_model, "fake-embedding")

    def test_semantic_index_exact_matches_cosine_scan(self) -> None:
        rag = LayeredMemoryRAG(embedding_provider=FakeEmbeddingProvider(), semantic_index="exact")
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook. "
            "Maya stores the incident response runbook in vault seven.",
            document_id="mixed",
            title="Mixed",
            cool_down_cycles=0,
        )

        query_embedding = rag.embedding_provider.embed_query("robotics deployment plan")
        items = [
            (chunk.chunk_id, chunk.embedding or [])
            for chunk in rag.chunks.values()
            if chunk.embedding
        ]
        expected_chunk_ids = [
            hit.chunk_id for hit in exact_navigation_hits(items, query_embedding, limit=5)
        ]
        evidence = rag._semantic_evidence("robotics deployment plan", limit=5)

        self.assertEqual([item.chunk_id for item in evidence], expected_chunk_ids)
        self.assertEqual(rag.semantic_navigation_stats()["last_strategy"], "exact")
        self.assertFalse(rag.semantic_navigation_stats()["index_built"])

    def test_semantic_index_auto_uses_exact_for_small_store(self) -> None:
        rag = LayeredMemoryRAG(
            embedding_provider=FakeEmbeddingProvider(),
            semantic_index="auto",
            semantic_index_min_items=99,
        )
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook.",
            document_id="launch",
            title="Launch Notes",
            cool_down_cycles=0,
        )

        answer = rag.answer("robotics deployment plan", limit=3)
        navigation = answer.diagnostics["semantic_navigation"]

        self.assertEqual(navigation["last_strategy"], "exact")
        self.assertFalse(navigation["index_built"])
        self.assertEqual(navigation["embedded_chunks"], 1)

    def test_semantic_index_ann_recalls_clustered_chunk(self) -> None:
        rag = LayeredMemoryRAG(
            embedding_provider=FakeEmbeddingProvider(),
            semantic_index="ann",
            semantic_index_min_items=1,
            semantic_index_m=4,
            semantic_index_ef_construction=16,
            semantic_index_ef_search=12,
        )
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook.",
            document_id="launch",
            title="Launch Notes",
            cool_down_cycles=0,
        )
        for index in range(10):
            rag.add_document(
                f"Maya stores incident response runbook copy {index} in vault seven.",
                document_id=f"ops-{index}",
                title=f"Ops {index}",
                cool_down_cycles=0,
            )

        answer = rag.answer("Where is the robotics deployment plan stored?", limit=5)
        navigation = answer.diagnostics["semantic_navigation"]

        self.assertEqual(navigation["last_strategy"], "ann")
        self.assertTrue(navigation["index_built"])
        self.assertGreaterEqual(navigation["last_visited"], 1)
        self.assertIn("blue notebook", answer.answer.lower())
        self.assertTrue(any(item.chunk_id == "launch:0" for item in answer.evidence))

    def test_semantic_index_invalidates_after_document_change(self) -> None:
        rag = LayeredMemoryRAG(
            embedding_provider=FakeEmbeddingProvider(),
            semantic_index="ann",
            semantic_index_min_items=1,
        )
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook.",
            document_id="launch",
            title="Launch Notes",
            cool_down_cycles=0,
        )
        first = rag.answer("robotics deployment plan", limit=3)
        first_build_count = first.diagnostics["semantic_navigation"]["build_count"]

        rag.add_document(
            "Maya stores the incident response runbook in vault seven.",
            document_id="ops",
            title="Ops Runbook",
            cool_down_cycles=0,
        )
        after_add = rag.semantic_navigation_stats()
        second = rag.answer("incident response runbook", limit=3)

        self.assertEqual(after_add["last_strategy"], "none")
        self.assertFalse(after_add["index_built"])
        self.assertGreater(second.diagnostics["semantic_navigation"]["build_count"], first_build_count)
        self.assertEqual(second.diagnostics["semantic_navigation"]["last_strategy"], "ann")

    def test_semantic_index_rebuilds_after_json_round_trip(self) -> None:
        rag = LayeredMemoryRAG(
            embedding_provider=FakeEmbeddingProvider(),
            semantic_index="ann",
            semantic_index_min_items=1,
        )
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook.",
            document_id="launch",
            title="Launch Notes",
            cool_down_cycles=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            rag.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("semantic_navigation", payload["rag_config"])
            restored = LayeredMemoryRAG.load_with_embeddings(
                path,
                embedding_provider=FakeEmbeddingProvider(),
                semantic_index="ann",
            )
            answer = restored.answer("robotics deployment plan", limit=3)

        self.assertEqual(answer.diagnostics["semantic_navigation"]["last_strategy"], "ann")
        self.assertEqual(answer.diagnostics["semantic_navigation"]["build_count"], 1)

    def test_load_with_embeddings_adds_provider_for_existing_store(self) -> None:
        rag = LayeredMemoryRAG()
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook.",
            document_id="launch",
            title="Launch Notes",
            cool_down_cycles=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            rag.save(path)
            restored = LayeredMemoryRAG.load_with_embeddings(
                path,
                embedding_provider=FakeEmbeddingProvider(),
            )
            embedded_count = restored.embed_missing_chunks()
            answer = restored.answer("robotics deployment plan", limit=3)

        self.assertGreater(embedded_count, 0)
        self.assertGreater(len(answer.evidence), 0)
        self.assertTrue(any(item.embedding_score is not None for item in answer.evidence))

    def test_bottom_layer_detail_can_surface_through_anchor(self) -> None:
        rag = LayeredMemoryRAG(total_layers=5, sealed_bottom_layers=2)
        rag.add_document(
            "topmemory links bottommemory.",
            document_id="linked",
            title="Linked Detail",
            cool_down_cycles=0,
        )

        top_id = next(
            fragment_id
            for fragment_id, fragment in rag.space.fragments.items()
            if fragment.normalized_text == "topmemory"
        )
        bottom_id = next(
            fragment_id
            for fragment_id, fragment in rag.space.fragments.items()
            if fragment.normalized_text == "bottommemory"
        )
        rag.space.fragments[top_id].layer = 0
        rag.space.fragments[top_id].z = rag.space._layer_to_height(0)
        rag.space.fragments[bottom_id].layer = 4
        rag.space.fragments[bottom_id].z = rag.space._layer_to_height(4)
        rag.space._rebuild_cross_layer_flags()

        evidence = rag.retrieve("topmemory", limit=10)
        evidence_ids = {item.fragment_id for item in evidence}

        self.assertIn(bottom_id, evidence_ids)
        bottom = next(item for item in evidence if item.fragment_id == bottom_id)
        self.assertIsNotNone(bottom.via_relation)

    def test_old_store_without_depth_is_migrated_on_load(self) -> None:
        rag = LayeredMemoryRAG()
        rag.add_document(
            "Maya stores the incident response runbook in vault seven.",
            document_id="ops",
            title="Ops Runbook",
            cool_down_cycles=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            rag.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["version"] = 1
            for fragment in payload["memory_space"]["fragments"]:
                fragment.pop("depth", None)
            path.write_text(json.dumps(payload), encoding="utf-8")

            restored = LayeredMemoryRAG.load(path)

        self.assertTrue(all(fragment.depth >= 0 for fragment in restored.space.fragments.values()))
        self.assertTrue(all(0.0 <= fragment.z <= 1.0 for fragment in restored.space.fragments.values()))

    def test_accessibility_prefers_upper_memory_but_still_returns_lower_memory(self) -> None:
        rag = LayeredMemoryRAG(total_layers=5, sealed_bottom_layers=0)
        rag.add_document(
            "uppermemory appears here. lowermemory appears here.",
            document_id="layers",
            title="Layers",
            cool_down_cycles=0,
        )
        upper_id = next(
            fragment_id
            for fragment_id, fragment in rag.space.fragments.items()
            if fragment.normalized_text == "uppermemory"
        )
        lower_id = next(
            fragment_id
            for fragment_id, fragment in rag.space.fragments.items()
            if fragment.normalized_text == "lowermemory"
        )
        rag.space.fragments[upper_id].layer = 0
        rag.space.fragments[lower_id].layer = 4
        rag.space._refresh_fragment_depth(rag.space.fragments[upper_id])
        rag.space._refresh_fragment_depth(rag.space.fragments[lower_id])

        evidence = rag.retrieve("uppermemory lowermemory", limit=10)
        evidence_by_id = {item.fragment_id: item for item in evidence}

        self.assertIn(upper_id, evidence_by_id)
        self.assertIn(lower_id, evidence_by_id)
        self.assertGreater(evidence_by_id[upper_id].score, evidence_by_id[lower_id].score)
        self.assertGreater(evidence_by_id[upper_id].accessibility, evidence_by_id[lower_id].accessibility)
        self.assertIsNotNone(evidence_by_id[upper_id].raw_keyword_score)
        self.assertEqual(evidence_by_id[upper_id].final_score, evidence_by_id[upper_id].score)

    def test_reinforce_and_decay_move_depth(self) -> None:
        rag = LayeredMemoryRAG(total_layers=5)
        rag.add_document("memorydepth marker.", document_id="depth", title="Depth", cool_down_cycles=0)
        fragment = next(
            item
            for item in rag.space.fragments.values()
            if item.normalized_text == "memorydepth"
        )
        fragment.layer = 2
        fragment.activation = 0.8
        fragment.strength = 0.8
        rag.space._refresh_fragment_depth(fragment)

        initial_depth = fragment.depth
        rag.reinforce(fragment.fragment_id, amount=0.12)
        reinforced_depth = fragment.depth
        rag.decay(step=0.14, cycles=1)

        self.assertLessEqual(reinforced_depth, initial_depth)
        self.assertGreater(fragment.depth, reinforced_depth)

    def test_linear_forgetting_model_preserves_legacy_decay(self) -> None:
        rag = LayeredMemoryRAG(total_layers=5, dynamics={"forgetting_model": "linear"})
        fragment_id = rag.add_memory("linearmemory", source="test")[0]
        fragment = rag.space.fragments[fragment_id]
        fragment.layer = 2
        fragment.activation = 0.8
        fragment.strength = 0.8
        fragment.ease = 0.7

        rag.decay(step=0.14, cycles=1)

        self.assertAlmostEqual(fragment.activation, 0.66)
        self.assertAlmostEqual(fragment.strength, 0.709)
        self.assertAlmostEqual(fragment.ease, 0.6888)
        self.assertEqual(fragment.layer, 2)
        self.assertEqual(rag.space.dynamics.forgetting_model, "linear")

    def test_ebbinghaus_forgetting_sinks_one_time_detail(self) -> None:
        rag = LayeredMemoryRAG(total_layers=5)
        rag.add_document(
            "The temporary code 45123789 appeared once.",
            document_id="forgetting",
            title="Forgetting",
            cool_down_cycles=0,
        )
        fragment = next(
            item
            for item in rag.space.fragments.values()
            if item.normalized_text == "45123789"
        )
        start_layer = fragment.layer
        start_activation = fragment.activation

        rag.decay(step=0.14, cycles=7)

        self.assertEqual(rag.space.dynamics.forgetting_model, "ebbinghaus")
        self.assertLess(fragment.activation, start_activation)
        self.assertLess(rag.space.retention_for_fragment(fragment), 0.24)
        self.assertGreater(fragment.layer, start_layer)

    def test_reinforced_memory_retains_more_than_plain_memory(self) -> None:
        rag = LayeredMemoryRAG(total_layers=5)
        reinforced_id = rag.add_memory("reinforcedmemory", source="test")[0]
        plain_id = rag.add_memory("plainmemory", source="test")[0]
        for fragment_id in (reinforced_id, plain_id):
            fragment = rag.space.fragments[fragment_id]
            fragment.layer = 2
            fragment.activation = 0.8
            fragment.strength = 0.8
            fragment.ease = 0.7
            fragment.retrievals = 0
            fragment.reinforcements = 1
            fragment.forgettings = 0
            rag.space.refresh_fragment_state(fragment)

        rag.reinforce(reinforced_id, amount=0.5)
        reinforced_start = rag.space.fragments[reinforced_id].activation
        plain_start = rag.space.fragments[plain_id].activation
        rag.decay(step=0.14, cycles=6)
        reinforced = rag.space.fragments[reinforced_id]
        plain = rag.space.fragments[plain_id]

        self.assertGreater(reinforced.activation / reinforced_start, plain.activation / plain_start)
        self.assertLessEqual(reinforced.layer, plain.layer)

    def test_negative_feedback_accelerates_forgetting(self) -> None:
        rag = LayeredMemoryRAG(total_layers=5)
        suppressed_id = rag.add_memory("suppressedmemory", source="test")[0]
        plain_id = rag.add_memory("neutralmemory", source="test")[0]
        for fragment_id in (suppressed_id, plain_id):
            fragment = rag.space.fragments[fragment_id]
            fragment.layer = 2
            fragment.activation = 0.8
            fragment.strength = 0.8
            fragment.ease = 0.7
            fragment.forgettings = 0
            rag.space.refresh_fragment_state(fragment)

        rag.apply_feedback(
            negative_fragment_ids=[suppressed_id],
            reason="unit_test",
            negative_amount=0.1,
        )
        rag.decay(step=0.14, cycles=5)
        suppressed = rag.space.fragments[suppressed_id]
        plain = rag.space.fragments[plain_id]

        self.assertLess(suppressed.activation, plain.activation)
        self.assertLess(rag.space.retention_for_fragment(suppressed), rag.space.retention_for_fragment(plain))
        self.assertGreaterEqual(suppressed.layer, plain.layer)

    def test_consolidation_anchor_resists_ordinary_forgetting(self) -> None:
        rag = LayeredMemoryRAG(total_layers=5)
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook. "
            "The robot launch checklist helps deployment readiness. "
            "Alice reviews deployment readiness before launch.",
            document_id="launch",
            title="Launch Notes",
            cool_down_cycles=0,
        )
        result = rag.consolidate(min_support=3, keywords_per_anchor=4)
        anchor = rag.space.fragments[result.created_anchor_ids[0]]

        rag.decay(step=0.14, cycles=10)

        self.assertEqual(anchor.layer, 0)
        self.assertGreater(anchor.activation, 0.5)
        self.assertGreater(rag.space.retention_for_fragment(anchor), 0.5)

    def test_dynamics_round_trip_preserves_forgetting_model(self) -> None:
        rag = LayeredMemoryRAG(
            dynamics={
                "forgetting_model": "linear",
                "base_forgetting_rate": 0.22,
            }
        )
        rag.add_document(
            "Maya stores the incident response runbook in vault seven.",
            document_id="ops",
            title="Ops Runbook",
            cool_down_cycles=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            rag.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            restored = LayeredMemoryRAG.load(path)

        dynamics = payload["memory_space"]["config"]["dynamics"]
        self.assertEqual(dynamics["forgetting_model"], "linear")
        self.assertEqual(restored.space.dynamics.forgetting_model, "linear")
        self.assertEqual(restored.space.dynamics.base_forgetting_rate, 0.22)

    def test_embedding_layout_places_similar_memories_closer(self) -> None:
        rag = LayeredMemoryRAG(total_layers=5)
        alpha_ids = rag.add_memory("alpha project launch checklist", source="demo")
        beta_ids = rag.add_memory("beta project launch checklist", source="demo")
        incident_ids = rag.add_memory("incident response runbook vault", source="demo")

        alpha = alpha_ids[0]
        beta = beta_ids[0]
        incident = incident_ids[0]
        result = apply_memory_layout(
            rag.space,
            fragment_embeddings={
                alpha: [1.0, 0.0, 0.0],
                beta: [0.96, 0.04, 0.0],
                incident: [0.0, 1.0, 0.0],
            },
            iterations=80,
        )

        self.assertEqual(result.layout_model, "embedding-force")
        self.assertTrue(result.has_embedding_layout)
        self.assertEqual(result.embedding_scope, "none")
        self.assertLess(
            _fragment_distance(rag, alpha, beta),
            _fragment_distance(rag, alpha, incident),
        )

    def test_chunk_scope_layout_reuses_existing_chunk_embeddings(self) -> None:
        rag = LayeredMemoryRAG(embedding_provider=FakeEmbeddingProvider())
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook. "
            "Maya stores the incident response runbook in vault seven.",
            document_id="mixed",
            title="Mixed",
            cool_down_cycles=0,
        )
        rag.embedding_provider = None

        result = rag.layout_memory_space(embedding_scope="chunk", iterations=20)

        self.assertTrue(result.has_embedding_layout)
        self.assertEqual(result.embedding_scope, "chunk")
        self.assertGreater(result.semantic_edge_count, 0)

    def test_fragment_scope_layout_caches_fragment_embeddings(self) -> None:
        rag = LayeredMemoryRAG(embedding_provider=FakeEmbeddingProvider())
        ids = rag.add_memory("robot launch checklist", source="demo")
        rag.add_memory("incident response runbook", source="demo")

        result = rag.layout_memory_space(
            embedding_scope="fragment",
            embed_fragments=True,
            iterations=20,
        )

        self.assertTrue(result.has_embedding_layout)
        self.assertEqual(result.embedding_scope, "fragment")
        cached = rag.space.fragments[ids[0]].metadata.get("embedding")
        self.assertIsInstance(cached, list)
        self.assertEqual(rag.space.fragments[ids[0]].metadata.get("embedding_model"), "fake-embedding")

        rag.embedding_provider = None
        cached_result = rag.layout_memory_space(embedding_scope="fragment", iterations=20)
        self.assertTrue(cached_result.has_embedding_layout)
        self.assertEqual(cached_result.embedding_scope, "fragment")


class MemoryVisualizationTests(unittest.TestCase):
    def test_graph_data_exports_fragments_relations_and_meta(self) -> None:
        rag = LayeredMemoryRAG()
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook.",
            document_id="launch",
            title="Launch Notes",
            cool_down_cycles=0,
        )
        rag.consolidate(min_support=3, keywords_per_anchor=4)

        graph = build_graph_data(rag)

        self.assertEqual(graph["meta"]["totalLayers"], rag.total_layers)
        self.assertEqual(graph["meta"]["layerHistogram"], rag.layer_histogram())
        self.assertEqual(graph["meta"]["fragmentCount"], len(rag.space.fragments))
        self.assertEqual(graph["meta"]["relationCount"], len(rag.space.relations))
        self.assertEqual(len(graph["nodes"]), len(rag.space.fragments))
        self.assertEqual(len(graph["links"]), len(rag.space.relations))

        node = next(item for item in graph["nodes"] if item["source"] is not None)
        self.assertEqual(node["source"]["documentId"], "launch")
        self.assertEqual(node["source"]["title"], "Launch Notes")
        self.assertIsNotNone(node["chunkText"])
        self.assertIn("depth", node)
        self.assertIn("accessibility", node)
        self.assertIn("layoutModel", node)
        self.assertIn("embeddingScope", node)
        self.assertIn("semanticEdgeCount", node)
        self.assertIn("layoutUpdatedAt", graph["meta"])
        self.assertEqual(graph["meta"]["retrievalProfile"], "balanced")
        self.assertGreaterEqual(graph["meta"]["consolidationAnchorCount"], 1)

        link = graph["links"][0]
        self.assertIn(link["source"], rag.space.fragments)
        self.assertIn(link["target"], rag.space.fragments)

    def test_graph_data_handles_memories_without_sources(self) -> None:
        rag = LayeredMemoryRAG()
        fragment_ids = rag.add_memory("User prefers concise technical explanations.")

        graph = build_graph_data(rag)

        exported = {node["id"]: node for node in graph["nodes"]}
        for fragment_id in fragment_ids:
            self.assertIn(fragment_id, exported)
            self.assertIsNone(exported[fragment_id]["source"])
            self.assertIsNone(exported[fragment_id]["chunkText"])

    def test_graph_data_handles_empty_store(self) -> None:
        rag = LayeredMemoryRAG(total_layers=4)

        graph = build_graph_data(rag)

        self.assertEqual(graph["nodes"], [])
        self.assertEqual(graph["links"], [])
        self.assertEqual(graph["meta"]["totalLayers"], 4)
        self.assertEqual(graph["meta"]["layerHistogram"], [0, 0, 0, 0])


class SQLiteStorageTests(unittest.TestCase):
    def test_sqlite_round_trip_preserves_memory_graph(self) -> None:
        rag = LayeredMemoryRAG(
            embedding_provider=FakeEmbeddingProvider(),
            retrieval_profile="conservative",
            semantic_index="ann",
            semantic_index_min_items=1,
        )
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook.",
            document_id="launch",
            title="Launch Notes",
            cool_down_cycles=0,
        )
        rag.layout_memory_space(embedding_scope="chunk", iterations=20)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.db"
            save_rag(path, rag, event_type="test_save")
            restored = load_rag(path)

        self.assertEqual(len(restored.documents), len(rag.documents))
        self.assertEqual(len(restored.chunks), len(rag.chunks))
        self.assertEqual(len(restored.space.fragments), len(rag.space.fragments))
        self.assertEqual(len(restored.space.relations), len(rag.space.relations))
        self.assertEqual(restored.layer_histogram(), rag.layer_histogram())
        self.assertEqual(restored.retrieval_profile.name, "conservative")
        self.assertEqual(restored.semantic_navigation_config.mode, "ann")
        self.assertEqual(restored.semantic_navigation_config.min_items, 1)
        self.assertTrue(any(chunk.embedding for chunk in restored.chunks.values()))
        self.assertTrue(all(fragment.depth >= 0 for fragment in restored.space.fragments.values()))
        self.assertTrue(any("layout_model" in fragment.metadata for fragment in restored.space.fragments.values()))
        restored.embedding_provider = FakeEmbeddingProvider()
        answer = restored.answer("robotics deployment plan", limit=3)
        self.assertEqual(answer.diagnostics["semantic_navigation"]["last_strategy"], "ann")

    def test_json_to_sqlite_migration_preserves_answer_behavior(self) -> None:
        rag = LayeredMemoryRAG()
        rag.add_document(
            "Maya stores the incident response runbook in vault seven.",
            document_id="ops",
            title="Ops Runbook",
            cool_down_cycles=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "memory.json"
            db_path = Path(temp_dir) / "memory.db"
            rag.save(json_path)
            migrated = migrate_json_to_sqlite(json_path, db_path)
            restored = load_rag(db_path)
            answer = restored.answer("incident response", limit=3)

        self.assertEqual(len(migrated.space.fragments), len(restored.space.fragments))
        self.assertGreater(len(answer.evidence), 0)
        self.assertIn("runbook", answer.answer.lower())

    def test_sqlite_round_trip_preserves_dynamics_config(self) -> None:
        rag = LayeredMemoryRAG(
            dynamics={
                "forgetting_model": "linear",
                "retention_floor": 0.07,
                "base_forgetting_rate": 0.22,
            }
        )
        rag.add_document(
            "Maya stores the incident response runbook in vault seven.",
            document_id="ops",
            title="Ops Runbook",
            cool_down_cycles=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "memory.db"
            save_rag(db_path, rag, event_type="test_save")
            restored = load_rag(db_path)

        self.assertEqual(restored.space.dynamics.forgetting_model, "linear")
        self.assertEqual(restored.space.dynamics.retention_floor, 0.07)
        self.assertEqual(restored.space.dynamics.base_forgetting_rate, 0.22)

    def test_cli_commands_work_with_sqlite_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "story.txt"
            db_path = Path(temp_dir) / "memory.db"
            text_path.write_text(
                "Alice keeps the robot launch checklist in the blue notebook. "
                "Maya stores the incident response runbook in vault seven.",
                encoding="utf-8",
            )

            ingest = _run_cli("ingest", str(text_path), "--store", str(db_path), "--title", "Story")
            stats = _run_cli("stats", "--store", str(db_path))
            ask = _run_cli(
                "ask",
                "robot launch checklist",
                "--store",
                str(db_path),
                "--semantic-index",
                "exact",
                "--json",
            )
            ask_payload = json.loads(ask.stdout)
            ask_ann = _run_cli(
                "ask",
                "robot launch checklist",
                "--store",
                str(db_path),
                "--semantic-index",
                "ann",
                "--json",
            )
            ask_ann_payload = json.loads(ask_ann.stdout)
            feedback = _run_cli(
                "feedback",
                "--store",
                str(db_path),
                "--query",
                "robot launch checklist",
                "--negative",
                ask_payload["evidence"][0]["fragment_id"],
                "--json",
            )
            consolidate = _run_cli(
                "consolidate",
                "--store",
                str(db_path),
                "--min-support",
                "3",
                "--keywords-per-anchor",
                "4",
                "--json",
            )
            layout = _run_cli("layout", "--store", str(db_path), "--embedding-scope", "none", "--iterations", "5")

        stats_payload = json.loads(stats.stdout)
        feedback_payload = json.loads(feedback.stdout)
        consolidate_payload = json.loads(consolidate.stdout)
        layout_payload = json.loads(layout.stdout)

        self.assertIn("ingested document", ingest.stdout)
        self.assertGreater(stats_payload["fragments"], 0)
        self.assertIn("semantic_navigation", stats_payload)
        self.assertGreater(len(ask_payload["evidence"]), 0)
        self.assertIn("raw_keyword_score", ask_payload["evidence"][0])
        self.assertEqual(ask_payload["diagnostics"]["semantic_navigation"]["mode"], "exact")
        self.assertEqual(ask_ann_payload["diagnostics"]["semantic_navigation"]["mode"], "ann")
        self.assertEqual(len(feedback_payload["negative"]), 1)
        self.assertEqual(feedback_payload["diagnostics"]["retrieval_profile"], "balanced")
        self.assertEqual(len(consolidate_payload["created_anchor_ids"]), 1)
        self.assertGreaterEqual(consolidate_payload["support_relations"], 3)
        self.assertEqual(layout_payload["layout_model"], "relation-force")

    def test_visualization_graph_supports_sqlite_store(self) -> None:
        rag = LayeredMemoryRAG()
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook.",
            document_id="launch",
            title="Launch Notes",
            cool_down_cycles=0,
        )
        rag.layout_memory_space(embedding_scope="none", iterations=10)

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "memory.db"
            save_rag(db_path, rag, event_type="layout")
            graph = build_graph_data(load_rag(db_path))

        self.assertGreater(len(graph["nodes"]), 0)
        self.assertGreater(len(graph["links"]), 0)
        self.assertIn("depth", graph["nodes"][0])
        self.assertIn("accessibility", graph["nodes"][0])
        self.assertIn("layoutModel", graph["nodes"][0])
        self.assertIn("layoutModel", graph["meta"])


def _fragment_distance(rag: LayeredMemoryRAG, left_id: str, right_id: str) -> float:
    left = rag.space.fragments[left_id]
    right = rag.space.fragments[right_id]
    return (
        (left.x - right.x) ** 2
        + (left.y - right.y) ** 2
        + (left.z - right.z) ** 2
    ) ** 0.5


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "humem_product.cli", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


if __name__ == "__main__":
    unittest.main()
