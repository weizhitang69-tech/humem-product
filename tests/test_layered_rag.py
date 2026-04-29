from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humem_product import LayeredMemoryRAG  # noqa: E402
from humem_product.memory_layout import apply_memory_layout  # noqa: E402
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
            self.assertEqual(payload["version"], 2)

            restored = LayeredMemoryRAG.load(path)
            answer = restored.answer("incident response", limit=3)

        self.assertEqual(len(restored.documents), 1)
        self.assertGreater(len(answer.evidence), 0)
        self.assertIn("runbook", answer.answer.lower())

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
        stored_chunk = next(iter(rag.chunks.values()))
        self.assertEqual(stored_chunk.embedding_model, "fake-embedding")

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
        self.assertLess(
            _fragment_distance(rag, alpha, beta),
            _fragment_distance(rag, alpha, incident),
        )


class MemoryVisualizationTests(unittest.TestCase):
    def test_graph_data_exports_fragments_relations_and_meta(self) -> None:
        rag = LayeredMemoryRAG()
        rag.add_document(
            "Alice keeps the robot launch checklist in the blue notebook.",
            document_id="launch",
            title="Launch Notes",
            cool_down_cycles=0,
        )

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


def _fragment_distance(rag: LayeredMemoryRAG, left_id: str, right_id: str) -> float:
    left = rag.space.fragments[left_id]
    right = rag.space.fragments[right_id]
    return (
        (left.x - right.x) ** 2
        + (left.y - right.y) ** 2
        + (left.z - right.z) ** 2
    ) ** 0.5


if __name__ == "__main__":
    unittest.main()
