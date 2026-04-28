from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humem_product import LayeredMemoryRAG  # noqa: E402


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
            self.assertEqual(payload["version"], 1)

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


if __name__ == "__main__":
    unittest.main()
