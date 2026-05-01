from __future__ import annotations

import argparse
import json
from pathlib import Path

from .policy import RETRIEVAL_PROFILES, make_retrieval_profile
from .rag import FeedbackResult, LayeredMemoryRAG, MemoryConsolidationResult, RAGAnswer
from .storage import load_rag, migrate_json_to_sqlite, save_rag
from .visualization import run_visualization_server


PROFILE_CHOICES = sorted(RETRIEVAL_PROFILES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HuMem layered memory RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="ingest a UTF-8 text file into a memory store")
    ingest.add_argument("input", type=Path)
    ingest.add_argument("--store", type=Path, required=True)
    ingest.add_argument("--document-id")
    ingest.add_argument("--title")
    ingest.add_argument("--embedding-provider", choices=["openai"])
    ingest.add_argument("--embedding-model", default="text-embedding-3-small")
    ingest.add_argument("--retrieval-profile", choices=PROFILE_CHOICES)

    ask = subparsers.add_parser("ask", help="query a memory store")
    ask.add_argument("query")
    ask.add_argument("--store", type=Path, required=True)
    ask.add_argument("--limit", type=int, default=6)
    ask.add_argument("--json", action="store_true")
    ask.add_argument("--embedding-provider", choices=["openai"])
    ask.add_argument("--embedding-model", default="text-embedding-3-small")
    ask.add_argument("--retrieval-profile", choices=PROFILE_CHOICES)
    ask.add_argument("--no-auto-embed", action="store_true")

    feedback = subparsers.add_parser("feedback", help="reinforce or suppress retrieved fragments")
    feedback.add_argument("--store", type=Path, required=True)
    feedback.add_argument("--query")
    feedback.add_argument("--positive", action="append", default=[], help="fragment id or comma-separated ids")
    feedback.add_argument("--negative", action="append", default=[], help="fragment id or comma-separated ids")
    feedback.add_argument("--reason", default="cli_feedback")
    feedback.add_argument("--retrieval-profile", choices=PROFILE_CHOICES)
    feedback.add_argument("--json", action="store_true")

    stats = subparsers.add_parser("stats", help="print memory store diagnostics")
    stats.add_argument("--store", type=Path, required=True)

    decay = subparsers.add_parser("decay", help="run forgetting cycles and save the store")
    decay.add_argument("--store", type=Path, required=True)
    decay.add_argument("--cycles", type=int, default=1)
    decay.add_argument("--step", type=float, default=0.14)

    visualize = subparsers.add_parser("visualize", help="open an interactive 3D memory graph")
    visualize.add_argument("--store", type=Path, required=True)
    visualize.add_argument("--host", default="127.0.0.1")
    visualize.add_argument("--port", type=int, default=8765)
    visualize.add_argument("--no-open", action="store_true")

    layout = subparsers.add_parser("layout", help="compute continuous memory-space coordinates")
    layout.add_argument("--store", type=Path, required=True)
    layout.add_argument("--embedding-provider", choices=["openai"])
    layout.add_argument("--embedding-model", default="text-embedding-3-small")
    layout.add_argument("--embedding-scope", choices=["chunk", "fragment", "none"], default="chunk")
    layout.add_argument("--embed-missing", action="store_true")
    layout.add_argument("--embed-missing-chunks", action="store_true", help=argparse.SUPPRESS)
    layout.add_argument("--embed-fragments", action="store_true", help=argparse.SUPPRESS)
    layout.add_argument("--no-embeddings", action="store_true", help=argparse.SUPPRESS)
    layout.add_argument("--iterations", type=int, default=120)
    layout.add_argument("--semantic-neighbors", type=int, default=4)

    consolidate = subparsers.add_parser("consolidate", help="create upper-layer anchors from recurring memories")
    consolidate.add_argument("--store", type=Path, required=True)
    consolidate.add_argument("--scope", choices=["document", "chunk", "global"], default="document")
    consolidate.add_argument("--max-anchors", type=int, default=8)
    consolidate.add_argument("--keywords-per-anchor", type=int, default=5)
    consolidate.add_argument("--min-support", type=int, default=3)
    consolidate.add_argument("--json", action="store_true")

    migrate = subparsers.add_parser("migrate", help="migrate a JSON memory store to SQLite")
    migrate.add_argument("--from", dest="source", type=Path, required=True)
    migrate.add_argument("--to", dest="target", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        if args.store.exists():
            rag = _load_store(args.store, args.embedding_provider, args.embedding_model, args.retrieval_profile)
        else:
            rag = LayeredMemoryRAG(
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model,
                retrieval_profile=args.retrieval_profile,
            )
        text = args.input.read_text(encoding="utf-8")
        doc_id = rag.add_document(
            text,
            document_id=args.document_id,
            title=args.title or args.input.stem,
            metadata={"path": str(args.input)},
        )
        save_rag(args.store, rag, event_type="ingest", event_payload={"document_id": doc_id})
        print(f"ingested document={doc_id} fragments={len(rag.space.fragments)} store={args.store}")
        return

    if args.command == "visualize":
        try:
            run_visualization_server(
                args.store,
                host=args.host,
                port=args.port,
                open_browser=not args.no_open,
            )
        except (FileNotFoundError, OSError) as exc:
            parser.error(str(exc))
        return

    if args.command == "layout":
        rag = _load_store(args.store, args.embedding_provider, args.embedding_model)
        if args.no_embeddings:
            args.embedding_scope = "none"
        if args.embed_fragments:
            args.embedding_scope = "fragment"

        if args.embedding_provider and (args.embed_missing or args.embed_missing_chunks) and args.embedding_scope == "chunk":
            embedded_count = rag.embed_missing_chunks()
        else:
            embedded_count = 0
        result = rag.layout_memory_space(
            use_embeddings=args.embedding_scope != "none",
            embed_fragments=args.embed_missing and args.embedding_scope == "fragment",
            embedding_scope=args.embedding_scope,
            iterations=args.iterations,
            semantic_neighbors=args.semantic_neighbors,
        )
        save_rag(
            args.store,
            rag,
            event_type="layout",
            event_payload={
                "layout_model": result.layout_model,
                "embedding_scope": result.embedding_scope,
                "semantic_edges": result.semantic_edge_count,
                "relation_edges": result.relation_edge_count,
            },
        )
        print(
            json.dumps(
                {
                    "store": str(args.store),
                    "layout_model": result.layout_model,
                    "has_embedding_layout": result.has_embedding_layout,
                    "nodes": result.node_count,
                    "semantic_edges": result.semantic_edge_count,
                    "relation_edges": result.relation_edge_count,
                    "embedding_scope": result.embedding_scope,
                    "layout_updated_at": result.layout_updated_at,
                    "embedded_chunks": embedded_count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "consolidate":
        rag = _load_store(args.store, None, "text-embedding-3-small")
        result = rag.consolidate(
            scope=args.scope,
            max_anchors=args.max_anchors,
            keywords_per_anchor=args.keywords_per_anchor,
            min_support=args.min_support,
        )
        save_rag(args.store, rag, event_type="consolidate", event_payload=_consolidation_to_dict(result))
        if args.json:
            print(json.dumps(_consolidation_to_dict(result), ensure_ascii=False, indent=2))
            return
        print(
            "consolidated "
            f"created={len(result.created_anchor_ids)} "
            f"refreshed={len(result.reinforced_anchor_ids)} "
            f"relations={result.support_relations} store={args.store}"
        )
        return

    if args.command == "migrate":
        rag = migrate_json_to_sqlite(args.source, args.target)
        print(
            json.dumps(
                {
                    "from": str(args.source),
                    "to": str(args.target),
                    "documents": len(rag.documents),
                    "chunks": len(rag.chunks),
                    "fragments": len(rag.space.fragments),
                    "relations": len(rag.space.relations),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    rag = _load_store(
        args.store,
        getattr(args, "embedding_provider", None),
        getattr(args, "embedding_model", "text-embedding-3-small"),
        getattr(args, "retrieval_profile", None),
    )

    if args.command == "ask":
        if args.embedding_provider and not args.no_auto_embed:
            embedded_count = rag.embed_missing_chunks()
            if embedded_count:
                save_rag(args.store, rag, event_type="embed_missing_chunks", event_payload={"count": embedded_count})
        answer = rag.answer(args.query, limit=args.limit)
        save_rag(
            args.store,
            rag,
            event_type="retrieve",
            event_payload={"query": args.query, "evidence_count": len(answer.evidence)},
        )
        if args.json:
            print(json.dumps(_answer_to_dict(answer), ensure_ascii=False, indent=2))
            return
        print(answer.answer)
        print("\nEvidence:")
        for index, item in enumerate(answer.evidence, start=1):
            title = item.title or "memory"
            relation = f" via={item.via_relation}" if item.via_relation else ""
            print(f"{index}. [{title} layer={item.layer} score={item.score:.2f}{relation}] {item.text}")
        return

    if args.command == "feedback":
        result = rag.apply_feedback(
            query=args.query,
            positive_fragment_ids=_parse_fragment_ids(args.positive),
            negative_fragment_ids=_parse_fragment_ids(args.negative),
            reason=args.reason,
        )
        save_rag(args.store, rag, event_type="feedback", event_payload=_feedback_to_dict(result))
        if args.json:
            print(json.dumps(_feedback_to_dict(result), ensure_ascii=False, indent=2))
            return
        print(
            "feedback applied "
            f"positive={len(result.positive)} negative={len(result.negative)} "
            f"ignored={len(result.ignored)} store={args.store}"
        )
        return

    if args.command == "stats":
        print(
            json.dumps(
                {
                    "documents": len(rag.documents),
                    "chunks": len(rag.chunks),
                    "fragments": len(rag.space.fragments),
                    "relations": len(rag.space.relations),
                    "layer_histogram": rag.layer_histogram(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "decay":
        rag.decay(step=args.step, cycles=args.cycles)
        save_rag(
            args.store,
            rag,
            event_type="decay",
            event_payload={"cycles": args.cycles, "step": args.step},
        )
        print(f"decayed cycles={args.cycles} step={args.step} store={args.store}")


def _answer_to_dict(answer: RAGAnswer) -> dict[str, object]:
    return {
        "query": answer.query,
        "answer": answer.answer,
        "diagnostics": answer.diagnostics,
        "evidence": [
            {
                "fragment_id": item.fragment_id,
                "text": item.text,
                "kind": item.kind,
                "layer": item.layer,
                "depth": item.depth,
                "score": item.score,
                "accessibility": item.accessibility,
                "memory_score": item.memory_score,
                "embedding_score": item.embedding_score,
                "raw_keyword_score": item.raw_keyword_score,
                "raw_embedding_score": item.raw_embedding_score,
                "relation_bonus": item.relation_bonus,
                "final_score": item.final_score,
                "spatial_score": item.spatial_score,
                "layout_score": item.layout_score,
                "via_relation": item.via_relation,
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
                "title": item.title,
                "chunk_text": item.chunk_text,
            }
            for item in answer.evidence
        ],
    }


def _feedback_to_dict(result: FeedbackResult) -> dict[str, object]:
    return {
        "query": result.query,
        "positive": result.positive,
        "negative": result.negative,
        "ignored": result.ignored,
        "diagnostics": result.diagnostics,
    }


def _consolidation_to_dict(result: MemoryConsolidationResult) -> dict[str, object]:
    return {
        "created_anchor_ids": result.created_anchor_ids,
        "reinforced_anchor_ids": result.reinforced_anchor_ids,
        "support_relations": result.support_relations,
        "skipped_groups": result.skipped_groups,
        "diagnostics": result.diagnostics,
        "candidates": [
            {
                "group_key": candidate.group_key,
                "title": candidate.title,
                "anchor_text": candidate.anchor_text,
                "theme_terms": candidate.theme_terms,
                "support_fragment_ids": candidate.support_fragment_ids,
                "score": candidate.score,
            }
            for candidate in result.candidates
        ],
    }


def _parse_fragment_ids(values: list[str]) -> list[str]:
    fragment_ids: list[str] = []
    for value in values:
        fragment_ids.extend(part.strip() for part in value.split(",") if part.strip())
    return fragment_ids


def _load_store(
    store: Path,
    embedding_provider: str | None,
    embedding_model: str,
    retrieval_profile: str | None = None,
) -> LayeredMemoryRAG:
    rag = load_rag(
        store,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )
    if retrieval_profile:
        rag.retrieval_profile = make_retrieval_profile(retrieval_profile)
        rag.memory_weight = rag.retrieval_profile.memory_weight
        rag.embedding_weight = rag.retrieval_profile.embedding_weight
    return rag


if __name__ == "__main__":
    main()
