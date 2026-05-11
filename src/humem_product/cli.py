from __future__ import annotations

import argparse
import json
from pathlib import Path

from .navigation import SEMANTIC_INDEX_MODES
from .policy import RETRIEVAL_PROFILES, make_retrieval_profile
from .event_rag import EventAnswer, EventFeedbackResult, EventRAG
from .rag import FeedbackResult, LayeredMemoryRAG, MemoryConsolidationResult, RAGAnswer
from .storage import (
    EventMemoryDB,
    is_event_store,
    load_event_rag,
    load_rag,
    migrate_json_to_sqlite,
    migrate_v3_to_event_rag,
    save_event_rag,
    save_rag,
)
from .visualization import run_visualization_server


PROFILE_CHOICES = sorted(RETRIEVAL_PROFILES)
SEMANTIC_INDEX_CHOICES = sorted(SEMANTIC_INDEX_MODES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HuMem layered memory RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    remember = subparsers.add_parser("remember", help="write text into a v4 event memory store")
    remember.add_argument("text", nargs="?")
    remember.add_argument("--input", type=Path)
    remember.add_argument("--store", type=Path, required=True)
    remember.add_argument("--source", default="memory")
    remember.add_argument("--captured-at")
    remember.add_argument("--collection", default="default")
    remember.add_argument("--json", action="store_true")
    _add_event_provider_args(remember)

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
    ask.add_argument("--embedding-api-key")
    _add_llm_args(ask)
    ask.add_argument("--retrieval-profile", choices=PROFILE_CHOICES)
    ask.add_argument("--semantic-index", choices=SEMANTIC_INDEX_CHOICES)
    ask.add_argument("--collection")
    ask.add_argument("--filter-json")
    ask.add_argument("--filter")
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

    collection = subparsers.add_parser("collection", help="manage v4 event collections")
    collection_subparsers = collection.add_subparsers(dest="collection_command", required=True)
    collection_create = collection_subparsers.add_parser("create")
    collection_create.add_argument("--store", type=Path, required=True)
    collection_create.add_argument("name")
    collection_create.add_argument("--id")
    collection_create.add_argument("--schema-json")
    collection_create.add_argument("--metadata-json")
    collection_create.add_argument("--json", action="store_true")
    collection_list = collection_subparsers.add_parser("list")
    collection_list.add_argument("--store", type=Path, required=True)
    collection_list.add_argument("--json", action="store_true")
    collection_show = collection_subparsers.add_parser("show")
    collection_show.add_argument("--store", type=Path, required=True)
    collection_show.add_argument("collection_id")
    collection_show.add_argument("--json", action="store_true")
    collection_update = collection_subparsers.add_parser("update")
    collection_update.add_argument("--store", type=Path, required=True)
    collection_update.add_argument("collection_id")
    collection_update.add_argument("--name")
    collection_update.add_argument("--schema-json")
    collection_update.add_argument("--metadata-json")
    collection_update.add_argument("--status")
    collection_update.add_argument("--json", action="store_true")

    index = subparsers.add_parser("index", help="build persisted HNSW indexes for a v4 SQLite event store")
    index.add_argument("--store", type=Path, required=True)
    index.add_argument("--collection")
    index.add_argument("--semantic-index", choices=SEMANTIC_INDEX_CHOICES, default="ann")
    index.add_argument("--semantic-index-min-items", type=int)
    index.add_argument("--semantic-index-m", type=int)
    index.add_argument("--semantic-index-ef-construction", type=int)
    index.add_argument("--semantic-index-ef-search", type=int)
    index.add_argument("--semantic-index-seed", type=int)
    index.add_argument("--json", action="store_true")

    delete = subparsers.add_parser("delete", help="soft-delete a v4 event")
    delete.add_argument("--store", type=Path, required=True)
    delete.add_argument("--event-id", required=True)
    delete.add_argument("--hard", action="store_true")
    delete.add_argument("--json", action="store_true")

    replace = subparsers.add_parser("replace", help="replace a v4 event with newly extracted text")
    replace.add_argument("--store", type=Path, required=True)
    replace.add_argument("--event-id", required=True)
    replace.add_argument("--text")
    replace.add_argument("--input", type=Path)
    replace.add_argument("--source", default="memory")
    replace.add_argument("--collection")
    replace.add_argument("--captured-at")
    replace.add_argument("--json", action="store_true")
    _add_event_provider_args(replace)

    compact = subparsers.add_parser("compact", help="compact a v4 event store and rebuild clean ANN indexes")
    compact.add_argument("--store", type=Path, required=True)
    compact.add_argument("--purge-deleted", action="store_true")
    compact.add_argument("--older-than-days", type=int)
    compact.add_argument("--json", action="store_true")

    backup = subparsers.add_parser("backup", help="backup a v4 event store")
    backup.add_argument("--store", type=Path, required=True)
    backup.add_argument("--to", type=Path, required=True)
    backup.add_argument("--json", action="store_true")

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

    migrate_v4 = subparsers.add_parser("migrate-v4", help="migrate a legacy v3 store to v4 event memory")
    migrate_v4.add_argument("--from", dest="source", type=Path, required=True)
    migrate_v4.add_argument("--to", dest="target", type=Path, required=True)
    _add_event_provider_args(migrate_v4)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "remember":
        if args.input:
            text = args.input.read_text(encoding="utf-8")
        elif args.text:
            text = args.text
        else:
            parser.error("remember requires text or --input")
        if args.store.exists():
            rag = _load_event_store(args.store, args, require_providers=True)
        else:
            rag = _new_event_store(args)
        event_ids = rag.remember(
            text,
            source=args.source,
            metadata={"path": str(args.input)} if args.input else {},
            captured_at=args.captured_at,
            collection=args.collection,
        )
        save_event_rag(args.store, rag, event_type="remember", event_payload={"event_ids": event_ids})
        payload = {"store": str(args.store), "event_ids": event_ids, "events": len(rag.events)}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        print(f"remembered events={len(event_ids)} store={args.store}")
        return

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

    if args.command == "collection":
        if args.collection_command == "create":
            rag = _load_event_store(args.store, args, require_providers=False) if args.store.exists() else EventRAG(require_providers=False)
            collection = rag.create_collection(
                args.name,
                collection_id=args.id,
                schema=_parse_json_arg(args.schema_json, "schema-json"),
                metadata=_parse_json_arg(args.metadata_json, "metadata-json"),
            )
            save_event_rag(
                args.store,
                rag,
                event_type="collection_create",
                event_payload={"collection_id": collection.collection_id},
            )
            payload = {"collection": _collection_to_cli_dict(collection), "store": str(args.store)}
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return
            print(f"collection created id={collection.collection_id} store={args.store}")
            return

        rag = _load_event_store(args.store, args, require_providers=False)
        if args.collection_command == "list":
            payload = {"collections": [_collection_to_cli_dict(item) for item in rag.list_collections()]}
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return
            for item in payload["collections"]:
                print(f"{item['collection_id']}\t{item['name']}\t{item['status']}")
            return

        if args.collection_command == "show":
            collection = rag.collections.get(args.collection_id)
            if collection is None:
                parser.error(f"unknown collection: {args.collection_id}")
            payload = {"collection": _collection_to_cli_dict(collection)}
            print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else json.dumps(payload["collection"], ensure_ascii=False, indent=2))
            return

        if args.collection_command == "update":
            try:
                collection = rag.update_collection(
                    args.collection_id,
                    name=args.name,
                    schema=_parse_json_arg(args.schema_json, "schema-json") if args.schema_json else None,
                    metadata=_parse_json_arg(args.metadata_json, "metadata-json") if args.metadata_json else None,
                    status=args.status,
                )
            except KeyError as exc:
                parser.error(str(exc))
            save_event_rag(
                args.store,
                rag,
                event_type="collection_update",
                event_payload={"collection_id": collection.collection_id},
            )
            payload = {"collection": _collection_to_cli_dict(collection), "store": str(args.store)}
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return
            print(f"collection updated id={collection.collection_id} store={args.store}")
            return

    if args.command == "index":
        if not is_event_store(args.store):
            parser.error("index currently supports v4 event stores only")
        if args.store.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
            parser.error("persisted HNSW indexes are supported for SQLite stores only")
        rag = _load_event_store(args.store, args, require_providers=False)
        _apply_event_semantic_args(rag, args)
        result = rag.build_semantic_indexes(force=True, collection=args.collection)
        save_event_rag(
            args.store,
            rag,
            event_type="index",
            event_payload={
                "semantic_index": args.semantic_index,
                "built": result["built"],
                "item_count": result["item_count"],
                "collection": args.collection,
            },
        )
        payload = {
            "store": str(args.store),
            "semantic_navigation": rag.semantic_navigation_stats(),
            "result": result,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        print(
            "indexed "
            f"items={result['item_count']} built={len(result['built'])} "
            f"store={args.store}"
        )
        return

    if args.command == "delete":
        if not is_event_store(args.store):
            parser.error("delete currently supports v4 event stores only")
        rag = _load_event_store(args.store, args, require_providers=False)
        result = rag.delete_event(args.event_id, soft=not args.hard)
        save_event_rag(args.store, rag, event_type="delete", event_payload=_maintenance_to_dict(result))
        if args.json:
            print(json.dumps(_maintenance_to_dict(result), ensure_ascii=False, indent=2))
            return
        print(f"deleted events={len(result.deleted)} soft={not args.hard} store={args.store}")
        return

    if args.command == "replace":
        if args.input:
            text = args.input.read_text(encoding="utf-8")
        elif args.text:
            text = args.text
        else:
            parser.error("replace requires --text or --input")
        if not is_event_store(args.store):
            parser.error("replace currently supports v4 event stores only")
        rag = _load_event_store(args.store, args, require_providers=True)
        event_ids = rag.replace_event(
            args.event_id,
            text,
            source=args.source,
            metadata={"path": str(args.input)} if args.input else {},
            captured_at=args.captured_at,
            collection=args.collection,
        )
        save_event_rag(
            args.store,
            rag,
            event_type="replace",
            event_payload={"event_id": args.event_id, "replacement_event_ids": event_ids},
        )
        payload = {"replacement_event_ids": event_ids, "store": str(args.store)}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        print(f"replaced event={args.event_id} replacements={len(event_ids)} store={args.store}")
        return

    if args.command == "compact":
        if not is_event_store(args.store):
            parser.error("compact currently supports v4 event stores only")
        db = EventMemoryDB(args.store)
        result = db.compact(purge_deleted=args.purge_deleted, older_than_days=args.older_than_days)
        payload = _maintenance_to_dict(result)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        print(
            f"compacted purged={result.compacted} ann_rebuilt={result.ann_rebuilt} store={args.store}"
        )
        return

    if args.command == "backup":
        if not is_event_store(args.store):
            parser.error("backup currently supports v4 event stores only")
        db = EventMemoryDB(args.store)
        result = db.backup(args.to)
        payload = _maintenance_to_dict(result)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        print(f"backup written to={args.to}")
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

    if args.command == "migrate-v4":
        rag = migrate_v3_to_event_rag(
            args.source,
            args.target,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            llm_base_url=args.llm_base_url,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            embedding_api_key=args.embedding_api_key,
        )
        print(
            json.dumps(
                {
                    "from": str(args.source),
                    "to": str(args.target),
                    "version": 4,
                    "events": len(rag.events),
                    "subtags": sum(len(event.subtags) for event in rag.events.values()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command in {"ask", "feedback", "stats"} and is_event_store(args.store):
        rag = _load_event_store(
            args.store,
            args,
            require_providers=args.command == "ask",
        )
        _apply_event_semantic_args(rag, args)
        if args.command == "ask":
            answer = rag.answer(
                args.query,
                limit=args.limit,
                collection=args.collection,
                filter=_load_filter_arg(args),
            )
            save_event_rag(
                args.store,
                rag,
                event_type="ask",
                event_payload={"query": args.query, "evidence_count": len(answer.evidence)},
            )
            if args.json:
                print(json.dumps(_event_answer_to_dict(answer), ensure_ascii=False, indent=2))
                return
            print(answer.answer)
            print("\nEvidence:")
            for index, item in enumerate(answer.evidence, start=1):
                print(
                    f"{index}. [event score={item.score:.2f} difficulty={item.recall_difficulty:.2f}] "
                    f"{item.main_label}"
                )
            return

        if args.command == "feedback":
            result = rag.apply_feedback(
                positive_event_ids=_parse_fragment_ids(args.positive),
                negative_event_ids=_parse_fragment_ids(args.negative),
                reason=args.reason,
            )
            save_event_rag(args.store, rag, event_type="feedback", event_payload=_event_feedback_to_dict(result))
            if args.json:
                print(json.dumps(_event_feedback_to_dict(result), ensure_ascii=False, indent=2))
                return
            print(
                "event feedback applied "
                f"positive={len(result.positive)} negative={len(result.negative)} "
                f"ignored={len(result.ignored)} store={args.store}"
            )
            return

        if args.command == "stats":
            print(json.dumps(rag.stats(), ensure_ascii=False, indent=2))
            return

    rag = _load_store(
        args.store,
        getattr(args, "embedding_provider", None),
        getattr(args, "embedding_model", "text-embedding-3-small"),
        getattr(args, "retrieval_profile", None),
        getattr(args, "semantic_index", None),
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
                    "forgetting_model": rag.space.dynamics.forgetting_model,
                    "semantic_navigation": rag.semantic_navigation_stats(),
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


def _event_answer_to_dict(answer: EventAnswer) -> dict[str, object]:
    return {
        "query": answer.query,
        "answer": answer.answer,
        "retrieval_plan": {
            "key_question": answer.retrieval_plan.key_question,
            "retrieval_terms": answer.retrieval_plan.retrieval_terms,
            "target_roles": answer.retrieval_plan.target_roles,
            "target_slots": [
                {"role": slot.role, "position": slot.position}
                for slot in answer.retrieval_plan.target_slots
            ],
            "time_hint": answer.retrieval_plan.time_hint,
            "recall_precision": answer.retrieval_plan.recall_precision,
        },
        "diagnostics": answer.diagnostics,
        "evidence": [
            {
                "event_id": item.event_id,
                "collection_id": item.collection_id,
                "main_label": item.main_label,
                "captured_at": item.captured_at,
                "compressed_trace": item.compressed_trace,
                "score": item.score,
                "keyword_score": item.keyword_score,
                "embedding_score": item.embedding_score,
                "recall_difficulty": item.recall_difficulty,
                "matched_slots": item.matched_slots,
                "stages": item.stages,
                "matched_subtags": [
                    {
                        "subtag_id": subtag.subtag_id,
                        "role": subtag.role,
                        "value": subtag.value,
                        "position": subtag.position,
                        "embedding_text": subtag.embedding_text,
                        "confidence": subtag.confidence,
                    }
                    for subtag in item.matched_subtags
                ],
                "source": {
                    "type": item.source.get("type"),
                    "metadata": item.source.get("metadata", {}),
                },
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


def _event_feedback_to_dict(result: EventFeedbackResult) -> dict[str, object]:
    return {
        "positive": result.positive,
        "negative": result.negative,
        "ignored": result.ignored,
        "diagnostics": result.diagnostics,
    }


def _maintenance_to_dict(result: object) -> dict[str, object]:
    return {
        "deleted": getattr(result, "deleted", []),
        "updated": getattr(result, "updated", []),
        "compacted": getattr(result, "compacted", 0),
        "backed_up": getattr(result, "backed_up", None),
        "ann_dirty": getattr(result, "ann_dirty", False),
        "ann_rebuilt": getattr(result, "ann_rebuilt", False),
        "diagnostics": getattr(result, "diagnostics", {}),
    }


def _collection_to_cli_dict(collection: object) -> dict[str, object]:
    schema = getattr(collection, "schema", None)
    return {
        "collection_id": getattr(collection, "collection_id", ""),
        "name": getattr(collection, "name", ""),
        "schema": {
            "allowed_roles": getattr(schema, "allowed_roles", []),
            "required_roles": getattr(schema, "required_roles", []),
            "embedding_model": getattr(schema, "embedding_model", None),
            "metadata_fields": getattr(schema, "metadata_fields", {}),
        },
        "metadata": getattr(collection, "metadata", {}),
        "created_at": getattr(collection, "created_at", None),
        "updated_at": getattr(collection, "updated_at", None),
        "status": getattr(collection, "status", None),
    }


def _parse_json_arg(value: str | None, label: str) -> dict[str, object] | None:
    if not value:
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--{label} must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"--{label} must be a JSON object")
    return payload


def _load_filter_arg(args: argparse.Namespace) -> dict[str, object] | None:
    value = getattr(args, "filter_json", None) or getattr(args, "filter", None)
    if not value:
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--filter-json/--filter must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--filter-json/--filter must be a JSON object")
    return payload


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
    semantic_index: str | None = None,
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
    if semantic_index:
        rag.set_semantic_index(semantic_index)
    return rag


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-provider", choices=["openai", "openai-compatible"], default="openai-compatible")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1")


def _add_event_provider_args(parser: argparse.ArgumentParser) -> None:
    _add_llm_args(parser)
    parser.add_argument("--embedding-provider", choices=["openai"], required=True)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-api-key")


def _new_event_store(args: argparse.Namespace) -> EventRAG:
    return EventRAG(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        llm_base_url=args.llm_base_url,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_api_key=args.embedding_api_key,
    )


def _load_event_store(
    store: Path,
    args: argparse.Namespace,
    *,
    require_providers: bool,
) -> EventRAG:
    return load_event_rag(
        store,
        llm_provider=getattr(args, "llm_provider", None),
        llm_model=getattr(args, "llm_model", None),
        llm_api_key=getattr(args, "llm_api_key", None),
        llm_base_url=getattr(args, "llm_base_url", "https://api.openai.com/v1"),
        embedding_provider=getattr(args, "embedding_provider", None),
        embedding_model=getattr(args, "embedding_model", "text-embedding-3-small"),
        embedding_api_key=getattr(args, "embedding_api_key", None),
        require_providers=require_providers,
    )


def _apply_event_semantic_args(rag: EventRAG, args: argparse.Namespace) -> None:
    semantic_index = getattr(args, "semantic_index", None)
    if semantic_index:
        rag.set_semantic_index(
            semantic_index,
            min_items=getattr(args, "semantic_index_min_items", None),
            m=getattr(args, "semantic_index_m", None),
            ef_construction=getattr(args, "semantic_index_ef_construction", None),
            ef_search=getattr(args, "semantic_index_ef_search", None),
            seed=getattr(args, "semantic_index_seed", None),
        )


if __name__ == "__main__":
    main()
