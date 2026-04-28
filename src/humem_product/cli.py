from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rag import LayeredMemoryRAG, RAGAnswer


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

    ask = subparsers.add_parser("ask", help="query a memory store")
    ask.add_argument("query")
    ask.add_argument("--store", type=Path, required=True)
    ask.add_argument("--limit", type=int, default=6)
    ask.add_argument("--json", action="store_true")
    ask.add_argument("--embedding-provider", choices=["openai"])
    ask.add_argument("--embedding-model", default="text-embedding-3-small")
    ask.add_argument("--no-auto-embed", action="store_true")

    stats = subparsers.add_parser("stats", help="print memory store diagnostics")
    stats.add_argument("--store", type=Path, required=True)

    decay = subparsers.add_parser("decay", help="run forgetting cycles and save the store")
    decay.add_argument("--store", type=Path, required=True)
    decay.add_argument("--cycles", type=int, default=1)
    decay.add_argument("--step", type=float, default=0.14)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        if args.store.exists():
            rag = _load_store(args.store, args.embedding_provider, args.embedding_model)
        else:
            rag = LayeredMemoryRAG(
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model,
            )
        text = args.input.read_text(encoding="utf-8")
        doc_id = rag.add_document(
            text,
            document_id=args.document_id,
            title=args.title or args.input.stem,
            metadata={"path": str(args.input)},
        )
        rag.save(args.store)
        print(f"ingested document={doc_id} fragments={len(rag.space.fragments)} store={args.store}")
        return

    rag = _load_store(
        args.store,
        getattr(args, "embedding_provider", None),
        getattr(args, "embedding_model", "text-embedding-3-small"),
    )

    if args.command == "ask":
        if args.embedding_provider and not args.no_auto_embed:
            embedded_count = rag.embed_missing_chunks()
            if embedded_count:
                rag.save(args.store)
        answer = rag.answer(args.query, limit=args.limit)
        rag.save(args.store)
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
        rag.save(args.store)
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
                "score": item.score,
                "memory_score": item.memory_score,
                "embedding_score": item.embedding_score,
                "via_relation": item.via_relation,
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
                "title": item.title,
                "chunk_text": item.chunk_text,
            }
            for item in answer.evidence
        ],
    }


def _load_store(
    store: Path,
    embedding_provider: str | None,
    embedding_model: str,
) -> LayeredMemoryRAG:
    if embedding_provider:
        return LayeredMemoryRAG.load_with_embeddings(
            store,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
    return LayeredMemoryRAG.load(store)


if __name__ == "__main__":
    main()
