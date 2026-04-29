from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .embeddings import make_embedding_provider
from .models import MemoryFragment, MemoryRelation
from .rag import LayeredMemoryRAG, SourceChunk, SourceDocument


SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
STORE_VERSION = 2


def load_rag(
    path: str | Path,
    *,
    embedding_provider: str | None = None,
    embedding_model: str = "text-embedding-3-small",
) -> LayeredMemoryRAG:
    store = Path(path)
    if _is_sqlite_store(store):
        rag = _load_sqlite(store)
        if embedding_provider:
            rag.embedding_provider = make_embedding_provider(
                embedding_provider,
                model=embedding_model,
            )
        return rag
    if embedding_provider:
        return LayeredMemoryRAG.load_with_embeddings(
            store,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
    return LayeredMemoryRAG.load(store)


def save_rag(
    path: str | Path,
    rag: LayeredMemoryRAG,
    *,
    event_type: str | None = None,
    event_payload: dict[str, Any] | None = None,
) -> None:
    store = Path(path)
    if _is_sqlite_store(store):
        _save_sqlite(store, rag, event_type=event_type, event_payload=event_payload)
        return
    rag.save(store)


def migrate_json_to_sqlite(json_path: str | Path, db_path: str | Path) -> LayeredMemoryRAG:
    rag = LayeredMemoryRAG.load(json_path)
    save_rag(
        db_path,
        rag,
        event_type="migrate",
        event_payload={"from": str(json_path), "to": str(db_path)},
    )
    return rag


def log_event(
    path: str | Path,
    event_type: str,
    *,
    fragment_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    store = Path(path)
    if not _is_sqlite_store(store):
        return
    with closing(sqlite3.connect(store)) as connection:
        _ensure_schema(connection)
        with connection:
            _insert_event(connection, event_type, fragment_id=fragment_id, payload=payload)


def _is_sqlite_store(path: Path) -> bool:
    return path.suffix.lower() in SQLITE_SUFFIXES


def _load_sqlite(path: Path) -> LayeredMemoryRAG:
    if not path.exists():
        raise FileNotFoundError(path)

    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_schema(connection)
        meta = connection.execute("SELECT config_json FROM store_meta WHERE id = 1").fetchone()
        config = json.loads(meta["config_json"]) if meta else {}
        rag = LayeredMemoryRAG(**config)

        rag.documents = {
            row["document_id"]: SourceDocument(
                document_id=row["document_id"],
                title=row["title"],
                text=row["text"],
                metadata=_loads_json(row["metadata_json"], {}),
            )
            for row in connection.execute("SELECT * FROM documents")
        }
        rag.chunks = {
            row["chunk_id"]: SourceChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                text=row["text"],
                ordinal=row["ordinal"],
                metadata=_loads_json(row["metadata_json"], {}),
                embedding=_loads_json(row["embedding_json"], None),
                embedding_model=row["embedding_model"],
            )
            for row in connection.execute("SELECT * FROM chunks")
        }

        space_payload = {
            "fragments": [
                {
                    "fragment_id": row["fragment_id"],
                    "text": row["text"],
                    "normalized_text": row["normalized_text"],
                    "kind": row["kind"],
                    "layer": row["layer"],
                    "depth": row["depth"],
                    "x": row["x"],
                    "y": row["y"],
                    "z": row["z"],
                    "activation": row["activation"],
                    "strength": row["strength"],
                    "ease": row["ease"],
                    "retrievals": row["retrievals"],
                    "reinforcements": row["reinforcements"],
                    "forgettings": row["forgettings"],
                    "source_count": row["source_count"],
                    "metadata": _loads_json(row["metadata_json"], {}),
                }
                for row in connection.execute("SELECT * FROM fragments")
            ],
            "relations": [
                {
                    "relation_id": row["relation_id"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "relation_type": row["relation_type"],
                    "weight": row["weight"],
                    "cross_layer": bool(row["cross_layer"]),
                    "metadata": _loads_json(row["metadata_json"], {}),
                }
                for row in connection.execute("SELECT * FROM relations")
            ],
        }
        rag._restore_space(space_payload)
        return rag


def _save_sqlite(
    path: Path,
    rag: LayeredMemoryRAG,
    *,
    event_type: str | None,
    event_payload: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _ensure_schema(connection)
        with connection:
            _replace_core_tables(connection, rag)
            if event_type:
                _insert_event(connection, event_type, payload=event_payload)


def _replace_core_tables(connection: sqlite3.Connection, rag: LayeredMemoryRAG) -> None:
    now = _utc_now()
    config_json = _dumps_json(rag.space.snapshot()["config"])
    connection.execute(
        """
        INSERT INTO store_meta (id, version, created_at, updated_at, config_json)
        VALUES (1, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          version = excluded.version,
          updated_at = excluded.updated_at,
          config_json = excluded.config_json
        """,
        (STORE_VERSION, now, now, config_json),
    )
    for table in ("relations", "fragments", "chunks", "documents"):
        connection.execute(f"DELETE FROM {table}")

    connection.executemany(
        """
        INSERT INTO documents (document_id, title, text, metadata_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            (document.document_id, document.title, document.text, _dumps_json(document.metadata))
            for document in rag.documents.values()
        ),
    )
    connection.executemany(
        """
        INSERT INTO chunks (
          chunk_id, document_id, text, ordinal, metadata_json, embedding_json, embedding_model
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.text,
                chunk.ordinal,
                _dumps_json(chunk.metadata),
                _dumps_json(chunk.embedding),
                chunk.embedding_model,
            )
            for chunk in rag.chunks.values()
        ),
    )
    connection.executemany(
        """
        INSERT INTO fragments (
          fragment_id, text, normalized_text, kind, layer, depth, x, y, z,
          activation, strength, ease, retrievals, reinforcements, forgettings,
          source_count, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                fragment.fragment_id,
                fragment.text,
                fragment.normalized_text,
                fragment.kind,
                fragment.layer,
                fragment.depth,
                fragment.x,
                fragment.y,
                fragment.z,
                fragment.activation,
                fragment.strength,
                fragment.ease,
                fragment.retrievals,
                fragment.reinforcements,
                fragment.forgettings,
                fragment.source_count,
                _dumps_json(fragment.metadata),
            )
            for fragment in rag.space.fragments.values()
        ),
    )
    connection.executemany(
        """
        INSERT INTO relations (
          relation_id, source_id, target_id, relation_type, weight, cross_layer, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                relation.relation_id,
                relation.source_id,
                relation.target_id,
                relation.relation_type,
                relation.weight,
                int(relation.cross_layer),
                _dumps_json(relation.metadata),
            )
            for relation in rag.space.relations.values()
        ),
    )


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS store_meta (
          id INTEGER PRIMARY KEY CHECK (id = 1),
          version INTEGER NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          config_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
          document_id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          text TEXT NOT NULL,
          metadata_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
          chunk_id TEXT PRIMARY KEY,
          document_id TEXT NOT NULL,
          text TEXT NOT NULL,
          ordinal INTEGER NOT NULL,
          metadata_json TEXT NOT NULL,
          embedding_json TEXT,
          embedding_model TEXT,
          FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS fragments (
          fragment_id TEXT PRIMARY KEY,
          text TEXT NOT NULL,
          normalized_text TEXT NOT NULL,
          kind TEXT NOT NULL,
          layer INTEGER NOT NULL,
          depth REAL NOT NULL,
          x REAL NOT NULL,
          y REAL NOT NULL,
          z REAL NOT NULL,
          activation REAL NOT NULL,
          strength REAL NOT NULL,
          ease REAL NOT NULL,
          retrievals INTEGER NOT NULL,
          reinforcements INTEGER NOT NULL,
          forgettings INTEGER NOT NULL,
          source_count INTEGER NOT NULL,
          metadata_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS relations (
          relation_id TEXT PRIMARY KEY,
          source_id TEXT NOT NULL,
          target_id TEXT NOT NULL,
          relation_type TEXT NOT NULL,
          weight REAL NOT NULL,
          cross_layer INTEGER NOT NULL,
          metadata_json TEXT NOT NULL,
          FOREIGN KEY (source_id) REFERENCES fragments(fragment_id) ON DELETE CASCADE,
          FOREIGN KEY (target_id) REFERENCES fragments(fragment_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS events (
          event_id TEXT PRIMARY KEY,
          event_type TEXT NOT NULL,
          fragment_id TEXT,
          payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_fragments_layer ON fragments(layer);
        CREATE INDEX IF NOT EXISTS idx_fragments_kind ON fragments(kind);
        CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
        CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
        """
    )


def _insert_event(
    connection: sqlite3.Connection,
    event_type: str,
    *,
    fragment_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO events (event_id, event_type, fragment_id, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (str(uuid4()), event_type, fragment_id, _dumps_json(payload or {}), _utc_now()),
    )


def _dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads_json(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
