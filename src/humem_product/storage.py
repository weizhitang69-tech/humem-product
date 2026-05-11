from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .embeddings import EmbeddingProvider, make_embedding_provider
from .event_rag import (
    ACTIVE_STATUS,
    DEFAULT_COLLECTION_ID,
    EVENT_SCHEMA_REVISION,
    EVENT_STORE_VERSION,
    CollectionSchema,
    EventMaintenanceResult,
    EventRAG,
)
from .llm import LLMProvider
from .models import MemoryFragment, MemoryRelation
from .rag import LayeredMemoryRAG, SourceChunk, SourceDocument


SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
STORE_VERSION = 3


def load_rag(
    path: str | Path,
    *,
    embedding_provider: str | EmbeddingProvider | None = None,
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
            rag._invalidate_semantic_navigation_index()
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


def load_event_rag(
    path: str | Path,
    *,
    llm_provider: str | LLMProvider | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str = "https://api.openai.com/v1",
    embedding_provider: str | EmbeddingProvider | None = None,
    embedding_model: str = "text-embedding-3-small",
    embedding_api_key: str | None = None,
    require_providers: bool = False,
) -> EventRAG:
    store = Path(path)
    if _is_sqlite_store(store):
        return _load_event_sqlite(
            store,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            require_providers=require_providers,
        )
    return EventRAG.load(
        store,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        require_providers=require_providers,
    )


def save_event_rag(
    path: str | Path,
    rag: EventRAG,
    *,
    event_type: str | None = None,
    event_payload: dict[str, Any] | None = None,
) -> None:
    store = Path(path)
    if _is_sqlite_store(store):
        _save_event_sqlite(store, rag, event_type=event_type, event_payload=event_payload)
        return
    if event_type:
        rag.events_log.append(
            {
                "log_id": str(uuid4()),
                "event_type": event_type,
                "payload": event_payload or {},
                "created_at": _utc_now(),
            }
        )
    rag.save(store)


def migrate_v3_to_event_rag(
    source_path: str | Path,
    target_path: str | Path,
    *,
    llm_provider: str | LLMProvider,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str = "https://api.openai.com/v1",
    embedding_provider: str | EmbeddingProvider,
    embedding_model: str = "text-embedding-3-small",
    embedding_api_key: str | None = None,
) -> EventRAG:
    source = Path(source_path)
    legacy = load_rag(source)
    event_rag = EventRAG(
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
    )

    chunks = sorted(legacy.chunks.values(), key=lambda item: (item.document_id, item.ordinal))
    for chunk in chunks:
        document = legacy.documents.get(chunk.document_id)
        metadata = {
            "legacy_store_version": 3,
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "ordinal": chunk.ordinal,
        }
        if document:
            metadata["title"] = document.title
            metadata.update(document.metadata)
        event_rag.remember(
            chunk.text,
            source=document.title if document else chunk.document_id,
            metadata=metadata,
        )

    save_event_rag(
        target_path,
        event_rag,
        event_type="migrate_v4",
        event_payload={"from": str(source_path), "to": str(target_path), "chunks": len(chunks)},
    )
    return event_rag


def is_event_store(path: str | Path) -> bool:
    store = Path(path)
    if not store.exists():
        return False
    if _is_sqlite_store(store):
        with closing(sqlite3.connect(store)) as connection:
            meta = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='store_meta'"
            ).fetchone()
            if not meta:
                return False
            row = connection.execute("SELECT version FROM store_meta WHERE id = 1").fetchone()
            return bool(row and int(row[0]) == EVENT_STORE_VERSION)
    try:
        payload = json.loads(store.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(payload.get("version", 0)) == EVENT_STORE_VERSION and payload.get("kind") == "event_memory"


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


class EventMemoryDB:
    """SQLite-first management wrapper for EventRAG stores."""

    def __init__(
        self,
        path: str | Path,
        *,
        llm_provider: str | LLMProvider | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str = "https://api.openai.com/v1",
        embedding_provider: str | EmbeddingProvider | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        self._lock = threading.RLock()

    def load(self, *, require_providers: bool = False) -> EventRAG:
        with self._lock:
            return load_event_rag(
                self.path,
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
                llm_api_key=self.llm_api_key,
                llm_base_url=self.llm_base_url,
                embedding_provider=self.embedding_provider,
                embedding_model=self.embedding_model,
                embedding_api_key=self.embedding_api_key,
                require_providers=require_providers,
            )

    def save(
        self,
        rag: EventRAG,
        *,
        event_type: str | None = None,
        event_payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            save_event_rag(self.path, rag, event_type=event_type, event_payload=event_payload)

    def create_collection(
        self,
        name: str,
        *,
        collection_id: str | None = None,
        schema: CollectionSchema | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with self._lock:
            rag = self.load(require_providers=False) if self.path.exists() else EventRAG(require_providers=False)
            collection = rag.create_collection(
                name,
                collection_id=collection_id,
                schema=schema,
                metadata=metadata,
            )
            self.save(rag, event_type="collection_create", event_payload={"collection_id": collection.collection_id})
            return collection.collection_id

    def compact(
        self,
        *,
        purge_deleted: bool = False,
        older_than_days: int | None = None,
    ) -> EventMaintenanceResult:
        with self._lock:
            rag = self.load(require_providers=False)
            result = rag.compact(purge_deleted=purge_deleted, older_than_days=older_than_days)
            self.save(rag, event_type="compact", event_payload=result.diagnostics)
            if _is_sqlite_store(self.path):
                with closing(sqlite3.connect(self.path)) as connection:
                    _configure_event_connection(connection)
                    connection.execute("VACUUM")
            return result

    def backup(self, target: str | Path) -> EventMaintenanceResult:
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if _is_sqlite_store(self.path):
                with closing(sqlite3.connect(self.path)) as source:
                    _configure_event_connection(source)
                    with closing(sqlite3.connect(target_path)) as destination:
                        source.backup(destination)
            else:
                shutil.copy2(self.path, target_path)
        return EventMaintenanceResult(backed_up=str(target_path))

    def prefilter_event_ids(
        self,
        filter_payload: dict[str, Any] | None = None,
        *,
        collection: str | None = None,
    ) -> set[str]:
        payload = filter_payload or {}
        if not _is_sqlite_store(self.path):
            return set(self.load(require_providers=False).events)
        where = ["1 = 1"]
        params: list[Any] = []
        lifecycle = payload.get("lifecycle", {}) if isinstance(payload, dict) else {}
        include_deleted = bool(lifecycle.get("include_deleted")) if isinstance(lifecycle, dict) else False
        if not include_deleted:
            where.append("e.status != 'deleted'")
        collections = _sql_filter_collections(payload, collection)
        if collections:
            where.append(f"e.collection_id IN ({_placeholders(collections)})")
            params.extend(collections)
        event_conditions, subtag_conditions, roles = _sql_filter_conditions(payload)
        for field, condition in event_conditions.items():
            column = _EVENT_SQL_FIELDS.get(field)
            if column:
                clause, values = _sql_condition(column, condition)
                where.append(clause)
                params.extend(values)
        if roles:
            where.append(
                "EXISTS (SELECT 1 FROM subtags s WHERE s.event_id = e.event_id "
                f"AND s.role IN ({_placeholders(roles)}))"
            )
            params.extend(roles)
        if subtag_conditions:
            sub_where = ["s.event_id = e.event_id"]
            sub_params: list[Any] = []
            for field, condition in subtag_conditions.items():
                column = _SUBTAG_SQL_FIELDS.get(field)
                if column:
                    clause, values = _sql_condition(column, condition)
                    sub_where.append(clause)
                    sub_params.extend(values)
            if len(sub_where) > 1:
                where.append(f"EXISTS (SELECT 1 FROM subtags s WHERE {' AND '.join(sub_where)})")
                params.extend(sub_params)
        sql = f"SELECT e.event_id FROM events e WHERE {' AND '.join(where)}"
        with self._lock, closing(sqlite3.connect(self.path)) as connection:
            _configure_event_connection(connection)
            _ensure_event_schema(connection)
            return {str(row[0]) for row in connection.execute(sql, params)}


def _configure_event_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA journal_mode = WAL")


def _load_event_sqlite(
    path: Path,
    *,
    llm_provider: str | LLMProvider | None,
    llm_model: str | None,
    llm_api_key: str | None,
    llm_base_url: str,
    embedding_provider: str | EmbeddingProvider | None,
    embedding_model: str,
    embedding_api_key: str | None,
    require_providers: bool,
) -> EventRAG:
    if not path.exists():
        raise FileNotFoundError(path)

    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        _configure_event_connection(connection)
        _ensure_event_schema(connection)
        meta = connection.execute("SELECT version, config_json FROM store_meta WHERE id = 1").fetchone()
        if not meta or int(meta["version"]) != EVENT_STORE_VERSION:
            raise ValueError(f"not a v{EVENT_STORE_VERSION} event memory store: {path}")
        config = _loads_json(meta["config_json"], {})

        traces = {
            row["event_id"]: row["text"]
            for row in connection.execute("SELECT event_id, text FROM compressed_traces")
        }
        sources = {
            row["event_id"]: {
                "type": row["source_type"],
                "text": row["text"],
                "metadata": _loads_json(row["metadata_json"], {}),
            }
            for row in connection.execute("SELECT * FROM source_records")
        }
        collections = [
            {
                "collection_id": row["collection_id"],
                "name": row["name"],
                "schema": _loads_json(row["schema_json"], {}),
                "metadata": _loads_json(row["metadata_json"], {}),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "status": row["status"],
            }
            for row in connection.execute("SELECT * FROM collections ORDER BY collection_id")
        ]

        subtags: dict[str, list[dict[str, Any]]] = {}
        for row in connection.execute("SELECT * FROM subtags ORDER BY event_id, position, role"):
            subtags.setdefault(row["event_id"], []).append(
                {
                    "subtag_id": row["subtag_id"],
                    "role": row["role"],
                    "value": row["value"],
                    "position": row["position"],
                    "embedding_text": row["embedding_text"],
                    "confidence": row["confidence"],
                    "embedding": _loads_json(row["embedding_json"], None),
                    "embedding_model": row["embedding_model"],
                    "metadata": _loads_json(row["metadata_json"], {}),
                }
            )

        events = []
        for row in connection.execute("SELECT * FROM events ORDER BY captured_at, event_id"):
            events.append(
                {
                    "event_id": row["event_id"],
                    "main_label": row["main_label"],
                    "captured_at": row["captured_at"],
                    "subtags": subtags.get(row["event_id"], []),
                    "compressed_trace": traces.get(row["event_id"], ""),
                    "source": sources.get(row["event_id"], _loads_json(row["source_json"], {})),
                    "recall_state": _loads_json(row["recall_state_json"], {}),
                    "metadata": _loads_json(row["metadata_json"], {}),
                    "main_embedding": _loads_json(row["main_embedding_json"], None),
                    "embedding_model": row["embedding_model"],
                    "x": row["x"],
                    "y": row["y"],
                    "z": row["z"],
                    "collection_id": row["collection_id"],
                    "status": row["status"],
                    "deleted_at": row["deleted_at"],
                    "updated_at": row["updated_at"],
                    "version": row["version"],
                }
            )

        logs = [
            {
                "log_id": row["log_id"],
                "event_type": row["event_type"],
                "payload": _loads_json(row["payload_json"], {}),
                "created_at": row["created_at"],
            }
            for row in connection.execute("SELECT * FROM events_log ORDER BY created_at, log_id")
        ]
        ann_snapshots = _load_ann_index_snapshots(connection)

    rag = EventRAG.from_snapshot(
        {
            "version": EVENT_STORE_VERSION,
            "kind": "event_memory",
            "event_rag_config": config,
            "collections": collections,
            "events": events,
            "events_log": logs,
        },
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        require_providers=require_providers,
    )
    rag.restore_semantic_index_snapshots(ann_snapshots)
    return rag


def _save_event_sqlite(
    path: Path,
    rag: EventRAG,
    *,
    event_type: str | None,
    event_payload: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if event_type:
        rag.events_log.append(
            {
                "log_id": str(uuid4()),
                "event_type": event_type,
                "payload": event_payload or {},
                "created_at": _utc_now(),
            }
        )
    with closing(sqlite3.connect(path)) as connection:
        _configure_event_connection(connection)
        _ensure_event_schema(connection)
        with connection:
            _replace_event_tables(connection, rag)


def _replace_event_tables(connection: sqlite3.Connection, rag: EventRAG) -> None:
    now = _utc_now()
    snapshot = rag.snapshot()
    config_payload = dict(snapshot["event_rag_config"])
    config_payload["schema_revision"] = EVENT_SCHEMA_REVISION
    config_json = _dumps_json(config_payload)
    connection.execute(
        """
        INSERT INTO store_meta (id, version, created_at, updated_at, config_json)
        VALUES (1, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          version = excluded.version,
          updated_at = excluded.updated_at,
          config_json = excluded.config_json
        """,
        (EVENT_STORE_VERSION, now, now, config_json),
    )
    for table in (
        "ann_index_tombstones",
        "ann_index_edges",
        "ann_index_nodes",
        "ann_index_meta",
        "events_log",
        "embedding_items",
        "source_records",
        "compressed_traces",
        "subtags",
        "events",
        "collections",
    ):
        connection.execute(f"DELETE FROM {table}")

    for collection in snapshot.get("collections", []):
        connection.execute(
            """
            INSERT INTO collections (
              collection_id, name, schema_json, metadata_json, created_at, updated_at, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                collection["collection_id"],
                collection["name"],
                _dumps_json(collection.get("schema", {})),
                _dumps_json(collection.get("metadata", {})),
                collection.get("created_at") or now,
                collection.get("updated_at") or now,
                collection.get("status", ACTIVE_STATUS),
            ),
        )

    for event in snapshot["events"]:
        connection.execute(
            """
            INSERT INTO events (
              event_id, main_label, captured_at, source_json, recall_state_json,
              metadata_json, main_embedding_json, embedding_model, x, y, z,
              collection_id, status, deleted_at, updated_at, version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["event_id"],
                event["main_label"],
                event["captured_at"],
                _dumps_json(event.get("source", {})),
                _dumps_json(event.get("recall_state", {})),
                _dumps_json(event.get("metadata", {})),
                _dumps_json(event.get("main_embedding")),
                event.get("embedding_model"),
                event.get("x", 0.0),
                event.get("y", 0.0),
                event.get("z", 1.0),
                event.get("collection_id", DEFAULT_COLLECTION_ID),
                event.get("status", ACTIVE_STATUS),
                event.get("deleted_at"),
                event.get("updated_at"),
                event.get("version", 1),
            ),
        )
        connection.execute(
            "INSERT INTO compressed_traces (event_id, text) VALUES (?, ?)",
            (event["event_id"], event.get("compressed_trace", "")),
        )
        source = event.get("source", {}) if isinstance(event.get("source"), dict) else {}
        connection.execute(
            """
            INSERT INTO source_records (source_id, event_id, source_type, text, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                f"{event['event_id']}:source",
                event["event_id"],
                str(source.get("type", "memory")),
                str(source.get("text", "")),
                _dumps_json(source.get("metadata", {})),
            ),
        )
        connection.execute(
            """
            INSERT INTO embedding_items (
              item_id, event_id, subtag_id, collection_id, item_type, text,
              embedding_json, embedding_model, status, deleted_at, updated_at
            )
            VALUES (?, ?, NULL, ?, 'main_label', ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{event['event_id']}:main",
                event["event_id"],
                event.get("collection_id", DEFAULT_COLLECTION_ID),
                event["main_label"],
                _dumps_json(event.get("main_embedding")),
                event.get("embedding_model"),
                event.get("status", ACTIVE_STATUS),
                event.get("deleted_at"),
                event.get("updated_at"),
            ),
        )
        for subtag in event.get("subtags", []):
            connection.execute(
                """
                INSERT INTO subtags (
                  subtag_id, event_id, role, value, position, embedding_text,
                  confidence, embedding_json, embedding_model, metadata_json,
                  collection_id, status, deleted_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subtag["subtag_id"],
                    event["event_id"],
                    subtag["role"],
                    subtag["value"],
                    subtag["position"],
                    subtag["embedding_text"],
                    subtag.get("confidence", 1.0),
                    _dumps_json(subtag.get("embedding")),
                    subtag.get("embedding_model"),
                    _dumps_json(subtag.get("metadata", {})),
                    event.get("collection_id", DEFAULT_COLLECTION_ID),
                    event.get("status", ACTIVE_STATUS),
                    event.get("deleted_at"),
                    event.get("updated_at"),
                ),
            )
            connection.execute(
                """
                INSERT INTO embedding_items (
                  item_id, event_id, subtag_id, collection_id, item_type, text,
                  embedding_json, embedding_model, status, deleted_at, updated_at
                )
                VALUES (?, ?, ?, ?, 'subtag', ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{event['event_id']}:{subtag['subtag_id']}",
                    event["event_id"],
                    subtag["subtag_id"],
                    event.get("collection_id", DEFAULT_COLLECTION_ID),
                    subtag["embedding_text"],
                    _dumps_json(subtag.get("embedding")),
                    subtag.get("embedding_model"),
                    event.get("status", ACTIVE_STATUS),
                    event.get("deleted_at"),
                    event.get("updated_at"),
                ),
            )

    connection.executemany(
        """
        INSERT INTO events_log (log_id, event_type, payload_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            (
                log.get("log_id") or str(uuid4()),
                log.get("event_type", "event"),
                _dumps_json(log.get("payload", {})),
                log.get("created_at") or now,
            )
            for log in snapshot.get("events_log", [])
        ),
    )
    _insert_ann_index_snapshots(connection, rag.semantic_index_snapshots())


def _load_ann_index_snapshots(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    table = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ann_index_meta'"
    ).fetchone()
    if not table:
        return []

    nodes_by_index: dict[str, list[dict[str, Any]]] = {}
    for row in connection.execute("SELECT * FROM ann_index_nodes ORDER BY index_name, item_id"):
        nodes_by_index.setdefault(row["index_name"], []).append(
            {
                "item_id": row["item_id"],
                "vector": _loads_json(row["vector_json"], []),
                "level": row["node_level"],
            }
        )

    edges_by_index: dict[str, list[dict[str, Any]]] = {}
    for row in connection.execute("SELECT * FROM ann_index_edges ORDER BY index_name, source_item_id, target_item_id"):
        edges_by_index.setdefault(row["index_name"], []).append(
            {
                "source": row["source_item_id"],
                "target": row["target_item_id"],
                "level": row["edge_level"],
            }
        )

    tombstones_by_index: dict[str, list[str]] = {}
    tombstone_table = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ann_index_tombstones'"
    ).fetchone()
    if tombstone_table:
        for row in connection.execute("SELECT * FROM ann_index_tombstones ORDER BY index_name, item_id"):
            tombstones_by_index.setdefault(row["index_name"], []).append(row["item_id"])

    snapshots: list[dict[str, Any]] = []
    for row in connection.execute("SELECT * FROM ann_index_meta ORDER BY index_name"):
        config = _loads_json(row["config_json"], {})
        snapshots.append(
            {
                "index_name": row["index_name"],
                "embedding_model": row["embedding_model"],
                "item_count": row["item_count"],
                "signature": _loads_json(row["signature_json"], []),
                "tombstones": tombstones_by_index.get(row["index_name"], []),
                "built_at": row["built_at"],
                "snapshot": {
                    "config": config,
                    "entry_id": row["entry_id"],
                    "max_level": row["max_level"],
                    "dimension": row["dimension"],
                    "nodes": nodes_by_index.get(row["index_name"], []),
                    "edges": edges_by_index.get(row["index_name"], []),
                },
            }
        )
    return snapshots


def _insert_ann_index_snapshots(
    connection: sqlite3.Connection,
    snapshots: list[dict[str, Any]],
) -> None:
    now = _utc_now()
    for payload in snapshots:
        index_name = str(payload["index_name"])
        snapshot = payload.get("snapshot", {})
        if not isinstance(snapshot, dict):
            continue
        config = snapshot.get("config", {})
        connection.execute(
            """
            INSERT INTO ann_index_meta (
              index_name, embedding_model, dimension, item_count, signature_json,
              config_json, entry_id, max_level, dirty, built_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                index_name,
                payload.get("embedding_model"),
                snapshot.get("dimension"),
                payload.get("item_count", 0),
                _dumps_json(payload.get("signature", [])),
                _dumps_json(config),
                snapshot.get("entry_id"),
                snapshot.get("max_level", 0),
                1 if payload.get("tombstones") else 0,
                payload.get("built_at") or now,
            ),
        )
        connection.executemany(
            """
            INSERT INTO ann_index_nodes (index_name, item_id, node_level, vector_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                (
                    index_name,
                    node.get("item_id"),
                    node.get("level", 0),
                    _dumps_json(node.get("vector", [])),
                )
                for node in snapshot.get("nodes", [])
                if isinstance(node, dict)
            ),
        )
        connection.executemany(
            """
            INSERT INTO ann_index_edges (index_name, source_item_id, target_item_id, edge_level)
            VALUES (?, ?, ?, ?)
            """,
            (
                (
                    index_name,
                    edge.get("source"),
                    edge.get("target"),
                    edge.get("level", 0),
                )
                for edge in snapshot.get("edges", [])
                if isinstance(edge, dict)
            ),
        )
        connection.executemany(
            """
            INSERT INTO ann_index_tombstones (index_name, item_id, marked_at)
            VALUES (?, ?, ?)
            """,
            (
                (index_name, str(item_id), now)
                for item_id in payload.get("tombstones", [])
            ),
        )


def _ensure_event_schema(connection: sqlite3.Connection) -> None:
    existing_events = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
    ).fetchone()
    if existing_events:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(events)").fetchall()
        }
        if "main_label" not in columns and columns:
            raise ValueError("SQLite store uses the legacy events log schema; use a fresh v4 target")

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS store_meta (
          id INTEGER PRIMARY KEY CHECK (id = 1),
          version INTEGER NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          config_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS collections (
          collection_id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          schema_json TEXT NOT NULL,
          metadata_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          status TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS events (
          event_id TEXT PRIMARY KEY,
          main_label TEXT NOT NULL,
          captured_at TEXT NOT NULL,
          source_json TEXT NOT NULL,
          recall_state_json TEXT NOT NULL,
          metadata_json TEXT NOT NULL,
          main_embedding_json TEXT,
          embedding_model TEXT,
          x REAL NOT NULL,
          y REAL NOT NULL,
          z REAL NOT NULL,
          collection_id TEXT NOT NULL DEFAULT 'default',
          status TEXT NOT NULL DEFAULT 'active',
          deleted_at TEXT,
          updated_at TEXT,
          version INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS subtags (
          subtag_id TEXT PRIMARY KEY,
          event_id TEXT NOT NULL,
          role TEXT NOT NULL,
          value TEXT NOT NULL,
          position INTEGER NOT NULL,
          embedding_text TEXT NOT NULL,
          confidence REAL NOT NULL,
          embedding_json TEXT,
          embedding_model TEXT,
          metadata_json TEXT NOT NULL,
          collection_id TEXT NOT NULL DEFAULT 'default',
          status TEXT NOT NULL DEFAULT 'active',
          deleted_at TEXT,
          updated_at TEXT,
          FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS compressed_traces (
          event_id TEXT PRIMARY KEY,
          text TEXT NOT NULL,
          FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS embedding_items (
          item_id TEXT PRIMARY KEY,
          event_id TEXT NOT NULL,
          subtag_id TEXT,
          item_type TEXT NOT NULL,
          text TEXT NOT NULL,
          embedding_json TEXT,
          embedding_model TEXT,
          collection_id TEXT NOT NULL DEFAULT 'default',
          status TEXT NOT NULL DEFAULT 'active',
          deleted_at TEXT,
          updated_at TEXT,
          FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
          FOREIGN KEY (subtag_id) REFERENCES subtags(subtag_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS source_records (
          source_id TEXT PRIMARY KEY,
          event_id TEXT NOT NULL,
          source_type TEXT NOT NULL,
          text TEXT NOT NULL,
          metadata_json TEXT NOT NULL,
          FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS events_log (
          log_id TEXT PRIMARY KEY,
          event_type TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ann_index_meta (
          index_name TEXT PRIMARY KEY,
          embedding_model TEXT,
          dimension INTEGER,
          item_count INTEGER NOT NULL,
          signature_json TEXT NOT NULL,
          config_json TEXT NOT NULL,
          entry_id TEXT,
          max_level INTEGER NOT NULL,
          dirty INTEGER NOT NULL DEFAULT 0,
          built_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ann_index_nodes (
          index_name TEXT NOT NULL,
          item_id TEXT NOT NULL,
          node_level INTEGER NOT NULL,
          vector_json TEXT NOT NULL,
          PRIMARY KEY (index_name, item_id),
          FOREIGN KEY (index_name) REFERENCES ann_index_meta(index_name) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ann_index_edges (
          index_name TEXT NOT NULL,
          source_item_id TEXT NOT NULL,
          target_item_id TEXT NOT NULL,
          edge_level INTEGER NOT NULL,
          PRIMARY KEY (index_name, source_item_id, target_item_id, edge_level),
          FOREIGN KEY (index_name) REFERENCES ann_index_meta(index_name) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ann_index_tombstones (
          index_name TEXT NOT NULL,
          item_id TEXT NOT NULL,
          marked_at TEXT NOT NULL,
          PRIMARY KEY (index_name, item_id),
          FOREIGN KEY (index_name) REFERENCES ann_index_meta(index_name) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_events_captured_at ON events(captured_at);
        CREATE INDEX IF NOT EXISTS idx_events_collection ON events(collection_id);
        CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
        CREATE INDEX IF NOT EXISTS idx_subtags_event ON subtags(event_id);
        CREATE INDEX IF NOT EXISTS idx_subtags_role ON subtags(role);
        CREATE INDEX IF NOT EXISTS idx_subtags_collection_role ON subtags(collection_id, role);
        CREATE INDEX IF NOT EXISTS idx_embedding_items_event ON embedding_items(event_id);
        CREATE INDEX IF NOT EXISTS idx_embedding_items_collection ON embedding_items(collection_id);
        CREATE INDEX IF NOT EXISTS idx_source_records_event ON source_records(event_id);
        CREATE INDEX IF NOT EXISTS idx_events_log_type ON events_log(event_type);
        CREATE INDEX IF NOT EXISTS idx_ann_index_edges_source ON ann_index_edges(index_name, source_item_id);
        """
    )
    _ensure_event_schema_columns(connection)
    now = _utc_now()
    connection.execute(
        """
        INSERT INTO collections (collection_id, name, schema_json, metadata_json, created_at, updated_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(collection_id) DO NOTHING
        """,
        (
            DEFAULT_COLLECTION_ID,
            DEFAULT_COLLECTION_ID,
            _dumps_json(
                {
                    "allowed_roles": [
                        "action",
                        "cause",
                        "from_where",
                        "intent",
                        "object",
                        "outcome",
                        "place",
                        "state",
                        "to_where",
                        "when",
                        "who",
                        "with_who",
                    ],
                    "required_roles": [],
                    "embedding_model": None,
                    "metadata_fields": {},
                }
            ),
            "{}",
            now,
            now,
            ACTIVE_STATUS,
        ),
    )


def _ensure_event_schema_columns(connection: sqlite3.Connection) -> None:
    _add_column_if_missing(connection, "events", "collection_id", "TEXT NOT NULL DEFAULT 'default'")
    _add_column_if_missing(connection, "events", "status", "TEXT NOT NULL DEFAULT 'active'")
    _add_column_if_missing(connection, "events", "deleted_at", "TEXT")
    _add_column_if_missing(connection, "events", "updated_at", "TEXT")
    _add_column_if_missing(connection, "events", "version", "INTEGER NOT NULL DEFAULT 1")
    _add_column_if_missing(connection, "subtags", "collection_id", "TEXT NOT NULL DEFAULT 'default'")
    _add_column_if_missing(connection, "subtags", "status", "TEXT NOT NULL DEFAULT 'active'")
    _add_column_if_missing(connection, "subtags", "deleted_at", "TEXT")
    _add_column_if_missing(connection, "subtags", "updated_at", "TEXT")
    _add_column_if_missing(connection, "embedding_items", "collection_id", "TEXT NOT NULL DEFAULT 'default'")
    _add_column_if_missing(connection, "embedding_items", "status", "TEXT NOT NULL DEFAULT 'active'")
    _add_column_if_missing(connection, "embedding_items", "deleted_at", "TEXT")
    _add_column_if_missing(connection, "embedding_items", "updated_at", "TEXT")
    _add_column_if_missing(connection, "ann_index_meta", "dirty", "INTEGER NOT NULL DEFAULT 0")


def _add_column_if_missing(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _load_sqlite(path: Path) -> LayeredMemoryRAG:
    if not path.exists():
        raise FileNotFoundError(path)

    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_schema(connection)
        meta = connection.execute("SELECT config_json FROM store_meta WHERE id = 1").fetchone()
        config_payload = json.loads(meta["config_json"]) if meta else {}
        space_config, rag_config = _split_config_payload(config_payload)
        semantic_navigation = rag_config.get("semantic_navigation", {})
        if not isinstance(semantic_navigation, dict):
            semantic_navigation = {}
        rag = LayeredMemoryRAG(
            **space_config,
            retrieval_profile=rag_config.get("retrieval_profile"),
            memory_weight=rag_config.get("memory_weight"),
            embedding_weight=rag_config.get("embedding_weight"),
            semantic_index=semantic_navigation.get("mode", "auto"),
            semantic_index_min_items=semantic_navigation.get("min_items", 256),
            semantic_index_m=semantic_navigation.get("m", 8),
            semantic_index_ef_construction=semantic_navigation.get("ef_construction", 64),
            semantic_index_ef_search=semantic_navigation.get("ef_search", 32),
            semantic_index_seed=semantic_navigation.get("seed", 13),
        )

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
    config_json = _dumps_json(
        {
            "space": rag.space.snapshot()["config"],
            "rag": rag.rag_config(),
        }
    )
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


def _split_config_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "space" in payload or "rag" in payload:
        return dict(payload.get("space", {})), dict(payload.get("rag", {}))
    return dict(payload), {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
