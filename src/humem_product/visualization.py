from __future__ import annotations

import json
import mimetypes
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .rag import LayeredMemoryRAG
from .storage import load_rag


def build_graph_data(rag: LayeredMemoryRAG) -> dict[str, Any]:
    """Convert a RAG store into the compact graph shape used by the viewer."""
    nodes: list[dict[str, Any]] = []
    fragments = rag.space.fragments

    for fragment in fragments.values():
        source = _best_source(fragment.metadata.get("sources"))
        chunk_id = source.get("chunkId") if source else None
        chunk = rag.chunks.get(chunk_id or "")
        consolidation = fragment.metadata.get("consolidation")
        is_anchor = isinstance(consolidation, dict) and bool(consolidation.get("anchor"))
        nodes.append(
            {
                "id": fragment.fragment_id,
                "text": fragment.text,
                "kind": fragment.kind,
                "layer": fragment.layer,
                "depth": fragment.depth,
                "x": fragment.x,
                "y": fragment.y,
                "z": fragment.z,
                "activation": fragment.activation,
                "strength": fragment.strength,
                "accessibility": rag.space.accessibility_weight(fragment.depth),
                "retrievals": fragment.retrievals,
                "source": source,
                "chunkText": chunk.text if chunk else None,
                "isConsolidationAnchor": is_anchor,
                "consolidation": consolidation if is_anchor else None,
                "layoutModel": fragment.metadata.get("layout_model", "hash-fallback"),
                "hasEmbeddingLayout": bool(fragment.metadata.get("layout_has_embedding")),
                "embeddingScope": fragment.metadata.get("layout_embedding_scope", "none"),
                "semanticEdgeCount": fragment.metadata.get("layout_semantic_edges", 0),
                "relationEdgeCount": fragment.metadata.get("layout_relation_edges", 0),
                "layoutUpdatedAt": fragment.metadata.get("layout_updated_at"),
            }
        )

    links: list[dict[str, Any]] = []
    for relation in rag.space.relations.values():
        if relation.source_id not in fragments or relation.target_id not in fragments:
            continue
        links.append(
            {
                "id": relation.relation_id,
                "source": relation.source_id,
                "target": relation.target_id,
                "type": relation.relation_type,
                "weight": relation.weight,
                "crossLayer": relation.cross_layer,
            }
        )

    layout_models = {
        str(fragment.metadata.get("layout_model", "hash-fallback"))
        for fragment in fragments.values()
    }
    has_embedding_layout = any(
        bool(fragment.metadata.get("layout_has_embedding"))
        for fragment in fragments.values()
    )
    embedding_scopes = {
        str(fragment.metadata.get("layout_embedding_scope", "none"))
        for fragment in fragments.values()
    }
    semantic_edge_count = max(
        (int(fragment.metadata.get("layout_semantic_edges", 0)) for fragment in fragments.values()),
        default=0,
    )
    relation_edge_count = max(
        (int(fragment.metadata.get("layout_relation_edges", 0)) for fragment in fragments.values()),
        default=0,
    )
    consolidation_anchor_count = sum(
        1
        for fragment in fragments.values()
        if isinstance(fragment.metadata.get("consolidation"), dict)
        and bool(fragment.metadata["consolidation"].get("anchor"))
    )
    layout_updated_at = max(
        (str(fragment.metadata.get("layout_updated_at", "")) for fragment in fragments.values()),
        default="",
    ) or None

    return {
        "nodes": nodes,
        "links": links,
        "meta": {
            "totalLayers": rag.total_layers,
            "layerHistogram": rag.layer_histogram(),
            "documentCount": len(rag.documents),
            "chunkCount": len(rag.chunks),
            "fragmentCount": len(rag.space.fragments),
            "relationCount": len(rag.space.relations),
            "layoutModel": "+".join(sorted(layout_models)) if layout_models else "empty",
            "hasEmbeddingLayout": has_embedding_layout,
            "embeddingScope": "+".join(sorted(embedding_scopes)) if embedding_scopes else "none",
            "semanticEdgeCount": semantic_edge_count,
            "relationEdgeCount": relation_edge_count,
            "consolidationAnchorCount": consolidation_anchor_count,
            "layoutUpdatedAt": layout_updated_at,
            "retrievalProfile": rag.retrieval_profile.name,
            "memoryWeight": rag.memory_weight,
            "embeddingWeight": rag.embedding_weight,
        },
    }


def load_graph_data(store_path: str | Path) -> dict[str, Any]:
    rag = load_rag(store_path)
    if _needs_layout(rag):
        rag.layout_memory_space(use_embeddings=True, iterations=60)
    return build_graph_data(rag)


def run_visualization_server(
    store_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    store = Path(store_path)
    if not store.exists():
        raise FileNotFoundError(f"Memory store does not exist: {store}")

    handler = _make_handler(store)
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        raise OSError(f"Could not start visualization server on {host}:{port}: {exc}") from exc

    url = f"http://{host}:{port}/"
    print(f"HuMem memory visualization running at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping visualization server.")
    finally:
        server.server_close()


def _best_source(value: Any) -> dict[str, str] | None:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return {
                "documentId": str(first.get("document_id", "")),
                "chunkId": str(first.get("chunk_id", "")),
                "title": str(first.get("title", "")),
            }
    return None


def _make_handler(store_path: Path) -> type[BaseHTTPRequestHandler]:
    cache: dict[str, Any] = {"mtime": None, "graph": None}

    class VisualizationHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
            parsed = urlparse(self.path)
            route = unquote(parsed.path)

            if route == "/api/graph":
                mtime = store_path.stat().st_mtime
                if cache["mtime"] != mtime or cache["graph"] is None:
                    cache["mtime"] = mtime
                    cache["graph"] = load_graph_data(store_path)
                self._send_json(cache["graph"])
                return

            if route in {"", "/"}:
                route = "/index.html"

            self._send_static(route.lstrip("/"))

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_static(self, relative_path: str) -> None:
            path_parts = Path(relative_path).parts
            if ".." in path_parts:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            try:
                static_root = resources.files("humem_product").joinpath("viewer")
                resource = static_root.joinpath(*path_parts)
                if not resource.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                body = resource.read_bytes()
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            content_type = mimetypes.guess_type(relative_path)[0] or "application/octet-stream"
            if relative_path.endswith(".js"):
                content_type = "text/javascript"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return VisualizationHandler


def _needs_layout(rag: LayeredMemoryRAG) -> bool:
    if not rag.space.fragments:
        return False
    return not all("layout_model" in fragment.metadata for fragment in rag.space.fragments.values())
