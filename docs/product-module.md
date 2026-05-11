# HuMem Product Module: LayeredMemoryRAG

## 2026 Event Memory Direction

The product mainline is now `EventRAG`. `LayeredMemoryRAG` remains available as
the legacy local baseline, but new work should use the event-centric temporal
memory model:

- input text is chunked by LLM-extracted events, not by surface sentences alone;
- every event stores a `main_label`, wall-clock `captured_at`, ordered subtags,
  and a compressed trace;
- subtags encode roles such as `when`, `who`, `action`, `object`, `place`,
  `state`, `cause`, `outcome`, and `intent`;
- embeddings are created for the main label and subtag `embedding_text` only,
  while original source text is retained for audit/debug;
- query planning is also LLM-driven and produces retrieval terms, target slots
  such as `place@3`, compatible target roles, a time hint, and recall precision;
- older memories are not deleted, but wall-clock Ebbinghaus difficulty raises
  the recall threshold so they need more precise prompts;
- the v4 viewer uses vertical time and a horizontal embedding-derived semantic
  plane, replacing the old arbitrary 3D spatial interpretation.

SQLite v4 stores also provide a lightweight database core for event memory:
collections/namespaces, per-collection role schemas, a shared filter DSL,
incremental HNSW maintenance, soft delete/replace, compaction, backup, and
WAL-backed local concurrency. JSON stores remain a portable event-memory format,
while SQLite is the target for database-level operations.

Minimal API shape:

```python
from humem_product import EventRAG

rag = EventRAG(
    llm_provider="openai-compatible",
    llm_model="your-chat-model",
    embedding_provider="openai",
)

rag.remember("我上个月因为 offer 还没确定，所以暂时没申请贷款。现在 offer 下来了，我想看额度。")
answer = rag.answer("我之前为什么没申请贷款？")
print(answer.answer)
```

`LayeredMemoryRAG` turns the research prototype into a deployable memory/RAG
module. It keeps the original HuMem idea:

- upper layers store sparse, easy-to-recall anchors;
- lower layers store dense details that should not always be retrieved directly;
- relations can pull sealed lower-layer details into the answer path;
- reads reinforce useful memories, while decay cycles push weak memories downward.

## Python API

```python
from humem_product import LayeredMemoryRAG

rag = LayeredMemoryRAG()
rag.add_document(
    "Alice keeps the robot launch checklist in the blue notebook. "
    "The temporary launch code 45123789 appeared once on the receipt.",
    document_id="launch-notes",
    title="Launch Notes",
)

answer = rag.answer("robot launch checklist")
print(answer.answer)
for item in answer.evidence:
    print(item.title, item.layer, item.score, item.text)

rag.save("memory-store.json")
restored = LayeredMemoryRAG.load("memory-store.json")
```

## CLI

```bash
humem-product ingest notes.txt --store memory-store.json --title "Launch Notes"
humem-product ask "robot launch checklist" --store memory-store.json
humem-product stats --store memory-store.json
humem-product decay --store memory-store.json --cycles 3
```

If the package is installed, the same commands are available through the
`humem` console script.

## Returned Object

`rag.answer(query)` returns `RAGAnswer`:

- `answer`: extractive answer assembled from the best source chunks;
- `evidence`: scored `MemoryEvidence` items with layer, source document, chunk,
  relation path, activation, and strength;
- `diagnostics`: fragment count, relation count, layer histogram, and evidence
  count.

## Production Boundary

This module is intentionally local and deterministic. It can already be used as
a product memory layer behind an API service. The trainable `MemoryCore` remains
available as the research path for replacing or reranking the current rule-based
retriever later.
