# HuMem Product Module: LayeredMemoryRAG

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
