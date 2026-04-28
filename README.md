# HuMem Product

HuMem Product 是一个可直接嵌入应用的本地分层记忆 RAG 模块。它从 HuMem 研究原型中拆出稳定的产品层，保留“上层稀疏锚点、下层稠密细节、关系牵引召回、读取强化、遗忘下沉”的记忆机制，同时把 embedding 做成可选增强。

默认模式下，它不需要向量数据库、不需要模型 API、不需要 PyTorch；启用 OpenAI embedding 后，它会变成混合检索：`HuMem 分层记忆分数 + embedding 语义相似度`。

它适合做：

- AI 应用的长期记忆层；
- 个人知识库、客服工单、项目日志的轻量 RAG；
- 不想一开始就部署向量数据库的本地检索与证据层；
- 需要“记忆会随使用变化”的 Agent memory；
- 后续接入 LLM 前的可解释上下文召回模块。

## 架构图

![HuMem Product layered memory RAG architecture](docs/assets/layered-memory-rag.svg)

## 两种运行模式

| 模式 | 是否需要 API Key | 适合场景 | 检索方式 |
| --- | --- | --- | --- |
| Local memory mode | 否 | 本地、离线、零成本、可解释记忆 | 分层规则检索 + 关系扩展 |
| Hybrid embedding mode | 是，默认 `OPENAI_API_KEY` | 查询改写、同义表达、语义相似召回 | HuMem 分层分数 + embedding 相似度 |

默认就是 Local memory mode。只有显式传入 `embedding_provider="openai"` 或 CLI 传 `--embedding-provider openai` 时，才会调用 OpenAI Embeddings API。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 文档写入 | `add_document()` 会切分文本、解析记忆片段、建立来源映射 |
| 记忆写入 | `add_memory()` 适合写入短事实、用户偏好、会话摘要 |
| Chunking | 默认 `chunk_size=700`、`chunk_overlap=80`，可在写入时调整 |
| 分层记忆 | 上层保存易召回锚点，下层保存噪声更高但可能有用的细节 |
| 关系扩展 | 被 sealed 的底层细节不会轻易直接命中，但可以被上层锚点通过关系带出 |
| 可选 embedding | 默认模型为 `text-embedding-3-small`，embedding 保存在 JSON store 的 chunk 中 |
| 混合检索 | 默认权重 `memory_weight=0.65`、`embedding_weight=0.35` |
| 证据对象 | 查询返回 `MemoryEvidence`，包含来源、层级、分数、关系路径和 chunk |
| 抽取式答案 | `answer()` 会基于最佳证据组装一个可直接返回或交给 LLM 的上下文答案 |
| 读取强化 | 每次 retrieve/answer 会温和强化被命中的片段 |
| 遗忘衰减 | `decay()` 会让弱激活片段下沉，模拟长期记忆的可提取性变化 |
| 本地持久化 | `save()` / `load()` 使用 JSON 保存完整记忆图、文档、chunk、embedding 和计数器 |

## 安装

从仓库根目录安装：

```bash
pip install -e .
```

安装后会得到命令行工具：

```bash
humem-product --help
```

如果不安装，也可以从仓库根目录临时运行：

```powershell
$env:PYTHONPATH = "src"
python -m humem_product.cli --help
```

## API Key 放在哪里？

如果不启用 embedding，不需要任何 API Key。

如果启用 OpenAI embedding，推荐把 key 放到环境变量：

```powershell
$env:OPENAI_API_KEY = "你的 OpenAI API Key"
```

Linux/macOS：

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

代码中也可以直接传入，但不建议把 key 写进仓库：

```python
rag = LayeredMemoryRAG(
    embedding_provider="openai",
    embedding_api_key="your-openai-api-key",
)
```

## 快速开始：本地模式

```python
from humem_product import LayeredMemoryRAG

rag = LayeredMemoryRAG()

rag.add_document(
    "Alice keeps the robot launch checklist in the blue notebook. "
    "The temporary launch code 45123789 appeared once on the receipt.",
    document_id="launch-notes",
    title="Launch Notes",
)

answer = rag.answer("robot launch checklist", limit=5)

print(answer.answer)
for item in answer.evidence:
    print(item.title, item.layer, item.score, item.text)

rag.save("memory-store.json")
```

## 快速开始：启用 OpenAI embedding

```python
from humem_product import LayeredMemoryRAG

rag = LayeredMemoryRAG(
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
    memory_weight=0.65,
    embedding_weight=0.35,
)

rag.add_document(
    "Alice keeps the robot launch checklist in the blue notebook.",
    title="Launch Notes",
)

# 字面没有 checklist/launch，也能通过 embedding 语义召回
answer = rag.answer("Where is the robotics deployment plan stored?")
print(answer.answer)

rag.save("memory-store.json")
```

如果已有 store 是之前本地模式写入的，可以加载后补 embedding：

```python
rag = LayeredMemoryRAG.load_with_embeddings(
    "memory-store.json",
    embedding_provider="openai",
)

embedded_count = rag.embed_missing_chunks()
rag.save("memory-store.json")
```

## Chunk 和 embedding 细节

Chunking 在 `src/humem_product/rag.py` 的 `_chunk_text()` 中实现：

```python
rag.add_document(
    text,
    chunk_size=700,
    chunk_overlap=80,
)
```

默认策略：

- 先按句子/换行切分；
- 尽量把句子合并到不超过 `chunk_size`；
- 相邻 chunk 保留 `chunk_overlap` 字符的上下文；
- 每个 chunk 会保存 `document_id`、`chunk_id`、`title`；
- 启用 embedding 时，每个 chunk 会保存 `embedding` 和 `embedding_model`。

Embedding provider 在 `src/humem_product/embeddings.py` 中实现。当前内置：

- `OpenAIEmbeddingProvider`
- 默认模型：`text-embedding-3-small`
- 接口：`POST https://api.openai.com/v1/embeddings`
- 不依赖 OpenAI SDK，使用 Python 标准库发请求

OpenAI 官方 Embeddings API 参考见：

- https://platform.openai.com/docs/api-reference/embeddings/create
- https://platform.openai.com/docs/guides/embeddings

## 命令行使用

本地模式写入文档：

```bash
humem-product ingest notes.txt --store memory-store.json --title "Launch Notes"
```

启用 OpenAI embedding 写入文档：

```bash
humem-product ingest notes.txt \
  --store memory-store.json \
  --title "Launch Notes" \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small
```

查询：

```bash
humem-product ask "robot launch checklist" --store memory-store.json
```

启用 embedding 查询。若已有 chunk 没有 embedding，会自动补齐并保存：

```bash
humem-product ask "robotics deployment plan" \
  --store memory-store.json \
  --embedding-provider openai
```

如果不想自动补齐旧 chunk：

```bash
humem-product ask "robotics deployment plan" \
  --store memory-store.json \
  --embedding-provider openai \
  --no-auto-embed
```

返回 JSON，适合服务端或脚本集成：

```bash
humem-product ask "robot launch checklist" --store memory-store.json --json
```

查看状态：

```bash
humem-product stats --store memory-store.json
```

执行遗忘衰减：

```bash
humem-product decay --store memory-store.json --cycles 3 --step 0.14
```

## 产品接口

### `LayeredMemoryRAG.add_document`

```python
doc_id = rag.add_document(
    text,
    document_id="optional-stable-id",
    title="Human Readable Title",
    metadata={"source": "crm"},
    chunk_size=700,
    chunk_overlap=80,
    cool_down_cycles=1,
)
```

适合写入长文本。模块会生成 `SourceDocument`、`SourceChunk`，并把每个记忆片段和来源 chunk 绑定。

### `LayeredMemoryRAG.add_memory`

```python
fragment_ids = rag.add_memory(
    "User prefers concise technical explanations.",
    source="profile",
    metadata={"user_id": "u_123"},
)
```

适合写入短期或长期事实，例如用户偏好、会话摘要、任务状态。

### `LayeredMemoryRAG.retrieve`

```python
evidence = rag.retrieve("incident response runbook", limit=8)
```

返回 `MemoryEvidence`：

| 字段 | 含义 |
| --- | --- |
| `fragment_id` | 记忆片段 ID |
| `text` | 命中的片段文本 |
| `kind` | `clause`、`concept`、`action`、`number`、`term` 等 |
| `layer` | 当前所在记忆层，数字越小越容易召回 |
| `score` | 混合后的最终分数 |
| `memory_score` | HuMem 分层记忆原始分数 |
| `embedding_score` | embedding cosine similarity |
| `via_relation` | 关系扩展类型；embedding 命中会标记为 `embedding` |
| `document_id` | 来源文档 |
| `chunk_id` | 来源 chunk |
| `title` | 来源标题 |
| `chunk_text` | 可交给 LLM 的完整 chunk 文本 |

### `LayeredMemoryRAG.answer`

```python
answer = rag.answer("incident response runbook", limit=6)
```

`answer()` 不调用 LLM。它返回的是抽取式答案和证据。你可以直接展示，也可以把 `answer.evidence` 拼进自己的 LLM prompt。

### `LayeredMemoryRAG.embed_missing_chunks`

```python
count = rag.embed_missing_chunks(batch_size=32)
```

当你给旧 store 开启 embedding 时，用这个方法补齐缺失的 chunk embedding。

### `LayeredMemoryRAG.decay`

```python
rag.decay(cycles=3, step=0.14)
```

遗忘衰减会降低激活和强度，让弱记忆逐渐下沉。这个机制用于把“长期不用的细节”推向较深层，而不是让所有内容永远同等容易被召回。

### `LayeredMemoryRAG.save/load`

```python
rag.save("memory-store.json")
rag = LayeredMemoryRAG.load("memory-store.json")
```

如果要加载并启用 embedding：

```python
rag = LayeredMemoryRAG.load_with_embeddings(
    "memory-store.json",
    embedding_provider="openai",
)
```

JSON store 保存：

- memory space 配置；
- fragments；
- relations；
- source documents；
- chunks；
- optional chunk embeddings；
- activation / strength / retrieval counters；
- relation cross-layer 状态。

## 后端集成示例

```python
from humem_product import LayeredMemoryRAG

STORE_PATH = "memory-store.json"

try:
    rag = LayeredMemoryRAG.load_with_embeddings(
        STORE_PATH,
        embedding_provider="openai",
    )
except FileNotFoundError:
    rag = LayeredMemoryRAG(
        embedding_provider="openai",
    )


def ingest_note(user_id: str, text: str) -> str:
    doc_id = rag.add_document(
        text,
        title=f"user:{user_id}",
        metadata={"user_id": user_id},
    )
    rag.save(STORE_PATH)
    return doc_id


def answer_question(query: str) -> dict:
    result = rag.answer(query)
    rag.save(STORE_PATH)
    return {
        "answer": result.answer,
        "evidence": [
            {
                "text": item.text,
                "title": item.title,
                "layer": item.layer,
                "score": item.score,
                "memory_score": item.memory_score,
                "embedding_score": item.embedding_score,
                "chunk": item.chunk_text,
            }
            for item in result.evidence
        ],
        "diagnostics": result.diagnostics,
    }
```

## 和普通 RAG 的区别

普通 RAG 往往把所有 chunk 放进同一个向量索引里，默认每条记忆有类似的可召回机会。HuMem Product 使用分层记忆图：

- 高频、核心、重复出现的内容更容易上浮；
- 一次性数字、临时编号、低显著细节更容易下沉；
- 下层 sealed 内容不会轻易直接命中，减少噪声；
- 但如果它和上层锚点有关，仍然能通过关系扩展进入证据；
- 查询本身会改变记忆状态，让常用内容更容易被再次召回；
- 启用 embedding 后，语义相似召回会补上规则检索的短板。

这让它更像一个“会变化的长期记忆层”，而不只是静态检索器。

## 项目结构

```text
humem-product/
  src/humem_product/
    __init__.py
    cli.py             # command-line interface
    embeddings.py      # optional OpenAI embedding provider
    memory_space.py    # layered memory graph runtime
    models.py          # dataclasses
    parser.py          # lightweight parser and relation extraction
    rag.py             # product-facing RAG API
  docs/
    assets/
      layered-memory-rag.svg
    product-module.md
  tests/
    test_layered_rag.py
  pyproject.toml
  README.md
```

## 测试

```bash
python -m unittest discover -s tests
```

当前测试覆盖：

- 文档写入和来源追踪；
- answer/evidence 返回；
- JSON save/load；
- sealed 底层细节通过上层锚点关系召回；
- fake embedding provider 的语义召回；
- 旧 store 补齐缺失 embedding。

## 本地评测和可视化

仓库内置一个零依赖评测脚本，会生成 SVG 图和 Markdown 报告：

```bash
python scripts/evaluate_memory_system.py --output reports
```

输出文件：

```text
reports/
  memory_system_report.md
  memory_system_metrics.json
  forgetting_curve.svg
  recall_reinforcement.svg
  linked_recall.svg
```

这组评测会检查：

- 一次性数字记忆是否随 `decay()` 衰减并下沉；
- 重复查询是否强化目标记忆并提升原始召回分数；
- sealed 底层细节是否能通过上层锚点关系被带出。

## 与 V2 研究代码的边界

这个仓库不包含：

- `TrainableMemoryCore`；
- neural encoder / attention / tensor batch；
- V2 bootstrap dataset；
- 训练脚本；
- PyTorch。

这些仍留在原始研究仓库中。`humem-product` 只保留可以直接上线的本地记忆 RAG 模块。

## 路线图

- 增加可选 LLM answer composer；
- 增加 SQLite store；
- 增加 HTTP service wrapper；
- 增加更好的中文分词和关系抽取；
- 增加多租户 namespace；
- 增加检索 trace 可视化。

## License

MIT
