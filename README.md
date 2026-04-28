# HuMem Product

HuMem Product 是一个可直接嵌入应用的本地分层记忆 RAG 模块。它从 HuMem 研究原型中拆出稳定的产品层，保留“上层稀疏锚点、下层稠密细节、关系牵引召回、读取强化、遗忘下沉”的记忆机制。

它适合做：

- AI 应用的长期记忆层；
- 个人知识库、客服工单、项目日志的轻量 RAG；
- 不想引入向量数据库时的本地检索与证据层；
- 需要“记忆会随使用变化”的 Agent memory；
- 后续接入 LLM 前的可解释上下文召回模块。

## 架构图

![HuMem Product layered memory RAG architecture](docs/assets/layered-memory-rag.svg)

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 文档写入 | `add_document()` 会切分文本、解析记忆片段、建立来源映射 |
| 记忆写入 | `add_memory()` 适合写入短事实、用户偏好、会话摘要 |
| 分层记忆 | 上层保存易召回锚点，下层保存噪声更高但可能有用的细节 |
| 关系扩展 | 被 sealed 的底层细节不会轻易直接命中，但可以被上层锚点通过关系带出 |
| 证据对象 | 查询返回 `MemoryEvidence`，包含来源、层级、分数、关系路径和 chunk |
| 抽取式答案 | `answer()` 会基于最佳证据组装一个可直接返回或交给 LLM 的上下文答案 |
| 读取强化 | 每次 retrieve/answer 会温和强化被命中的片段 |
| 遗忘衰减 | `decay()` 会让弱激活片段下沉，模拟长期记忆的可提取性变化 |
| 本地持久化 | `save()` / `load()` 使用 JSON 保存完整记忆图、文档、chunk 和计数器 |
| 零运行依赖 | Python 3.11+ 即可运行，不需要 Torch、向量数据库或外部服务 |

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

## 快速开始

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
restored = LayeredMemoryRAG.load("memory-store.json")
```

返回的 `RAGAnswer` 包含三部分：

```python
answer.query          # 原始查询
answer.answer         # 基于证据 chunk 组装的抽取式答案
answer.evidence       # MemoryEvidence 列表
answer.diagnostics    # fragment_count, relation_count, layer_histogram 等诊断信息
```

## 命令行使用

写入文档：

```bash
humem-product ingest notes.txt --store memory-store.json --title "Launch Notes"
```

查询：

```bash
humem-product ask "robot launch checklist" --store memory-store.json
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
| `score` | 综合匹配分数 |
| `via_relation` | 如果是关系扩展带出的命中，会显示关系类型 |
| `document_id` | 来源文档 |
| `chunk_id` | 来源 chunk |
| `title` | 来源标题 |
| `chunk_text` | 可交给 LLM 的完整 chunk 文本 |

### `LayeredMemoryRAG.answer`

```python
answer = rag.answer("incident response runbook", limit=6)
```

`answer()` 不调用 LLM。它返回的是抽取式答案和证据。你可以直接展示，也可以把 `answer.evidence` 拼进自己的 LLM prompt。

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

JSON store 保存：

- memory space 配置；
- fragments；
- relations；
- source documents；
- chunks；
- activation / strength / retrieval counters；
- relation cross-layer 状态。

## 后端集成示例

一个典型服务端可以在启动时加载 JSON store，在请求后保存：

```python
from humem_product import LayeredMemoryRAG

STORE_PATH = "memory-store.json"

try:
    rag = LayeredMemoryRAG.load(STORE_PATH)
except FileNotFoundError:
    rag = LayeredMemoryRAG()


def ingest_note(user_id: str, text: str) -> str:
    return rag.add_document(
        text,
        title=f"user:{user_id}",
        metadata={"user_id": user_id},
    )


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
- 查询本身会改变记忆状态，让常用内容更容易被再次召回。

这让它更像一个“会变化的长期记忆层”，而不只是静态检索器。

## 项目结构

```text
humem-product/
  src/humem_product/
    __init__.py
    cli.py             # command-line interface
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
- sealed 底层细节通过上层锚点关系召回。


## 路线图

- 增加可选 LLM answer composer；
- 增加 SQLite store；
- 增加 HTTP service wrapper；
- 增加更好的中文分词和关系抽取；
- 增加多租户 namespace；
- 增加检索 trace 可视化。

## License

MIT
