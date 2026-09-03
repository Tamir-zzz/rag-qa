# 企业级 RAG 知识库问答系统

基于 **FastAPI + LangChain + ChromaDB + 本地 Embedding/Ollama** 的检索增强生成（RAG）服务。

**查询改写 → BM25 + 向量混合召回 → RRF 融合 → CrossEncoder 重排 → 父子块展开 → LLM 流式生成**，全程可溯源。

全链路可在无外网、无 API Key 的环境下运行（Embedding、Rerank 与 LLM 均本地化）。

---

## 一、核心链路

```
                  ┌─ 查询改写（多轮指代消解）─┐
用户提问 ─────────┤                          ├─→ 1~N 个查询
                  └─ Multi-Query 变体扩展 ───┘        │
                                                      ▼
                                      ┌─ 向量召回（语义相似，余弦）─┐
                            每查询    │                            ├─ RRF ─→ 查询间 RRF ─→ 候选集
                                      └─ BM25 召回（字面匹配）─────┘                        │
                                                                                            ▼
                                                                                  CrossEncoder 重排
                                                                                            │
                                                                            Small-to-Big 父块展开
                                                                                            │
                                                                                    LLM 流式生成 ─→ SSE
```

| 环节 | 解决的问题 |
| --- | --- |
| **查询改写** | 多轮追问含指代（"他的学历呢？"），直接检索必然失败。改写成独立查询后再召回 |
| **Multi-Query** | 单一表述召回不全，生成多个变体并行检索（可选，默认关闭） |
| **BM25 + 向量** | 向量擅长语义、弱于专有名词；BM25 反之。双路互补 |
| **RRF 融合** | `score = Σ 1/(k+rank)`，只用排名不看分数，天然融通不同量纲的多路结果 |
| **重排** | CrossEncoder 逐对精算，只把最相关片段送进上下文 |
| **父子块** | 小块（500）建索引保证匹配精度，命中后返回父块（1500）保证上下文完整 |

## 二、目录结构

```
rag-qa/
├── app/
│   ├── main.py                       # FastAPI 入口：lifespan 预热、CORS、静态资源
│   ├── core/config.py                # 统一配置（.env 驱动，设备自适应）
│   ├── services/
│   │   ├── rag_engine.py             # 编排层：索引 / 查询改写 / 检索 / 组装 / 生成 / 流式
│   │   ├── retrieval.py              # 检索：向量 + BM25 + 多查询 RRF 融合
│   │   ├── rerank.py                 # 重排：CrossEncoder 精排（懒加载）
│   │   ├── generation.py             # 生成：LLM、查询改写、流式、思维链清洗
│   │   └── store.py                  # 父块存储：SQLite（Small-to-Big）
│   ├── api/v1/endpoints/
│   │   ├── documents.py              # 上传 / 列表 / 删除 / 统计
│   │   └── chat.py                   # 知识库问答（同步 + SSE 流式）
│   └── utils/
│       ├── file_utils.py             # 按扩展名分发 Loader（txt / pdf / docx）
│       └── text_utils.py             # 中英混合分词（BM25 语料切分）
├── streamlit_app.py                  # Streamlit 前端界面（独立进程运行）
├── eval/
│   ├── run_eval.py                   # 评估脚本（内置 / RAGAS 双后端）
│   └── golden_set.example.json       # 黄金测试集模板
├── data/
│   ├── uploads/                      # 原始文件落盘（UUID 重命名）
│   └── parent_store.db               # 父块存储（运行时生成）
├── chroma_db/                        # ChromaDB 持久化目录
├── requirements.txt                  # 主依赖
├── requirements-eval.txt             # 评估可选依赖（RAGAS）
└── .env.example                      # 配置模板
```

## 三、快速开始

### 1. 安装依赖

```bash
cd rag-qa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> GPU 环境请先按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版本，再安装其余依赖。

### 2. 配置环境

```bash
cp .env.example .env      # 按需修改
```

### 3. 启动本地大模型（使用 Ollama 时）

```bash
ollama pull deepseek-r1:1.5b
ollama serve
```

### 4. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- **API 文档**：<http://localhost:8000/docs>

> 首次启动会下载 Embedding 模型（约 100 MB）；首次问答时再下载 Rerank 模型（约 1.1 GB）。

### 5. 启动前端（另开一个终端）

```bash
streamlit run streamlit_app.py
```

前端默认访问 <http://localhost:8501>。若后端不在本机，可通过环境变量指定：

```bash
RAG_API_BASE=http://192.168.1.10:8000 streamlit run streamlit_app.py
```

## 四、前端界面（Streamlit）

`streamlit_app.py` 为独立的 Streamlit 应用，与后端通过 HTTP/SSE 通信，两者互不耦合，可分开部署。

```bash
streamlit run streamlit_app.py     # 默认 http://localhost:8501
```

**侧边栏**
- 后端地址配置（支持远程后端）
- 知识库统计：文档数 / 分块数 / 父块数
- 能力开关可视化：混合检索、重排、父子块、查询改写、多查询
- 文档上传与列表，支持单个删除
- 检索设置：`top_k` 滑杆、检索范围下拉（限定在某文档内）

**主对话区**
- **打字机输出**：逐 token 渲染并带闪烁光标，做了 40ms 节流以平衡流畅度与刷新开销
- **引用溯源**：`references` 事件先于答案渲染，来源卡片展示文件名 / 页码 / 相关度 / 片段；正文中的 `[1]` 自动渲染为上标角标
- **多轮对话**：自动维护最近 10 轮历史并回传后端用于指代消解
- **错误处理**：后端未启动或请求异常时给出明确提示，而非白屏

> 前端只依赖 `streamlit` 与 `requests`，不引入任何后端重量级依赖。

## 五、API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 服务探针（返回前端启动提示） |
| GET | `/health` | 健康检查 |
| POST | `/api/v1/documents/upload` | 上传并索引文档（txt / pdf / docx，≤50 MB） |
| GET | `/api/v1/documents` | 已索引文档列表 |
| DELETE | `/api/v1/documents/{source}` | 按文件名删除文档（清理向量 + 父块 + 磁盘文件） |
| GET | `/api/v1/documents/stats` | 知识库统计（文档数、分块数、父块数、能力开关） |
| POST | `/api/v1/chat` | 知识库问答（一次性返回） |
| POST | `/api/v1/chat/stream` | 知识库问答（**SSE 流式**） |

### 普通问答

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "他的学历是什么？",
    "history": [["这份文档是关于谁的？", "这份文档是关于周阳的简历。"]],
    "top_k": 4
  }'
```

```json
{
  "answer": "根据知识库内容，周阳的学历为……",
  "sources": ["简历-周阳.pdf"],
  "references": [
    {"source": "简历-周阳.pdf", "page": 2, "score": 0.8123, "snippet": "……"}
  ],
  "retrieved": 3,
  "reranked": true,
  "parent_child": true,
  "hybrid": true
}
```

`history` 支持 `{"question": "...", "answer": "..."}` 或 `["问", "答"]` 两种写法。

### 流式问答（SSE）

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "总结一下这份文档的核心内容"}'
```

```
event: references
data: {"sources": ["简历-周阳.pdf"], "references": [...], "retrieved": 4, "reranked": true, ...}

event: token
data: {"content": "根据"}

event: done
data: {"answer": "根据知识库……", "sources": [...], "retrieved": 4, ...}
```

| 事件 | 时机 | 用途 |
| --- | --- | --- |
| `references` | 生成开始前 | 先拿到引用来源，前端可立即渲染引用卡片 |
| `token` | 生成过程中 | 答案增量，已剔除 `<think>` 思维链 |
| `done` | 生成结束 | 完整答案与引用，便于前端落库 |
| `error` | 出错时 | 响应头已发出，只能以事件形式通知，前端需监听 |

客户端消费示例见 `streamlit_app.py` 中的 `iter_sse()` 与 `handle_question()`。

> 若走 Nginx 反代，需关闭缓冲（`proxy_buffering off;`）；响应头已带 `X-Accel-Buffering: no`。

## 六、质量评估

黄金测试集 + 评估脚本，让每一次调参都有据可依。

```bash
cp eval/golden_set.example.json eval/golden_set.json   # 按实际文档填写
python eval/run_eval.py --set eval/golden_set.json
python eval/run_eval.py --set eval/golden_set.json --output eval/report.json
```

输出示例：

```
用例              命中   召回   MRR   忠实   相关   相似   拒答  耗时
--------------------------------------------------------------------------------
q1                1.00  1.00  1.00  0.92  0.78  0.81  1.00  检索0.31s 生成4.21s
q3-multi-turn     1.00  1.00  1.00  0.88  0.74  0.76  1.00  检索0.52s 生成3.98s
q4-out-of-scope     -     -     -      -  0.31    -   1.00  检索0.12s 生成1.87s
--------------------------------------------------------------------------------
汇总              0.85  0.80  0.82  0.90  0.75  0.71  0.95

平均耗时: 检索 0.32s / 生成 3.35s
```

指标说明：

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| 命中 / 召回 / MRR | 检索 | 期望来源是否出现、出现比例、首个正确来源的排名倒数 |
| 忠实度 | 生成 | 用 LLM 逐句判定答案是否可由上下文推出，**衡量幻觉** |
| 相关性 | 生成 | 问题与答案的 Embedding 余弦相似度 |
| 相似度 | 生成 | 答案与 `ground_truth` 的语义相似度（需填写标准答案） |
| 拒答正确性 | 生成 | 知识库外的问题是否正确拒绝回答 |

官方 RAGAS 指标（可选）：

```bash
pip install -r requirements-eval.txt
python eval/run_eval.py --set eval/golden_set.json --backend ragas
```

会额外输出 `faithfulness` / `answer_relevancy` / `context_precision` / `context_recall`。

> 测试集中的 `q3-multi-turn`（含指代追问）与 `q4-out-of-scope`（库外问题）是两类关键用例，分别用于验证查询改写与反幻觉能力，建议保留。

## 七、关键配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama` 本地推理 / `openai` 兼容接口（DeepSeek） |
| `OLLAMA_MODEL` | `deepseek-r1:1.5b` | 本地模型名 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 中文 Embedding |
| `EMBEDDING_DEVICE` | `auto` | 有 CUDA 自动用 GPU，否则 CPU |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | **子块**大小与重叠（用于召回） |
| `PARENT_CHUNK_SIZE` / `PARENT_CHUNK_OVERLAP` | `1500` / `100` | **父块**大小与重叠（用于生成） |
| `TOP_K` / `SCORE_THRESHOLD` | `4` / `0.3` | 返回数量与向量召回相关性下限 |
| `MAX_HISTORY_TURNS` | `3` | 单次问答携带的历史轮数 |
| `ENABLE_QUERY_REWRITE` | `true` | 多轮指代消解式查询改写 |
| `ENABLE_MULTI_QUERY` | `false` | 多查询变体并行检索 |
| `MULTI_QUERY_COUNT` | `3` | 查询变体数量（含原始问题） |
| `ENABLE_HYBRID_SEARCH` | `true` | 关闭则退化为纯向量检索 |
| `CANDIDATE_K` | `20` | 送入重排的候选数，须 ≥ `TOP_K` |
| `RRF_K` | `60` | RRF 融合常数，越大各路差异越平滑 |
| `ENABLE_RERANK` | `true` | 关闭则直接返回融合排序结果 |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | 中文重排模型；`large` 更准但更慢 |
| `RERANK_DEVICE` | `auto` | 重排模型推理设备 |
| `RERANK_THRESHOLD` | `-5.0` | 重排分数下限（logits），剔除明显不相关片段 |
| `ENABLE_PARENT_CHILD` | `true` | 关闭则退化为单级分块 |
| `CORS_ORIGINS` | `*` | 生产环境务必指定具体域名 |

切换到 DeepSeek 云端模型：`.env` 中设置 `LLM_PROVIDER=openai` 并填入 `DEEPSEEK_API_KEY` 即可，无需改动代码。

### 性能调优建议

| 场景 | 建议 |
| --- | --- |
| 快速验证链路 | `ENABLE_RERANK=false`、`ENABLE_MULTI_QUERY=false` |
| 追求延迟（实时对话） | 关重排，或让重排跑 GPU；`CANDIDATE_K` 调至 10~15 |
| 追求准确率（离线分析） | 全开 + `ENABLE_MULTI_QUERY=true`，`CANDIDATE_K=50`、`TOP_K=8` |
| 长文档、答案常被截断 | 调大 `PARENT_CHUNK_SIZE` |
| CPU 环境 | 建议 `ENABLE_RERANK=false`，否则每次问答额外增加秒级开销 |

## 八、设计要点

- **多轮可用**：查询改写消解指代后再检索，而非把含代词的原句直接丢给检索引擎。
- **混合检索**：BM25（字面）+ 向量（语义）双路召回，RRF 融合，互补覆盖专有名词与同义表述。
- **上下文完整**：Small-to-Big 用小块召回、父块生成，兼顾匹配精度与上下文完整性。
- **重排精筛**：CrossEncoder 只放行最相关片段，降低噪声与幻觉。
- **流式优先**：引用先行推送、答案逐 token 返回；思维链清洗器正确处理跨 chunk 的半个 `<think>` 标签。
- **反幻觉**：Prompt 强制只依据上下文作答，无依据时统一返回"知识库中暂无相关信息，无法回答。"。
- **非阻塞**：索引、检索、重排、LLM 推理统一放入线程池；SSE 用同步生成器由 Starlette 自动线程池化。
- **优雅降级**：`rank_bm25` 缺失、BM25 异常、重排加载/推理失败、查询改写失败，均自动回退到上一环节。
- **懒加载**：Embedding 启动时预热，Rerank 首次问答时加载；模块导入不触发任何模型加载。
- **上传安全**：UUID 重命名 + 目录成分剥离，杜绝路径穿越与同名覆盖；索引失败自动回滚。
- **可插拔**：`RetrievalService` / `RerankService` / `GenerationService` / `ParentStore` 相互独立，可单独替换。

## 九、已知限制与后续规划

- **BM25 需全量语料**：当前从 Chroma 拉取全量文档在内存构建索引（带缓存，写入时失效）。万级分块以内无压力，更大规模应改用 Elasticsearch 或 Milvus 稀疏向量。
- **查询改写增加一次 LLM 调用**：多轮对话会多一次推理开销；单轮对话自动跳过，无额外成本。
- **多 worker 部署**（如 `uvicorn --workers 4`）时各进程缓存独立，BM25 索引不同步，需改为单 worker 或接入外部向量库。
- **会话历史**由调用方携带，服务端未持久化（前端维护最近 10 轮）。
- **接口无鉴权与限流**，仅适用于内网或开发环境。
- 规划中：鉴权与限流、Docker 部署、Redis 会话持久化、异步索引与进度回调、表格与扫描件解析、Milvus 生产部署。

> **重要**：本次改造引入了 `chunk_id` 元数据与独立的父块存储，且向量空间为余弦（早期版本为 L2）。**升级后请删除 `chroma_db/` 与 `data/parent_store.db` 并重新上传文档**，否则旧数据缺少 `chunk_id` 会导致 BM25 与向量结果无法对齐、父子块检索失效。
