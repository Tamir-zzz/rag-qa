# 企业级 RAG 知识库问答系统

基于 **FastAPI + LangChain + ChromaDB + 本地大模型（Ollama）** 的检索增强生成（RAG）知识库问答系统。
支持文档上传、混合检索、重排、多轮对话与流式回答，**全链路可离线运行，数据不出本地**。

核心链路：**查询改写 → BM25 + 向量混合召回 → RRF 融合 → CrossEncoder 重排 → 父子块展开 → LLM 流式生成（SSE）**，全程可溯源。

---

## 简介

大语言模型存在知识截止、易产生幻觉、不了解企业内部私有数据三大问题。本项目用 RAG（检索增强生成）思路解决：
先根据用户问题从知识库**检索相关片段**，再让大模型**只基于这些片段**作答，相当于给模型"开卷考试"。

系统面向中文场景优化（中文 Embedding + 中文重排模型 + 中文分词），并针对工程落地做了大量健壮性设计：
混合检索互补、重排精筛、多轮指代消解、流式输出、引用溯源、优雅降级与可量化评估。

---

## 核心特性

- **混合检索**：BM25（字面匹配）+ 向量（语义相似）双路召回，RRF 融合互补，兼顾专有名词与同义表述。
- **CrossEncoder 重排**：召回候选经重排模型逐对精算，只把最相关片段送进上下文，降低噪声与幻觉。
- **父子块检索（Small-to-Big）**：小块建索引保证匹配精度，命中后返回父块保证上下文完整，避免答案被截断。
- **多轮对话**：查询改写消解指代（"他的学历？"→"xx的学历？"），让追问也能正确检索；改写失败自动降级。
- **流式输出（SSE）**：答案逐 token 返回（打字机效果），引用来源先于答案推送，前端可即时渲染溯源卡片。
- **全链路本地化**：Embedding、重排、LLM 均本地运行，无需外网与 API Key，适合内网 / 涉密场景。
- **优雅降级**：BM25 缺失、重排失败、改写失败均自动回退上一环节，单点故障不拖垮主链路。
- **可量化评估**：内置黄金测试集与评估脚本，量化检索 / 生成质量，让调参有据可依（兼容 RAGAS）。

---

## 技术架构

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

---

## 技术栈

| 层面 | 技术 |
| --- | --- |
| Web 框架 | FastAPI / Uvicorn / SSE |
| 编排 | LangChain |
| 向量库 | ChromaDB（余弦空间，HNSW） |
| 父块存储 | SQLite |
| Embedding | BAAI/bge-small-zh-v1.5（本地，可 CUDA） |
| 重排 | BAAI/bge-reranker-base（CrossEncoder） |
| 大模型 | Ollama（本地方便切换 qwen2.5 / deepseek-r1 等） |
| 混合检索 | rank-bm25 + jieba 中文分词 |
| 前端 | Streamlit |

---

## 快速开始

### 环境要求

- Python 3.10+
- [Ollama](https://ollama.com/)（本地大模型运行时）
- 可选：NVIDIA GPU + CUDA（embedding 与重排可跑 GPU，速度更快）

### 1. 安装依赖

```bash
cd rag-qa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> GPU 环境请先按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版本的 torch，再安装其余依赖。

### 2. 配置环境

```bash
cp .env.example .env      # 按需修改，默认已面向 Ollama + qwen2.5:7b
```

### 3. 启动本地大模型（Ollama）

Ollama 在 Linux 上默认启用 CUDA，无需额外参数，服务启动后会自动把模型加载到 GPU。

```bash
ollama pull qwen2.5:7b    # 拉取模型（首次）
ollama serve              # 启动 Ollama 服务（监听 11434）
ollama run qwen2.5:7b     # 启动千问模型 ollama ps 可以查看运行状态
```

`.env` 中 `OLLAMA_KEEP_ALIVE=-1` 会让模型常驻显存，避免每次问答冷启动。

### 4. 启动后端

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

---

## 前端界面（Streamlit）

`streamlit_app.py` 为独立的 Streamlit 应用，与后端通过 HTTP/SSE 通信，两者互不耦合，可分开部署。

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

---

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 服务探针 |
| GET | `/health` | 健康检查 |
| POST | `/api/v1/documents/upload` | 上传并索引文档（txt / pdf / docx，≤50 MB） |
| GET | `/api/v1/documents` | 已索引文档列表 |
| DELETE | `/api/v1/documents/{source}` | 按文件名删除文档（清理向量 + 父块 + 磁盘文件） |
| GET | `/api/v1/documents/stats` | 知识库统计（文档数、分块数、父块数、能力开关） |
| POST | `/api/v1/chat` | 知识库问答（一次性返回） |
| POST | `/api/v1/chat/stream` | 知识库问答（**SSE 流式**） |

### 流式问答（SSE）

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "总结一下这份文档的核心内容"}'
```

```
event: references
data: {"sources": ["简历-xx.pdf"], "references": [...], "retrieved": 4, "reranked": true, ...}

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

> 若走 Nginx 反代，需关闭缓冲（`proxy_buffering off;`）；响应头已带 `X-Accel-Buffering: no`。

---

## 质量评估

黄金测试集 + 评估脚本，让每一次调参都有据可依。

```bash
cp eval/golden_set.example.json eval/golden_set.json   # 按实际文档填写
python eval/run_eval.py --set eval/golden_set.json
python eval/run_eval.py --set eval/golden_set.json --output eval/report.json
```

官方 RAGAS 指标（可选）：

```bash
pip install -r requirements-eval.txt
python eval/run_eval.py --set eval/golden_set.json --backend ragas
```

会额外输出 `faithfulness` / `answer_relevancy` / `context_precision` / `context_recall`。

---

## 配置说明

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama` 本地推理 / `openai` 兼容接口（DeepSeek） |
| `OLLAMA_MODEL` | `qwen2.5:7b` | 本地模型名（也可换 deepseek-r1 等） |
| `OLLAMA_KEEP_ALIVE` | `-1` | 模型驻留显存时长；`-1` 永久，或 `"5m"`/`"1h"` 等带单位写法 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 中文 Embedding |
| `EMBEDDING_DEVICE` | `auto` | 有 CUDA 自动用 GPU，否则 CPU |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | **子块**大小与重叠（用于召回） |
| `PARENT_CHUNK_SIZE` / `PARENT_CHUNK_OVERLAP` | `1500` / `100` | **父块**大小与重叠（用于生成） |
| `TOP_K` / `SCORE_THRESHOLD` | `4` / `0.3` | 返回数量与向量召回相关性下限 |
| `MAX_HISTORY_TURNS` | `3` | 单次问答携带的历史轮数 |
| `ENABLE_QUERY_REWRITE` | `true` | 多轮指代消解式查询改写 |
| `ENABLE_MULTI_QUERY` | `false` | 多查询变体并行检索 |
| `ENABLE_HYBRID_SEARCH` | `true` | 关闭则退化为纯向量检索 |
| `CANDIDATE_K` | `20` | 送入重排的候选数，须 ≥ `TOP_K` |
| `RRF_K` | `60` | RRF 融合常数，越大各路差异越平滑 |
| `ENABLE_RERANK` | `true` | 关闭则直接返回融合排序结果 |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | 中文重排模型；`large` 更准但更慢 |
| `ENABLE_PARENT_CHILD` | `true` | 关闭则退化为单级分块 |
| `CORS_ORIGINS` | `*` | 生产环境务必指定具体域名 |

切换到 DeepSeek 等云端模型：`.env` 中设置 `LLM_PROVIDER=openai` 并填入 `DEEPSEEK_API_KEY` 即可，无需改动代码。

### 性能调优建议

| 场景 | 建议 |
| --- | --- |
| 快速验证链路 | `ENABLE_RERANK=false`、`ENABLE_MULTI_QUERY=false` |
| 追求延迟（实时对话） | 关重排，或让重排跑 GPU；`CANDIDATE_K` 调至 10~15 |
| 追求准确率（离线分析） | 全开 + `ENABLE_MULTI_QUERY=true`，`CANDIDATE_K=50`、`TOP_K=8` |
| 长文档、答案常被截断 | 调大 `PARENT_CHUNK_SIZE` |
| CPU 环境 | 建议 `ENABLE_RERANK=false`，否则每次问答额外增加秒级开销 |

---

## 设计要点

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

---

## 已知限制与后续规划

- **BM25 需全量语料**：当前从 Chroma 拉取全量文档在内存构建索引（带缓存，写入时失效）。万级分块以内无压力，更大规模应改用 Elasticsearch 或 Milvus 稀疏向量。
- **查询改写增加一次 LLM 调用**：多轮对话会多一次推理开销；单轮对话自动跳过，无额外成本。
- **多 worker 部署**（如 `uvicorn --workers 4`）时各进程缓存独立，BM25 索引不同步，需改为单 worker 或接入外部向量库。
- **会话历史**由调用方携带，服务端未持久化（前端维护最近 10 轮）。
- **接口无鉴权与限流**，仅适用于内网或开发环境。
- 规划中：鉴权与限流、Docker 部署、Redis 会话持久化、异步索引与进度回调、表格与扫描件解析、Milvus 生产部署。

---

## License

本项目以 MIT 协议开源。
