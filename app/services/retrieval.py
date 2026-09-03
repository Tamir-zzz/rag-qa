# app/services/retrieval.py
"""检索服务：向量召回 + BM25 稀疏召回 + RRF 融合 + 精排（Rerank）。

支持 Multi-Query：多个查询变体各自召回后，再对结果做一次 RRF 融合。
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.config import settings
from app.services.rerank import RerankService
from app.utils.text_utils import tokenize

logger = logging.getLogger(__name__)

try:  # BM25 为可选能力，缺失时自动降级为纯向量检索
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None
    logger.warning("未安装 rank_bm25，BM25 混合检索将不可用（pip install rank-bm25）")


class RetrievalService:
    """
    混合检索服务。

    单查询链路：向量检索（语义）┐
                               ├─ RRF 融合 ─┐
               BM25 检索（字面）┘            │
                                            ├─ 多查询融合 ─→ 候选集 ─→ CrossEncoder 重排 ─→ Top-K
             查询变体 2..N 各自重复上述链路 ─┘
    """

    def __init__(self, embeddings: Embeddings, reranker: Optional[RerankService] = None):
        self.embeddings = embeddings
        self.reranker = reranker
        self.vectorstore: Optional[Chroma] = None

        # BM25 语料缓存：chunk_id -> {text, metadata}
        self._corpus: Optional[Dict[str, Dict[str, Any]]] = None
        self._bm25 = None

    # ---------------- 向量库生命周期 ----------------

    def ensure_vectorstore(self) -> Chroma:
        """懒加载向量库：首次使用时才连接/创建，避免导入即阻塞。"""
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                collection_name=settings.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=settings.CHROMA_PERSIST_DIR,
                # 余弦空间：配合归一化 embedding，relevance score 即为余弦相似度
                collection_metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "向量库已就绪: dir=%s collection=%s",
                settings.CHROMA_PERSIST_DIR,
                settings.COLLECTION_NAME,
            )
        return self.vectorstore

    def _invalidate_cache(self) -> None:
        """语料变更（新增/删除）后让 BM25 与语料缓存失效"""
        self._corpus = None
        self._bm25 = None

    # ---------------- 语料与 BM25 ----------------

    def _ensure_corpus(self) -> Dict[str, Dict[str, Any]]:
        """加载全量语料（用于 BM25 与统计），带缓存。"""
        if self._corpus is None:
            data = self.ensure_vectorstore().get(include=["documents", "metadatas"])
            corpus: Dict[str, Dict[str, Any]] = {}
            for doc_id, text, meta in zip(
                data.get("ids") or [],
                data.get("documents") or [],
                data.get("metadatas") or [],
            ):
                meta = meta or {}
                key = meta.get("chunk_id") or doc_id
                corpus[key] = {"text": text or "", "metadata": meta}
            self._corpus = corpus
            logger.debug("语料缓存已构建，共 %d 个分块", len(corpus))
        return self._corpus

    def _ensure_bm25(self):
        """构建 BM25 索引（首次检索时触发，后续复用）"""
        if BM25Okapi is None:
            return None
        corpus = self._ensure_corpus()
        if self._bm25 is None and corpus:
            tokenized = [tokenize(item["text"]) for item in corpus.values()]
            self._bm25 = BM25Okapi(tokenized)
            logger.debug("BM25 索引已构建，共 %d 篇文档", len(tokenized))
        return self._bm25

    # ---------------- 写入 ----------------

    def add_documents(self, chunks: List[Document]) -> None:
        if not chunks:
            return
        self.ensure_vectorstore().add_documents(chunks)
        self._invalidate_cache()
        self.persist()

    def persist(self) -> None:
        if self.vectorstore is not None:
            try:
                self.vectorstore.persist()
            except Exception as exc:  # 新版 Chroma 会自动持久化，失败不致命
                logger.debug("persist 调用跳过: %s", exc)

    # ---------------- 检索入口 ----------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """单查询检索"""
        return self.search_multi(
            [query], top_k=top_k, score_threshold=score_threshold, source_filter=source_filter
        )

    def search_multi(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        多查询检索（Multi-Query + 混合召回 + 重排）。

        未开启重排时 score 为 RRF 融合分；
        开启重排时 score 为 CrossEncoder 打分（logits）。
        """
        top_k = top_k or settings.TOP_K
        threshold = settings.SCORE_THRESHOLD if score_threshold is None else score_threshold

        if not self._ensure_corpus():
            return []

        queries = [q.strip() for q in (queries or []) if q and q.strip()]
        if not queries:
            return []

        candidate_k = max(settings.CANDIDATE_K, top_k)

        # 1) 每个查询各自做「向量 + BM25」混合召回
        per_query_hits = [
            self._search_one(q, candidate_k, threshold, source_filter) for q in queries
        ]

        # 2) 查询间再做一次 RRF 融合
        if len(per_query_hits) == 1:
            fused = per_query_hits[0]
        else:
            fused = self._rrf_merge(per_query_hits)
        logger.info(
            "混合召回: %d 个查询 -> 候选 %d 条", len(queries), len(fused)
        )

        # 3) 精排（以首个查询，即原始/改写后的主查询为准）
        if self.reranker and settings.ENABLE_RERANK:
            fused = self.reranker.rerank(queries[0], fused, top_k)
        else:
            fused = fused[:top_k]

        return fused

    # ---------------- 单路召回与融合 ----------------

    def _search_one(
        self,
        query: str,
        candidate_k: int,
        threshold: float,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """单个查询的混合召回：向量 + BM25，RRF 融合"""
        vector_hits = self._vector_search(query, candidate_k, threshold, source_filter)
        if not settings.ENABLE_HYBRID_SEARCH:
            return vector_hits

        bm25_hits = self._bm25_search(query, candidate_k, source_filter)
        if not bm25_hits:
            return vector_hits
        if not vector_hits:
            return bm25_hits

        return self._rrf_merge([vector_hits, bm25_hits])

    def _vector_search(
        self,
        query: str,
        candidate_k: int,
        threshold: float,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """稠密向量检索，按余弦相似度下限过滤"""
        kwargs: Dict[str, Any] = {"k": candidate_k}
        if source_filter:
            kwargs["filter"] = {"source": source_filter}

        try:
            pairs = self.ensure_vectorstore().similarity_search_with_relevance_scores(
                query, **kwargs
            )
        except Exception as exc:
            logger.exception("向量检索失败: %s", exc)
            return []

        hits = [(doc, float(score)) for doc, score in pairs if score >= threshold]
        if not hits:
            logger.info("向量召回 %d 条但均低于阈值 %.2f，已丢弃", len(pairs), threshold)
        return hits

    def _bm25_search(
        self,
        query: str,
        candidate_k: int,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """BM25 字面召回，擅长专有名词、型号、人名等精确匹配"""
        bm25 = self._ensure_bm25()
        corpus = self._ensure_corpus()
        if bm25 is None or not corpus:
            return []

        keys = list(corpus.keys())
        try:
            scores = bm25.get_scores(tokenize(query))
        except Exception as exc:
            logger.exception("BM25 检索失败，本次跳过该路召回: %s", exc)
            return []

        ranked = sorted(range(len(keys)), key=lambda i: scores[i], reverse=True)[:candidate_k]

        hits: List[Tuple[Document, float]] = []
        for i in ranked:
            if scores[i] <= 0:  # 0 分表示查询词完全未命中该文档
                break
            item = corpus[keys[i]]
            meta = item["metadata"]
            if source_filter and meta.get("source") != source_filter:
                continue
            hits.append((Document(page_content=item["text"], metadata=meta), float(scores[i])))
        return hits

    @staticmethod
    def _rrf_merge(hit_lists: List[List[Tuple[Document, float]]]) -> List[Tuple[Document, float]]:
        """
        倒数排名融合：score = Σ 1/(k + rank)。

        只用排名、不看原始分数，因此天然适用于
        量纲不同的多路结果（向量余弦 / BM25 / 多个查询变体）。
        """
        rrf: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for hits in hit_lists:
            for rank, (doc, _) in enumerate(hits):
                # 跨路结果靠 chunk_id 对齐；缺失时退化为独立条目
                key = doc.metadata.get("chunk_id") or f"anon_{id(doc)}"
                rrf[key] = rrf.get(key, 0.0) + 1.0 / (settings.RRF_K + rank + 1)
                doc_map.setdefault(key, doc)

        merged = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
        return [(doc_map[key], score) for key, score in merged]

    # ---------------- 文档管理 ----------------

    def delete_by_source(self, source: str) -> None:
        """按原始文件名删除该文档的全部向量"""
        if not self._ensure_corpus():
            return
        self.ensure_vectorstore().delete(where={"source": source})
        self._invalidate_cache()
        self.persist()

    def list_sources(self) -> List[Dict[str, Any]]:
        """聚合统计已索引的文档：文件名、分块数、磁盘路径"""
        corpus = self._ensure_corpus()
        if not corpus:
            return []
        stats: Dict[str, Dict[str, Any]] = {}
        for item in corpus.values():
            meta = item["metadata"]
            name = meta.get("source", "未知来源")
            entry = stats.setdefault(
                name, {"source": name, "chunk_count": 0, "file_path": meta.get("file_path")}
            )
            entry["chunk_count"] += 1
        return sorted(stats.values(), key=lambda x: x["source"])

    def count(self) -> int:
        return len(self._ensure_corpus())
