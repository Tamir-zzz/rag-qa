# app/services/rag_engine.py
"""RAG 引擎：编排「加载 -> 分块 -> 索引」与「查询改写 -> 混合召回 -> 重排 -> 生成」链路。"""
import logging
import os
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.generation import GenerationService
from app.services.rerank import RerankService
from app.services.retrieval import RetrievalService
from app.services.store import ParentStore
from app.utils.file_utils import load_document

logger = logging.getLogger(__name__)

# 上下文条目：文本、得分、元数据
ContextItem = Tuple[str, float, Dict[str, Any]]

_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


class RAGEngine:
    """组合检索、重排与生成服务，对外提供索引 / 问答 / 流式问答 / 文档管理能力"""

    def __init__(self):
        # 1. Embedding 模型（本地运行，设备自适应）
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": settings.EMBEDDING_NORMALIZE},
        )

        # 2. 文本分割器：子块用于召回，父块用于生成
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=_SEPARATORS,
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.PARENT_CHUNK_SIZE,
            chunk_overlap=settings.PARENT_CHUNK_OVERLAP,
            separators=_SEPARATORS,
        )

        # 3. 子服务（向量库与重排模型均懒加载）
        self.reranker = RerankService()
        self.retriever = RetrievalService(self.embeddings, reranker=self.reranker)
        self.generator = GenerationService()
        self.parent_store = ParentStore()

        logger.info(
            "RAG 引擎初始化完成 | embedding=%s(%s) | rerank=%s | parent_child=%s | llm=%s",
            settings.EMBEDDING_MODEL,
            settings.EMBEDDING_DEVICE,
            settings.RERANK_MODEL if settings.ENABLE_RERANK else "off",
            settings.ENABLE_PARENT_CHILD,
            settings.LLM_PROVIDER,
        )

    # ---------------- 索引 ----------------

    def index_document(self, file_path: str, source_name: Optional[str] = None) -> int:
        """
        索引单个文档：加载 -> 分块 -> 标注元数据 -> 写入向量库（父块写入 SQLite）
        :param file_path: 磁盘上的实际路径
        :param source_name: 对外展示的原始文件名（不暴露服务器路径）
        :return: 子块数量
        """
        docs = load_document(file_path)
        if not docs:
            raise ValueError("文档加载失败或内容为空")

        source = source_name or os.path.basename(file_path)

        # 同名文档视为覆盖更新：先清理旧向量与旧父块
        self.retriever.delete_by_source(source)
        self.parent_store.delete_by_source(source)

        if settings.ENABLE_PARENT_CHILD:
            chunks, parent_items = self._split_parent_child(docs, source, file_path)
            self.parent_store.put_many(parent_items)
        else:
            chunks = self.text_splitter.split_documents(docs)
            parent_items = []
            for chunk in chunks:
                chunk.metadata.update(
                    {"source": source, "file_path": file_path, "chunk_id": uuid.uuid4().hex}
                )

        if not chunks:
            raise ValueError("文档切分后无有效内容")

        self.retriever.add_documents(chunks)
        logger.info(
            "文档已索引: %s -> %d 个子块 / %d 个父块",
            source,
            len(chunks),
            len(parent_items),
        )
        return len(chunks)

    def _split_parent_child(
        self, docs: List[Document], source: str, file_path: str
    ) -> Tuple[List[Document], List[Dict[str, Any]]]:
        """
        Small-to-Big 分块：先切大块（父，用于生成），再在每个父块内切小块（子，用于召回）。
        这样子块不会跨越父块边界，父子归属天然明确。
        """
        parents = self.parent_splitter.split_documents(docs)
        chunks: List[Document] = []
        parent_items: List[Dict[str, Any]] = []

        for parent in parents:
            parent_id = uuid.uuid4().hex
            children = self.text_splitter.split_documents([parent])
            if not children:
                continue

            parent_items.append(
                {
                    "id": parent_id,
                    "source": source,
                    "text": parent.page_content,
                    # 父块可能跨页，记录其起始页
                    "page": children[0].metadata.get("page"),
                    "file_path": file_path,
                }
            )

            for child in children:
                child.metadata.update(
                    {
                        "source": source,
                        "file_path": file_path,
                        "chunk_id": uuid.uuid4().hex,
                        "parent_id": parent_id,
                    }
                )
                chunks.append(child)

        return chunks, parent_items

    # ---------------- 查询改写 ----------------

    def _prepare_queries(
        self,
        question: str,
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> List[str]:
        """
        生成用于检索的查询列表。
        Multi-Query 优先；否则在多轮场景下做指代消解式改写；单轮直接用原问题。
        """
        if settings.ENABLE_MULTI_QUERY:
            try:
                queries = self.generator.expand_queries(question, history)
                if queries:
                    return queries
            except Exception as exc:
                logger.warning("Multi-Query 失败，降级为单查询: %s", exc)

        if settings.ENABLE_QUERY_REWRITE and history:
            try:
                return [self.generator.rewrite_query(question, history)]
            except Exception as exc:
                logger.warning("查询改写失败，使用原始问题: %s", exc)

        return [question]

    # ---------------- 检索 ----------------

    def retrieve(
        self,
        question: str,
        history: Optional[Sequence[Tuple[str, str]]] = None,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """查询改写 -> 混合召回（含 Multi-Query）-> 重排，返回子块命中列表"""
        t0 = time.time()
        queries = self._prepare_queries(question, history)
        t1 = time.time()
        hits = self.retriever.search_multi(
            queries, top_k=top_k, source_filter=source_filter
        )
        t2 = time.time()
        logger.info(
            "检索耗时 | 查询改写 %.2fs | 召回+重排 %.2fs | 查询数 %d | 命中 %d 条",
            t1 - t0, t2 - t1, len(queries), len(hits),
        )
        return hits

    def expand_to_parents(
        self, hits: List[Tuple[Document, float]]
    ) -> List[ContextItem]:
        """
        Small-to-Big：命中子块后，回取所属父块作为 LLM 上下文。
        同一父块的多个子块只保留一次（保留最高得分）。
        """
        if not settings.ENABLE_PARENT_CHILD:
            return [
                (doc.page_content, score, dict(doc.metadata)) for doc, score in hits
            ]

        parent_ids = list(
            dict.fromkeys(
                doc.metadata.get("parent_id")
                for doc, _ in hits
                if doc.metadata.get("parent_id")
            )
        )
        parents = self.parent_store.get_many(parent_ids) if parent_ids else {}

        items: List[ContextItem] = []
        seen: set = set()
        for doc, score in hits:
            parent_id = doc.metadata.get("parent_id")
            parent = parents.get(parent_id) if parent_id else None
            if parent:
                if parent_id in seen:
                    continue
                seen.add(parent_id)
                items.append(
                    (
                        parent["text"],
                        score,
                        {
                            "source": parent["source"],
                            "page": parent["page"],
                            "parent_id": parent_id,
                        },
                    )
                )
            else:
                # 无 parent_id（旧数据或父块缺失）时回退为子块本身
                items.append((doc.page_content, score, dict(doc.metadata)))
        return items

    @staticmethod
    def compose(items: List[ContextItem]) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """
        把上下文条目组装成 LLM 输入，并提取来源信息。
        :return: (context, sources, references)
        """
        context_parts = []
        for idx, (text, _, meta) in enumerate(items, start=1):
            tag = f"[{idx}] 来源: {meta.get('source', '未知来源')}"
            if meta.get("page") is not None:
                tag += f"（第 {int(meta['page']) + 1} 页）"
            context_parts.append(f"{tag}\n{text}")

        context = "\n\n".join(context_parts)

        # dict.fromkeys 去重且保持相关性排序（set 会打乱顺序）
        sources = list(
            dict.fromkeys(meta.get("source", "未知来源") for _, _, meta in items)
        )

        references = [
            {
                "source": meta.get("source", "未知来源"),
                "page": int(meta["page"]) + 1 if meta.get("page") is not None else None,
                "score": round(float(score), 4),
                "snippet": text[:200],
            }
            for text, score, meta in items
        ]
        return context, sources, references

    # ---------------- 问答 ----------------

    def ask(
        self,
        question: str,
        history: Optional[Sequence[Tuple[str, str]]] = None,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        问答入口（一次性返回）：查询改写 -> 混合召回 -> 重排 -> 父块展开 -> LLM 生成
        """
        hits = self.retrieve(question, history, top_k, source_filter)

        if not hits:
            return self._empty_result("知识库为空或未检索到相关内容，请先上传文档。")

        t0 = time.time()
        items = self.expand_to_parents(hits)
        context, sources, references = self.compose(items)
        answer = self.generator.generate(question=question, context=context, history=history)
        logger.info("LLM 生成耗时 %.2fs（输出 %d 字）", time.time() - t0, len(answer))

        return self._result(answer, sources, references, len(items))

    def stream_answer(
        self,
        question: str,
        history: Optional[Sequence[Tuple[str, str]]] = None,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        问答入口（流式）。依次产出结构化事件：
          - {"event": "references", ...} 召回结果，供前端先行展示引用
          - {"event": "token", "content": "..."} 答案增量
          - {"event": "done", ...} 完成，携带完整答案
        """
        hits = self.retrieve(question, history, top_k, source_filter)

        if not hits:
            yield {
                "event": "done",
                **self._empty_result("知识库为空或未检索到相关内容，请先上传文档。"),
            }
            return

        items = self.expand_to_parents(hits)
        context, sources, references = self.compose(items)

        yield {
            "event": "references",
            "sources": sources,
            "references": references,
            "retrieved": len(items),
            **self._meta(),
        }

        buffer: List[str] = []
        started = time.time()
        first_token_at = None
        for chunk in self.generator.stream(
            question=question, context=context, history=history
        ):
            if first_token_at is None:
                first_token_at = time.time()
                logger.info("LLM 首字延迟 %.2fs", first_token_at - started)
            buffer.append(chunk)
            yield {"event": "token", "content": chunk}
        logger.info(
            "LLM 生成耗时 %.2fs（输出 %d 字）", time.time() - started, len("".join(buffer))
        )

        yield {
            "event": "done",
            "answer": "".join(buffer),
            "sources": sources,
            "references": references,
            "retrieved": len(items),
            **self._meta(),
        }

    # ---------------- 结果组装 ----------------

    @staticmethod
    def _meta() -> Dict[str, Any]:
        """随问答结果返回的能力开关，便于前端与排查"""
        return {
            "reranked": settings.ENABLE_RERANK,
            "parent_child": settings.ENABLE_PARENT_CHILD,
            "hybrid": settings.ENABLE_HYBRID_SEARCH,
        }

    def _result(
        self,
        answer: str,
        sources: List[str],
        references: List[Dict[str, Any]],
        retrieved: int,
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "sources": sources,
            "references": references,
            "retrieved": retrieved,
            **self._meta(),
        }

    @staticmethod
    def _empty_result(message: str) -> Dict[str, Any]:
        return {
            "answer": message,
            "sources": [],
            "references": [],
            "retrieved": 0,
            "reranked": False,
            "parent_child": False,
            "hybrid": False,
        }

    # ---------------- 文档管理 ----------------

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.retriever.list_sources()

    def delete_document(self, source: str) -> bool:
        """删除文档：清理向量 + 父块 + 磁盘文件"""
        items = [item for item in self.retriever.list_sources() if item["source"] == source]
        if not items:
            return False

        self.retriever.delete_by_source(source)
        self.parent_store.delete_by_source(source)

        file_path = items[0].get("file_path")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as exc:
                logger.warning("删除磁盘文件失败 %s: %s", file_path, exc)
        return True

    def stats(self) -> Dict[str, Any]:
        return {
            "chunk_count": self.retriever.count(),
            "parent_count": self.parent_store.count(),
            "document_count": len(self.retriever.list_sources()),
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_device": settings.EMBEDDING_DEVICE,
            "llm_provider": settings.LLM_PROVIDER,
            "hybrid_search": settings.ENABLE_HYBRID_SEARCH,
            "multi_query": settings.ENABLE_MULTI_QUERY,
            "query_rewrite": settings.ENABLE_QUERY_REWRITE,
            "parent_child": settings.ENABLE_PARENT_CHILD,
            "rerank_enabled": bool(
                self.reranker and settings.ENABLE_RERANK and self.reranker.is_ready()
            ),
            "rerank_model": settings.RERANK_MODEL if settings.ENABLE_RERANK else None,
        }


# ---------------- 单例管理（懒加载，避免 import 即阻塞） ----------------
_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """获取全局引擎实例，首次调用时才真正加载模型"""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


def shutdown_rag_engine() -> None:
    """应用关闭时释放资源"""
    global _engine
    if _engine is not None:
        try:
            _engine.retriever.persist()
        except Exception as exc:  # 关闭阶段不因持久化失败而中断
            logger.warning("关闭时持久化失败: %s", exc)
        _engine = None
