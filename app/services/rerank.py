# app/services/rerank.py
"""重排服务：使用 CrossEncoder 对召回候选做精排序，提升上下文质量。"""
import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.core.config import settings

logger = logging.getLogger(__name__)


class RerankService:
    """
    CrossEncoder 重排器（默认 BAAI/bge-reranker-base，中文效果好）。
    模型采用懒加载，首次真正重排时才载入显存/内存。
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANK_MODEL
        self.enabled = settings.ENABLE_RERANK
        self._model = None

    # ---------------- 模型管理 ----------------

    def _ensure_model(self):
        if not self.enabled:
            return None
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("加载重排模型: %s (device=%s)", self.model_name, settings.RERANK_DEVICE)
                self._model = CrossEncoder(
                    self.model_name, max_length=512, device=settings.RERANK_DEVICE
                )
            except Exception as exc:
                logger.error(
                    "重排模型加载失败，将自动降级为仅混合检索: %s", exc
                )
                self.enabled = False
                return None
        return self._model

    # ---------------- 重排 ----------------

    def rerank(
        self,
        query: str,
        hits: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        """
        对 (Document, score) 候选列表精排。
        :return: 重排后的 [(Document, rerank_score)]，最多 top_k 条
        """
        if not hits:
            return []

        model = self._ensure_model()
        if model is None:
            return hits[:top_k]

        pairs = [[query, doc.page_content] for doc, _ in hits]
        try:
            scores = model.predict(pairs)
        except Exception as exc:
            logger.exception("重排推理失败，回退到融合排序: %s", exc)
            return hits[:top_k]

        scored = sorted(
            ((doc, float(score)) for (doc, _), score in zip(hits, scores)),
            key=lambda item: item[1],
            reverse=True,
        )

        # CrossEncoder 输出的是 logits（可正可负），阈值用于剔除明显不相关的片段
        if settings.RERANK_THRESHOLD is not None:
            scored = [item for item in scored if item[1] >= settings.RERANK_THRESHOLD]

        logger.debug("重排完成: %d -> %d 条", len(hits), len(scored[:top_k]))
        return scored[:top_k]

    def is_ready(self) -> bool:
        return self.enabled and self._model is not None
