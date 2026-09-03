# app/utils/text_utils.py
"""文本处理工具：中文分词等，供 BM25 检索使用。"""
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# 连续的中/英数片段：英文数字整体成词，中文串交给 jieba 细切
_SEGMENT_RE = re.compile(r"[A-Za-z0-9_]+|[一-鿿]+")

_jieba = None


def _get_jieba():
    """懒加载 jieba 并关闭其冗余初始化日志"""
    global _jieba
    if _jieba is None:
        import jieba

        jieba.setLogLevel(logging.ERROR)  # 屏蔽 "Building prefix dict..." 等噪音
        _jieba = jieba
    return _jieba


def tokenize(text: str) -> List[str]:
    """
    中英混合分词：
      - 英文/数字统一小写后成词
      - 中文片段使用 jieba 精确模式切分
    """
    if not text:
        return []
    tokenizer = _get_jieba()
    tokens: List[str] = []
    for seg in _SEGMENT_RE.findall(text):
        if seg.isascii():
            tokens.append(seg.lower())
        else:
            tokens.extend(tokenizer.lcut(seg))
    return tokens
