# app/utils/file_utils.py
"""文档加载工具：按扩展名分发到对应的 LangChain Loader。"""
import logging
import os
from typing import List

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_document(file_path: str) -> List[Document]:
    """
    根据文件扩展名自动选择合适的加载器
    :raises ValueError: 不支持的类型，或文件损坏/为空
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

    docs = loader.load()
    # 过滤空页面，避免把纯空白页写进向量库造成噪声
    docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
    if not docs:
        raise ValueError("文档无有效文本内容（可能是扫描件或加密文件）")

    logger.info("已加载 %s，共 %d 个原始段落", os.path.basename(file_path), len(docs))
    return docs
