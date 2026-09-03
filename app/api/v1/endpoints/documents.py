# app/api/v1/endpoints/documents.py
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.services.rag_engine import get_rag_engine

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = settings.UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

_UNSAFE_CHARS = re.compile(r"[^\w.\-()一-龥]+")


class DocumentItem(BaseModel):
    source: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentItem]


class UploadResponse(BaseModel):
    message: str
    source: str
    chunk_count: int


def _safe_filename(filename: str) -> str:
    """
    生成安全的落盘文件名，防止路径穿越（../../etc/passwd）与覆盖攻击。
    仅取文件名部分，并追加 uuid 前缀避免同名冲突。
    """
    name = Path(filename or "").name          # 剥离任何目录成分，杜绝路径穿越
    name = _UNSAFE_CHARS.sub("_", name).strip("._")
    if not name:
        name = "unnamed"
    return f"{uuid.uuid4().hex}_{name}"       # uuid 前缀避免同名覆盖


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传并索引一个文档（同名文件视为覆盖更新）"""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {ext or '(空)'}，仅支持: {list(settings.ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"文件过大，单个文件不得超过 {settings.MAX_UPLOAD_MB} MB"
        )
    if not content:
        raise HTTPException(status_code=400, detail="文件为空")

    source_name = Path(file.filename).name      # 对外展示的原始文件名
    file_path = os.path.join(UPLOAD_DIR, _safe_filename(file.filename))

    with open(file_path, "wb") as buffer:
        buffer.write(content)

    engine = get_rag_engine()
    try:
        # 索引涉及 CPU/GPU 密集计算，放入线程池避免阻塞事件循环
        chunk_count = await run_in_threadpool(engine.index_document, file_path, source_name)
    except Exception as exc:
        if os.path.exists(file_path):
            os.remove(file_path)               # 索引失败则回滚已保存文件
        logger.exception("索引失败: %s", source_name)
        raise HTTPException(status_code=500, detail=f"索引失败: {exc}") from exc

    return UploadResponse(
        message=f"文档 '{source_name}' 上传并索引成功",
        source=source_name,
        chunk_count=chunk_count,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """列出知识库中已索引的所有文档"""
    engine = get_rag_engine()
    docs = await run_in_threadpool(engine.list_documents)
    # 不对外暴露服务器磁盘路径
    return DocumentListResponse(
        total=len(docs),
        documents=[DocumentItem(source=d["source"], chunk_count=d["chunk_count"]) for d in docs],
    )


@router.delete("/{source}")
async def delete_document(source: str):
    """按原始文件名删除文档（同时清理向量与磁盘文件）"""
    engine = get_rag_engine()
    ok = await run_in_threadpool(engine.delete_document, source)
    if not ok:
        raise HTTPException(status_code=404, detail=f"知识库中不存在文档: {source}")
    return {"message": f"文档 '{source}' 已删除"}


@router.get("/stats")
async def stats() -> Dict[str, Any]:
    """知识库统计信息"""
    engine = get_rag_engine()
    return await run_in_threadpool(engine.stats)
