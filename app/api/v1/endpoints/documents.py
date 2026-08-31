# app/api/v1/endpoints/documents.py
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.rag_engine import rag_engine

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传并索引一个文档"""
    # 校验文件类型
    allowed_extensions = [".txt", ".pdf", ".docx"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型，仅支持: {allowed_extensions}")
    
    # 保存文件到服务器
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 调用RAG引擎进行索引
        chunk_count = rag_engine.index_document(file_path)
        return {
            "message": f"文档 '{file.filename}' 上传并索引成功",
            "chunk_count": chunk_count,
            "file_path": file_path
        }
    except Exception as e:
        # 如果索引失败，删除已保存的文件
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"索引失败: {str(e)}")