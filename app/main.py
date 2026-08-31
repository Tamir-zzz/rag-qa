# app/main.py
from fastapi import FastAPI
from app.api.v1.endpoints import documents, chat

app = FastAPI(
    title="企业级RAG知识库问答系统",
    description="铂金标准 RAG 系统，支持文档上传与智能问答",
    version="1.0.0"
)

# 注册路由
app.include_router(documents.router, prefix="/api/v1/documents", tags=["文档管理"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["智能问答"])

@app.get("/")
async def root():
    return {"message": "RAG 企业知识库系统已启动", "docs": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}