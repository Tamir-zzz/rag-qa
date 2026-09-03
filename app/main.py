# app/main.py
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.api.v1.endpoints import chat, documents
from app.core.config import settings
from app.services.rag_engine import get_rag_engine, shutdown_rag_engine

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时预热模型，关闭时释放资源"""
    try:
        # 首次调用会加载 Embedding 模型并连接向量库，放进线程池避免阻塞启动流程
        await run_in_threadpool(get_rag_engine)
        logger.info("RAG 引擎预热完成")
    except Exception as exc:
        # 不阻断服务启动：LLM/向量库未就绪时接口会返回明确错误，便于排查
        logger.error("RAG 引擎初始化失败，请检查模型与向量库配置: %s", exc)

    yield

    shutdown_rag_engine()
    logger.info("RAG 引擎已释放")


app = FastAPI(
    title=settings.APP_NAME,
    description="铂金标准 RAG 系统，支持文档上传、管理与带引用溯源的知识库问答",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# 跨域支持（生产环境请通过 CORS_ORIGINS 指定具体域名，不要用 *）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(documents.router, prefix="/api/v1/documents", tags=["文档管理"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["智能问答"])


@app.get("/", summary="服务根路径")
async def root():
    """服务探针（前端由 Streamlit 独立进程提供）"""
    return {
        "message": "RAG 企业知识库系统已启动",
        "docs": "/docs",
        "frontend": "streamlit run streamlit_app.py",
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """消除浏览器自动请求 /favicon.ico 造成的 404 日志噪音"""
    return Response(status_code=204)


@app.get("/health", summary="健康检查")
async def health_check():
    """健康检查：liveness，不依赖外部模型"""
    return {"status": "healthy", "version": settings.APP_VERSION}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
