# app/api/v1/endpoints/chat.py
import json
import logging
from typing import Any, Dict, Iterator, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool

from app.services.rag_engine import get_rag_engine

logger = logging.getLogger(__name__)
router = APIRouter()


class HistoryItem(BaseModel):
    """一轮历史对话"""
    question: str
    answer: str

    @model_validator(mode="before")
    @classmethod
    def _accept_pair(cls, data: Any) -> Any:
        """兼容 {"question":..,"answer":..} 与 ["问","答"] 两种写法"""
        if isinstance(data, (list, tuple)) and len(data) == 2:
            return {"question": data[0], "answer": data[1]}
        return data


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    history: Optional[List[HistoryItem]] = Field(default=None, description="多轮历史，按时间正序")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="最终返回片段数量")
    source: Optional[str] = Field(default=None, description="限定在某个文档内检索")


class Reference(BaseModel):
    source: str
    page: Optional[int] = None
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    references: List[Reference]
    retrieved: int
    reranked: bool = False
    parent_child: bool = False
    hybrid: bool = False


def _sse(event: str, data: Dict[str, Any]) -> str:
    """按 SSE 规范序列化单个事件"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """基于知识库问答（一次性返回完整答案）"""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    history = [(h.question, h.answer) for h in (request.history or [])]

    engine = get_rag_engine()
    try:
        # 检索 + 重排 + LLM 推理均为阻塞操作，放到线程池执行，避免卡住事件循环
        result: Dict[str, Any] = await run_in_threadpool(
            engine.ask,
            question,
            history or None,
            request.top_k,
            request.source,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("问答失败")
        raise HTTPException(status_code=500, detail=f"问答失败: {exc}") from exc

    return ChatResponse(**result)


@router.post("/stream", summary="流式问答（SSE）")
async def chat_stream(request: ChatRequest):
    """
    基于知识库问答，以 Server-Sent Events 逐段返回。

    事件序列：
      - `references`：召回的引用来源，在生成开始前推送，便于前端先行渲染引用卡片
      - `token`    ：答案增量（已剔除推理模型的思维链）
      - `done`     ：完成，携带完整答案与引用
      - `error`    ：生成过程中出错（此时响应头已发出，只能以事件形式通知）
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    history = [(h.question, h.answer) for h in (request.history or [])]
    engine = get_rag_engine()

    def event_generator() -> Iterator[str]:
        # 同步生成器：Starlette 会自动放到线程池中迭代，不会阻塞事件循环
        try:
            for item in engine.stream_answer(
                question, history or None, request.top_k, request.source
            ):
                event = item.pop("event")
                yield _sse(event, item)
        except Exception as exc:
            logger.exception("流式问答失败")
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 等反向代理的响应缓冲
        },
    )
