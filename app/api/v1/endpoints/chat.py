# app/api/v1/endpoints/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_engine import rag_engine

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """基于知识库问答"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    result = rag_engine.ask(request.question)
    return ChatResponse(**result)