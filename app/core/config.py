# app/core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # LLM 配置（以DeepSeek为例，兼容OpenAI接口）
    OPENAI_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")  # 环境变量名写DEEPSEEK_API_KEY
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "deepseek-chat")
    
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
    # Embedding 配置（使用本地模型，不需要API Key）
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    
    # ChromaDB 持久化路径
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    
    # 分块参数
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

settings = Settings()