# app/core/config.py
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

# 项目根目录（rag-qa/）
BASE_DIR = Path(__file__).resolve().parents[2]


def _get_bool(key: str, default: bool = False) -> bool:
    """将环境变量中的 "1/true/yes/on" 解析为布尔值"""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _get_keep_alive(key: str, default: str = "-1"):
    """
    Ollama keep_alive 的两种合法形式：
      - 整数（秒）：-1 表示永久驻留。注意必须是数字类型，不能是字符串 "-1"，
        否则 Ollama 会按 Go 的 time.ParseDuration 解析，因无单位报错：
        time: missing unit in duration "-1" (status code: 400)
      - 带单位的字符串："5m" / "1h" / "30s"
    这里：纯数字（可带负号）转 int；其余按 duration 字符串原样返回。
    """
    raw = os.getenv(key, default)
    if raw is None:
        return default
    raw = raw.strip()
    if raw.lstrip("-").isdigit():   # -1 / 0 / 3600 等 → 数字
        return int(raw)
    return raw


def _resolve_device(env_key: str) -> str:
    """
    推理设备自适应：
      - auto（默认）：有 CUDA 用 cuda，否则 cpu
      - 也可通过对应环境变量（如 EMBEDDING_DEVICE=cuda）强制指定
    """
    device = os.getenv(env_key, "auto").strip().lower()
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Settings:
    # ---------------- 应用基础 ----------------
    APP_NAME: str = "企业级RAG知识库问答系统"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = _get_bool("DEBUG", False)

    # 允许跨域的前端地址，逗号分隔；* 表示全部放通
    CORS_ORIGINS: List[str] = [
        o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
    ]

    # ---------------- 目录配置 ----------------
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads"))
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_qa")

    # ---------------- LLM 配置 ----------------
    # provider 可选：ollama（本地） / openai（DeepSeek 等 OpenAI 兼容接口）
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    TEMPERATURE: float = _get_float("TEMPERATURE", 0.1)

    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
    # 模型在显存中的驻留时长。Ollama 默认 5 分钟就卸载，下次提问要冷启动重新加载，
    # 大模型（几 GB）加载一次要好几秒。-1 = 永久驻留；也可用 "10m"、"24h" 等写法。
    OLLAMA_KEEP_ALIVE = _get_keep_alive("OLLAMA_KEEP_ALIVE", "-1")

    # OpenAI 兼容接口（DeepSeek）
    OPENAI_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")
    OPENAI_MODEL: str = os.getenv("MODEL_NAME", "deepseek-chat")

    # 生成最大 token 数
    MAX_TOKENS: int = _get_int("MAX_TOKENS", 2048)

    # ---------------- Embedding 配置 ----------------
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    EMBEDDING_DEVICE: str = _resolve_device("EMBEDDING_DEVICE")
    EMBEDDING_NORMALIZE: bool = _get_bool("EMBEDDING_NORMALIZE", True)

    # ---------------- 分块参数 ----------------
    CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 500)
    CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 50)

    # ---------------- 检索参数 ----------------
    TOP_K: int = _get_int("TOP_K", 4)
    # 相关性下限（0~1，余弦相似度）。低于该值的片段直接丢弃，避免"硬凑上下文"
    SCORE_THRESHOLD: float = _get_float("SCORE_THRESHOLD", 0.3)
    # 单次问答最多携带的历史轮数
    MAX_HISTORY_TURNS: int = _get_int("MAX_HISTORY_TURNS", 3)

    # ---------------- 混合检索 ----------------
    # 关闭后仅使用向量检索
    ENABLE_HYBRID_SEARCH: bool = _get_bool("ENABLE_HYBRID_SEARCH", True)
    # 送入重排的候选数量（须 >= TOP_K）
    CANDIDATE_K: int = _get_int("CANDIDATE_K", 20)
    # RRF 融合常数，越大则各路排序差异被平滑得越明显
    RRF_K: int = _get_int("RRF_K", 60)

    # ---------------- 重排（Rerank） ----------------
    ENABLE_RERANK: bool = _get_bool("ENABLE_RERANK", True)
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    RERANK_DEVICE: str = _resolve_device("RERANK_DEVICE")
    # 重排分数下限（CrossEncoder 输出为 logits，可正可负）；设为 None 表示不过滤
    RERANK_THRESHOLD: float = _get_float("RERANK_THRESHOLD", -5.0)

    # ---------------- 查询改写 ----------------
    # 多轮对话时，把含指代的追问（"他的学历呢？"）改写成可独立检索的查询
    ENABLE_QUERY_REWRITE: bool = _get_bool("ENABLE_QUERY_REWRITE", True)
    # 生成多个查询变体并行检索（召回率更高，但检索开销成倍增加）
    ENABLE_MULTI_QUERY: bool = _get_bool("ENABLE_MULTI_QUERY", False)
    MULTI_QUERY_COUNT: int = _get_int("MULTI_QUERY_COUNT", 3)

    # ---------------- 父子块检索（Small-to-Big） ----------------
    # 小块建索引保证匹配精度，命中后返回所属父块保证上下文完整
    ENABLE_PARENT_CHILD: bool = _get_bool("ENABLE_PARENT_CHILD", True)
    PARENT_CHUNK_SIZE: int = _get_int("PARENT_CHUNK_SIZE", 1500)
    PARENT_CHUNK_OVERLAP: int = _get_int("PARENT_CHUNK_OVERLAP", 100)
    PARENT_STORE_PATH: str = os.getenv(
        "PARENT_STORE_PATH", str(BASE_DIR / "data" / "parent_store.db")
    )


    # ---------------- 上传限制 ----------------
    ALLOWED_EXTENSIONS: tuple = (".txt", ".pdf", ".docx")
    MAX_UPLOAD_MB: int = _get_int("MAX_UPLOAD_MB", 50)


settings = Settings()
