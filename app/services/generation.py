# app/services/generation.py
"""生成服务：负责 LLM 初始化与答案生成（含反幻觉约束）。"""
import logging
import re
from typing import Iterator, List, Optional, Sequence, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import settings

logger = logging.getLogger(__name__)

# 无答案时的统一兜底话术（便于前端识别与统计）
NO_ANSWER = "知识库中暂无相关信息，无法回答。"

SYSTEM_PROMPT = f"""你是一个严谨的企业知识库助手。请严格依据下面【上下文】中的信息回答问题。

规则：
1. 只使用上下文提供的信息，不得依赖你自身的先验知识；
2. 如果上下文不足以回答问题，请原样输出："{NO_ANSWER}" 严禁编造或猜测；
3. 回答使用与提问相同的语言，条理清晰，必要时分点说明；
4. 若上下文存在多个文档来源，请综合判断，不要遗漏关键信息。"""

HUMAN_PROMPT = """【上下文】
{context}

【问题】
{question}"""

# 多轮对话下，追问常含指代（"他的学历呢？"），需改写成可独立检索的语句
REWRITE_PROMPT = """你是查询改写助手。请把用户的【当前问题】结合【对话历史】，改写为一句可以独立理解的检索查询语句。

要求：
1. 只输出改写后的查询语句本身，不要解释、不要引号、不要结尾标点；
2. 必须消解指代：把"他/她/它/这个/那个/前者/后者"等替换为历史中的具体对象；
3. 补全被省略的主语或限定条件，但不得引入历史中不存在的新信息；
4. 如果【当前问题】本身已独立完整（无指代、无省略），则原样输出。

【对话历史】
{history}

【当前问题】
{question}"""

# Multi-Query：生成多个表述变体，并行检索以提升召回率
MULTI_QUERY_PROMPT = """你是检索优化助手。请为下面的问题生成 {count} 个不同表述的检索查询语句，用于提升向量检索的召回率。

要求：
1. 每行一句，严格输出 {count} 行，不要编号、不要引号、不要空行、不要任何解释；
2. 各句须从不同角度表述同一信息需求（同义词替换、句式变换、更概括或更具体）；
3. 保持原意，不得引入新的问题或无关内容。

【对话历史】
{history}

【原始问题】
{question}"""

# deepseek-r1 等推理模型会把思维链包在 <think> 标签里返回，需剔除
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """移除推理模型的思维链内容，只保留最终答案"""
    if not text:
        return text
    cleaned = _THINK_PATTERN.sub("", text)
    # 标签未闭合时取最后一个 <think> 之前的内容
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0]
    return cleaned.strip()


class StreamingReasoningStripper:
    """
    流式场景下的思维链清洗器。

    <think> ... </think> 可能横跨多个 token chunk，
    这里用状态机 + 尾部缓冲处理跨 chunk 的半个标签。
    """

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self):
        self._in_think = False
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        """喂入一个增量片段，返回可安全输出的文本"""
        if not chunk:
            return ""
        self._buffer += chunk
        out: List[str] = []

        while self._buffer:
            if self._in_think:
                end = self._buffer.find(self.CLOSE_TAG)
                if end == -1:
                    keep = self._tag_prefix_len(self.CLOSE_TAG)
                    self._buffer = self._buffer[-keep:] if keep else ""
                    break
                self._buffer = self._buffer[end + len(self.CLOSE_TAG):]
                self._in_think = False
            else:
                start = self._buffer.find(self.OPEN_TAG)
                if start == -1:
                    keep = self._tag_prefix_len(self.OPEN_TAG)
                    if keep:
                        out.append(self._buffer[:-keep])
                        self._buffer = self._buffer[-keep:]
                    else:
                        out.append(self._buffer)
                        self._buffer = ""
                    break
                out.append(self._buffer[:start])
                self._buffer = self._buffer[start + len(self.OPEN_TAG):]
                self._in_think = True

        return "".join(out)

    def close(self) -> str:
        """流结束时冲刷缓冲区：思维链内的残留直接丢弃"""
        if self._in_think:
            self._buffer = ""
            return ""
        tail = self._buffer
        self._buffer = ""
        return tail

    def _tag_prefix_len(self, tag: str) -> int:
        """缓冲区尾部与 tag 前缀重合的最大长度，用于保住半个标签"""
        n = min(len(tag) - 1, len(self._buffer))
        for i in range(n, 0, -1):
            if self._buffer.endswith(tag[:i]):
                return i
        return 0


class GenerationService:
    """LLM 生成服务，支持 Ollama 与 OpenAI 兼容接口两种后端"""

    def __init__(self):
        self.llm: BaseChatModel = self._build_llm()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("history", optional=True),
                ("human", HUMAN_PROMPT),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    # ---------------- LLM 构建 ----------------

    def _build_llm(self) -> BaseChatModel:
        provider = settings.LLM_PROVIDER
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            if not settings.OPENAI_API_KEY:
                logger.warning("LLM_PROVIDER=openai 但未配置 DEEPSEEK_API_KEY")
            logger.info("使用 OpenAI 兼容接口: %s / %s", settings.OPENAI_API_BASE, settings.OPENAI_MODEL)
            return ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY or None,
                base_url=settings.OPENAI_API_BASE,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )

        if provider != "ollama":
            logger.warning("未知 LLM_PROVIDER=%s，回退到 ollama", provider)

        from langchain_ollama import ChatOllama

        logger.info(
            "使用 Ollama: %s / %s (keep_alive=%s)",
            settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL, settings.OLLAMA_KEEP_ALIVE,
        )
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.TEMPERATURE,
            num_predict=settings.MAX_TOKENS,
            # 让模型常驻显存，避免每次问答冷启动重新加载（大模型加载耗时数秒）
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
        )

    # ---------------- 生成 ----------------

    def generate(
        self,
        question: str,
        context: str,
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> str:
        """
        调用 LLM 生成答案。
        history: [(用户问题, 助手回答), ...]，仅保留最近若干轮。
        """
        messages = self._build_history(history)
        try:
            answer = self.chain.invoke(
                {"context": context, "question": question, "history": messages}
            )
        except Exception as exc:
            logger.exception("LLM 调用失败: %s", exc)
            raise RuntimeError(f"模型调用失败，请确认 LLM 服务已启动：{exc}") from exc

        return strip_reasoning(answer)

    def stream(
        self,
        question: str,
        context: str,
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Iterator[str]:
        """
        流式生成，逐段产出已清洗掉思维链的答案文本。
        :raises RuntimeError: 模型调用失败
        """
        messages = self._build_history(history)
        stripper = StreamingReasoningStripper()
        try:
            for chunk in self.chain.stream(
                {"context": context, "question": question, "history": messages}
            ):
                if not chunk:
                    continue
                text = stripper.feed(chunk)
                if text:
                    yield text
        except Exception as exc:
            logger.exception("LLM 流式调用失败: %s", exc)
            raise RuntimeError(f"模型调用失败，请确认 LLM 服务已启动：{exc}") from exc

        tail = stripper.close()
        if tail:
            yield tail

    # ---------------- 查询改写 ----------------

    def rewrite_query(
        self,
        question: str,
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> str:
        """
        把含指代的追问改写成可独立检索的查询语句。
        失败时静默回退为原始问题，不影响主流程。
        """
        if not history or not settings.ENABLE_QUERY_REWRITE:
            return question

        chain = ChatPromptTemplate.from_template(REWRITE_PROMPT) | self.llm | StrOutputParser()
        try:
            raw = chain.invoke(
                {"history": self._format_history(history), "question": question}
            )
        except Exception as exc:
            logger.warning("查询改写失败，回退为原始问题: %s", exc)
            return question

        rewritten = strip_reasoning(raw).strip().strip("\"'“”‘’").strip()
        if not rewritten:
            return question

        if rewritten != question:
            logger.info("查询改写: %r -> %r", question, rewritten)
        return rewritten

    def expand_queries(
        self,
        question: str,
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> List[str]:
        """
        Multi-Query：生成多个表述变体并行检索，提升召回率。
        返回列表中第一项为原始问题（保留字面匹配能力）。
        """
        count = max(1, settings.MULTI_QUERY_COUNT)
        chain = (
            ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
            | self.llm
            | StrOutputParser()
        )
        try:
            raw = chain.invoke(
                {
                    "history": self._format_history(history),
                    "question": question,
                    "count": count,
                }
            )
        except Exception as exc:
            logger.warning("Multi-Query 生成失败，回退为原始问题: %s", exc)
            return [question]

        queries = self._parse_query_lines(strip_reasoning(raw), question, count)
        logger.info("Multi-Query 扩展: %r -> %r", question, queries)
        return queries

    @staticmethod
    def _parse_query_lines(raw: str, fallback: str, limit: int) -> List[str]:
        """解析 LLM 输出的多行查询，清理编号与引号"""
        queries: List[str] = []
        seen = set()
        for line in (raw or "").splitlines():
            line = re.sub(r"^\s*(\d+[.、)）]|[-*•])\s*", "", line.strip())
            line = line.strip().strip("\"'“”‘’").strip()
            if len(line) < 2 or line in seen:
                continue
            seen.add(line)
            queries.append(line)
            if len(queries) >= limit:
                break

        if not queries:
            return [fallback]
        if fallback not in seen:
            queries.insert(0, fallback)  # 始终保留原始问题参与检索
            queries = queries[:limit]
        return queries

    @staticmethod
    def _format_history(history: Optional[Sequence[Tuple[str, str]]]) -> str:
        if not history:
            return "（无历史对话）"
        return "\n".join(f"用户：{q}\n助手：{a}" for q, a in history)

    @staticmethod
    def _build_history(
        history: Optional[Sequence[Tuple[str, str]]],
    ) -> List:
        """把 [(q, a)] 转成 LangChain 消息列表，并截断到最近 N 轮"""
        if not history:
            return []
        turns = list(history)[-settings.MAX_HISTORY_TURNS :]
        messages: List = []
        for question, answer in turns:
            messages.append(HumanMessage(content=question))
            messages.append(AIMessage(content=answer))
        return messages
