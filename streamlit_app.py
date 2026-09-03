# streamlit_app.py
"""
企业知识库问答系统 - Streamlit 前端

启动：
    streamlit run streamlit_app.py

通过环境变量可覆盖后端地址：
    RAG_API_BASE=http://192.168.1.10:8000 streamlit run streamlit_app.py
"""
import json
import os
import re
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import quote

import requests
import streamlit as st

st.set_page_config(
    page_title="企业知识库问答",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000")
CITE_PATTERN = re.compile(r"\[(\d+)\]")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 1rem; }
      .stChatMessage { border-radius: 12px; }
      section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
      .cap-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0 12px; }
      .cap {
        font-size: 11px; padding: 2px 9px; border-radius: 20px;
        background: #f1f5f9; color: #64748b;
      }
      .cap.on { background: #ecfdf5; color: #059669; }
      .stat-box {
        text-align: center; background: #f8fafc; border-radius: 10px;
        padding: 10px 4px; border: 1px solid #e5e7eb;
      }
      .stat-box b { display: block; font-size: 19px; color: #4f46e5; }
      .stat-box span { font-size: 11px; color: #64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- 会话状态 ----------------

def init_state() -> None:
    st.session_state.setdefault("messages", [])      # 界面消息：{role, content, meta}
    st.session_state.setdefault("history", [])       # 多轮历史：[[question, answer]]
    st.session_state.setdefault("api_base", DEFAULT_API_BASE)
    st.session_state.setdefault("top_k", 4)
    st.session_state.setdefault("scope", "")
    st.session_state.setdefault("_uploaded_key", "")


init_state()


def api(endpoint: str) -> str:
    return f"{st.session_state.api_base.rstrip('/')}{endpoint}"


# ---------------- 后端接口 ----------------

@st.cache_data(ttl=5, show_spinner=False)
def fetch_json(endpoint: str, api_base: str) -> Optional[dict]:
    """带短缓存的 GET 请求，避免每次交互都拉取"""
    try:
        resp = requests.get(f"{api_base.rstrip('/')}{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def post_upload(file) -> Tuple[bool, str]:
    try:
        resp = requests.post(
            api("/api/v1/documents/upload"),
            files={"file": (file.name, file.getvalue())},
            timeout=300,
        )
    except Exception as exc:
        return False, f"无法连接后端服务：{exc}"
    if resp.status_code == 200:
        data = resp.json()
        return True, f"「{data['source']}」索引成功，共 {data['chunk_count']} 个分块"
    try:
        return False, resp.json().get("detail", f"上传失败（{resp.status_code}）")
    except Exception:
        return False, f"上传失败（{resp.status_code}）"


def delete_document(source: str) -> Tuple[bool, str]:
    try:
        resp = requests.delete(api(f"/api/v1/documents/{quote(source)}"), timeout=30)
    except Exception as exc:
        return False, str(exc)
    if resp.status_code == 200:
        return True, "已删除"
    return False, resp.json().get("detail", "删除失败")


# ---------------- SSE 流式问答 ----------------

def iter_sse(payload: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """请求流式接口并逐事件解析 SSE"""
    resp = requests.post(
        api("/api/v1/chat/stream"), json=payload, stream=True, timeout=(10, 600)
    )
    resp.raise_for_status()

    buffer = ""
    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
        if not chunk:
            continue
        buffer += chunk
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            event = data = None
            for line in block.splitlines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
            if not event or not data:
                continue
            try:
                yield event, json.loads(data)
            except ValueError:
                continue


def to_html(text: str) -> str:
    """把 [n] 引用标记渲染成上标角标（保留 Markdown 其他语法）"""
    return CITE_PATTERN.sub(
        r'<sup style="color:#4f46e5;font-weight:700;padding:0 2px">\1</sup>', text
    )


def render_references(refs: List[Dict[str, Any]], expanded: bool = False) -> None:
    if not refs:
        return
    with st.expander(f"📎 引用来源 · {len(refs)} 条", expanded=expanded):
        for idx, ref in enumerate(refs, start=1):
            page = f" · 第 {ref['page']} 页" if ref.get("page") else ""
            st.markdown(
                f"`[{idx}]` **{ref['source']}**{page} · 相关度 `{ref['score']:.3f}`"
            )
            if ref.get("snippet"):
                st.caption(ref["snippet"])


# ---------------- 侧边栏 ----------------

def render_sidebar() -> None:
    with st.sidebar:
        st.title("📚 企业知识库")
        st.caption("RAG · 混合检索 · 可溯源")

        api_base = st.text_input("后端地址", value=st.session_state.api_base, key="api_input")
        if api_base != st.session_state.api_base:
            st.session_state.api_base = api_base
            st.rerun()

        base = st.session_state.api_base
        stats = fetch_json("/api/v1/documents/stats", base)
        docs = fetch_json("/api/v1/documents", base)

        if stats is None:
            st.error("无法连接后端服务，请确认已启动：\n`uvicorn app.main:app`")
            return

        # 统计
        c1, c2, c3 = st.columns(3)
        for col, value, label in (
            (c1, stats.get("document_count", 0), "文档"),
            (c2, stats.get("chunk_count", 0), "分块"),
            (c3, stats.get("parent_count", 0), "父块"),
        ):
            col.markdown(
                f'<div class="stat-box"><b>{value}</b><span>{label}</span></div>',
                unsafe_allow_html=True,
            )

        # 能力开关
        caps = [
            ("混合检索", stats.get("hybrid_search")),
            ("重排", stats.get("rerank_enabled")),
            ("父子块", stats.get("parent_child")),
            ("查询改写", stats.get("query_rewrite")),
            ("多查询", stats.get("multi_query")),
        ]
        st.markdown(
            '<div class="cap-row">'
            + "".join(
                f'<span class="cap {"on" if on else ""}">{name}{" ✓" if on else " ✕"}</span>'
                for name, on in caps
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # 上传
        st.subheader("上传文档")
        uploaded = st.file_uploader(
            "支持 PDF / DOCX / TXT", type=["pdf", "docx", "txt"], label_visibility="collapsed"
        )
        if uploaded is not None:
            file_key = f"{uploaded.name}:{uploaded.size}"
            if st.session_state._uploaded_key != file_key:
                st.session_state._uploaded_key = file_key
                with st.spinner(f"正在索引「{uploaded.name}」…"):
                    ok, msg = post_upload(uploaded)
                st.success(msg) if ok else st.error(msg)
                fetch_json.clear()
                st.rerun()

        # 文档列表
        st.subheader("知识库")
        documents = (docs or {}).get("documents", [])
        if not documents:
            st.caption("暂无文档，请先上传")
        for doc in documents:
            col1, col2 = st.columns([6, 1])
            col1.markdown(f"📄 **{doc['source']}**  \n`{doc['chunk_count']}` 分块")
            if col2.button("🗑", key=f"del_{doc['source']}", help="删除该文档"):
                ok, msg = delete_document(doc["source"])
                st.success(msg) if ok else st.error(msg)
                if st.session_state.scope == doc["source"]:
                    st.session_state.scope = ""
                fetch_json.clear()
                st.rerun()

        st.divider()

        # 检索设置
        st.subheader("检索设置")
        st.session_state.top_k = st.slider("召回数量 top_k", 1, 20, st.session_state.top_k)
        options = [""] + [d["source"] for d in documents]
        st.session_state.scope = st.selectbox(
            "检索范围",
            options,
            index=options.index(st.session_state.scope)
            if st.session_state.scope in options
            else 0,
            format_func=lambda x: "全部文档" if not x else x,
        )

        st.divider()
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

        st.caption(
            f"Embedding：`{stats.get('embedding_model', '-')}`  \n"
            f"LLM：`{stats.get('llm_provider', '-')}`"
        )


# ---------------- 主对话区 ----------------

def render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            meta = msg.get("meta") or {}
            if meta.get("references"):
                render_references(meta["references"])
            st.markdown(to_html(msg["content"]), unsafe_allow_html=True)


def handle_question(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        refs_holder = st.empty()
        answer_holder = st.empty()
        state: Dict[str, Any] = {}

        try:
            events = iter_sse(
                {
                    "question": question,
                    "history": st.session_state.history,
                    "top_k": st.session_state.top_k,
                    "source": st.session_state.scope or None,
                }
            )

            full = ""
            last_render = 0.0
            for event, data in events:
                if event == "references":
                    with refs_holder.container():
                        render_references(data.get("references", []))
                    state["meta"] = data
                elif event == "token":
                    full += data.get("content", "")
                    # 节流渲染，保证打字机效果的同时避免频繁刷新
                    now = time.time()
                    if now - last_render > 0.04:
                        answer_holder.markdown(full + "▌")
                        last_render = now
                elif event == "error":
                    full += f"\n\n⚠️ {data.get('detail', '生成失败')}"
                elif event == "done":
                    state["meta"] = data
                    full = data.get("answer") or full

            answer_holder.markdown(to_html(full), unsafe_allow_html=True)

        except requests.exceptions.ConnectionError:
            full = "⚠️ 无法连接后端服务，请确认已启动 `uvicorn app.main:app`"
            answer_holder.error(full)
        except requests.exceptions.HTTPError as exc:
            full = f"⚠️ 请求失败：{exc}"
            answer_holder.error(full)
        except Exception as exc:
            full = f"⚠️ {exc}"
            answer_holder.error(full)

    st.session_state.messages.append(
        {"role": "assistant", "content": full, "meta": state.get("meta")}
    )
    if not full.startswith("⚠️"):
        st.session_state.history.append([question, full])
        if len(st.session_state.history) > 10:
            st.session_state.history.pop(0)
    fetch_json.clear()


def main() -> None:
    render_sidebar()

    st.header("知识库问答")
    if not st.session_state.messages:
        st.info(
            "上传文档后即可开始提问。回答严格基于知识库内容并附带引用来源；"
            "支持多轮追问（自动消解指代），无法回答时会明确说明。",
            icon="💡",
        )

    render_history()

    if question := st.chat_input("输入问题，Enter 发送"):
        handle_question(question)
        st.rerun()


main()
