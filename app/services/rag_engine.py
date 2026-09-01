# app/services/rag_engine.py
import os
from typing import List, Dict, Any
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_community.chat_models import ChatOpenAI,ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.core.config import settings
from app.utils.file_utils import load_document

class RAGEngine:
    def __init__(self):
        # 1. 初始化 Embedding 模型（本地运行）
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda'},  # 如果有GPU可改为 'cuda'
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 2. 初始化 LLM (DeepSeek)
        # self.llm = ChatOpenAI(
        #     model=settings.MODEL_NAME,
        #     openai_api_key=settings.OPENAI_API_KEY,
        #     openai_api_base=settings.OPENAI_API_BASE,
        #     temperature=0.1  # 温度低，回答更严谨
        # )
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.1,
            # 可选参数：num_predict, top_k 等
        )
        # 3. 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        # 4. 向量存储对象（延迟初始化）
        self.vectorstore = None
        
        # 5. 提示词模板（强制基于上下文回答）
        self.prompt = ChatPromptTemplate.from_template("""
你是一个严谨的知识助手。请严格基于以下【上下文】来回答问题。
如果上下文中没有相关信息，请直接回答“知识库中暂无相关信息，无法回答”，严禁编造或猜测。

【上下文】
{context}

【问题】
{question}
""")
    
    def index_document(self, file_path: str) -> int:
        """
        索引单个文档：加载 -> 分割 -> 向量化 -> 存入Chroma
        返回分块数量
        """
        # 加载文档
        docs = load_document(file_path)
        if not docs:
            raise ValueError("文档加载失败或为空")
        
        # 分割文档
        chunks = self.text_splitter.split_documents(docs)
        
        # 如果向量存储尚未初始化，则创建；否则追加
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=settings.CHROMA_PERSIST_DIR
            )
        else:
            self.vectorstore.add_documents(chunks)
        
        # 持久化到磁盘
        self.vectorstore.persist()
        return len(chunks)
    
    def search(self, query: str, top_k: int = 4) -> List[Document]:
        """检索最相关的文档片段"""
        if self.vectorstore is None:
            # 尝试从磁盘加载已有的向量库
            if os.path.exists(settings.CHROMA_PERSIST_DIR):
                self.vectorstore = Chroma(
                    persist_directory=settings.CHROMA_PERSIST_DIR,
                    embedding_function=self.embeddings
                )
            else:
                return []
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        # return retriever.get_relevant_documents(query)
        return retriever.invoke(query)
    
    def ask(self, query: str) -> Dict[str, Any]:
        """
        问答入口：检索 -> 构建上下文 -> 调用LLM生成
        返回包含答案和引用来源的字典
        """
        # 1. 检索相关文档
        relevant_docs = self.search(query)
        
        if not relevant_docs:
            return {
                "answer": "知识库为空或未检索到相关内容，请先上传文档。",
                "sources": []
            }
        
        # 2. 拼接上下文
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # 3. 构建Prompt并调用LLM
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": query
        })
        
        # 4. 提取来源（文件名或页码等元数据）
        sources = list(set([doc.metadata.get("source", "未知来源") for doc in relevant_docs]))
        
        return {
            "answer": response.content,
            "sources": sources
        }

# 创建全局单例（方便在API中复用）
rag_engine = RAGEngine()