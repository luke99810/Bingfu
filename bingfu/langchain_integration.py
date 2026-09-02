"""
LangChain Integration Module (LangChain集成模块)
Provides LangChain-based implementations for BingFu agents.

- LangChainAgent: 使用LangChain重构的Agent框架
- LangChainMemory: 基于LangChain的记忆模块
- RAGRetriever: RAG检索增强功能

Note: LangChain is an optional dependency. If not installed, these classes will raise ImportError.
"""

from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field, PrivateAttr
import os

# Try to import LangChain dependencies
_LANGCHAIN_AVAILABLE = False

#: 导入失败的具体原因。
#:
#: ★ 这一行是为了让「装了却用不了」这件事说得出话来。
#:
#:   下面那个 try 覆盖十几个子导入。原来失败时是 `except ImportError: pass`，
#:   于是无论缺的是 langchain 本身、还是 langchain-classic 换了模块路径、
#:   还是 faiss 没装，外面看到的都是同一句「LangChain is not installed」——
#:   一个已经 pip install 过的人拿着这句话无从下手。
_LANGCHAIN_IMPORT_ERROR = ""
try:
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_classic.chains import LLMChain, RetrievalQA
    from langchain_classic.memory import (
        ConversationBufferMemory,
        ConversationSummaryMemory,
        ConversationBufferWindowMemory,
        VectorStoreRetrieverMemory,
    )
    from langchain_classic.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
    )
    from langchain_classic.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        BaseMessage,
    )
    from langchain_classic.tools import BaseTool, StructuredTool, Tool
    from langchain_classic.agents import (
        initialize_agent,
        AgentType,
        Tool as AgentTool,
    )
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS, Chroma
    from langchain_core.output_parsers import StrOutputParser
    
    # CharacterTextSplitter is deprecated but may still be needed by some code
    try:
        from langchain_text_splitters import CharacterTextSplitter
    except ImportError:
        CharacterTextSplitter = RecursiveCharacterTextSplitter  # fallback
    
    # Load environment variables
    load_dotenv()
    _LANGCHAIN_AVAILABLE = True
except ImportError as exc:
    _LANGCHAIN_IMPORT_ERROR = str(exc)


def langchain_status():
    """返回 (是否可用, 不可用时的具体原因)。"""

    return _LANGCHAIN_AVAILABLE, _LANGCHAIN_IMPORT_ERROR


class LangChainMemory(BaseModel):
    """
    LangChain Memory (LangChain记忆模块)
    基于LangChain的高级记忆系统，支持多种记忆类型。
    
    记忆类型：
    - buffer: 对话缓冲区记忆
    - summary: 对话摘要记忆
    - window: 滑动窗口记忆
    - vector: 向量存储记忆（支持长期记忆检索）
    
    Note: Requires langchain package. Install with: pip install langchain langchain-openai
    """

    memory_type: str = Field(default="buffer", description="记忆类型")
    llm: Optional[Any] = Field(default=None, description="LLM实例")
    vector_store: Optional[Any] = Field(default=None, description="向量存储")
    window_size: int = Field(default=5, description="窗口大小")

    _memory: Any = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def _create_memory(self):
        """创建LangChain记忆实例"""
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain 不可用：%s。 "
                "安装：pip install 'bingfu[langchain,rag]'"
                % (_LANGCHAIN_IMPORT_ERROR or "未安装")
            )
        
        if self.memory_type == "buffer":
            self._memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        elif self.memory_type == "summary":
            if not self.llm:
                raise ValueError("summary memory requires an LLM instance")
            self._memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True
            )
        elif self.memory_type == "window":
            self._memory = ConversationBufferWindowMemory(
                k=self.window_size,
                memory_key="chat_history",
                return_messages=True
            )
        elif self.memory_type == "vector":
            if not self.vector_store:
                raise ValueError("vector memory requires a vector store")
            self._memory = VectorStoreRetrieverMemory(
                retriever=self.vector_store.as_retriever(),
                memory_key="chat_history",
                return_messages=True
            )
        else:
            raise ValueError(f"Unknown memory type: {self.memory_type}")

    @property
    def memory(self):
        """获取记忆实例"""
        if self._memory is None:
            self._create_memory()
        return self._memory

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """保存对话上下文"""
        self.memory.save_context(inputs, outputs)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """加载记忆变量"""
        return self.memory.load_memory_variables(inputs)

    def clear(self) -> None:
        """清除记忆"""
        if self._memory:
            self._memory.clear()


class RAGRetriever(BaseModel):
    """
    RAG Retriever (RAG检索器)
    实现检索增强生成功能，支持从文档中检索相关信息。
    
    功能：
    - 文档加载与分割
    - 向量存储与检索
    - 基于检索结果的问答
    
    Note: Requires langchain and vector store packages. 
    Install with: pip install langchain langchain-openai faiss-cpu
    """

    embedding_model: str = Field(default="text-embedding-3-small", description="嵌入模型")
    vector_store_type: str = Field(default="faiss", description="向量存储类型")
    persist_directory: Optional[str] = Field(default=None, description="持久化目录")
    _vector_store: Any = PrivateAttr(default=None)
    _embeddings: Any = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def _init_embeddings(self):
        """初始化嵌入模型 — 支持OpenAI/DeepSeek/本地HuggingFace回退"""
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain 不可用：%s。 "
                "安装：pip install 'bingfu[langchain,rag]'"
                % (_LANGCHAIN_IMPORT_ERROR or "未安装")
            )
        
        # 尝试OpenAI embeddings（需要OPENAI_API_KEY）
        #
        # ★ 这个探测必须带超时，且不能重试。
        #
        #   原来是裸的 OpenAIEmbeddings(...) + embed_query("test")，
        #   靠 `except Exception` 兜底往下走。但 except 拦得住**报错**，
        #   拦不住**卡住** —— 环境里若设了 OPENAI_BASE_URL 指向一个
        #   不支持 embeddings 或干脆没响应的中转端点，SDK 会带着默认
        #   重试一直等下去。实测这里挂了整整十分钟才被强杀，
        #   而后面两级本地回退一次都没轮到。
        #
        #   一个「失败了会自动降级」的链条，只在失败能被**及时判定**时才成立。
        try:
            self._embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                timeout=8.0,
                max_retries=0,
            )
            # 快速验证是否可用
            _ = self._embeddings.embed_query("test")
            return
        except Exception:
            self._embeddings = None
        
        # 回退1: 本地HuggingFace模型
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            return
        except ImportError:
            pass
        
        # 回退2: langchain_community (deprecated but still works)
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            return
        except Exception:
            pass
        
        raise RuntimeError(
            "无可用的embedding模型！请选择以下任一方案：\n"
            "  1. 设置 OPENAI_API_KEY 使用 OpenAI embeddings\n"
            "  2. pip install langchain-huggingface 使用本地模型\n"
            "  3. pip install sentence-transformers 使用本地模型"
        )

    def _init_vector_store(self):
        """初始化向量存储"""
        if self._embeddings is None:
            self._init_embeddings()

        if self.vector_store_type == "faiss":
            if self.persist_directory and os.path.exists(self.persist_directory):
                self._vector_store = FAISS.load_local(
                    self.persist_directory,
                    self._embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                self._vector_store = FAISS.from_texts(
                    ["初始化文档"],
                    self._embeddings
                )
        elif self.vector_store_type == "chroma":
            self._vector_store = Chroma(
                embedding_function=self._embeddings,
                persist_directory=self.persist_directory
            )
        else:
            raise ValueError(f"Unknown vector store type: {self.vector_store_type}")

    @property
    def vector_store(self):
        """获取向量存储实例"""
        if self._vector_store is None:
            self._init_vector_store()
        return self._vector_store

    def add_documents(self, documents: List[str]) -> None:
        """
        添加文档到向量存储
        
        Args:
            documents: 文档内容列表
        """
        self.vector_store.add_texts(documents)

    def add_document_from_file(self, file_path: str) -> None:
        """
        从文件加载文档
        
        Args:
            file_path: 文件路径
        """
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        self.vector_store.add_documents(texts)

    def add_documents_from_directory(self, dir_path: str) -> None:
        """
        从目录加载所有文档
        
        Args:
            dir_path: 目录路径
        """
        loader = DirectoryLoader(dir_path, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = text_splitter.split_documents(documents)
        self.vector_store.add_documents(texts)

    def similarity_search(self, query: str, k: int = 3) -> List[Any]:
        """
        相似性检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
        
        Returns:
            检索到的文档列表
        """
        return self.vector_store.similarity_search(query, k=k)

    def create_retrieval_qa_chain(self, llm: Any) -> Any:
        """
        创建检索问答链
        
        Args:
            llm: LLM实例
        
        Returns:
            RetrievalQA链
        """
        retriever = self.vector_store.as_retriever()
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

    def save(self) -> None:
        """保存向量存储到磁盘"""
        if self.persist_directory and self.vector_store_type == "faiss":
            self.vector_store.save_local(self.persist_directory)
        elif self.vector_store_type == "chroma":
            self.vector_store.persist()


class LangChainAgent(BaseModel):
    """
    LangChain Agent (LangChain智能体)
    使用LangChain重构的Agent框架，支持：
    - 标准Agent执行
    - RAG检索增强
    - 多种记忆模式
    - 工具调用
    
    Note: Requires langchain package. Install with: pip install langchain langchain-openai
    """

    name: str = Field(..., description="Agent名称（将领名号）")
    role: Optional[str] = Field(default=None, description="Agent角色/职位")
    description: Optional[str] = Field(default=None, description="Agent描述")
    llm_model: str = Field(default="gpt-3.5-turbo", description="LLM模型")
    temperature: float = Field(default=0.7, description="温度参数")
    memory_type: str = Field(default="buffer", description="记忆类型")
    use_rag: bool = Field(default=False, description="是否启用RAG")
    rag_retriever: Optional[RAGRetriever] = Field(default=None, description="RAG检索器")

    # 内部状态
    _llm: Any = PrivateAttr(default=None)
    _agent: Any = PrivateAttr(default=None)
    _memory: Any = PrivateAttr(default=None)
    _tools: List[Any] = PrivateAttr(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def _init_llm(self):
        """初始化LLM — 支持DeepSeek和OpenAI"""
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain 不可用：%s。 "
                "安装：pip install 'bingfu[langchain,rag]'"
                % (_LANGCHAIN_IMPORT_ERROR or "未安装")
            )
        
        # 检测DeepSeek模型并设置对应base_url
        model_lower = self.llm_model.lower()
        is_deepseek = "deepseek" in model_lower
        
        kwargs = dict(
            model=self.llm_model,
            temperature=self.temperature,
        )
        
        if is_deepseek:
            kwargs["openai_api_key"] = os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY"))
            kwargs["openai_api_base"] = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        else:
            kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        
        self._llm = ChatOpenAI(**kwargs)

    def _init_memory(self):
        """初始化记忆"""
        langchain_memory = LangChainMemory(
            memory_type=self.memory_type,
            llm=self._llm if self.memory_type == "summary" else None,
            vector_store=self.rag_retriever.vector_store if self.memory_type == "vector" and self.rag_retriever else None
        )
        self._memory = langchain_memory.memory

    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        prompt = f"你是将领「{self.name}」"
        if self.role:
            prompt += f"，职位「{self.role}」"
        if self.description:
            prompt += f"。{self.description}"
        
        prompt += """
你是一个古代军事风格的智能体，用中文回复，风格简练有力如军令。
你可以使用提供的工具来完成任务，遵循思考-行动-观察循环执行，直到给出最终结论。
"""
        return prompt

    def add_tool(self, name: str, func: Callable, description: str) -> None:
        """
        添加工具
        
        Args:
            name: 工具名称
            func: 工具函数
            description: 工具描述
        """
        tool = Tool(
            name=name,
            func=func,
            description=description
        )
        self._tools.append(tool)

    def add_structured_tool(self, func: Callable) -> None:
        """
        添加结构化工具
        
        Args:
            func: 工具函数（需要有类型注解）
        """
        tool = StructuredTool.from_function(func)
        self._tools.append(tool)

    def initialize(self) -> None:
        """初始化Agent"""
        self._init_llm()
        self._init_memory()
        
        # 创建系统提示模板
        system_prompt = self._build_system_prompt()
        
        # 初始化Agent (LangChain v1.x expects string, not SystemMessage object)
        self._agent = initialize_agent(
            self._tools,
            self._llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self._memory,
            verbose=True,
            agent_kwargs={
                "system_message": system_prompt
            }
        )

    def run(self, task: str) -> str:
        """
        执行任务
        
        Args:
            task: 任务描述
        
        Returns:
            执行结果
        """
        if self._agent is None:
            self.initialize()
        
        try:
            result = self._agent.run(task)
            return result
        except Exception as e:
            return f"⚠️ Agent执行失败: {str(e)}"

    def chat(self, message: str) -> str:
        """
        对话模式
        
        Args:
            message: 用户消息
        
        Returns:
            Agent回复
        """
        return self.run(message)

    def run_with_rag(self, query: str) -> Dict[str, Any]:
        """
        使用RAG执行查询
        
        Args:
            query: 查询文本
        
        Returns:
            包含答案和来源文档的字典
        """
        if not self.use_rag or not self.rag_retriever:
            raise ValueError("RAG未启用或未配置检索器")
        
        if self._llm is None:
            self._init_llm()
        
        qa_chain = self.rag_retriever.create_retrieval_qa_chain(self._llm)
        result = qa_chain({"query": query})
        
        return {
            "answer": result.get("result", ""),
            "sources": [doc.page_content for doc in result.get("source_documents", [])]
        }

    def add_rag_document(self, content: str) -> None:
        """
        向RAG添加文档
        
        Args:
            content: 文档内容
        """
        if not self.rag_retriever:
            self.rag_retriever = RAGRetriever()
        
        self.rag_retriever.add_documents([content])
        self.use_rag = True

    def __str__(self) -> str:
        return f"LangChainAgent(name='{self.name}', role='{self.role}', model='{self.llm_model}')"

    def __repr__(self) -> str:
        return self.__str__()