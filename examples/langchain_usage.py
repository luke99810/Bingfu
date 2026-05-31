"""
LangChain Integration Examples (LangChain集成示例)

展示如何使用BingFu的LangChain集成功能：
1. LangChainAgent - 使用LangChain重构的Agent框架
2. LangChainMemory - 多种记忆模式
3. RAGRetriever - RAG检索增强功能
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 确保设置了OpenAI API Key
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("请设置 OPENAI_API_KEY 环境变量")


def example_langchain_agent():
    """示例1: 使用LangChainAgent"""
    from bingfu import LangChainAgent
    
    # 创建LangChain Agent
    agent = LangChainAgent(
        name="诸葛亮",
        role="军师",
        description="精通兵法谋略，善于运筹帷幄",
        llm_model="gpt-3.5-turbo",
        temperature=0.3,
        memory_type="buffer"
    )
    
    # 添加工具
    def get_weather(city: str) -> str:
        """获取指定城市的天气"""
        return f"{city}天气晴朗，温度25°C"
    
    agent.add_tool(
        name="get_weather",
        func=get_weather,
        description="获取指定城市的天气信息"
    )
    
    # 初始化并执行任务
    agent.initialize()
    result = agent.run("帮我分析一下当前局势，需要考虑天气因素")
    print(f"🎯 LangChainAgent执行结果:\n{result}\n")


def example_langchain_memory():
    """示例2: 使用LangChainMemory的多种模式"""
    from bingfu import LangChainMemory
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # 示例1: Buffer记忆
    buffer_memory = LangChainMemory(memory_type="buffer")
    buffer_memory.save_context(
        {"input": "你好"},
        {"output": "您好！我是您的AI助手。"}
    )
    print(f"📝 Buffer记忆内容: {buffer_memory.load_memory_variables({})}")
    
    # 示例2: Summary记忆（需要LLM）
    summary_memory = LangChainMemory(
        memory_type="summary",
        llm=llm
    )
    summary_memory.save_context(
        {"input": "我叫张三，今年25岁"},
        {"output": "很高兴认识你，张三！"}
    )
    print(f"📝 Summary记忆内容: {summary_memory.load_memory_variables({})}")
    
    # 示例3: Window记忆（滑动窗口）
    window_memory = LangChainMemory(
        memory_type="window",
        window_size=3
    )
    for i in range(5):
        window_memory.save_context(
            {"input": f"消息{i}"},
            {"output": f"回复{i}"}
        )
    print(f"📝 Window记忆内容: {window_memory.load_memory_variables({})}\n")


def example_rag_retriever():
    """示例3: 使用RAGRetriever进行检索增强"""
    from bingfu import RAGRetriever
    from langchain_openai import ChatOpenAI
    
    # 创建RAG检索器
    retriever = RAGRetriever(
        embedding_model="text-embedding-3-small",
        vector_store_type="faiss",
        persist_directory="./data/faiss_index"
    )
    
    # 添加示例文档
    documents = [
        "孙子兵法是中国古代著名的军事著作，作者是孙武。",
        "孙子兵法共有十三篇，包括始计、作战、谋攻等。",
        "兵者，国之大事，死生之地，存亡之道，不可不察也。",
        "百战百胜，非善之善者也；不战而屈人之兵，善之善者也。",
        "知己知彼，百战不殆；不知彼而知己，一胜一负；不知彼不知己，每战必殆。"
    ]
    retriever.add_documents(documents)
    print("📚 已添加5篇孙子兵法相关文档")
    
    # 执行检索
    results = retriever.similarity_search("知己知彼", k=3)
    print(f"\n🔍 检索结果（与'知己知彼'相关）:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")
    
    # 使用RAG进行问答
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa_chain = retriever.create_retrieval_qa_chain(llm)
    result = qa_chain({"query": "孙子兵法中关于战争的核心思想是什么？"})
    print(f"\n💡 RAG问答结果:\n{result['result']}")
    
    # 保存向量存储
    retriever.save()
    print("\n✅ 向量存储已保存")


def example_langchain_agent_with_rag():
    """示例4: LangChainAgent结合RAG"""
    from bingfu import LangChainAgent, RAGRetriever
    
    # 创建RAG检索器并添加文档
    rag_retriever = RAGRetriever(vector_store_type="faiss")
    rag_retriever.add_documents([
        "BingFu是一个轻量级多智能体框架，灵感来自中国古代军事思想。",
        "BingFu支持击鼓鸣金控制机制，采用孙子兵法战术引擎。",
        "BingFu v0.6.0新增了LangChain集成，支持RAG检索增强功能。"
    ])
    
    # 创建带有RAG支持的Agent
    agent = LangChainAgent(
        name="情报官",
        role="信息检索专家",
        description="擅长从知识库中检索相关信息",
        use_rag=True,
        rag_retriever=rag_retriever
    )
    
    # 使用RAG查询
    result = agent.run_with_rag("BingFu框架有哪些主要功能？")
    print(f"📖 RAG增强查询结果:\n答案: {result['answer']}")
    print(f"📚 来源文档: {[s[:50]+'...' for s in result['sources']]}")


if __name__ == "__main__":
    print("=" * 60)
    print("BingFu LangChain 集成示例")
    print("=" * 60)
    
    try:
        # example_langchain_agent()  # 需要OpenAI API Key
        # example_langchain_memory()  # 需要OpenAI API Key
        example_rag_retriever()  # 需要OpenAI API Key
        # example_langchain_agent_with_rag()  # 需要OpenAI API Key
    except Exception as e:
        print(f"⚠️ 示例执行失败（可能需要配置OpenAI API Key）: {e}")