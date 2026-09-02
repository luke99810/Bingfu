"""
BingFu 一键环境安装脚本
运行: python setup_env.py
"""
import subprocess, sys, os

def run(cmd):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=False)

print("=" * 60)
print("  兵符 BingFu — 环境安装")
print("=" * 60)

# 1. Install packages
print("\n[1/4] 安装依赖包...")
packages = [
    "langchain-huggingface",
    "sentence-transformers",
]
for pkg in packages:
    run(f"{sys.executable} -m pip install {pkg} -q")

# 2. Pre-download embedding model (use HF mirror for China)
print("\n[2/4] 下载本地 embedding 模型...")
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vec = model.encode("test")
    print(f"  模型下载成功! 向量维度: {len(vec)}")
except Exception as e:
    print(f"  下载失败, 可手动运行: python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')\"")
    print(f"  如遇网络问题, 设置环境变量: set HF_ENDPOINT=https://hf-mirror.com")

# 3. Verify core imports
print("\n[3/4] 验证核心模块...")
try:
    from bingfu import BingFu, Agent, LangChainAgent, LangChainMemory, RAGRetriever
    from bingfu.llm import LLMFactory, LLMConfig
    print("  所有核心模块导入成功")
except Exception as e:
    print(f"  导入失败: {e}")

# 4. Test RAG
print("\n[4/4] 测试 RAG 检索...")
try:
    from bingfu.langchain_integration import RAGRetriever
    ret = RAGRetriever(vector_store_type="faiss")
    ret.add_documents(["孙子曰：兵者，国之大事。", "知己知彼，百战不殆。"])
    results = ret.similarity_search("知己知彼", k=1)
    print(f"  RAG 检索成功: {results[0].page_content}")
except Exception as e:
    print(f"  RAG 测试失败: {e}")

print("\n" + "=" * 60)
print("  安装完成! 运行 python launch.py 启动控制台")
print("=" * 60)
