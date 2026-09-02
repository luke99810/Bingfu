"""工具层（兵器谱）—— 让将领真的能做事，而不只是生成文本。

基准里 Code 类只有 12–25%，直接原因是 tools=None 让 ReAct 循环
恒定只跑一轮。这个包补上的是那个缺口。
"""

from .code_exec import ExecResult, execute_python, run_python, run_tests
from .registry import TOOLS_BY_CATEGORY, ToolBelt, belt_for
from .retrieval import BM25, Document, SourceStore, fetch_page_raw, tokenize, web_search_raw
from .workspace import Workspace, workspace_at

__all__ = [
    "ExecResult", "execute_python", "run_python", "run_tests",
    "ToolBelt", "belt_for", "TOOLS_BY_CATEGORY",
    "BM25", "Document", "SourceStore", "tokenize",
    "fetch_page_raw", "web_search_raw",
    "Workspace", "workspace_at",
]
