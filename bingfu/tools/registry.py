r"""工具集装配（兵器谱）。

════════════════════════════════════════════════════════════════
 这一层做的一件关键的事
════════════════════════════════════════════════════════════════

工具不只是"让模型能调用外部能力"，它还产生**证据**。

  · 检索工具抓回的原文 → 进入 SourceStore
  · SourceStore 既是 RAG 的语料，也是验收层做数字溯源的来源集

验收层的 ``check_numbers_traceable`` 之前被迫禁用，
理由是"将领没有工具，研究类任务的数字无处可溯"。
现在它重新成立了 —— 但**前提是工具真的被调用过**。

★ 因此这里不能只返回文本给模型看完就丢。
  ``ToolBelt.sources()`` 把证据留存下来交给验收层，
  否则工具用了等于没用：模型看过资料，
  而系统无法证明它引用的数字来自那份资料。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .code_exec import execute_python, run_tests
from .retrieval import BM25, SourceStore, fetch_page_raw, web_search_raw


@dataclass
class ToolBelt:
    """一次任务执行期间可用的工具集合与它产生的证据。

    每条任务用一个新的 ToolBelt —— 证据不能跨任务串。
    上一条任务抓到的资料如果留到下一条，
    溯源检查就会把"来自别的任务的数字"判成有出处。
    """

    store: SourceStore = field(default_factory=SourceStore)
    enable_web: bool = True
    enable_code: bool = True
    #: 跨任务知识库。传入后，抓回的页面会沉淀进去，供后续任务复用。
    #:
    #: ★ 沉淀的是**知识**，不是**溯源证据**。
    #:
    #:   store（本次的 SourceStore）每条任务新建、结束即弃 ——
    #:   否则上一条任务抓到的资料，会让下一条的数字溯源检查
    #:   把"来自别处的数字"判成有出处，验收层就失去了意义。
    #:
    #:   knowledge 则跨任务保留。两者分开，是因为
    #:   "不能串证据"与"不能留知识"是两件不同的事。
    knowledge: Optional[Any] = None
    #: 记录每个工具被调用了几次 —— 用来验证工具真的被用了，
    #: 而不是挂在那里从没被调用（那是最容易发生的失效）
    call_counts: Dict[str, int] = field(default_factory=dict)

    # ── 工具实现 ────────────────────────────────────────

    def _count(self, name: str) -> None:
        self.call_counts[name] = self.call_counts.get(name, 0) + 1

    def web_search(self, query: str) -> str:
        """搜索互联网获取资料。传入检索词，返回若干条标题、摘要与链接。

        用于研究类任务收集事实依据。看到有价值的链接后，
        可以再用 fetch_page 抓取全文。
        """

        self._count("web_search")
        hits = web_search_raw(query, max_results=5)
        if not hits:
            return "[检索无结果] 换一组关键词再试，或直接用 fetch_page 抓取已知链接。"
        lines = []
        for i, h in enumerate(hits, 1):
            # 摘要也存进来 —— 模型常常只看摘要就下结论，
            # 那些数字同样需要可溯源
            self.store.add(h["title"], h["url"], h["snippet"])
            lines.append(f"{i}. {h['title']}\n   {h['snippet']}\n   链接：{h['url']}")
        return "\n".join(lines)

    def fetch_page(self, url: str) -> str:
        """抓取指定网页的正文内容。传入完整链接。

        用于获取搜索结果里某个链接的详细内容。
        """

        self._count("fetch_page")
        title, text = fetch_page_raw(url)
        if not title and text.startswith("[抓取失败]"):
            return text
        doc = self.store.add(title, url, text)
        if self.knowledge is not None:
            # 沉淀原文供后续任务复用；短片段会被知识库自行忽略
            self.knowledge.learn(text, source=url, title=title)
        return f"[{doc.doc_id}] {title}\n{doc.snippet(1500)}"

    def search_knowledge(self, query: str) -> str:
        """在本次已抓取的资料中检索相关段落。

        当资料很多、需要回头查找某个具体信息时使用。
        """

        self._count("search_knowledge")
        docs = self.store.all()

        # ★ 先看跨任务积累里有没有 —— 命中就省下一次抓取。
        #
        #   实测一次调研任务有 9 次搜索、8 次抓取；同领域的连续任务
        #   会把同一批页面重新抓一遍，而工具让检索类的 token 涨 19.6 倍。
        #   复用已读过的内容是这一层最直接的成本收益。
        prior = ""
        if self.knowledge is not None:
            prior = self.knowledge.recall(query, top_k=2)

        if not docs:
            return prior or "[知识库为空] 尚未抓取任何资料，请先用 web_search 或 fetch_page。"

        hits = BM25().build(docs).search(query, top_k=3)
        if not hits:
            miss = f"[本次资料未命中] 已有 {len(docs)} 份，但没有与「{query}」相关的内容。"
            return f"{miss}\n\n{prior}" if prior else miss

        current = "\n\n".join(
            f"[{d.doc_id}] {d.title}（相关度 {s:.2f}）\n{d.snippet(600)}"
            for d, s in hits
        )
        return f"{current}\n\n{prior}" if prior else current

    # ── 装配 ────────────────────────────────────────────

    def as_functions(self) -> Dict[str, Callable]:
        """返回 {工具名: 可调用对象}，供 Agent 注册。"""

        funcs: Dict[str, Callable] = {}
        if self.enable_web:
            funcs["web_search"] = self.web_search
            funcs["fetch_page"] = self.fetch_page
            funcs["search_knowledge"] = self.search_knowledge
        if self.enable_code:
            # ★ 必须包一层计数，不能直接挂模块级函数。
            #
            #   第一版写的是 funcs["execute_python"] = execute_python ——
            #   模块级函数不经过 self._count()，于是**工具真的被调用了，
            #   而 call_counts 始终是空的**。
            #
            #   实测后果：我据此判断"模型不调用工具"，
            #   又去查 schema、查请求体、查 provider ——
            #   全都是好的，问题在我自己的计数器。
            #
            # ★ 这正是"可观测性本身出错"的样子：它不会报错，
            #   只会让你朝错误的方向排查。一个坏掉的仪表
            #   比没有仪表更费时间。
            funcs["execute_python"] = self._counted("execute_python", execute_python)
            funcs["run_tests"] = self._counted("run_tests", run_tests)
        return funcs

    def _counted(self, name: str, fn: Callable) -> Callable:
        """给无状态的工具函数包一层调用计数，同时保留签名与文档。

        签名和 docstring 必须保留 —— Agent 靠内省它们生成
        工具的参数 schema，包装丢了签名，schema 就退化成空参数表。
        """

        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            self._count(name)
            return fn(*args, **kwargs)

        return wrapper

    def sources(self) -> List[str]:
        """交给验收层做数字溯源的证据集。"""

        return self.store.texts()

    def used_any(self) -> bool:
        """工具到底被调用过没有。

        ★ 这个方法存在的理由：挂上工具但模型从不调用，
          是一个**完全静默**的失效 —— 成功率不会变，
          日志不会报错，看起来一切正常，
          而实际上什么都没改变。必须能被观察到。
        """

        return bool(self.call_counts)


#: 按任务类别决定给哪些工具。
#:
#: ★ 依据是任务本身需要什么，不是"多多益善"：
#:   给写作任务一个 Python 解释器不会有帮助，
#:   只会增加它跑偏去写代码的机会，并且每一轮都要
#:   把工具定义塞进上下文，白白消耗 token。
TOOLS_BY_CATEGORY: Dict[str, Dict[str, bool]] = {
    "Code":   {"enable_web": False, "enable_code": True},
    "IR":     {"enable_web": True,  "enable_code": False},
    "Data":   {"enable_web": True,  "enable_code": True},
    "Write":  {"enable_web": True,  "enable_code": False},
    "Reason": {"enable_web": True,  "enable_code": False},
}


def belt_for(category: str, knowledge: Optional[Any] = None) -> ToolBelt:
    """按任务类别装配工具带。

    ★ knowledge 必须能从外面传进来。

      编排层每派一次活就 belt_for() 新建一条工具带（见
      orchestration._configure）。若这里不收，调用方在外面给
      工具带挂的知识库每次都会被丢掉 —— 跨任务知识于是永远为空，
      而 search_knowledge 照常返回本次抓取的结果，看不出少了什么。
      这与工作区工具被重配冲掉是同一个坑。
    """

    cfg = TOOLS_BY_CATEGORY.get(category, {"enable_web": True, "enable_code": True})
    return ToolBelt(knowledge=knowledge, **cfg)
