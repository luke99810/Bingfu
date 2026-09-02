r"""舆图 —— 语义记忆（Semantic Memory）。

════════════════════════════════════════════════════════════════
 这一层补的是什么
════════════════════════════════════════════════════════════════

工具层抓回的资料存在 ``ToolBelt.store`` 里，而那个 store
**每条任务新建一个、任务结束即丢弃**。

这在当时是刻意的：证据不能跨任务串，否则上一条任务抓到的资料
会让下一条任务的数字溯源检查把"来自别处的数字"判成有出处。

★ 但"不能串证据"与"不能留知识"是两件事。

  同一次会话里连着做三个关于同一领域的任务，第二、三次
  会把第一次抓过的页面**重新抓一遍** —— 实测一次调研任务
  就有 9 次搜索、8 次抓取，成本是 token 的 19.6 倍。
  全部重来是纯浪费。

因此这里的设计是：**知识可以跨任务复用，但溯源证据不可以。**

  · KnowledgeBase 持久化"读到过什么"，供检索复用
  · 每条任务的 ToolBelt 仍持有自己的 SourceStore 做溯源
  · 从知识库取回的内容会**标注来源与轮次**，让调用方知道
    这不是本次新抓的

════════════════════════════════════════════════════════════════
 为什么仍然用 BM25
════════════════════════════════════════════════════════════════

与 tools/retrieval 同因：向量检索需要嵌入模型，
本机既无可用的 embedding key，也装不下本地模型（约 2.5 GB，
磁盘只剩几百 MB）。BM25 零依赖、零磁盘、完全确定 ——
同一查询永远得到同一批结果，这对可复现的实验是硬需求。
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Fact:
    """一条被记住的知识。"""

    text: str
    source: str = ""
    title: str = ""
    #: 第几次任务时学到的 —— 用于判断新鲜度
    learned_at: int = 0
    #: 被检索命中过几次。长期为 0 的可以先淘汰
    hits: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class KnowledgeBase:
    """跨任务的知识库：存事实，按 BM25 检索。"""

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        max_facts: int = 1000,
        max_chars_per_fact: int = 4000,
    ) -> None:
        self.path = Path(path) if path else None
        self.max_facts = max_facts
        self.max_chars_per_fact = max_chars_per_fact
        self._facts: List[Fact] = []
        self._lock = threading.Lock()
        self._epoch = 0
        if self.path and self.path.exists():
            self._load()

    # ── 写入 ────────────────────────────────────────────

    def learn(self, text: str, *, source: str = "", title: str = "") -> Optional[Fact]:
        """记住一条知识。重复内容不重复记。"""

        text = (text or "").strip()[: self.max_chars_per_fact]
        if len(text) < 40:
            # ★ 太短的片段没有检索价值，只会稀释语料。
            #   不是"存了也没坏处" —— BM25 的 idf 会被大量短文档带偏。
            return None

        with self._lock:
            key = text[:200]
            for f in self._facts:
                if f.text[:200] == key:
                    return f                        # 已经记过
            fact = Fact(text=text, source=source, title=title,
                        learned_at=self._epoch)
            self._facts.append(fact)
            if len(self._facts) > self.max_facts:
                self._evict()
        self._save()
        return fact

    def _evict(self) -> None:
        """淘汰：先丢从未被检索命中、且最早学到的。

        ★ 不按纯 FIFO 淘汰 —— 一条被反复用到的旧知识，
          比一条刚学到却从没用过的更值得留。
        """

        self._facts.sort(key=lambda f: (f.hits, f.learned_at))
        drop = len(self._facts) - self.max_facts
        self._facts = self._facts[drop:]

    def next_epoch(self) -> int:
        with self._lock:
            self._epoch += 1
            return self._epoch

    # ── 检索 ────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Fact, float]]:
        from bingfu.tools.retrieval import BM25, Document

        with self._lock:
            facts = list(self._facts)
        if not facts:
            return []

        docs = [
            Document(doc_id=f"K{i}", title=f.title or f.source or "知识",
                     url=f.source, text=f.text)
            for i, f in enumerate(facts)
        ]
        # ★ 用映射表回查，不要从 doc_id 里反解下标。
        #
        #   原来是 int(doc.doc_id[1:])，依赖 doc_id 恒为 "K<下标>"。
        #   检索层一旦分块，doc_id 会变成 "K3#2"，那行直接 ValueError ——
        #   而它离检索层很远，报错时根本看不出是分块引起的。
        by_id = {f"K{i}": f for i, f in enumerate(facts)}
        hits = BM25().build(docs).search(query, top_k=top_k)
        out: List[Tuple[Fact, float]] = []
        seen = set()
        for doc, score in hits:
            fact = by_id.get(doc.source_id)
            if fact is None or id(fact) in seen:
                continue        # 同一条知识的多个段落只算一次
            seen.add(id(fact))
            fact.hits += 1
            out.append((fact, score))
        return out

    def recall(self, query: str, top_k: int = 3) -> str:
        """检索并拼成可放进提示词的文字。

        ★ 必须标注"这是过去学到的"。

          不标注的话，模型会把旧知识与本次新抓的资料混同，
          而数字溯源检查只认本次抓回的原文 —— 于是一个
          引用了旧知识的正确结论，会被判成"数字无出处"。
        """

        hits = self.search(query, top_k=top_k)
        if not hits:
            return ""
        lines = ["【过往积累的资料，非本次抓取，引用时请注明】"]
        for fact, score in hits:
            head = fact.title or fact.source or "未命名"
            lines.append(f"· {head}（相关度 {score:.2f}）\n  {fact.text[:500]}")
        return "\n".join(lines)

    def __len__(self) -> int:
        with self._lock:
            return len(self._facts)

    # ── 持久化 ──────────────────────────────────────────

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:                          # noqa: BLE001
            self._facts = []
            return
        self._facts = [Fact.from_dict(d) for d in raw if isinstance(d, dict)]
        self._epoch = max((f.learned_at for f in self._facts), default=0)

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = [f.to_dict() for f in self._facts]
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:                          # noqa: BLE001
            pass
