r"""检索工具与 RAG（军需库 · 斥候）。

════════════════════════════════════════════════════════════════
 为什么是 BM25 而不是向量检索
════════════════════════════════════════════════════════════════

这是被约束逼出来的选择，写下来是为了让后来者知道可以改：

  · OpenAI 的 embedding key 实测返回 401，不可用
  · 本地嵌入模型（sentence-transformers）会拉进 torch，约 2.5 GB，
    而机器上 C 盘只剩 628 MB —— 装不下
  · DeepSeek 未提供 embedding 端点

BM25 的代价是失去语义泛化（同义词、跨语言），
收益是零依赖、零磁盘、可解释、完全确定 ——
同一个查询永远得到同一批结果，这对**可复现的实验**是硬需求。

★ 在基准这种量级的语料（几十篇抓取的网页）上，
  BM25 与稠密检索的差距远小于"有检索 vs 没检索"的差距。
  先把有无解决了，好坏可以之后再谈。

════════════════════════════════════════════════════════════════
 与 langchain_integration.RAGRetriever 的关系
════════════════════════════════════════════════════════════════

框架里有两套检索，用途不同，不要混淆：

  · 本模块（BM25）—— **默认路径**。零依赖，装好 bingfu 就能用，
    基准实验走的是这条。
  · langchain_integration.RAGRetriever —— **可选的重型路径**。
    需要额外安装 langchain 与 faiss；未安装时它会抛出带安装命令的
    ImportError，而不是静默返回空结果（这一点是对的：
    一个静默返回空列表的检索器，会让调用方以为"资料里没有"）。

所以：**不装 langchain 也能用检索**。
需要语义泛化（同义词、跨语言召回）时再考虑那条路径。

════════════════════════════════════════════════════════════════
 这一层同时解决了另一个问题
════════════════════════════════════════════════════════════════

验收层的 ``check_numbers_traceable`` 曾经被迫禁用，
原因是"基准里将领没有工具，研究类任务的数字只能出自模型知识，
根本没有可溯源的对象"。

现在检索到的原文进入 ``SourceStore``，
它既是 RAG 的语料，也是溯源检查的**来源集合** ——
那条断言因此重新成立。一个工具补上，两处缺口一起闭合。
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import requests

_UA = {"User-Agent": "Mozilla/5.0 (compatible; BingFuResearch/1.0)"}
_TIMEOUT = 15
_MAX_DOC_CHARS = 20000


# ══════════════════════════════════════════════════════════
#  分词：中英混排
# ══════════════════════════════════════════════════════════

_TOKEN = re.compile(r"[a-zA-Z]+|\d+(?:\.\d+)?|[一-鿿]")


def tokenize(text: str) -> List[str]:
    """英文按词、数字按数、中文按字切分。

    中文按**单字**而非分词，是刻意的：分词需要词典依赖，
    而单字切分在 BM25 下对短查询已经够用，且不引入外部包。
    代价是丢失词序信息 —— 对检索召回影响有限。
    """

    return [t.lower() for t in _TOKEN.findall(text or "")]


# ══════════════════════════════════════════════════════════
#  来源库：既是 RAG 语料，也是溯源检查的证据集
# ══════════════════════════════════════════════════════════

@dataclass
class Document:
    doc_id: str
    title: str
    url: str
    text: str
    #: 分块后指向原文的 doc_id；空 = 它本身就是原文
    parent_id: str = ""

    def snippet(self, n: int = 400) -> str:
        return self.text[:n] + ("..." if len(self.text) > n else "")

    @property
    def source_id(self) -> str:
        """无论是原文还是分块，都能问出"它出自哪一篇"。"""

        return self.parent_id or self.doc_id


# ══════════════════════════════════════════════════════════
#  段落切分
# ══════════════════════════════════════════════════════════

#: 一段的目标长度。取 700 是因为它接近一屏中文正文，
#: 既能容下一个完整论点，又不至于把无关内容拖进来。
CHUNK_CHARS = 700
#: 相邻段之间的重叠。答案被切在边界上是分块最典型的失败方式，
#: 留一点重叠比事后补救便宜得多。
CHUNK_OVERLAP = 120
#: 短于此的尾巴并进上一段，不单独成段 —— 碎片会带偏 idf。
MIN_CHUNK = 150

_SENT_END = "。！？；!?;\n"


def _split_sentences(text: str) -> List[str]:
    """按句末标点切句。切不动的超长句按硬长度截 —— 总得切开。"""

    out, buf = [], []
    for ch in text:
        buf.append(ch)
        if ch in _SENT_END:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))

    final: List[str] = []
    for s in out:
        while len(s) > CHUNK_CHARS:
            final.append(s[:CHUNK_CHARS])
            s = s[CHUNK_CHARS:]
        if s:
            final.append(s)
    return final


def split_passages(text: str, *, size: int = CHUNK_CHARS,
                   overlap: int = CHUNK_OVERLAP) -> List[str]:
    """把长文切成段落。短文原样返回单段。"""

    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    passages: List[str] = []
    buf = ""
    for sent in _split_sentences(text):
        if buf and len(buf) + len(sent) > size:
            passages.append(buf)
            # 重叠：把上一段的尾巴带进下一段
            buf = (buf[-overlap:] if overlap else "") + sent
        else:
            buf += sent
    if buf:
        if passages and len(buf) < MIN_CHUNK:
            passages[-1] += buf          # 太短的尾巴并回去
        else:
            passages.append(buf)
    return passages


def chunk_documents(docs: Sequence[Document], *,
                    size: int = CHUNK_CHARS) -> List[Document]:
    """把一批文档展开成段落级文档。

    ★ 短文档原样通过（``parent_id`` 留空）——
      于是"分块"对小语料完全是零影响，行为与从前一致。
    """

    out: List[Document] = []
    for d in docs:
        parts = split_passages(d.text, size=size)
        if len(parts) <= 1:
            out.append(d)
            continue
        for i, part in enumerate(parts, 1):
            out.append(Document(doc_id=f"{d.doc_id}#{i}", title=d.title,
                                url=d.url, text=part, parent_id=d.doc_id))
    return out


class SourceStore:
    """本次任务中抓取到的全部原文。

    ★ 线程安全 —— Graph 层会并行执行节点，
      多个节点可能同时往里写。不加锁的话，
      索引会在某些交错下损坏，而症状是"检索结果偶尔变少"，
      这种间歇性问题极难排查。
    """

    def __init__(self) -> None:
        self._docs: Dict[str, Document] = {}
        self._lock = threading.Lock()

    def add(self, title: str, url: str, text: str) -> Document:
        with self._lock:
            doc_id = f"D{len(self._docs) + 1}"
            doc = Document(doc_id=doc_id, title=title, url=url,
                           text=(text or "")[:_MAX_DOC_CHARS])
            self._docs[doc_id] = doc
            return doc

    def all(self) -> List[Document]:
        with self._lock:
            return list(self._docs.values())

    def texts(self) -> List[str]:
        """给验收层做数字溯源用的来源文本。"""

        return [d.text for d in self.all()]

    def __len__(self) -> int:
        with self._lock:
            return len(self._docs)


# ══════════════════════════════════════════════════════════
#  BM25
# ══════════════════════════════════════════════════════════

@dataclass
class BM25:
    """标准 BM25，纯 Python 实现。

    k1 控制词频饱和，b 控制文档长度归一化。
    取值用的是文献里的常规默认，没有在本项目上调过 ——
    调参需要一个检索质量的标注集，目前没有。
    写出来是为了避免"这些数字是调出来的"这种误解。
    """

    k1: float = 1.5
    b: float = 0.75
    _docs: List[List[str]] = field(default_factory=list)
    _meta: List[Document] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _avg_len: float = 0.0

    def build(self, docs: Sequence[Document], *,
              chunk: bool = True) -> "BM25":
        """建索引。默认按段落分块。

        ★ 为什么默认开。

          整篇进索引时，BM25 的长度归一化（b=0.75）会压低长文的分数：
          一篇几千字、只在某一段命中的正文，可能排在一篇顺带提到
          关键词的短文后面。而"命中在哪一段"恰恰是使用者要的东西。

        ★ 短文档不受影响 —— split_passages 对短文返回单段，
          于是小语料上的行为与从前逐字相同。
        """

        docs = chunk_documents(docs) if chunk else list(docs)
        self._meta = list(docs)
        self._docs = [tokenize(d.text) for d in docs]
        self._df = Counter()
        for toks in self._docs:
            for t in set(toks):
                self._df[t] += 1
        self._avg_len = (sum(len(t) for t in self._docs) / len(self._docs)
                         if self._docs else 0.0)
        return self

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        # 加一平滑，避免高频词得到负 idf
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        if not self._docs:
            return []
        q = tokenize(query)
        scored: List[Tuple[Document, float]] = []
        for idx, toks in enumerate(self._docs):
            tf = Counter(toks)
            dl = len(toks) or 1
            score = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1))
                score += self._idf(term) * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append((self._meta[idx], score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ══════════════════════════════════════════════════════════
#  网络检索
# ══════════════════════════════════════════════════════════

def _extract_text(html: str) -> str:
    """从 HTML 抽正文。"""

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return re.sub(r"<[^>]+>", " ", html)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    return re.sub(r"\s{2,}", " ", soup.get_text(" ", strip=True))


def web_search_raw(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """DuckDuckGo HTML 端点检索，返回标题/摘要/链接。"""

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query}, headers=_UA, timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    out: List[Dict[str, str]] = []
    for node in soup.select("div.result")[:max_results]:
        a = node.select_one("a.result__a")
        sn = node.select_one(".result__snippet")
        if not a:
            continue
        out.append({
            "title": a.get_text(strip=True),
            "url": a.get("href", ""),
            "snippet": sn.get_text(strip=True) if sn else "",
        })
    return out


def fetch_page_raw(url: str) -> Tuple[str, str]:
    """抓取网页，返回 (标题, 正文)。"""

    try:
        resp = requests.get(url, headers=_UA, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        return "", f"[抓取失败] {type(exc).__name__}: {exc}"
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else url
    except ImportError:
        title = url
    return title, _extract_text(resp.text)
