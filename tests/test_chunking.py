# -*- coding: utf-8 -*-
r"""段落分块的测试。

════════════════════════════════════════════════════════════════
 分块要解决的到底是什么
════════════════════════════════════════════════════════════════

不是排序 —— 实测长文本来就能排第一。真正的问题在下游：

``registry.search_knowledge`` 把命中文档的 ``snippet(600)`` 喂给模型，
而 ``snippet`` 取的是**开头** 600 字。于是一篇几千字的正文，
答案埋在第 1273 个字符处时：

    检索命中了它（排第一），而模型看到的是开头那段无关的填充文字。

**检索"成功"了，却什么有用的都没给出去。** 这类失效不报错、
指标也好看（命中率 100%），只是答不对题。
"""

import pytest

from bingfu.tools.retrieval import (
    CHUNK_CHARS, BM25, Document, chunk_documents, split_passages,
)


def _long_doc():
    filler = "兵符框架采用军事叙事承载多智能体概念，界面用中军帐的形式呈现。" * 40
    answer = ("庙算推导敌方战力：复杂度乘十，加能力数乘十，"
              "再乘难度系数，钳制在 10 到 200 之间。")
    return Document("D1", "兵符总览", "u1", filler + answer + filler), answer


# ════════════════════════════════════════════════════════════════
#  一、真正要解决的那个问题
# ════════════════════════════════════════════════════════════════

def test_buried_answer_reaches_the_model():
    """★ 整组里最要紧的一条。

    不分块时这个断言会红：文档排第一，但 snippet(600) 里没有答案。
    """

    doc, _ = _long_doc()
    hit, _score = BM25().build([doc]).search("敌方战力的算式，钳制范围", top_k=1)[0]
    assert "10 到 200" in hit.snippet(600), (
        "命中了文档，但喂给模型的那 600 字里没有答案 —— "
        "检索成功而内容没送到"
    )


def test_without_chunking_the_answer_does_not_reach_the_model():
    """把上一条的反面固定下来：这正是分块前的行为。

    ★ 写这条不是为了测旧代码，是为了让"分块有没有用"这件事
      有一个会变红的凭据。哪天有人把 chunk 默认关掉，
      上一条会红，而这一条会提醒他红的原因是什么。
    """

    doc, _ = _long_doc()
    hit, _ = BM25().build([doc], chunk=False).search(
        "敌方战力的算式，钳制范围", top_k=1)[0]
    assert hit.doc_id == "D1"
    assert "10 到 200" not in hit.snippet(600)


def test_hit_can_be_traced_back_to_its_source_document():
    """段落必须能问出"它出自哪一篇" —— 否则溯源就断了。"""

    doc, _ = _long_doc()
    hit, _ = BM25().build([doc]).search("钳制范围", top_k=1)[0]
    assert "#" in hit.doc_id
    assert hit.source_id == "D1"
    assert hit.title == "兵符总览" and hit.url == "u1"


# ════════════════════════════════════════════════════════════════
#  二、短文档必须完全不受影响
# ════════════════════════════════════════════════════════════════

def test_short_documents_pass_through_untouched():
    """★ 小语料上的行为要与从前逐字相同。

    否则这次改动就不是"加了个能力"，而是"顺手改了所有人的结果"。
    """

    docs = [Document("D1", "甲", "u1", "短文一段，讲的是点将。"),
            Document("D2", "乙", "u2", "短文另一段，讲的是庙算。")]
    out = chunk_documents(docs)
    assert out == docs, "短文档被改动了"
    assert all(d.parent_id == "" for d in out)


def test_short_text_is_a_single_passage():
    assert split_passages("很短的一句话。") == ["很短的一句话。"]
    assert split_passages("") == []
    assert split_passages("   ") == []


# ════════════════════════════════════════════════════════════════
#  三、切分本身
# ════════════════════════════════════════════════════════════════

def test_passages_respect_the_size_budget():
    text = "这是一个用于测试的句子，长度适中。" * 200
    parts = split_passages(text, size=400)
    assert len(parts) > 1
    # 允许重叠带来的少量超出，但不能失控
    assert max(len(p) for p in parts) <= 400 + 120 + 40


def test_nothing_is_lost_in_the_middle():
    """★ 切分不能丢内容。

    丢了不会报错 —— 只会表现为"某些资料检索不到"，
    而那与"资料里本来就没有"分不清。
    """

    marker = "唯一标记词ZZQ"
    text = ("填充句子。" * 300) + marker + ("填充句子。" * 300)
    parts = split_passages(text)
    assert any(marker in p for p in parts), "切分把中间的内容弄丢了"


def test_overlap_protects_the_boundary():
    """答案被切在边界上是分块最典型的失败方式，靠重叠兜住。"""

    parts = split_passages("句子甲。" * 200, size=300, overlap=100)
    assert len(parts) > 2
    # 相邻两段应当有公共的尾/头
    assert parts[0][-50:] in parts[1], "相邻段落之间没有重叠"


def test_a_single_overlong_sentence_still_gets_split():
    """一句话就超长时也必须切开 —— 切不动不能变成不切。"""

    parts = split_passages("啊" * (CHUNK_CHARS * 3))
    assert len(parts) >= 3
    assert all(len(p) <= CHUNK_CHARS + 200 for p in parts)


def test_tiny_tail_is_merged_not_left_alone():
    """碎片段会带偏 idf，尾巴太短就并回上一段。"""

    text = "正常长度的句子内容在这里。" * 60 + "尾。"
    parts = split_passages(text, size=300)
    assert all(len(p) >= 100 for p in parts), \
        "留下了过短的碎片：%r" % [len(p) for p in parts]


# ════════════════════════════════════════════════════════════════
#  四、知识库不能被 doc_id 的新格式打穿
# ════════════════════════════════════════════════════════════════

def test_knowledge_base_survives_chunked_doc_ids():
    """★ semantic.py 原来用 int(doc.doc_id[1:]) 反解下标。

    分块后 doc_id 变成 "K3#2"，那行会直接 ValueError ——
    而它离检索层很远，报错时看不出是分块引起的。
    """

    from bingfu.memory import KnowledgeBase

    kb = KnowledgeBase()
    long_text = ("兵符的庙算阶段用于评估任务难度。" * 60
                 + "敌方战力最终钳制在 10 到 200 之间，这是硬上限。")
    kb.learn(long_text, source="doc://庙算", title="庙算")

    hits = kb.search("钳制在多少之间", top_k=2)
    assert hits, "分块后知识库检索不到了"
    fact, score = hits[0]
    assert "10 到 200" in fact.text
    assert fact.hits == 1, "命中计数没有落到正确的知识条目上"


def test_one_fact_counted_once_even_if_several_passages_match():
    """同一条知识的多个段落都命中时，只应算作一条结果。"""

    from bingfu.memory import KnowledgeBase

    kb = KnowledgeBase()
    kb.learn("点将阶段为全体将领打分。" * 80, source="s", title="点将")
    hits = kb.search("点将 打分", top_k=5)
    assert len(hits) == 1, "同一条知识被重复返回了 %d 次" % len(hits)
