# -*- coding: utf-8 -*-
r"""记忆接线的回归测试。

════════════════════════════════════════════════════════════════
 这组测试针对的是同一类失效
════════════════════════════════════════════════════════════════

分层记忆的代码一直是齐的：``agent.py`` 里回放与记录的分支都在，
``memory/`` 下三层共 545 行也都实现了。缺的只是**没人把库建出来传进去** ——
``Agent.episodic`` 与 ``Agent.knowledge`` 恒为 ``None``。

这种失效不抛异常、不打日志，界面照常工作，只是将领打完一仗什么都不留下。
所以下面每条测试都刻意构造成：**接线断掉时它必须变红**。
"""

import io
import os

import pytest

from bingfu.memory import Episode, EpisodicMemory, KnowledgeBase, summarize
from bingfu.tools import belt_for


# ════════════════════════════════════════════════════════════════
#  一、回放的挑选策略
# ════════════════════════════════════════════════════════════════

def test_failures_are_replayed_even_when_buried_under_successes():
    """★ 失败优先，且不能被后来的成功挤出窗口。

    构造成「一条失败在最前，后面压六条成功」——
    旧实现取 ``recent(5)``，窗口里全是成功，那条失败**看不见了**。
    而失败恰恰是历史里唯一能改变下次选择的东西。
    """

    mem = EpisodicMemory()
    mem.record(Episode(task="旧任务", agent_name="白起", category="Code",
                       success=False, stopped_by="回炉轮次用尽"))
    for i in range(6):
        mem.record(Episode(task=f"任务{i}", agent_name="韩信",
                           category="Code", success=True))

    text = summarize(mem, category="Code")
    assert "白起" in text, "失败的战报被后来的成功挤出了回放窗口"
    assert "回炉轮次用尽" in text


def test_successes_still_appear_when_there_is_no_failure():
    """失败优先不等于只放失败 —— 没有失败时窗口该由成功填满。"""

    mem = EpisodicMemory()
    for i in range(3):
        mem.record(Episode(task=f"任务{i}", agent_name="韩信",
                           category="Code", success=True))
    text = summarize(mem, category="Code")
    assert text.count("韩信") == 3


def test_tool_sequence_is_replayed():
    """★ 工具序列比结论有用，而 Episode 一开始就记了 tool_calls。

    记了却不回放，等于没记。
    """

    mem = EpisodicMemory()
    mem.record(Episode(task="改代码", agent_name="猛将", category="Code",
                       success=True,
                       tool_calls={"read_file": 2, "write_file": 1}))
    text = summarize(mem, category="Code")
    assert "read_file×2" in text, "工具序列没有进入回放"
    assert "write_file×1" in text


def test_replay_has_a_length_cap_and_says_so_when_truncated():
    """★ 这段文字每一轮都进上下文，必须有上限；截断了要说出来。

    无上限的话，跑得越久提示词越长，而变长的过程是无声的。
    """

    mem = EpisodicMemory()
    for i in range(5):
        mem.record(Episode(task="x", agent_name="将领" + "长" * 60,
                           category="Code", success=False,
                           stopped_by="原因" + "很长" * 40,
                           tool_calls={"read_file": 3}))
    text = summarize(mem, category="Code", max_chars=300)
    assert len(text) < 400, "回放没有被截断"
    assert "已截断" in text, "截断了却没说"


def test_no_history_yields_empty_string_not_a_placeholder():
    """没有历史时返回空串 —— 不要塞一句「暂无历史」进系统提示。"""

    assert summarize(EpisodicMemory(), category="Code") == ""


# ════════════════════════════════════════════════════════════════
#  二、知识库能不能活过「按类别重配工具带」
# ════════════════════════════════════════════════════════════════

def test_belt_for_accepts_knowledge():
    kb = KnowledgeBase()
    belt = belt_for("Data", knowledge=kb)
    assert belt.knowledge is kb


def test_knowledge_survives_reconfiguration_by_the_orchestrator():
    """★ 这条是这组里最要紧的一个。

    编排层每派一次活就 ``belt_for()`` 新建一条工具带。若 knowledge
    不跟着传下去，跨任务知识**永远是空的**，而 ``search_knowledge``
    照常返回本次抓取的结果 —— 从外面完全看不出少了一半。
    """

    from bingfu.agent import Agent
    from bingfu.orchestration import SubTask, _configure

    kb = KnowledgeBase()
    # ★ 正文必须 ≥ 40 字：learn() 会静默丢弃更短的片段，
    #   因为大量短文档会把 BM25 的 idf 带偏。第一版测试写了 28 字，
    #   于是知识库是空的，测试红了 —— 红的是测试数据，不是被测代码。
    kb.learn("兵符的庙算阶段推导敌方战力：复杂度乘十，加上所需能力数乘十，"
             "再乘难度系数，最后钳制在 10 到 200 之间。难度系数按复杂度分三档。",
             source="doc://庙算", title="庙算")

    agent = Agent(name="斥候", role="侦察", knowledge=kb)
    configured = _configure(agent, SubTask(id="s1", description="查战力口径",
                                           category="Data"))

    fn = configured._tool_functions.get("search_knowledge")
    assert fn is not None, "Data 类应当配发 search_knowledge"

    out = fn("敌方战力怎么算")
    assert "过往积累" in out, "重配后知识库丢失了（belt_for 没收到 knowledge）"
    assert "10 到 200" in out


def test_knowledge_is_labelled_as_prior_not_as_this_run():
    """跨任务知识必须标注来源，否则数字溯源会把它当成本次抓取的原文。"""

    kb = KnowledgeBase()
    kb.learn("这是很久以前学到的一段资料，专门用于验证跨任务知识在回放时"
             "是否带上了来源标注，以免被数字溯源检查误判为本次抓取的原文。",
             source="doc://old", title="旧资料")
    belt = belt_for("Data", knowledge=kb)
    out = belt.search_knowledge("很久以前")
    assert "非本次抓取" in out


# ════════════════════════════════════════════════════════════════
#  三、launch.py 的接线
# ════════════════════════════════════════════════════════════════

def test_launch_injects_memory_into_every_general():
    """★ 结构性检查，作用有限，但不是空检查。

      它拦不住「传了但传错库」，能拦住的是「有人把这两行删了」——
      而那正是这个缺陷当初的形态：字段在、分支在、就是没人传。

      真正的验证是跑一次战役看 episodes.json 有没有长出来，
      那个不适合放在单元测试里（要真调 LLM）。
    """

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = io.open(os.path.join(here, "launch.py"), encoding="utf-8").read()

    assert "EpisodicMemory(" in src, "launch.py 没有建战报库"
    assert "KnowledgeBase(" in src, "launch.py 没有建知识库"
    assert "episodic=episodic" in src, "Agent 构造时没有注入战报库"
    assert "knowledge=knowledge" in src, "Agent 构造时没有注入知识库"


def test_memory_files_land_under_the_workspace(tmp_path):
    """记忆按工作区隔离：换个工作区就是换一段历史。"""

    d = tmp_path / ".bingfu"
    d.mkdir()
    mem = EpisodicMemory(path=str(d / "episodes.json"))
    mem.record(Episode(task="t", agent_name="韩信", category="Code"))

    assert (d / "episodes.json").exists()

    reloaded = EpisodicMemory(path=str(d / "episodes.json"))
    assert len(reloaded) == 1, "落盘了却读不回来"
    assert reloaded.all()[0].agent_name == "韩信"
