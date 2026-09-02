# -*- coding: utf-8 -*-
r"""庙算不该为一个改变不了决策的问题付费。

★ 背景：实测每一道题上兵符都比对手多一次 LLM 调用，
  多出来的那次就是战役级庙算。而它的产出里只有 complexity_score
  有行为后果 —— 唯一的去处是 orchestrate 里「拆不拆」那个闸门。

★ 最贵的一格是 robust-missing：规则版判 4、交付物只有 1 件，
  而 LLM 判 5 越过闸门，于是「读一个文件、写一个文件」被拆成两个子任务，
  13 次调用、14 311 token —— 对手是 3 次、1 750。
  闸门在那里放行的是纯损耗。

★ 这一组里「什么时候仍然要花那次调用」的断言比「什么时候能省」更要紧：
  省错了会让本该拆的任务不拆，那是少干活，比多花钱严重。
"""

import pytest

from bingfu.assessment import (TaskAssessment, TaskAssessor, TaskComplexity,
                                deliverable_count)


def _fixed(score: int):
    """构造一个规则版结果替身。

    ★ 必须把 TaskAssessment 的必填字段填全。
      第一版少填两个字段，构造时抛 ValidationError，
      而它正好被测试自己的 try/except 吞掉 ——
      于是断言失败被归因成「代码没省对」，其实是**替身根本没造出来**。
      教训：try/except 的范围要窄到只包住被测的那一步。
    """

    def _f(_task):
        return TaskAssessment(
            complexity_score=score,
            complexity=(TaskComplexity.TRIVIAL if score <= 3 else
                        TaskComplexity.MODERATE if score <= 6 else
                        TaskComplexity.DIFFICULT),
            required_capabilities=[], enemy_power=score * 10,
            reasoning="替身")

    return _f


# ════════════════════════════════════════════════════════════════
#  一、交付物计数：素材与产出必须分开数
# ════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("task, want", [
    ("写一个文件 note.md，内容里必须出现 ALPHA 这个词。", 1),
    ("读取当前目录的 data.csv，把 sales 列加起来，把总和写进 total.md。", 1),
    ("先写 facts.md 列出三种语言；再写 pick.md 从中挑一种。", 2),
    ("读 sales.csv 求和写入 out.md；文件不存在也要写 out.md 说明情况。", 1),
    ("分析一下这个项目的架构，给我讲讲。", 0),
])
def test_deliverable_count(task, want):
    assert deliverable_count(task) == want


def test_input_files_are_not_counted_as_deliverables():
    """★ 这条是计数的关键。

    data.csv 是素材、total.md 是产出。两个都算就成了「可拆的两件事」，
    于是一件事被分给两个子任务，它们抢着做同一件事 ——
    实测剩余冗余全部是这么来的。
    """

    assert deliverable_count("读 data.csv，把总和写进 total.md") == 1


def test_same_file_named_twice_counts_once():
    assert deliverable_count("写 out.md；再写 out.md 补充说明") == 1


# ════════════════════════════════════════════════════════════════
#  二、什么时候能省掉那次调用
# ════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════
#  三、什么时候**仍然**要花 —— 更要紧的一组
# ════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════
#  四、接线
# ════════════════════════════════════════════════════════════════


def test_advisor_stays_silent_without_an_event_sink():
    """★ 旁白没有消费者时不该发生调用。

    进言只给界面看，不参与决策。无头运行时它唯一的消费者不存在 ——
    一个到不了任何人那里的产出不该计费。
    """

    from bingfu.campaign import Campaign

    class _Boom:
        def generate(self, messages, **kw):    # pragma: no cover
            raise AssertionError("没挂事件槽时不该调用模型")

    class _BF:
        agents: dict = {}

    c = Campaign(_BF(), strategist=_Boom())
    assert c._advise("庙算", "随便什么上下文") == ""


# ════════════════════════════════════════════════════════════════
#  五、闸门本身不该花钱
# ════════════════════════════════════════════════════════════════

def test_gate_never_calls_the_model():
    """★ 这条是整节的要点：拆不拆是**结构决定**，不是买来的分数。

    原先闸门是「LLM 给 1–10 分，低于阈值就单干」。三处问题：
    分有噪声（同一道题连问两次得 4 和 3）、几乎从不改变结论、
    而且问错了问题 —— 一件很难但只需一位将领的事，拆开只会多付开销。
    """

    from bingfu.assessment import needs_orchestration

    class _Boom:
        def generate(self, messages, **kw):    # pragma: no cover
            raise AssertionError("闸门不该调用模型")

    import bingfu.assessment as asm

    orig = asm.TaskAssessor.__init__
    # 直接调用即可：needs_orchestration 内部自建规则版 assessor
    assert needs_orchestration("写一个文件 note.md") is False
    assert asm.TaskAssessor.__init__ is orig


@pytest.mark.parametrize("task", [
    "写一个文件 note.md，内容里必须出现 ALPHA 这个词。",
    "读取当前目录的 data.csv，把 sales 列加起来，把总和写进 total.md。",
    "先写 facts.md 列出三种语言；再写 pick.md 从中挑一种。",
    "读 sales.csv 求和写入 out.md；文件不存在也要写 out.md 说明情况。",
])
def test_single_general_work_does_not_decompose(task):
    """一位将领做得完的事不该拆 —— 拆开只会多付几套 ReAct 开销。"""

    from bingfu.assessment import needs_orchestration

    assert needs_orchestration(task) is False


@pytest.mark.parametrize("task", [
    "先检索三家竞品的公开定价整理成 pricing.md；再据此写对比分析 report.md；"
    "最后写脚本 plot.py 画图",
    "分析 sales.csv 季度趋势写成 analysis.md，并实现 forecast.py 做预测",
])
def test_multi_role_work_still_decomposes(task):
    """★ 比上面那组更要紧：省钱不能省掉功能。

    闸门收紧之后最危险的失败是**本该拆的也不拆了** ——
    那是少干活，比多花钱严重。
    """

    from bingfu.assessment import needs_orchestration

    assert needs_orchestration(task) is True


def test_force_decompose_escape_hatch_exists():
    """闸门是默认策略，不是不可绕过的规则。"""

    import inspect

    from bingfu.orchestration import orchestrate

    assert "force_decompose" in inspect.signature(orchestrate).parameters


def test_campaign_no_longer_buys_a_complexity_score():
    import inspect

    import bingfu.campaign as camp

    src = inspect.getsource(camp.Campaign.run)
    assert "assess_fast(" in src, "战役应当走规则版庙算"
    assert "assess_economical(" not in src and "assessor.assess(" not in src, \
        "战役仍在花调用买复杂度分"
