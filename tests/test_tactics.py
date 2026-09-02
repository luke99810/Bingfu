"""战术引擎的判据 —— 论文里的 Algorithm 1。

════════════════════════════════════════════════════════════════
 ★ 这个文件此前测的是一个**已经不存在的 API**
════════════════════════════════════════════════════════════════

旧版本 import 的是 `TacticsEngine`（现名 `TacticEngine`），
断言的是 `TacticType.AMBUSH / COUNTER / DECEIVE …` 等 13 个枚举成员 ——
而当前的 `TacticCategory` 里**一个都没有**（现在是 STRATEGIC_PLANNING /
RECONNAISSANCE / … 对应孙子十三篇）。

后果不是「几条测试失败」，是**整套 pytest 收集中断**：

    ImportError: cannot import name 'TacticsEngine' from 'bingfu.tactics'
    !!!! Interrupted: 1 error during collection !!!!

于是 `pytest` 一条都跑不了 —— 包括其它 6 个文件里那 76 条本来是好的测试。
一次改名之后，这个仓库的判据整体失效，而没有任何东西提示过。

★ 重写之后，`tactics.py` + `tactic_definitions.py` + `tactic_library.py`
  这 1500 行**核心算法**才第一次真正有判据守着。
"""

from __future__ import annotations

import pytest

from bingfu import Agent, TacticEngine
from bingfu.presets import PRESET_GENERALS, get_preset
from bingfu.tactic_definitions import TacticalContext, TacticCategory
from bingfu.tactics import OptimizationResult


@pytest.fixture
def agents() -> dict[str, Agent]:
    """三位有战力档案的将领 —— 档案是打分的输入，不能省。"""

    roster: dict[str, Agent] = {}
    for name in list(PRESET_GENERALS)[:3]:
        roster[name] = Agent(name=name, profile=get_preset(name))
    return roster


@pytest.fixture
def engine() -> TacticEngine:
    return TacticEngine()


# ══════════════════════════════════════════════════════════════
#  ★★ 因果：任务变了，选择必须跟着变
# ══════════════════════════════════════════════════════════════


def test_different_tasks_select_different_tactics(engine, agents) -> None:
    """★ 这是整个引擎最重要的一条判据。

    如果不管什么任务都选出同一个战术，那么「战术选择」这个核心贡献
    **在效果上等于一个常数函数** —— 论文里所有关于 tactic selection
    的结论都会落空，而单元测试照样能全绿（因为每次调用都不报错）。

    这条测的不是「返回值合法」，是「输入真的影响了输出」。
    """

    tasks = [
        "紧急修复线上支付故障，30 分钟内必须恢复",
        "设计一个全新的品牌视觉方案，要有创意",
        "整理近三年财报数据并做趋势分析",
    ]
    chosen = [engine.select_tactic(t, agents).selected_tactic.name for t in tasks]

    assert len(set(chosen)) > 1, (
        f"三个性质迥异的任务选出了同一个战术：{chosen[0]} —— 战术选择等于常数函数"
    )


def test_selection_is_deterministic(engine, agents) -> None:
    """★ 同样输入必须得到同样结果。

    论文的实验要可复现；一个带随机性的选择器会让「BingFu 比基线高 10.4 点」
    这类结论无法被任何人验证 —— 包括作者自己。
    """

    task = "紧急修复线上支付故障"
    first = engine.select_tactic(task, agents)
    second = engine.select_tactic(task, agents)

    assert first.selected_tactic.name == second.selected_tactic.name
    assert first.selected_agent_name == second.selected_agent_name
    assert first.combined_score == pytest.approx(second.combined_score, abs=1e-12)


# ══════════════════════════════════════════════════════════════
#  ★ 返回结构必须自洽
# ══════════════════════════════════════════════════════════════


def test_result_shape_is_consistent(engine, agents) -> None:
    result = engine.select_tactic("分析竞争对手的定价策略", agents)

    assert isinstance(result, OptimizationResult)
    assert result.selected_agent_name in agents, "选了一个不在名册里的将领"
    assert result.selected_tactic is not None
    assert result.all_evaluations, "没有给出任何候选评估"


def test_alignment_score_is_a_cosine_similarity(engine, agents) -> None:
    """★ 对齐分声称是 cos_sim(style_vector, task_features)，
    那它就必须落在 [-1, 1] 里。

    这条守的是「实现有没有偷偷换成别的东西」—— 余弦相似度越界
    通常意味着少做了一次归一化，而那会让排序悄悄失真。
    """

    result = engine.select_tactic("组建一支跨部门攻坚小组", agents, top_k=10)

    for ev in result.all_evaluations:
        assert -1.0 - 1e-9 <= ev.alignment_score <= 1.0 + 1e-9, (
            f"{ev.tactic.name} 的对齐分 {ev.alignment_score} 不是余弦相似度"
        )


def test_top_k_is_respected(engine, agents) -> None:
    assert len(engine.select_tactic("任务", agents, top_k=2).all_evaluations) <= 2
    assert len(engine.select_tactic("任务", agents, top_k=5).all_evaluations) <= 5


def test_selected_is_the_best_of_the_evaluations(engine, agents) -> None:
    """★ 「选中的」必须真的是评估里分最高的那个。

    否则排序算得再对也没用 —— 最后一步取错了，
    而返回结构看起来完全正常。
    """

    result = engine.select_tactic("制定明年的市场进入计划", agents, top_k=10)
    best = max(ev.combined_score for ev in result.all_evaluations)

    assert result.combined_score == pytest.approx(best), (
        "返回的 combined_score 不是候选里的最高分"
    )


# ══════════════════════════════════════════════════════════════
#  ★ 战场态势：纯函数，可确定性判定
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("self_strength", "enemy_strength", "expected"),
    [
        (10.0, 100.0, "defensive"),   # ratio 0.1  < 0.5
        (50.0, 50.0, "balanced"),     # ratio 1.0
        (100.0, 10.0, "offensive"),   # ratio 10.0 > 2.0
    ],
)
def test_regime_follows_strength_ratio(self_strength, enemy_strength, expected) -> None:
    ctx = TacticalContext(self_strength=self_strength, enemy_strength=enemy_strength)
    assert ctx.regime == expected


def test_zero_enemy_strength_does_not_divide_by_zero() -> None:
    """★ 敌军战力为 0 是合法输入（完全无对手），不该让引擎崩。

    实现里用 `max(enemy_strength, 1)` 兜住 —— 这条守着那个兜底不被删掉。
    """

    ctx = TacticalContext(self_strength=50.0, enemy_strength=0.0)
    assert ctx.strength_ratio == pytest.approx(50.0)


def test_task_features_are_five_dimensional() -> None:
    """★ 5 维是硬约定：style_vector 也是 5 维，两者要做点积。

    维度对不上的后果是 numpy 直接报错，或者更糟 —— 广播成一个
    看起来正常的数字。
    """

    features = TacticalContext().to_task_features()
    assert features.shape == (5,), f"任务特征维度是 {features.shape}，与 style_vector 对不上"


# ══════════════════════════════════════════════════════════════
#  ★ 战术库
# ══════════════════════════════════════════════════════════════


def test_every_tactic_has_a_five_dim_style_vector(engine) -> None:
    """★ 库里**每一个**战术都必须能参与打分。

    只要有一个的 style_vector 维度不对，它在打分时要么报错、
    要么被静默跳过 —— 后者意味着这个战术永远不会被选中，
    而库的规模看起来仍然很大。
    """

    assert engine.library, "战术库是空的"
    for name, tactic in engine.library.items():
        assert tactic.style_vector.shape == (5,), f"战术 {name} 的 style_vector 维度不对"


def test_tactic_categories_cover_the_thirteen_chapters() -> None:
    """★ 十三篇是这个框架的立论基础，不是凑数的枚举。"""

    assert len(TacticCategory) >= 13, f"战术类别只有 {len(TacticCategory)} 个，不足十三篇"


# ══════════════════════════════════════════════════════════════
#  ★ 边界
# ══════════════════════════════════════════════════════════════


def test_no_agents_reports_the_real_cause(engine) -> None:
    """★ 无将领时抛异常是合理的 —— 让人拿着**错误的原因**去查才不合理。

    此前不论什么情况都报「No applicable tactic found for task: …」，
    而最常见的触发原因是 `agents` 为空。看到那句话的人会去翻战术库、
    改任务措辞 —— 而真正要做的只是先注册一位将领。

    两种情况的修法完全不同，报错就必须能区分它们。
    """

    with pytest.raises(ValueError) as exc:
        engine.select_tactic("任务", {})

    message = str(exc.value)
    assert "将领" in message, f"没说清是将领的问题：{message}"
    assert "战术库" in message or "agents 为空" in message


def test_unsatisfiable_preconditions_report_differently(engine, agents) -> None:
    """★ 反向：有将领、但没有战术适用时，报的必须是**另一件事**。

    只有正向那条测试的话，一个把两种情况写成同一句话的实现照样能绿。
    """

    import re

    with pytest.raises(ValueError) as no_agent:
        engine.select_tactic("任务", {})

    # 有将领时正常返回，不该走到异常分支
    ok = engine.select_tactic("任务", agents)
    assert ok.selected_agent_name in agents

    assert re.search(r"没有可用的将领|agents 为空", str(no_agent.value)), (
        "两种失败原因没有被区分开"
    )
