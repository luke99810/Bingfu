"""战术选择机制的测试。

这组测试来自一次根因调查：为什么战术注入"对代码类有害"？

★ 结论是**它并不有害** —— 那个"−26 点"是 3/12 对 4/12，
  差一个任务，Fisher 精确检验 p = 1.000。

  但调查途中发现了三个真实缺陷，比那个不存在的负作用严重得多：

    ① power 归一化差 10 倍，把任务信号压掉 97%
    ② 所谓"(战术,将领) 联合优化"是可加可分的，联合并未发生
    ③ 全非负向量上的余弦相似度没有区分力 —— 20 条任务里 19 条选中同一战术

  这里逐条锁住。
"""

import numpy as np
import pytest

from bingfu.agent import Agent
from bingfu.presets import PRESET_GENERALS
from bingfu.tactic_library import get_tactic_library
from bingfu.tactics import TacticEngine


@pytest.fixture(scope="module")
def agents():
    return {n: Agent(name=n, role="将军", profile=p) for n, p in PRESET_GENERALS.items()}


@pytest.fixture(scope="module")
def library():
    return get_tactic_library()


#: 一条代码任务的实测特征（来自 LLM 评估路径）
CODE_FEATURES = np.array([0.60, 0.40, 0.15, 0.30, 0.54])


# ══════════════════════════════════════════════════════════
#  ① power 归一化
# ══════════════════════════════════════════════════════════

def test_power_score_is_actually_normalised(agents):
    """★ 曾经 ``agent_power / 10.0``，而属性定义域是 1–100。

    得到的是 0.1–10，不是文档声称的 0–1。
    后果：power 项贡献约 3.88，把跨度仅 0.28 的 alignment 压掉 97%，
    合计分变成 4.2–4.3 这种没有意义的数。
    """

    eng = TacticEngine(assessor=None)
    power = eng._compute_power_score(
        np.array([80, 65, 95, 85, 90], dtype=np.float64), CODE_FEATURES
    )
    assert 0.0 <= power <= 1.0, f"power={power}，未落在 [0,1]"


def test_combined_score_stays_in_unit_range(agents):
    """合计分是三个 [0,1] 项的加权和，本身也该在 [0,1]。"""

    eng = TacticEngine(assessor=None)
    r = eng.select_tactic("写一个 REST API 服务", agents, top_k=5)
    for e in r.all_evaluations:
        if e.is_applicable:
            assert 0.0 <= e.combined_score <= 1.0


# ══════════════════════════════════════════════════════════
#  ② 可加可分：「联合优化」并未联合
# ══════════════════════════════════════════════════════════

def test_scoring_is_additively_separable(agents):
    """★ Q(t,a) = w₁·align(t) + w₂·power(a) + w₃·history(a)

    align 只依赖战术、power/history 只依赖将领 ——
    于是 182 个组合上的 argmax 等价于两个独立 argmax。

    这条测试断言的是**当前实现的事实**，不是期望的设计。
    如果有人加入了交互项（例如战术风格与将领风格的匹配度），
    这里会红 —— 那时应当更新这条测试，而不是移除它，
    因为它记录着"联合优化"从声称到兑现的那一刻。
    """

    eng = TacticEngine(assessor=None)
    r = eng.select_tactic("分析销售数据并给出建议", agents, top_k=200)
    ev = [e for e in r.all_evaluations if e.is_applicable]

    # 同一战术在不同将领下，alignment 必须完全相同
    by_tactic = {}
    for e in ev:
        by_tactic.setdefault(e.tactic.name, set()).add(round(e.alignment_score, 9))
    for name, vals in by_tactic.items():
        assert len(vals) == 1, f"战术 {name} 的 alignment 随将领变化了"

    # 同一将领在不同战术下，power 必须完全相同
    by_agent = {}
    for e in ev:
        by_agent.setdefault(e.agent_name, set()).add(round(e.power_score, 9))
    for name, vals in by_agent.items():
        assert len(vals) == 1, f"将领 {name} 的 power 随战术变化了"


# ══════════════════════════════════════════════════════════
#  ③ 余弦在全非负向量上没有区分力
# ══════════════════════════════════════════════════════════

def test_style_vectors_are_all_non_negative(library):
    """这是余弦退化的前提条件，先把它钉住。"""

    V = np.array([t.style_vector for t in library.values()])
    assert (V >= 0).all()


def test_raw_cosine_barely_discriminates(library):
    """★ 全非负向量的余弦天然聚集在高位。

    实测：26 个战术对一条代码任务的对齐分全部 > 0.6，
    最高与最低相差不到 0.3 —— 这不是"择优"，是几乎无差别。
    """

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    scores = np.array([cos(t.style_vector, CODE_FEATURES) for t in library.values()])
    assert (scores > 0.6).all(), "前提变了：不再是全部高分"
    assert scores.max() - scores.min() < 0.4, "区分度意外变好了，请复核这条测试"


def test_centering_improves_discrimination(library):
    """中心化把跨度拉开一个数量级 —— 改善是真的。"""

    eng = TacticEngine(assessor=None, center_alignment=True)
    eng.library = library
    centered = np.array([eng._alignment(t, CODE_FEATURES) for t in library.values()])

    eng_raw = TacticEngine(assessor=None)
    eng_raw.library = library
    raw = np.array([eng_raw._alignment(t, CODE_FEATURES) for t in library.values()])

    assert (centered.max() - centered.min()) > 3 * (raw.max() - raw.min())


def test_centering_is_off_by_default():
    """★ 这条锁的是一个**决定**，不是一个能力。

    区分度变好是过程指标，成功率变高才是结果指标 ——
    两者不自动等价。

    唯二显著的正效应（Data p=0.039、IR p=0.005）恰恰是在
    "永远选五火之变"的状态下测出来的。换战术可能让 IR 的
    12/12 掉下来。在拿到对比数据之前打开它，
    就是用未经验证的改动替换已被验证的行为。
    """

    assert TacticEngine(assessor=None).center_alignment is False


# ══════════════════════════════════════════════════════════
#  ④ 退化本身要可见
# ══════════════════════════════════════════════════════════

def test_selection_degeneracy_is_measurable(agents):
    """把"选择器在不在真的选"变成一个可以观察的量。

    ★ 这条测试不断言"必须选出 N 种战术" —— 那会把一个
      尚未验证的目标写死。它只保证这个量能被算出来，
      使得任何针对选择器的改动都有一个可比较的基线。
    """

    eng = TacticEngine(assessor=None)
    tasks = [
        "研究量子计算硬件的现状并给出结构化报告",
        "用 FastAPI 写一个任务管理服务，含鉴权",
        "分析这份销售数据的季节性并给出建议",
        "写一篇面向工程师的技术博客",
        "推理这个分布式系统故障的根因",
    ]
    picked = {eng.select_tactic(t, agents).selected_tactic.name for t in tasks}
    assert 1 <= len(picked) <= len(tasks)
