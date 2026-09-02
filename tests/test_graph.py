"""Graph 编排层的测试。

★ 重点不是"能跑通"，而是三件容易悄悄失效的事：

  ① 并行是真的发生了 —— DAG 退化成链表时代码照样跑，只是慢
  ② 失败被隔离了，而且"部分完成"是**显式**的
  ③ 分流表与实测数据一致 —— 它是这一层全部价值的来源
"""

import time

import pytest

from bingfu.graph import (
    CycleError,
    GraphOrchestrator,
    NodeStatus,
    ROUTES,
    route_for,
)


# ══════════════════════════════════════════════════════════
#  ① 并行必须是真的
# ══════════════════════════════════════════════════════════

def _sleepy(name: str, seconds: float = 0.25):
    def fn(_inputs):
        time.sleep(seconds)
        return name
    return fn


def test_independent_nodes_run_in_parallel():
    """★ 这条测的是墙钟，不是结构。

    只断言"层里有多个节点"不够 —— 层内串行执行也满足那个断言。
    必须让串行与并行的耗时有数量级差异，才能真的分辨。
    """

    g = GraphOrchestrator()
    g.add("start", lambda i: "ok")
    for n in ("a", "b", "c"):
        g.add(n, _sleepy(n), depends_on=["start"])
    g.add("join", lambda i: sorted([i["a"], i["b"], i["c"]]),
          depends_on=["a", "b", "c"])

    t0 = time.time()
    r = g.run()
    elapsed = time.time() - t0

    assert r.ok
    assert r.output_of("join") == ["a", "b", "c"]
    assert elapsed < 0.6, f"三个 0.25s 的独立节点耗时 {elapsed:.2f}s —— 没有并行"


def test_degenerate_chain_is_detectable():
    """全串行的图 parallel_width == 1 —— 这是"伪 Graph"的自查判据。"""

    g = GraphOrchestrator()
    g.add("a", lambda i: 1)
    g.add("b", lambda i: 2, depends_on=["a"])
    g.add("c", lambda i: 3, depends_on=["b"])
    assert g.run().parallel_width == 1


def test_layers_respect_dependencies():
    g = GraphOrchestrator()
    g.add("root", lambda i: 0)
    g.add("mid1", lambda i: 1, depends_on=["root"])
    g.add("mid2", lambda i: 2, depends_on=["root"])
    g.add("leaf", lambda i: 3, depends_on=["mid1", "mid2"])
    layers = g.topological_sort()
    assert layers == [["root"], ["mid1", "mid2"], ["leaf"]]


# ══════════════════════════════════════════════════════════
#  ② 失败隔离 —— 跳过而非崩溃
# ══════════════════════════════════════════════════════════

def test_one_failure_does_not_kill_sibling_branches():
    g = GraphOrchestrator()
    g.add("root", lambda i: "ok")
    g.add("good", lambda i: "G", depends_on=["root"])
    g.add("bad", lambda i: 1 / 0, depends_on=["root"])
    g.add("after_good", lambda i: "AG", depends_on=["good"])
    g.add("after_bad", lambda i: "不该执行", depends_on=["bad"])

    r = g.run()
    assert r.nodes["good"].status is NodeStatus.OK
    assert r.nodes["after_good"].status is NodeStatus.OK
    assert r.nodes["bad"].status is NodeStatus.FAILED
    assert r.nodes["after_bad"].status is NodeStatus.SKIPPED


def test_partial_completion_is_explicit():
    """★ 悄悄返回半份结果比直接失败更危险。

    调用方必须能一眼看出"这次只完成了一部分"，
    再自己决定这半份能不能用 ——
    而不是由编排层替它假装一切正常。
    """

    g = GraphOrchestrator()
    g.add("ok_node", lambda i: 1)
    g.add("bad_node", lambda i: 1 / 0)
    r = g.run()
    assert r.ok is False
    assert r.partial is True


def test_failure_reason_is_recorded():
    g = GraphOrchestrator()
    g.add("bad", lambda i: 1 / 0)
    r = g.run()
    assert "ZeroDivisionError" in r.nodes["bad"].error


# ══════════════════════════════════════════════════════════
#  ③ 图本身的错误必须抛，不能静默
# ══════════════════════════════════════════════════════════

def test_cycle_raises_instead_of_silently_dropping_nodes():
    """★ 静默跳过的话，循环依赖会表现为"某些节点莫名没跑"，

    而整体看起来是成功的 —— 又一个不报错的坏结果。
    """

    g = GraphOrchestrator()
    g.add("a", lambda i: 1, depends_on=["b"])
    g.add("b", lambda i: 2, depends_on=["a"])
    with pytest.raises(CycleError):
        g.run()


def test_unknown_dependency_raises():
    g = GraphOrchestrator()
    g.add("a", lambda i: 1, depends_on=["不存在"])
    with pytest.raises(ValueError):
        g.topological_sort()


def test_duplicate_node_name_raises():
    g = GraphOrchestrator()
    g.add("a", lambda i: 1)
    with pytest.raises(ValueError):
        g.add("a", lambda i: 2)


# ══════════════════════════════════════════════════════════
#  ④ 分流表必须与实测一致
# ══════════════════════════════════════════════════════════

def test_tactic_stays_on_where_the_effect_was_only_noise():
    """★ 这条测试锁的是一条**判断规则**，不是某个百分比。

    这两行曾经断言 ``is False``，依据是"实测 Code −26 点、Write −12 点"。
    重算显著性之后那个依据不成立：

        Code   3/12 vs 4/12  →  差 1 个任务，p = 1.000
        Write   3/9 vs  5/9  →  差 2 个任务，p = 0.637

    n≈12 的格子里一个任务翻转就是 8 个百分点。
    把它读成"注入有害"，是在噪声里读故事。

    ★ 规则：**只在有显著证据时才偏离默认。**
      没有证据时保持默认，而不是顺着噪声的方向走 ——
      因为一旦偏离，之后就再也不会产生能反驳它的数据，
      这个错误会自我固化成"看起来被数据支持的决定"。
    """

    assert route_for("Code").inject_tactic is True
    assert route_for("Write").inject_tactic is True
    assert route_for("Reason").inject_tactic is True   # p=1.000，同理


def test_tactic_is_on_where_the_effect_is_significant():
    """实测显著有益：Data +6 任务 (p=0.039)、IR +7 任务 (p=0.005)。"""

    assert route_for("Data").inject_tactic is True
    assert route_for("IR").inject_tactic is True


def test_gate_is_off_where_there_is_no_headroom():
    """IR 实测已到 100% —— 开门禁只增成本（实测 token 涨 2.7 倍）。"""

    assert route_for("IR").verify_enabled is False
    assert route_for("IR").max_revisions == 0


def test_gate_is_on_where_headroom_is_largest():
    """Code 实测 12% —— 回炉收益远大于成本。"""

    assert route_for("Code").verify_enabled is True
    assert route_for("Code").max_revisions >= 1


def test_unknown_category_gets_a_conservative_default():
    plan = route_for("从未见过的类别")
    assert plan.verify_enabled is True and plan.max_revisions == 1


def test_every_route_explains_itself():
    """每条分流规则都要写明依据 —— 否则半年后没人知道能不能改。"""

    for plan in ROUTES.values():
        assert plan.rationale, f"{plan.category} 缺少 rationale"


# ══════════════════════════════════════════════════════════
#  ⑤ 消融档的语义必须诚实
# ══════════════════════════════════════════════════════════

def test_no_tactic_arm_reports_no_tactic():
    """★ no_tactic 这一档**根本没有使用任何战术**。

    它曾经把 tactic_used 报成 "random"。后果不是名字难看，
    而是任何按 tactic_used 分组的分析都会凭空多出一个
    不存在的战术，并把"未使用战术"的样本算进"使用了某战术"里。

    没有就是没有 —— 报表不能替实现圆场。
    """

    from bingfu.experiment import BENCHMARK_TASKS, BaselineRunner

    runner = BaselineRunner(None)          # 无 LLM，只验报表字段
    task = next(t for t in BENCHMARK_TASKS if t.id == "C1")
    assert runner.run_bingfu(task, seed=42, ablation="no_tactic").tactic_used == "none"


def test_ablation_arms_differ_by_exactly_one_variable():
    """★ full 相对每个消融档只能差一件事，否则差异无法归因。

    原设计里 full 相对两个消融档同时改了「选将方式」与
    「是否注入战术」，于是 32 个点的领先究竟来自哪里，
    那个设计答不出来 —— 而 32 点、p<0.00001 的结果
    看起来极有说服力，这正是它危险的地方。

      full       按战力选将 + 注入战术
      no_tactic  按战力选将 + **不注入**   ← 只差注入
      no_power   **随机选将** + 注入战术   ← 只差选将
    """

    from bingfu.experiment import BENCHMARK_TASKS, BaselineRunner

    runner = BaselineRunner(None)
    task = next(t for t in BENCHMARK_TASKS if t.id == "C1")
    full = runner.run_bingfu(task, seed=42, ablation="full")
    no_tactic = runner.run_bingfu(task, seed=42, ablation="no_tactic")
    no_power = runner.run_bingfu(task, seed=42, ablation="no_power")

    # no_tactic 不带战术；no_power 带战术（与 full 同名，因为选择器一致）
    assert no_tactic.tactic_used == "none"
    assert no_power.tactic_used != "none"
    assert full.tactic_used != "none"
