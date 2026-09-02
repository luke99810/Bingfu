# -*- coding: utf-8 -*-
r"""拆解校验的测试。

════════════════════════════════════════════════════════════════
 这一层对着一个实测出来的问题
════════════════════════════════════════════════════════════════

改造前用三个真实任务跑拆解，结果全是链：

    t1 → t2 → t3 → t4     parallel_width = 1

也就是说框架最核心的「同层并行」**一次都没有被触发过**，
而从结果上完全看不出来 —— 有多个子任务、有多份产出、
有多位将领参与，看起来相当像协作。

所以这组测试的重点不是「校验函数能跑」，而是
**退化的图必须被判定为退化**。
"""

import pytest

from bingfu.orchestration import Plan, SubTask
from bingfu.plan_check import (
    ERROR, WARN, has_error, issues_as_feedback, layers_of,
    parallel_width, stable_subtask_id, validate_plan,
)


def _plan(*specs):
    """specs: (id, deps) 或 (id, deps, description)"""
    subs = []
    for spec in specs:
        tid, deps = spec[0], spec[1]
        desc = spec[2] if len(spec) > 2 else "做 %s" % tid
        subs.append(SubTask(id=tid, description=desc, depends_on=list(deps)))
    return Plan(subs, source="llm")


def _codes(issues):
    return {i.code for i in issues}


# ════════════════════════════════════════════════════════════════
#  一、分层与并行宽度
# ════════════════════════════════════════════════════════════════

def test_diamond_has_width_two():
    p = _plan(("a", []), ("b", ["a"]), ("c", ["a"]), ("d", ["b", "c"]))
    assert p.layers == [["a"], ["b", "c"], ["d"]]
    assert p.parallel_width == 2


def test_pure_chain_has_width_one():
    p = _plan(("a", []), ("b", ["a"]), ("c", ["b"]), ("d", ["c"]))
    assert p.parallel_width == 1


def test_layering_is_deterministic():
    """同一张图每次分层结果必须一致 —— 对照实验按固定种子跑，
    执行顺序若随字典序漂移，那几个种子就控不住变量了。"""

    p = _plan(("z", []), ("a", []), ("m", ["z", "a"]))
    assert layers_of(p.subtasks) == layers_of(p.subtasks) == [["a", "z"], ["m"]]


# ════════════════════════════════════════════════════════════════
#  二、退化必须被判出来
# ════════════════════════════════════════════════════════════════

def test_degenerate_chain_is_reported():
    """★ 这条是整组的核心。"""

    issues = validate_plan(_plan(("a", []), ("b", ["a"]), ("c", ["b"])))
    assert "degenerate_chain" in _codes(issues), \
        "拆成一条链却没有被判定为退化：%r" % [str(i) for i in issues]
    assert not has_error(issues), "退化是坏味道，不是硬错误，不该触发重拆"


def test_two_step_chain_is_not_flagged():
    """两步的顺序任务本来就该是链 —— 不能把正常拆解也报成退化。"""

    issues = validate_plan(_plan(("a", []), ("b", ["a"])))
    assert "degenerate_chain" not in _codes(issues)


def test_a_healthy_dag_has_no_issues():
    issues = validate_plan(_plan(("a", []), ("b", ["a"]), ("c", ["a"]),
                                 ("d", ["b", "c"])))
    assert issues == [], [str(i) for i in issues]


def test_multiple_subtasks_without_edges_is_broadcast_not_collaboration():
    issues = validate_plan(_plan(("a", []), ("b", []), ("c", [])))
    assert "no_edges" in _codes(issues)


# ════════════════════════════════════════════════════════════════
#  三、结构性错误
# ════════════════════════════════════════════════════════════════

def test_cycle_is_an_error():
    issues = validate_plan(_plan(("a", ["b"]), ("b", ["a"])))
    assert "cycle" in _codes(issues)
    assert has_error(issues), "成环必须是硬错误 —— 跑下去只会悄悄少几个节点"


def test_unknown_dependency_is_an_error():
    issues = validate_plan(_plan(("a", []), ("b", ["nope"])))
    assert "unknown_dep" in _codes(issues) and has_error(issues)


def test_self_dependency_is_an_error():
    issues = validate_plan(_plan(("a", ["a"])))
    assert "self_dep" in _codes(issues) and has_error(issues)


def test_duplicate_ids_are_an_error():
    p = Plan([SubTask("a", "x"), SubTask("a", "y")], source="llm")
    assert "duplicate_id" in _codes(validate_plan(p))


def test_empty_plan_is_an_error():
    assert has_error(validate_plan(Plan([], source="llm")))


# ════════════════════════════════════════════════════════════════
#  四、臆造的文件中转
# ════════════════════════════════════════════════════════════════

def test_invented_file_handoff_is_flagged():
    """★ 这是实测里真出现过的坏拆解。

        t1: 解析 data.csv，计算总销量
        t2: 读取 total.txt（由其他任务生成），获取总销量

    t1 从未被要求创建 total.txt —— t2 只能失败或者编一个数出来。
    上游产物本来就会自动作为输入传下去，凭空约定的中间文件多半不存在。
    """

    p = _plan(("a", [], "解析 data.csv，计算所有销量的总和"),
              ("b", ["a"], "读取文件 total.txt（由其他任务生成），获取其中的总销量数字"))
    assert "invented_handoff" in _codes(validate_plan(p))


def test_referring_to_upstream_output_is_fine():
    """正确的写法不该被误报。"""

    p = _plan(("a", [], "解析 data.csv，计算所有销量的总和"),
              ("b", ["a"], "根据上游给出的总销量数值，撰写一份简洁的报告"))
    assert "invented_handoff" not in _codes(validate_plan(p))


# ════════════════════════════════════════════════════════════════
#  五、稳定 id
# ════════════════════════════════════════════════════════════════

def test_same_description_yields_the_same_id():
    """★ 断点靠它跨「重新拆解」存活。

    id 此前是模型随口给的 t1/t2/step-1，同一道军令重跑一次就换一批，
    图签名必然对不上，断点直接作废 ——
    表现是「存了但从来没续上过」。
    """

    a = stable_subtask_id("调研主流的多智能体框架")
    b = stable_subtask_id("  调研主流的多智能体框架  ")
    assert a == b


def test_different_descriptions_yield_different_ids():
    assert stable_subtask_id("查资料") != stable_subtask_id("写报告")


def test_ids_look_like_ids():
    assert stable_subtask_id("x").startswith("t")


# ════════════════════════════════════════════════════════════════
#  六、反馈文本
# ════════════════════════════════════════════════════════════════

def test_feedback_names_every_issue():
    issues = validate_plan(_plan(("a", ["b"]), ("b", ["a"])))
    text = issues_as_feedback(issues)
    assert "cycle" in text and "重新拆解" in text


def test_no_issues_yields_no_feedback():
    assert issues_as_feedback([]) == ""


# ════════════════════════════════════════════════════════════════
#  七、decompose 用上了校验
# ════════════════════════════════════════════════════════════════

def test_decompose_attaches_issues_to_the_plan():
    """拆出来的 Plan 必须带着校验结果 —— 否则上层无从告警。"""

    import json

    from bingfu.orchestration import decompose

    class _Stub:
        def generate(self, messages=None, **kw):
            class R:
                content = json.dumps({"subtasks": [
                    {"id": "s1", "description": "第一步：查资料", "category": "IR",
                     "depends_on": []},
                    {"id": "s2", "description": "第二步：写提纲", "category": "Write",
                     "depends_on": ["s1"]},
                    {"id": "s3", "description": "第三步：出终稿", "category": "Write",
                     "depends_on": ["s2"]},
                ]}, ensure_ascii=False)
            return R()

    class _Agent:
        role = "通用"
        profile = None

    plan = decompose("写点东西", {"谋士": _Agent()}, _Stub())
    assert len(plan.subtasks) == 3
    assert hasattr(plan, "issues")
    assert "degenerate_chain" in {i.code for i in plan.issues}, \
        "链式拆解没有被记进 plan.issues：%r" % [str(i) for i in plan.issues]


def test_decompose_gives_subtasks_stable_ids():
    import json

    from bingfu.orchestration import decompose

    class _Stub:
        def generate(self, messages=None, **kw):
            class R:
                content = json.dumps({"subtasks": [
                    {"id": "s1", "description": "查一查资料", "category": "IR",
                     "depends_on": []},
                    {"id": "s2", "description": "写一份报告", "category": "Write",
                     "depends_on": ["s1"]},
                ]}, ensure_ascii=False)
            return R()

    class _Agent:
        role = "通用"
        profile = None

    p1 = decompose("干活", {"谋士": _Agent()}, _Stub())
    p2 = decompose("干活", {"谋士": _Agent()}, _Stub())
    assert [t.id for t in p1.subtasks] == [t.id for t in p2.subtasks], \
        "同样的拆解结果给出了不同的 id —— 断点会永远续不上"
    assert p1.subtasks[1].depends_on == [p1.subtasks[0].id], \
        "换 id 时依赖没有跟着改，边断了"
