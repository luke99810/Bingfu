# -*- coding: utf-8 -*-
r"""断点续跑的测试。

每条都构造成「接线断掉时必须变红」。特别是那条最容易写成空检查的：
**光断言"续跑后拿到了结果"没有意义** —— 从头跑一遍也能拿到结果。
必须断言 fn **没有被再次调用**。
"""

import json
import os

import pytest

from bingfu.checkpoint import (
    CHECKPOINT_VERSION, JSONCheckpointer, MemoryCheckpointer,
    graph_signature, validate,
)
from bingfu.graph import GraphOrchestrator, NodeStatus


def _counting(name, calls, value=None, boom=False):
    def fn(inputs):
        calls.append(name)
        if boom:
            raise RuntimeError("炸了")
        return value if value is not None else f"{name}-out"
    return fn


# ════════════════════════════════════════════════════════════════
#  一、续跑真的跳过了执行
# ════════════════════════════════════════════════════════════════

def test_resume_skips_nodes_instead_of_rerunning_them():
    """★ 这条是整组的核心。

    断言的不是"续跑后有结果"（从头跑也有结果），
    而是**被跳过的节点的 fn 一次都没有再被调用**。
    """

    cp = MemoryCheckpointer()
    calls = []

    g1 = (GraphOrchestrator()
          .add("a", _counting("a", calls))
          .add("b", _counting("b", calls), depends_on=["a"]))
    r1 = g1.run(checkpointer=cp, thread_id="t1")
    assert r1.ok
    assert calls == ["a", "b"]

    calls.clear()
    g2 = (GraphOrchestrator()
          .add("a", _counting("a", calls))
          .add("b", _counting("b", calls), depends_on=["a"]))
    r2 = g2.run(checkpointer=cp, thread_id="t1")

    assert calls == [], "续跑时节点又被执行了一遍"
    assert r2.ok
    assert r2.output_of("b") == "b-out", "跳过了却没拿到产物"
    assert set(r2.resumed) == {"a", "b"}
    assert r2.nodes["a"].from_checkpoint is True


def test_partial_progress_survives_a_crash():
    """崩在中途：已完成的那一层必须已经落盘。

    ★ 每层存一次，不是最后存一次 —— 最后才存的话，
      崩在中途等于没存，而那正是这套机制唯一的用武之地。
    """

    cp = MemoryCheckpointer()
    calls = []

    g = (GraphOrchestrator()
         .add("a", _counting("a", calls))
         .add("b", _counting("b", calls, boom=True), depends_on=["a"]))
    r = g.run(checkpointer=cp, thread_id="t2")
    assert not r.ok and r.partial

    saved = cp.get("t2")
    assert "a" in saved["pool"], "已完成的节点没有落盘"
    assert "b" not in saved["pool"]

    # 修好 b 之后接着跑：a 不该重跑
    calls.clear()
    g2 = (GraphOrchestrator()
          .add("a", _counting("a", calls))
          .add("b", _counting("b", calls), depends_on=["a"]))
    r2 = g2.run(checkpointer=cp, thread_id="t2")
    assert calls == ["b"], "崩溃前已完成的节点被重跑了：%r" % calls
    assert r2.ok


def test_downstream_receives_the_restored_upstream_output():
    """续跑时下游拿到的必须是**存回来的**上游产物，不是 None、也不是重算的。

    ★ 两次必须是同一张图。

      第一版测试在第二次运行时多加了一个节点，签名对不上、
      续跑被拒 —— 于是 a 重跑、下游收到重算的值，测试红了。
      红得对：被拒是正确行为，错的是测试。
    """

    cp = MemoryCheckpointer()
    seen = {}

    def b_fails(inputs):
        raise RuntimeError("第一次先让下游失败，把上游的产物留在断点里")

    g = (GraphOrchestrator()
         .add("a", lambda i: {"n": 42})
         .add("b", b_fails, depends_on=["a"]))
    g.run(checkpointer=cp, thread_id="t3")

    def b_ok(inputs):
        seen.update(inputs.get("a") or {})
        return "ok"

    g2 = (GraphOrchestrator()
          .add("a", lambda i: {"n": 0})       # 不该被调用；调用了就会看到 0
          .add("b", b_ok, depends_on=["a"]))
    g2.run(checkpointer=cp, thread_id="t3")
    assert seen == {"n": 42}, "下游收到的不是断点里的产物：%r" % seen


def test_resume_requires_identical_subtask_ids():
    """★ 记一个真实的局限，免得日后当成 bug 去查。

    签名由「节点名 + 依赖」构成。战役路径上的节点名来自 LLM 拆解出的
    子任务 id —— 同一个任务重跑一次，若 LLM 给出的 id 不同，
    签名就对不上，断点等于作废。

    也就是说：**断点对"改一行再接着跑"有效，对"重新拆解一次"无效。**
    要让后者也能续，得先让 decompose 的 id 稳定（比如按内容哈希），
    那是另一件事。
    """

    from bingfu.checkpoint import graph_signature

    g1 = GraphOrchestrator().add("s1", lambda i: 1)
    g2 = GraphOrchestrator().add("step-1", lambda i: 1)
    assert graph_signature(g1.nodes) != graph_signature(g2.nodes)


# ════════════════════════════════════════════════════════════════
#  二、什么时候不能续
# ════════════════════════════════════════════════════════════════

def test_changed_graph_is_not_resumed_and_says_why():
    """★ 图变了就不能续，而且必须说出原因。

    产物对应的是旧图，喂给新图会得到一个看起来正常、
    实际上错位的结果 —— 那比从头跑一遍糟糕得多。
    """

    cp = MemoryCheckpointer()
    calls = []
    GraphOrchestrator().add("a", _counting("a", calls)).run(
        checkpointer=cp, thread_id="t4")

    calls.clear()
    g2 = (GraphOrchestrator()
          .add("a", _counting("a", calls))
          .add("c", _counting("c", calls), depends_on=["a"]))   # 图变了
    r = g2.run(checkpointer=cp, thread_id="t4")

    assert calls == ["a", "c"], "图变了却仍然续了跑"
    assert "形状已改变" in r.resume_note, "不可续却没说原因：%r" % r.resume_note
    assert r.resumed == []


def test_version_mismatch_is_rejected_with_a_reason():
    cp = MemoryCheckpointer()
    g = GraphOrchestrator().add("a", lambda i: 1)
    cp.put("t5", {"version": CHECKPOINT_VERSION + 99,
                  "signature": graph_signature(g.nodes),
                  "pool": {"a": 1}})
    r = g.run(checkpointer=cp, thread_id="t5")
    assert r.resumed == []
    assert "版本" in r.resume_note


def test_no_checkpoint_is_silent_not_noisy():
    """第一次跑没有断点 —— 这不是异常，不该产生说明。"""

    cp = MemoryCheckpointer()
    r = GraphOrchestrator().add("a", lambda i: 1).run(
        checkpointer=cp, thread_id="fresh")
    assert r.resume_note == ""


def test_running_without_a_checkpointer_still_works():
    """checkpointer 是可选的，不传时行为与从前完全一致。"""

    r = (GraphOrchestrator()
         .add("a", lambda i: 1)
         .add("b", lambda i: i["a"] + 1, depends_on=["a"])).run()
    assert r.ok and r.output_of("b") == 2
    assert r.resumed == [] and r.resume_note == ""


# ════════════════════════════════════════════════════════════════
#  三、存不下的产物
# ════════════════════════════════════════════════════════════════

def test_unserialisable_output_is_named_not_silently_dropped():
    """★ 存不下就说出来。

    产物不可 JSON 序列化时不能假装存下了 —— 那个节点下次一定重跑，
    而重跑意味着它的副作用会再发生一次。使用者有权提前知道。
    """

    cp = MemoryCheckpointer()

    class Opaque:
        pass

    g = (GraphOrchestrator()
         .add("a", lambda i: Opaque())
         .add("b", lambda i: "fine", depends_on=["a"]))
    r = g.run(checkpointer=cp, thread_id="t6")

    assert r.ok
    assert "a" in r.unresumable, "不可序列化的节点没有被记名"
    assert "a" not in (cp.get("t6")["pool"])
    assert "b" in cp.get("t6")["pool"]


def test_unserialisable_node_is_rerun_on_resume():
    cp = MemoryCheckpointer()
    calls = []

    class Opaque:
        pass

    def a(inputs):
        calls.append("a")
        return Opaque()

    g = GraphOrchestrator().add("a", a).add("b", lambda i: 1, depends_on=["a"])
    g.run(checkpointer=cp, thread_id="t7")
    calls.clear()

    g2 = GraphOrchestrator().add("a", a).add("b", lambda i: 1, depends_on=["a"])
    g2.run(checkpointer=cp, thread_id="t7")
    assert calls == ["a"], "存不下的节点续跑时没有重跑"


# ════════════════════════════════════════════════════════════════
#  四、落盘实现
# ════════════════════════════════════════════════════════════════

def test_json_checkpointer_round_trips(tmp_path):
    path = str(tmp_path / "sub" / "cp.json")
    cp = JSONCheckpointer(path)
    cp.put("t", {"version": CHECKPOINT_VERSION, "pool": {"a": [1, 2]}})
    assert os.path.exists(path)
    assert JSONCheckpointer(path).get("t")["pool"]["a"] == [1, 2]


def test_corrupt_checkpoint_file_does_not_break_the_run(tmp_path):
    """断点文件坏了就当没有断点，从头跑 —— 不能让它把整次执行带崩。"""

    path = tmp_path / "cp.json"
    path.write_text("{ 这不是 JSON", encoding="utf-8")
    cp = JSONCheckpointer(str(path))
    r = GraphOrchestrator().add("a", lambda i: 1).run(checkpointer=cp,
                                                      thread_id="t")
    assert r.ok, "坏掉的断点文件把执行搞崩了"


def test_write_is_atomic_no_partial_file_left(tmp_path):
    """写盘要原子：断点文件本身是用来从崩溃里恢复的，
    它自己不能因为崩在写到一半而变成半截 JSON。"""

    path = tmp_path / "cp.json"
    cp = JSONCheckpointer(str(path))
    for i in range(5):
        cp.put("t", {"version": CHECKPOINT_VERSION, "pool": {"n": i}})
        json.loads(path.read_text(encoding="utf-8"))   # 每次都必须是完整 JSON
    leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    assert leftovers == [], "留下了临时文件：%r" % leftovers


def test_every_layer_writes_a_checkpoint():
    """三层图应当写三次，而不是最后写一次。"""

    cp = MemoryCheckpointer()
    g = (GraphOrchestrator()
         .add("a", lambda i: 1)
         .add("b", lambda i: 2, depends_on=["a"])
         .add("c", lambda i: 3, depends_on=["b"]))
    g.run(checkpointer=cp, thread_id="t")
    assert cp.writes == 3, "每层都该存一次，实际存了 %d 次" % cp.writes


def test_validate_always_explains_why_it_refused():
    ok, note = validate({"version": 0, "signature": "x"}, "y")
    assert not ok and note
    ok2, note2 = validate(None, "y")
    assert not ok2 and note2 == ""      # 没有断点不是"拒绝"，无需解释


# ════════════════════════════════════════════════════════════════
#  五、接线：断点必须真的到得了战役
# ════════════════════════════════════════════════════════════════

def test_checkpointer_reaches_run_plan():
    """★ 接口做完不算完 —— 编排层收不到，它就只是 graph.py 里的摆设。"""

    import inspect
    from bingfu.orchestration import orchestrate, run_plan

    assert "checkpointer" in inspect.signature(run_plan).parameters
    assert "checkpointer" in inspect.signature(orchestrate).parameters


def test_campaign_accepts_a_checkpointer():
    import inspect
    from bingfu.campaign import Campaign

    assert "checkpointer" in inspect.signature(Campaign.__init__).parameters


def test_same_order_maps_to_the_same_checkpoint_slot():
    """★ 用任务文本的哈希做 thread_id，不用时间戳。

    时间戳每次都不同 —— 那样每次都是新槽位，断点永远命中不了自己，
    而表现是"存了但从来没续上过"，看不出哪里错。
    """

    from bingfu.campaign import Campaign

    a = Campaign.thread_id_for("整理 report 目录并统计文件数")
    b = Campaign.thread_id_for("  整理 report 目录并统计文件数  ")
    c = Campaign.thread_id_for("另一道军令")
    assert a == b, "同一道军令应当落到同一个槽位"
    assert a != c


# ════════════════════════════════════════════════════════════════
#  六、拆解成功时不能崩
# ════════════════════════════════════════════════════════════════

def test_successful_decomposition_actually_yields_subtasks():
    """★ 这条对着一个长期潜伏的缺陷。

    ``decompose`` 里写的是 ``result.data.get("subtasks")``，
    而 ``call_structured`` 返回的 ``HarnessResult`` 只有 ``output``。
    于是**拆解成功时必抛 AttributeError**，被上层兜成一句
    "执行失败：'HarnessResult' object has no attribute 'data'"。

    它能潜伏这么久，是因为拆解**失败**时会走 fallback 返回单子任务，
    那条路不碰这一行 —— 而单元测试里的桩 LLM 多半给不出合规 JSON，
    于是永远走 fallback，永远绿。

    所以这里必须喂一个**能通过校验的** JSON。
    """

    import json

    from bingfu.orchestration import decompose

    class _StubLLM:
        def generate(self, messages=None, **kw):
            class R:
                content = json.dumps({"subtasks": [
                    {"id": "s1", "description": "找资料", "category": "IR",
                     "depends_on": []},
                    {"id": "s2", "description": "写结论", "category": "Write",
                     "depends_on": ["s1"]},
                ]}, ensure_ascii=False)
            return R()

    class _StubAgent:
        role = "侦察"
        profile = None

    plan = decompose("查一下再写个结论", {"斥候": _StubAgent()}, _StubLLM())

    assert len(plan.subtasks) == 2, (
        "结构化拆解成功却没有得到子任务：%r" % (plan.subtasks,))
    assert plan.source != "single", "走成了 fallback —— 拆解结果被丢掉了"
    # ★ 不断言字面 id：子任务 id 现在按描述内容哈希生成（为了让断点
    #   能跨「重新拆解」存活）。要断言的是**边还在**，且指向第一个子任务。
    assert plan.subtasks[1].depends_on == [plan.subtasks[0].id],         "依赖边丢了或没跟着换 id，图退化成了广播"
