# -*- coding: utf-8 -*-
r"""证明**执行层**真的并行 —— 不涉及 LLM，不涉及提示词。

════════════════════════════════════════════════════════════════
 这组测试要挡住的是一种自欺
════════════════════════════════════════════════════════════════

改了拆解提示词之后，拆出来的图从「链」变成了「宽度 2」。
但那只说明**模型愿意给出并列的子任务**，完全不能说明
执行层真的会同时跑它们。两件事之间隔着整个 GraphOrchestrator。

如果执行层其实是串行的，那么：

  · 提示词的改进毫无价值（拆宽了也白拆）
  · 而从外部看**一切正常** —— 有多个子任务、多位将领、多份产出，
    只是慢，而慢没有人会归因到「并行没生效」

所以下面用**手工构造的图 + 会睡觉的桩将领**，把时间戳和线程名记下来，
直接断言重叠。整组不发一次模型请求。
"""

import threading
import time

import pytest

from bingfu.orchestration import Plan, SubTask, run_plan


class _RecordingAgent:
    """一个会睡一会儿、并记录自己何时在哪个线程上跑的桩将领。"""

    def __init__(self, name, log, lock, delay=0.35):
        self.name = name
        self.role = "将领"
        self.profile = None
        self.tools = []
        self._tool_functions = {}
        self._log = log
        self._lock = lock
        self._delay = delay
        self.last_prompt = None
        self.tool_call_counts = {}

    # run_plan 会 copy.copy 一份再执行，属性要能被浅复制
    def add_tool(self, name, fn):
        self._tool_functions[name] = fn

    def register_tool_function(self, name, fn):
        self._tool_functions[name] = fn

    def rearm_base_tools(self):
        pass

    def set_progress_callback(self, cb):
        pass

    def execute(self, prompt):
        start = time.time()
        with self._lock:
            self._log.append({
                "agent": self.name, "start": start,
                "thread": threading.current_thread().name,
                "prompt": prompt,
            })
        time.sleep(self._delay)
        end = time.time()
        with self._lock:
            for rec in self._log:
                if rec["agent"] == self.name and "end" not in rec:
                    rec["end"] = end
        self.last_prompt = prompt
        return "%s 的产出" % self.name


def _diamond():
    """a → (b, c) → d：第二层的两个节点互不依赖，应当并行。"""

    return Plan([
        SubTask(id="a", description="第一步", category="IR"),
        SubTask(id="b", description="第二步左", depends_on=["a"], category="Reason",
                agent_name="乙"),
        SubTask(id="c", description="第二步右", depends_on=["a"], category="Reason",
                agent_name="丙"),
        SubTask(id="d", description="汇总", depends_on=["b", "c"], category="Write",
                agent_name="丁"),
    ], source="manual")


def _run_diamond(max_workers=4, delay=0.35):
    log, lock = [], threading.Lock()
    agents = {n: _RecordingAgent(n, log, lock, delay) for n in ("甲", "乙", "丙", "丁")}
    plan = _diamond()
    plan.subtasks[0].agent_name = "甲"
    result = run_plan(plan, agents, matcher=None, max_workers=max_workers)
    return result, log, agents


# ════════════════════════════════════════════════════════════════
#  一、同层节点真的同时在跑
# ════════════════════════════════════════════════════════════════

def test_same_layer_nodes_overlap_in_time():
    """★ 整组最要紧的一条：b 与 c 的执行区间必须重叠。

    如果执行层是串行的，两段时间会首尾相接而不重叠 ——
    而结果、产出、日志都一模一样，只是慢一倍。
    """

    _result, log, _agents = _run_diamond()
    by_name = {r["agent"]: r for r in log}
    b, c = by_name["乙"], by_name["丙"]

    overlap = min(b["end"], c["end"]) - max(b["start"], c["start"])
    assert overlap > 0, (
        "同层的两个子任务没有时间重叠 —— 执行层是串行的。"
        "乙 %.3f→%.3f，丙 %.3f→%.3f"
        % (b["start"], b["end"], c["start"], c["end"]))


def test_same_layer_nodes_run_on_different_threads():
    _result, log, _agents = _run_diamond()
    by_name = {r["agent"]: r for r in log}
    assert by_name["乙"]["thread"] != by_name["丙"]["thread"], \
        "同层两个节点跑在同一个线程上，不可能是并行"


def test_wall_clock_is_closer_to_three_steps_than_four():
    """★ 用墙钟时间做第二重证明。

    四个节点、每个 0.35s。串行 ≈ 1.4s；三层并行 ≈ 1.05s。
    取 1.25s 作为分界 —— 留出足够余量，但仍能区分两种执行方式。
    """

    t0 = time.time()
    _run_diamond(delay=0.35)
    elapsed = time.time() - t0
    assert elapsed < 1.25, (
        "墙钟 %.2fs，接近串行的 4×0.35=1.4s —— 同层并行没有生效" % elapsed)


def test_serial_executor_would_fail_these_assertions():
    """★ 反证：把并发上限压到 1，上面那些断言必须失败。

    没有这一条，前面几条就可能是「无论如何都会通过」的空检查。
    """

    _result, log, _agents = _run_diamond(max_workers=1)
    by_name = {r["agent"]: r for r in log}
    b, c = by_name["乙"], by_name["丙"]
    overlap = min(b["end"], c["end"]) - max(b["start"], c["start"])
    assert overlap <= 0.01, (
        "max_workers=1 时竟然还有重叠 %.3fs —— 说明重叠不是并行造成的，"
        "上面的断言不可信" % overlap)


# ════════════════════════════════════════════════════════════════
#  二、依赖顺序与产物传递
# ════════════════════════════════════════════════════════════════

def test_downstream_starts_only_after_both_upstreams_finish():
    _result, log, _agents = _run_diamond()
    by_name = {r["agent"]: r for r in log}
    assert by_name["丁"]["start"] >= by_name["乙"]["end"] - 0.02
    assert by_name["丁"]["start"] >= by_name["丙"]["end"] - 0.02


def test_downstream_actually_receives_both_upstream_products():
    """★ 边不能是装饰。

    只按顺序跑而不传产物，与并行跑没有区别 ——
    那样的「依赖」只是一个装饰。
    """

    _result, log, _agents = _run_diamond()
    prompt = {r["agent"]: r["prompt"] for r in log}["丁"]
    assert "乙 的产出" in prompt, "上游 b 的产物没进下游提示词"
    assert "丙 的产出" in prompt, "上游 c 的产物没进下游提示词"
    assert "汇总" in prompt, "子任务自身的描述丢了"


def test_first_layer_agent_gets_no_context():
    _result, log, _agents = _run_diamond()
    prompt = {r["agent"]: r["prompt"] for r in log}["甲"]
    assert "的产出" not in prompt


# ════════════════════════════════════════════════════════════════
#  三、结果结构
# ════════════════════════════════════════════════════════════════

def test_result_reports_real_parallel_width():
    result, _log, _agents = _run_diamond()
    assert result.graph.parallel_width == 2
    assert result.parallel_width == 2
    assert result.is_real_collaboration is True


def test_每个子任务分派到不同将领():
    result, _log, _agents = _run_diamond()
    assert len(set(result.assignments.values())) == 4, \
        "四个子任务应当分给四位将领：%r" % result.assignments


def test_max_workers_is_reachable_from_the_campaign():
    """并发上限必须能从上层调下来 —— 写死在图里就等于不可调。"""

    import inspect

    from bingfu.campaign import Campaign
    from bingfu.orchestration import orchestrate

    assert "max_workers" in inspect.signature(run_plan).parameters
    assert "max_workers" in inspect.signature(orchestrate).parameters
    assert "max_workers" in inspect.signature(Campaign.run).parameters
