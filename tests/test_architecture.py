"""三层是否真的在框架的执行路径上。

★ 这组测试锁的不是某个功能，而是**架构约束**。

  重建之前，Harness / Loop / Graph 三个模块只被 experiment.py 导入，
  框架自身（agent / commander / bingfu）一处都没用，
  GraphOrchestrator 更是从未被实例化 —— 纯死代码。

  也就是说：一个用户 `from bingfu import Agent, Commander` 跑任务，
  这三层一个都用不上。它们是实验脚手架，不是框架骨架。

  这组测试确保那件事不会再次发生。
"""

import time

import pytest

from bingfu.agent import Agent
from bingfu.commander import Commander
from bingfu.loop import AgentLoop, LoopPolicy, Outcome
from bingfu.orchestration import Plan, SubTask, run_plan
from bingfu.presets import get_preset
from bingfu.runtime import Budget, Harness
from bingfu.verify import verify_output


class _Echo:
    """把收到的最后一条消息回显出来，便于断言上下文传递。"""

    def __init__(self, reply=None, delay=0.0):
        self.reply = reply
        self.delay = delay
        self.seen = []

    def generate(self, messages, **kwargs):
        self.seen.append(messages[-1].content or "")
        if self.delay:
            time.sleep(self.delay)
        text = self.reply if self.reply is not None else f"[回显]{self.seen[-1][:40]}"

        class R:
            content = text
            finish_reason = "stop"
            usage = {"total_tokens": 50}
            tool_calls = []
            has_tool_calls = False

        return R()


# ══════════════════════════════════════════════════════════
#  ① Agent 必须跑在 Harness + Loop 上
# ══════════════════════════════════════════════════════════

def test_agent_execution_goes_through_harness():
    """★ Agent 不再自己发请求。

    这是本次重建的核心：Agent 只持有身份（名号、画像、工具清单），
    机制交给 Harness。于是"某一层忘了接上"在结构上不可能发生。
    """

    agent = Agent(name="测试", role="将军", llm=_Echo("完成"))
    agent.execute("任务")
    assert agent._harness is not None, "执行没有经过 Harness"
    assert agent._harness.trace.generations >= 1


def test_agent_no_longer_owns_a_second_execution_path():
    """★ 旧的 _run_react / _generate_complete 必须删掉，不能留着。

    两套执行路径共存比一套坏的更危险：读者无法判断哪一套在生效，
    改错地方也不会有任何报错。
    """

    assert not hasattr(Agent, "_run_react")
    assert not hasattr(Agent, "_generate_complete")


def test_token_accounting_has_exactly_one_home():
    """★ token 记账只能有一处。

    先前它同时发生在两个地方，导致每轮把最后一次调用算两遍。
    token 效率是这个框架唯一被实测证实的优势，
    把它的计量搞坏，等于毁掉唯一站得住的结论 ——
    而报出的数字只是偏大，不会有任何报错。
    """

    llm = _Echo("完成")
    agent = Agent(name="测试", role="将军", llm=llm)
    agent.execute("任务")
    assert agent.last_run_tokens == 50, "一次调用 50 token，不该被重复累加"


# ══════════════════════════════════════════════════════════
#  ② Loop 的每个出口都要说明为什么
# ══════════════════════════════════════════════════════════

BAD_CODE = "```python\ndef f(x):\n    return g(\n```"
GOOD_CODE = "```python\ndef f(x):\n    return x + 1\n```"


class _Sequence:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.n = 0
        self.seen = []

    def generate(self, messages, **kwargs):
        self.seen.append(messages[-1].content or "")
        text = self.replies[min(self.n, len(self.replies) - 1)]
        self.n += 1

        class R:
            content = text
            finish_reason = "stop"
            usage = {"total_tokens": 100}
            tool_calls = []
            has_tool_calls = False

        return R()


def test_loop_distinguishes_its_exits():
    """★ "验收通过"和"预算耗尽只好交付"都会返回产出，

    但它们对调用方意味着完全不同的事。笼统的成功/失败会把
    这个区别抹掉 —— 先前正是如此，于是"一次过"和"回炉两轮
    才勉强交付"在数据上无法区分。
    """

    verify = lambda o: verify_output(o, category="Code")

    # 通过
    h = Harness(_Sequence(GOOD_CODE))
    r = AgentLoop(h, LoopPolicy(verify=verify, max_revisions=2)).run("写函数")
    assert r.outcome is Outcome.VERIFIED

    # 回炉后通过
    h = Harness(_Sequence(BAD_CODE, GOOD_CODE))
    r = AgentLoop(h, LoopPolicy(verify=verify, max_revisions=2)).run("写函数")
    assert r.outcome is Outcome.VERIFIED and r.revisions == 1

    # 一直不过 → 耗尽
    h = Harness(_Sequence(BAD_CODE))
    r = AgentLoop(h, LoopPolicy(verify=verify, max_revisions=2)).run("写函数")
    assert r.outcome is Outcome.EXHAUSTED
    assert r.stopped_by == "回炉轮次用尽"
    assert r.output, "耗尽时也要交付当前最好的结果"

    # 无验收策略
    h = Harness(_Sequence(GOOD_CODE))
    r = AgentLoop(h, LoopPolicy(verify=None)).run("写函数")
    assert r.outcome is Outcome.UNVERIFIED


def test_loop_feeds_back_specific_reasons():
    """回炉必须带**具体原因**，不能原样重试。

    对确定性失败（语法错），原样重试一百次还是同样的错，
    只是把成本乘以一百。
    """

    llm = _Sequence(BAD_CODE, GOOD_CODE)
    h = Harness(llm)
    AgentLoop(h, LoopPolicy(verify=lambda o: verify_output(o, category="Code"),
                            max_revisions=2)).run("写函数")
    assert "语法错误" in llm.seen[1]


@pytest.mark.parametrize("budget,expected", [
    (Budget(max_iterations=99, max_tokens=150), "token 预算耗尽"),
    (Budget(max_iterations=2, max_tokens=10 ** 9), "思考轮次用尽"),
])
def test_each_budget_names_itself(budget, expected):
    """★ 三条预算边界在结果上必须可区分，且名字不能撞。

    这条测试第一次跑就抓到一个歧义：内层的思考轮次和外层的
    回炉轮次**都叫"轮次"**。看到"轮次用尽"根本不知道是哪一个，
    而这两种情况该采取的措施完全不同（前者调 max_iterations，
    后者说明产出反复过不了验收）。已在 runtime.py 里改名消歧。
    """

    h = Harness(_Sequence(BAD_CODE), budget=budget)
    r = AgentLoop(h, LoopPolicy(verify=lambda o: verify_output(o, category="Code"),
                                max_revisions=9)).run("写函数")
    assert r.stopped_by == expected


# ══════════════════════════════════════════════════════════
#  ③ Graph 必须是真协作，而不是广播
# ══════════════════════════════════════════════════════════

@pytest.fixture
def agents():
    return {n: Agent(name=n, role="将军", profile=get_preset(n), llm=_Echo(delay=0.2))
            for n in ("韩信", "白起", "诸葛亮")}


def test_downstream_receives_upstream_output(agents):
    """★ "依赖"二字的全部含义就是上游产物真的喂给了下游。

    只按顺序跑而不传递产物，与并行跑没有区别，
    那样的依赖只是一个装饰。
    """

    plan = Plan([
        SubTask("t1", "收集市场数据"),
        SubTask("t2", "综合成报告", depends_on=["t1"]),
    ])
    r = run_plan(plan, agents)
    downstream = str(r.graph.nodes["t2"].output)
    assert "t1 的产出" in downstream or "收集市场数据" in downstream


def test_independent_subtasks_run_in_parallel(agents):
    """同层节点必须真的并行 —— 测墙钟，不测结构。

    只断言"层里有多个节点"不够：层内串行执行也满足那个断言，
    DAG 退化成链表时代码照样跑，只是慢，而测试全绿。
    """

    plan = Plan([SubTask(f"t{i}", f"任务{i}") for i in (1, 2, 3)])
    t0 = time.time()
    r = run_plan(plan, agents)
    elapsed = time.time() - t0
    assert r.parallel_width == 3
    assert elapsed < 0.5, f"三个 0.2s 的独立节点耗时 {elapsed:.2f}s —— 没有并行"


def test_collaboration_is_self_reported(agents):
    """★ "有没有在协作"必须可观察。

    判据是结构（多于一个子任务、且至少一条依赖边），不是意图。
    先前那个 round_robin 广播没有任何地方能看出它没在协作。
    """

    with_edges = run_plan(Plan([
        SubTask("t1", "甲"), SubTask("t2", "乙", depends_on=["t1"]),
    ]), agents)
    assert with_edges.is_real_collaboration is True

    no_edges = run_plan(Plan([SubTask("t1", "甲"), SubTask("t2", "乙")]), agents)
    assert no_edges.is_real_collaboration is False, "并行但无依赖，不算协作"

    single = run_plan(Plan([SubTask("t1", "整件事")]), agents)
    assert single.is_real_collaboration is False


def test_agent_selection_actually_varies(agents):
    """★ 点将必须真的按任务选人。

    第一版给 matcher 传错了参数类型，而外层一个
    `except Exception: pass` 把错误整个吞掉 ——
    于是每个子任务都静默退回"战力最高者"，
    三个不同的子任务全部点了同一员将，
    而功能看起来完全正常：有结果、不报错。
    """

    from bingfu.matcher import TaskMatcher

    plan = Plan([
        SubTask("t1", "侦察竞品情报，收集市场数据"),
        SubTask("t2", "快速突破实现原型代码"),
        SubTask("t3", "综合分析并撰写战略报告", depends_on=["t1", "t2"]),
    ])
    r = run_plan(plan, agents, matcher=TaskMatcher())
    assert len(set(r.assignments.values())) > 1, "所有子任务点了同一员将，matcher 未生效"


def test_commander_exposes_orchestration():
    """Commander 必须真的能做多智能体编排，而不只是广播。"""

    cmd = Commander(name="主帅")
    assert hasattr(cmd, "orchestrate")


def test_partial_failure_is_visible(agents):
    """一个子任务失败不该让整次执行归零，但"部分完成"必须显式。"""

    class _Boom:
        def generate(self, messages, **kwargs):
            raise RuntimeError("模型炸了")

    agents["白起"].llm = _Boom()
    plan = Plan([
        SubTask("t1", "甲", agent_name="韩信"),
        SubTask("t2", "乙", agent_name="白起"),
    ])
    r = run_plan(plan, agents)
    assert r.graph.partial is True, "部分完成必须可观察"
    assert r.output, "成功的那部分产出要保留"
