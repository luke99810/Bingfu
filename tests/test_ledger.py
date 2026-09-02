# -*- coding: utf-8 -*-
r"""冗余熔断与结果压缩的测试。

★ 熔断这类机制最危险的失败不是「没省下」，而是**省错了** ——
  把一次真实的覆盖写入当成重复跳过，或者把改动过的文件从缓存里
  返回旧内容。那两种错都不会报错，只会让结果悄悄不对。
  所以下面对「什么时候不能熔断」的断言比对「能熔断」的更多。
"""

import pytest

from bingfu.ledger import CallLedger, compress


def _args(name, content=None):
    d = {"filename": name}
    if content is not None:
        d["content"] = content
    return d


# ════════════════════════════════════════════════════════════════
#  一、该熔断的
# ════════════════════════════════════════════════════════════════

def test_identical_read_is_short_circuited():
    L = CallLedger()
    assert L.check("read_file", _args("a.txt")) is None      # 首次不熔断
    L.record("read_file", _args("a.txt"), "苹果 12")
    out = L.check("read_file", _args("a.txt"))
    assert out is not None and "已读过" in out
    assert "苹果 12" in out, "熔断要把上次的内容还回去，而不是只说一句已读过"


def test_identical_write_is_short_circuited():
    L = CallLedger()
    L.record("write_file", _args("a.txt", "X"), "已写入")
    out = L.check("write_file", _args("a.txt", "X"))
    assert out is not None and "已跳过" in out


def test_breaker_counts_what_it_saved():
    """省下的调用要看得见，否则无从判断它有没有用。"""

    L = CallLedger()
    L.record("read_file", _args("a.txt"), "x")
    L.check("read_file", _args("a.txt"))
    L.check("read_file", _args("a.txt"))
    assert L.stats()["breaker_hits"] == 2


# ════════════════════════════════════════════════════════════════
#  二、不该熔断的 —— 这几条更要紧
# ════════════════════════════════════════════════════════════════

def test_write_with_different_content_is_not_blocked():
    """★ 同一文件写第二次但内容不同，是覆盖，是合法动作。

    把它熔断掉会**静默丢掉一次真实修改** —— 比多花几个 token 严重得多。
    """

    L = CallLedger()
    L.record("write_file", _args("a.txt", "第一版"), "已写入")
    assert L.check("write_file", _args("a.txt", "第二版")) is None


def test_read_cache_is_invalidated_by_a_write():
    """★ 写完再读回来确认是正当动作，必须读到新内容。

    不失效的话，模型会拿到写入前的旧内容，而它据此下的结论
    没有任何地方看得出是基于过期材料。
    """

    L = CallLedger()
    L.record("read_file", _args("a.txt"), "旧内容")
    assert L.check("read_file", _args("a.txt")) is not None   # 此时可熔断
    L.record("write_file", _args("a.txt", "新内容"), "已写入")
    assert L.check("read_file", _args("a.txt")) is None, \
        "文件被写过之后，读缓存必须失效"


def test_different_files_do_not_share_cache():
    L = CallLedger()
    L.record("read_file", _args("a.txt"), "A")
    assert L.check("read_file", _args("b.txt")) is None


def test_failed_calls_are_not_cached():
    """报错的结果不该被记进账本 —— 否则一次偶发失败会被永久复读。

    记录由调用方决定（runtime 只在非错误时 record），
    这里断言账本本身不会凭空缓存没 record 过的东西。
    """

    L = CallLedger()
    assert L.check("read_file", _args("missing.txt")) is None


# ════════════════════════════════════════════════════════════════
#  三、压缩
# ════════════════════════════════════════════════════════════════

def test_short_text_is_untouched():
    assert compress("短文本") == "短文本"


def test_long_text_is_compressed_and_says_how_much_was_dropped():
    """★ 省略必须说出来。

    悄悄截断会让模型以为自己看到了全部内容，
    而它据此做的判断没有任何地方能看出是基于半份材料。
    """

    text = "甲" * 5000
    out = compress(text)
    assert len(out) < 600
    assert "省略" in out and "字符" in out
    assert out.startswith("甲") and out.endswith("甲")


def test_compressed_read_still_carries_the_head():
    """熔断回灌的是压缩版，但开头必须还在 —— 多数信息在开头。"""

    L = CallLedger()
    L.record("read_file", _args("big.txt"), "开头标记ABC" + "填" * 4000)
    out = L.check("read_file", _args("big.txt"))
    assert "开头标记ABC" in out


# ════════════════════════════════════════════════════════════════
#  四、接线：账本必须真的到得了分发层
# ════════════════════════════════════════════════════════════════

def test_harness_accepts_a_ledger():
    import inspect

    from bingfu.runtime import Harness

    assert "ledger" in inspect.signature(Harness.__init__).parameters


def test_agent_forwards_the_ledger():
    from bingfu.agent import Agent

    L = CallLedger()
    a = Agent(name="x", role="y", ledger=L)
    assert a.ledger is L


def test_run_plan_shares_one_ledger_across_subtasks():
    """★ 这条是整个机制的关键。

    实测冗余发生在**不同子任务之间**（两个子任务各读一遍同样的文件）。
    每个子任务各有一本账等于没做 —— 所以要断言它们拿到的是**同一本**。
    """

    import threading

    from bingfu.orchestration import Plan, SubTask, run_plan

    seen = []
    lock = threading.Lock()

    class _Agent:
        name = "甲"
        role = "将领"
        profile = None
        tools = []
        ledger = None

        def __init__(self):
            self._tool_functions = {}

        def add_tool(self, n, f):
            self._tool_functions[n] = f

        def register_tool_function(self, n, f):
            self._tool_functions[n] = f

        def rearm_base_tools(self):
            pass

        def execute(self, prompt):
            with lock:
                seen.append(id(self.ledger))
            return "ok"

    plan = Plan([SubTask(id="a", description="一"),
                 SubTask(id="b", description="二")], source="manual")
    run_plan(plan, {"甲": _Agent()}, matcher=None)

    assert len(seen) == 2
    assert seen[0] == seen[1], "两个子任务拿到了不同的账本，熔断不会跨子任务生效"
    assert seen[0] is not None and seen[0] != id(None)


def test_orchestration_result_reports_breaker_hits():
    from dataclasses import fields

    from bingfu.orchestration import OrchestrationResult

    assert "breaker_hits" in {f.name for f in fields(OrchestrationResult)}
