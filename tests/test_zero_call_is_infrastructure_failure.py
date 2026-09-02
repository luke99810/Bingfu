# -*- coding: utf-8 -*-
r"""零次 LLM 调用必须记成「跑不起来」，不能记成「没做对」。

════════════════════════════════════════════════════════════════
 这条守的是一次真实发生过的误记
════════════════════════════════════════════════════════════════

跑第 80 次时 DeepSeek 余额耗尽，此后每个请求都是 HTTP 402。
供应商把异常吞掉、战役返回空输出，而计分照常进行 ——
于是 45 次「什么都没发生」被记成 45 次任务失败，
正确率从 118/120 变成 74/120，**而且没有任何地方报错**。

★ 那批数字看上去完全正常：有 tok/得分点、有中位调用数、有成本，
  每一列都算得出来。**一个由停电产生的数据集，
  和一个由系统表现产生的数据集，在表格里长得一模一样。**

★ 所以判据必须是「模型侧有没有发生过请求」，
  而不是「有没有抛异常」—— 异常会被吞，请求数不会说谎。
"""

import inspect

import pytest


def test_run_one_flags_zero_call_runs_as_failed():
    import bench.run_one as ro

    src = inspect.getsource(ro)
    assert "r.llm_calls == 0" in src, "没有检查零调用"
    assert "r.failed = True" in src.split("r.llm_calls == 0")[1][:200], \
        "检查到零调用之后没有标成 failed"


def test_the_flag_carries_a_reason_that_points_at_infrastructure():
    """标记还要说清楚是哪一类失败，否则下次还是会归错因。"""

    import bench.run_one as ro

    src = inspect.getsource(ro)
    tail = src.split("r.llm_calls == 0")[1][:400]
    assert "配额" in tail or "余额" in tail, "错误信息没有指向基础设施"


def test_failed_runs_are_excluded_from_pass_counting():
    """★ 关键：`passed` 必须先看 failed。

    否则「跑不起来」会一路走到「该产出的文件不存在」，
    最后以「任务失败」的面目出现在正确率里。
    """

    from bench.cross_framework import Run

    r = Run(system="x", task_id="t", repeat=0, shape="solo")
    r.failed = True
    r.hit, r.total = 0, 0
    assert r.passed is False, "failed 的运行不能算通过"


def test_a_real_run_with_calls_is_not_flagged():
    """对照：正常跑起来的运行不该被这条误伤。"""

    from bench.cross_framework import Run

    r = Run(system="x", task_id="t", repeat=0, shape="solo")
    r.llm_calls = 2
    r.hit, r.total = 1, 1
    assert r.failed is False
    assert r.passed is True
