# -*- coding: utf-8 -*-
r"""拆解基准的测试。

基准本身是测量工具，不进运行路径 —— 但它给出的数字会被写进文档，
所以它的算法必须自己先被验一遍。尤其是统计检验：
**一个算错的 p 值比没有 p 值更糟**，因为它看起来是有根据的。
"""

import pytest

from bingfu.plan_bench import (
    PARALLEL, SEQUENTIAL, BenchTask, Sample, compare, fisher_exact_2x2,
    format_report, legacy_prompt, summarize,
)


# ════════════════════════════════════════════════════════════════
#  一、Fisher 精确检验
# ════════════════════════════════════════════════════════════════

def test_fisher_matches_scipy():
    """★ 自己实现的检验必须与权威实现对得上。

    对不上的话，文档里那个 p 值就是编的。
    """

    scipy_stats = pytest.importorskip("scipy.stats")
    cases = [(3, 9, 4, 8), (12, 0, 5, 7), (9, 3, 3, 9),
             (1, 1, 1, 1), (20, 0, 0, 20), (7, 5, 6, 6)]
    for a, b, c, d in cases:
        mine = fisher_exact_2x2(a, b, c, d)
        theirs = scipy_stats.fisher_exact([[a, b], [c, d]])[1]
        assert abs(mine - theirs) < 1e-9, \
            "(%d,%d,%d,%d) 我算 %.6f，scipy 算 %.6f" % (a, b, c, d, mine, theirs)


def test_the_historical_mistake_would_now_be_caught():
    """★ 用项目自己踩过的那组数字做回归。

    Code 类 3/12 对 4/12 曾被读成「−26 点，有害」，据此关掉了整类的
    战术注入。真实的 p 是 1.000 —— 差一个任务而已。
    """

    p = fisher_exact_2x2(3, 9, 4, 8)
    assert p > 0.9, "3/12 vs 4/12 竟然算出了 p=%.3f" % p


def test_a_real_effect_is_detected():
    """12/12 对 5/12 是真效应，不能被判成不显著。"""

    assert fisher_exact_2x2(12, 0, 5, 7) < 0.01


def test_empty_table_is_not_significant():
    assert fisher_exact_2x2(0, 0, 0, 0) == 1.0


# ════════════════════════════════════════════════════════════════
#  二、汇总口径
# ════════════════════════════════════════════════════════════════

def _s(shape, width, codes=(), failed=False, tid="t", variant="v"):
    return Sample(task_id=tid, shape=shape, variant=variant, repeat=0,
                  width=width, codes=list(codes), failed=failed)


def test_parallel_rate_counts_only_parallelizable_tasks():
    """该串行的任务不该被算进「并行率」的分母 ——
    否则一个全是串行任务的集合会让并行率永远难看。"""

    s = summarize([_s(PARALLEL, 2), _s(PARALLEL, 1), _s(SEQUENTIAL, 1)])
    assert s["n_parallel_tasks"] == 2
    assert s["parallel_hit"] == 1
    assert s["parallel_rate"] == 0.5


def test_over_split_is_measured_separately():
    """★ 把该串行的任务硬拆成并行，也是一种拆坏了。

    只看并行率的话，一个「什么都并行」的模型会拿满分。
    """

    s = summarize([_s(SEQUENTIAL, 3), _s(SEQUENTIAL, 1)])
    assert s["over_split"] == 1 and s["over_split_rate"] == 0.5


def test_failed_samples_are_kept_in_the_denominator():
    """★ 丢掉失败样本会让成功率凭空变高 —— 最常见的一种自欺。"""

    s = summarize([_s(PARALLEL, 2), _s(PARALLEL, 0, failed=True)])
    assert s["n_total"] == 2 and s["n_failed"] == 1
    assert s["n_parallel_tasks"] == 1, "失败样本不该算进有效分母"


def test_structural_breakage_is_counted():
    s = summarize([_s(PARALLEL, 2, ["cycle"]), _s(PARALLEL, 2)])
    assert s["structurally_broken"] == 1


def test_invented_handoff_is_counted():
    s = summarize([_s(SEQUENTIAL, 1, ["invented_handoff"]), _s(SEQUENTIAL, 1)])
    assert s["invented_handoff"] == 1


# ════════════════════════════════════════════════════════════════
#  三、报告必须说实话
# ════════════════════════════════════════════════════════════════

def test_report_says_not_significant_when_it_is_not():
    """★ 这是这个模块存在的全部理由。"""

    a = [_s(PARALLEL, 1)] * 5 + [_s(PARALLEL, 2)] * 5
    b = [_s(PARALLEL, 1)] * 4 + [_s(PARALLEL, 2)] * 6
    text = format_report(compare(a, b))
    assert "不显著" in text
    assert "噪声里读故事" in text, "不显著时必须把理由说清楚"


def test_report_does_not_cry_wolf_on_a_real_effect():
    a = [_s(PARALLEL, 1)] * 12
    b = [_s(PARALLEL, 2)] * 12
    cmp = compare(a, b)
    assert cmp["parallel_significant"]
    assert "不显著" not in format_report(cmp).split("Fisher")[0]


def test_report_always_carries_n():
    """每个比例都要带 n —— 光给百分比就是在请人读错。"""

    text = format_report(compare([_s(PARALLEL, 2)], [_s(PARALLEL, 1)]))
    assert "/" in text and "n" in text.lower() or "1/1" in text


# ════════════════════════════════════════════════════════════════
#  四、A/B 的对照必须是受控的
# ════════════════════════════════════════════════════════════════

def test_legacy_prompt_differs_only_in_the_requirements_block():
    """★ 只有「要求」那一段不同，其余逐字相同。

    否则比出来的差异说不清是哪一处带来的 —— 那就不是受控对照。
    """

    from bingfu.orchestration import _DECOMPOSE_PROMPT

    old, new = legacy_prompt(), _DECOMPOSE_PROMPT
    marker = "只输出 JSON"
    assert old[:old.index("要求：")] == new[:new.index("要求：")], "任务描述段不同"
    assert old[old.index(marker):] == new[new.index(marker):], "JSON 格式段不同"
    assert "自动" in new and "自动" not in old
    assert "{task}" in old and "{agents}" in old


def test_bench_tasks_are_labelled_with_a_reason():
    """★ 标注要写依据，好让后来者能反驳它。"""

    from bingfu.plan_bench import BENCH_TASKS

    assert len(BENCH_TASKS) >= 6
    assert all(t.shape in (SEQUENTIAL, PARALLEL) for t in BENCH_TASKS)
    assert all(t.note for t in BENCH_TASKS), "有任务没写标注依据"
    assert sum(1 for t in BENCH_TASKS if t.shape == PARALLEL) >= 3
    assert sum(1 for t in BENCH_TASKS if t.shape == SEQUENTIAL) >= 2


def test_samples_round_trip_through_disk(tmp_path):
    """原始观测要能存下来复算 —— 结论可被反驳的前提。"""

    from bingfu.plan_bench import load, save

    p = str(tmp_path / "s.json")
    save([_s(PARALLEL, 2, ["x"], tid="par-1")], p)
    back = load(p)
    assert back[0].task_id == "par-1" and back[0].width == 2
