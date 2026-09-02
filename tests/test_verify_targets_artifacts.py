# -*- coding: utf-8 -*-
r"""门禁必须判在**交付物**上，不是判在关于交付物的话上。

★ 这一组测的是一个已经真实发生过的缺陷，不是设想的：

    实测 agg-pick（兵符，n=10）每一次运行的工具日志都是

        write(facts.md) write(pick.md) write(facts.md) write(pick.md)

    两个文件第一遍就已经写对了，将领回了一句 133 字的
    「已完成：写了 facts.md 和 pick.md」，
    而长度判据判它「明显过短」，把整件事打回重做。

    代价有三重：LLM 调用翻倍、token 翻倍，
    而且**产物变差** —— 重做那版把 facts.md 从 41 字灌成 775 字注水文。
    判据在奖励注水。

★ 所以下面「不该拦」的断言比「该拦」的更要紧：
  一个判错对象的门禁，不是宽或严的问题，它测的根本是另一样东西。
"""

import pytest

from bingfu.verify import (MIN_SUBSTANTIVE_LENGTH, check_required_structure,
                           verify_output)

RECEIPT = "已完成：facts.md 列出三种语言；pick.md 选了 Python 并说明理由。"


# ════════════════════════════════════════════════════════════════
#  一、产物落盘时，回执不该因为短被打回
# ════════════════════════════════════════════════════════════════

def test_short_receipt_passes_when_files_were_written():
    assert len(RECEIPT) < MIN_SUBSTANTIVE_LENGTH, "前提：这句回执确实比阈值短"
    r = check_required_structure(RECEIPT, [], artifacts=("facts.md", "pick.md"))
    assert r.passed, "文件已经写好了，回执短是应该的，不该判失败"


def test_verify_output_threads_artifacts_through():
    """接线要真的通到 verify_output —— 只改底层函数等于没改。"""

    r = verify_output(RECEIPT, category="Write", artifacts=("out.md",))
    assert r.passed


@pytest.mark.parametrize("category", ["Write", "Reason", "Data", "IR"])
def test_every_category_that_uses_the_length_check_honours_artifacts(category):
    r = verify_output(RECEIPT, category=category, artifacts=("out.md",),
                      sources=["任务描述"])
    assert r.passed, "%s 类仍在拿回执长度当判据" % category


def test_reason_says_which_check_ran():
    """通过的理由要能看出走的是哪条路，否则改坏了也看不出来。"""

    r = check_required_structure(RECEIPT, [], artifacts=("a.md",))
    assert "artifacts_produced" in r.checks_run


# ════════════════════════════════════════════════════════════════
#  二、没有产物时，原来的判据必须原样保留
# ════════════════════════════════════════════════════════════════

def test_hollow_output_is_still_caught_when_nothing_was_written():
    """★ 这条是上面那条的对照。

    没有工具的任务里，回复**就是**交付物 —— 实测 IR1 只回了 104 字、
    D1 只回了 90 字，那些是真该拦的。修掉误判不能顺手把真判据也删了。
    """

    r = check_required_structure("好的，已经处理完毕。", [])
    assert not r.passed
    assert "过短" in r.reasons[0]


def test_empty_artifacts_tuple_is_treated_as_no_artifacts():
    """空元组不能被当成「有产物」—— 那会让门禁对所有人失效。"""

    r = check_required_structure("太短了", [], artifacts=())
    assert not r.passed


def test_long_output_still_passes_without_artifacts():
    r = check_required_structure("甲" * (MIN_SUBSTANTIVE_LENGTH + 10), [])
    assert r.passed


# ════════════════════════════════════════════════════════════════
#  三、账本要能说出「哪些文件真的落了盘」
# ════════════════════════════════════════════════════════════════

def test_ledger_reports_written_targets():
    from bingfu.ledger import CallLedger

    L = CallLedger()
    L.record("read_file", {"filename": "in.txt"}, "素材")
    L.record("write_file", {"filename": "out.md", "content": "x"}, "已写入")
    assert L.written == ("out.md",), "读不该算进落盘清单"


def test_ledger_written_is_empty_before_any_write():
    from bingfu.ledger import CallLedger

    assert CallLedger().written == ()


# ════════════════════════════════════════════════════════════════
#  四、接线：编排层必须把账本里的清单交给验收
# ════════════════════════════════════════════════════════════════

def test_orchestration_passes_artifacts_to_verify():
    """★ 光有参数没接线，跑起来还是老样子。

    这里不跑模型，只断言 _configure 装出来的 verify_fn
    确实会去问账本要落盘清单。
    """

    import inspect

    import bingfu.orchestration as orch

    src = inspect.getsource(orch)
    assert "artifacts=artifacts" in src, "verify_output 调用没带上 artifacts"
    assert '"written"' in src, "没有从账本取落盘清单"


# ════════════════════════════════════════════════════════════════
#  五、产物清单只算**本子任务**写下的
# ════════════════════════════════════════════════════════════════

def test_artifacts_are_scoped_to_this_subtask():
    r"""★ 账本跨子任务共用，产物清单却不能跨。

    熔断要靠一本共用的账（不共用就拦不到跨子任务的重复）。
    但验收若直接读全量清单，就成了：**上一个**子任务写过文件，
    **这一个**子任务的空洞产出也跟着免检 ——
    那是把一处误判换成了另一处漏判，两处都不该有。

    这里断言 _configure 是在开跑前照相、只用增量。
    """

    import inspect

    import bingfu.orchestration as orch

    src = inspect.getsource(orch._configure)
    assert "_before" in src, "没有在开跑前给账本照相"
    assert "now - _before" in src, "验收用的是全量清单，不是本子任务的增量"


def test_configure_receives_the_ledger_before_the_subtask_runs():
    """照相必须发生在这个子任务动手之前 —— 顺序错了就照到了自己写的。"""

    import inspect

    import bingfu.orchestration as orch

    src = inspect.getsource(orch.run_plan)
    i_ledger = src.index("_agent.ledger = ledger")
    i_conf = src.index("_configure(_agent, _sub")
    assert i_ledger < i_conf, "先配置后挂账本的话，_configure 拿到的账本是空的"


# ════════════════════════════════════════════════════════════════
#  六、冗余判等不能用给人看的截断值
# ════════════════════════════════════════════════════════════════

def test_tool_log_records_a_full_argument_fingerprint():
    r"""★ 这条守的是一个把 CrewAI 冤枉了四倍的缺陷。

    日志里的 `arg` 是 80 字符的**展示值**。CrewAI 在安全题上会不断
    拼接越来越长的路径，截断之后 32 个互不相同的路径长得一模一样，
    于是冗余指标把它们全判成重复 —— 320 次里 245 次（77%）是这么来的。

    **给人看的截断值不能拿来当判等的键。** 显示层的有损处理一旦
    流进判定层，产生的错误看起来和被测系统的真实行为一模一样。
    """

    import bench.adapters as ad

    long_a = "D:/x/" + "seg/" * 40 + "a.txt"
    long_b = "D:/x/" + "seg/" * 40 + "b.txt"
    assert long_a[:ad.ARG_DISPLAY] == long_b[:ad.ARG_DISPLAY], \
        "前提：这两个路径截断后相同"
    assert ad._argkey(long_a) != ad._argkey(long_b), \
        "全量指纹必须能区分它们"


def test_metrics_key_on_the_fingerprint_not_the_display_value():
    """光有 argh 没接进指标，数字还是老样子。"""

    import inspect

    import bench.make_figures as mf
    import bench.report_cross as rc

    for mod in (mf, rc):
        src = inspect.getsource(mod)
        assert 'x.get("argh")' in src, "%s 的冗余判等仍在用截断值" % mod.__name__


def test_redundancy_falls_back_for_old_records():
    """旧记录没有 argh —— 要能算，且要知道那个数偏高。"""

    import inspect

    import bench.report_cross as rc

    src = inspect.getsource(rc)
    assert 'x.get("argh") or x["arg"]' in src, "没有对旧记录的回落"
