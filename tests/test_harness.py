"""Harness 五级降级链的测试。

★ 每条测试都对应一个**真实发生过**的故障，不是为覆盖率而写。
"""

import json
import re

import pytest

from bingfu.harness import (
    AgentHarnessFailure,
    call_structured,
    extract_json_object,
    repair_json,
    validate_schema,
)


# ── 那个骗过整套实验的 bug ──────────────────────────────

JUDGE_REPLY = (
    '好的，评估如下：\n```json\n'
    '{"success": true, "completion_score": 4.5, '
    '"criteria_scores": {"Factual Accuracy": 4, "Coverage": 5}, '
    '"reasoning": "结构完整"}\n```\n以上。'
)


def test_the_regex_that_broke_everything_is_not_used():
    """复现原 bug：`\{[^{}]*\}` 取到的是嵌套的最内层，不是外层信封。

    ★ 这条测试的价值在于它**同时断言了旧写法是错的**和新写法是对的。
      只断言新写法正确，无法阻止有人"优化"回正则。
    """

    inner = re.search(r"\{[^{}]*\}", JUDGE_REPLY, re.DOTALL)
    assert inner is not None
    wrong = json.loads(inner.group())
    # 旧写法拿到的是 criteria_scores，于是这两个字段恒取默认值
    assert wrong.get("success", False) is False
    assert wrong.get("completion_score", 3.0) == 3.0

    right = extract_json_object(JUDGE_REPLY)
    assert right["success"] is True
    assert right["completion_score"] == 4.5


def test_braces_inside_strings_do_not_break_matching():
    """字符串字面量里的花括号不能干扰配对计数。"""

    text = '{"success": true, "note": "用 {} 表示占位", "completion_score": 3}'
    got = extract_json_object(text)
    assert got is not None and got["completion_score"] == 3


@pytest.mark.parametrize("dirty", [
    '```json\n{"success": true, "completion_score": 4}\n```',
    '{"success": true, "completion_score": 4,}',
    '{"success": true, "completion_score": 4',
    '评估：{"success": true, "completion_score": 4} 完毕',
])
def test_level1_repairs_common_dirt(dirty):
    """Level 1 只做字符串修复，不调模型。"""

    got = repair_json(dirty)
    assert got is not None and got["success"] is True


def test_plain_text_is_not_forced_into_json():
    """没有 JSON 就是没有 —— 不能编一个出来。"""

    assert repair_json("这个任务完成得不错。") is None


# ── 五级链 ────────────────────────────────────────────

REQUIRED = ("success", "completion_score")


def _gen_from(*replies):
    it = iter(replies)

    def gen(temperature=None):
        try:
            return next(it)
        except StopIteration:
            return replies[-1]

    return gen


def test_level0_clean_json_passes_without_retry():
    r = call_structured(_gen_from('{"success":true,"completion_score":4}'),
                        required=REQUIRED)
    assert r.success and r.degradation_level == 0 and r.attempts == 1


def test_level2_regenerates_at_low_temperature():
    """首次不合法 → 降温重生成。断言**确实又调了一次**。"""

    seen = []

    def gen(temperature=None):
        seen.append(temperature)
        return "不是JSON" if len(seen) == 1 else '{"success":true,"completion_score":5}'

    r = call_structured(gen, required=REQUIRED)
    assert r.success and r.degradation_level == 2 and r.attempts == 2
    assert seen[1] == 0.1, "重生成必须降温，原样重试对确定性失败没有意义"


def test_level3_template_is_marked_degraded():
    tpl = {"success": False, "completion_score": 3.0, "degraded": True}
    r = call_structured(_gen_from("不是JSON", "还不是"), required=REQUIRED, template=tpl)
    assert r.degradation_level == 3 and r.degraded is True


def test_level5_raises_instead_of_returning_a_plausible_default():
    """★ 这是整个 Harness 最重要的一条。

    静默返回一个合法默认值，正是裁判那个 bug 能存活至今的原因：
    它让"没测成"和"测出来是 3.0 分"在数据上无法区分。
    """

    with pytest.raises(AgentHarnessFailure):
        call_structured(_gen_from("纯文本", "纯文本"), required=REQUIRED, template=None)


def test_missing_required_field_is_not_silently_accepted():
    """Level 0 校验：缺字段不能当成功。"""

    r = call_structured(_gen_from('{"completion_score":4}', '{"completion_score":4}'),
                        required=REQUIRED, template=None)
    assert r.success is False and r.degradation_level == 4


def test_schema_validation_reports_every_missing_field():
    missing = validate_schema({"a": 1}, ("a", "b", "c"))
    assert len(missing) == 2


def test_degradation_level_is_carried_in_result():
    """降级等级必须可追溯 —— 否则模板兜底的分数会混进正常统计。"""

    tpl = {"success": False, "completion_score": 3.0}
    r = call_structured(_gen_from("坏", "坏"), required=REQUIRED, template=tpl)
    assert r.to_dict()["degradation_level"] == 3
    assert r.to_dict()["attempts"] == 2
