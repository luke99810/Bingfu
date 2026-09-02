"""长度截断的自动续写。

★ 这一条修的是本项目里最隐蔽的一个失效。

  基准任务 C3（要求一个含读取、清洗、统计、可视化、导出五个环节
  的数据管线脚本）在**全部 9 次运行中零成功**，分数每次都精确
  等于 2.0 —— 三个档位、三个 seed、有工具无工具，无一例外。

  裁判的理由是："脚本不完整，可视化函数中途被截断，
  JSON 导出步骤完全缺失。"

  原因是 max_tokens 默认 2048，脚本写不完。对照实验：

      max_tokens=2048 → finish_reason='length'，停在 logger.info(f"Removed {
      max_tokens=8192 → finish_reason='stop'，25731 字符，完整

★ 最值得记的是：**API 每一次都明确说了它被截断了**。
  finish_reason 就在响应里，值是 'length'。
  而全仓库只检查它等不等于 'error'，从不看 'length'。

★ 验收层也抓不到：在语句边界处截断的代码语法完全合法，
  ast.parse 照样通过。只查语法的断言对"写了一半"是盲的。

  于是症状是：产出看起来正常、语法正确、长度可观，
  只是缺了最后两个功能 —— 没有任何一层报错。
"""

import pytest

from bingfu.agent import Agent


class _Chunked:
    """按段返回，最后一段之前 finish_reason 都是 'length'。"""

    def __init__(self, *chunks):
        self.chunks = list(chunks)
        self.n = 0
        self.prompts = []

    def generate(self, messages, **kwargs):
        self.prompts.append(messages[-1].content or "")
        chunk = self.chunks[min(self.n, len(self.chunks) - 1)]
        self.n += 1
        is_last = self.n >= len(self.chunks)

        class R:
            content = chunk
            finish_reason = "stop" if is_last else "length"
            usage = {"total_tokens": 100}
            tool_calls = []
            has_tool_calls = False

        return R()


class _AlwaysTruncated:
    def __init__(self):
        self.n = 0

    def generate(self, messages, **kwargs):
        self.n += 1

        class R:
            content = "x"
            finish_reason = "length"
            usage = {"total_tokens": 10}
            tool_calls = []
            has_tool_calls = False

        return R()


def test_truncated_output_is_continued():
    """被截断的产出必须接着写完，而不是当成最终答案。"""

    llm = _Chunked("def f():\n    a = 1\n", "    b = 2\n", "    return a + b\n")
    agent = Agent(name="测试", role="将军", llm=llm)
    out = agent.execute("写个函数")

    assert "a = 1" in out and "b = 2" in out and "return a + b" in out
    assert agent._last_continuations == 2


def test_continuation_prompt_forbids_repeating():
    """续写提示必须明确要求"从中断处继续、不要重复"。

    否则模型会从头再写一遍，合并后得到重复的内容 ——
    那比截断更糟，因为它看起来是完整的。
    """

    llm = _Chunked("前半", "后半")
    Agent(name="测试", role="将军", llm=llm).execute("任务")
    assert "不要重复" in llm.prompts[1]
    assert "继续" in llm.prompts[1]


def test_no_continuation_when_not_truncated():
    """没被截断就不该多发请求 —— 续写是修复手段，不是常规路径。"""

    llm = _Chunked("完整回答")
    agent = Agent(name="测试", role="将军", llm=llm)
    agent.execute("任务")
    assert llm.n == 1
    assert agent._last_continuations == 0


def test_continuation_is_bounded():
    """★ 续写必须有上界。

    模型可能永远返回 'length'（例如被要求写一本书）。
    没有上界的话，一次调用会无限烧钱下去。
    """

    from bingfu.runtime import MAX_CONTINUATIONS

    llm = _AlwaysTruncated()
    agent = Agent(name="测试", role="将军", llm=llm)
    agent.execute("任务")
    # ★ 续写上限现在属于 Harness（运行时），不再是 Agent 的类属性。
    #   这正是重建要达到的效果：截断处理是运行时的职责，
    #   任何跑在 Harness 上的调用方都自动获得它，
    #   而不需要 Agent 自己记得实现一遍。
    assert agent._last_continuations == MAX_CONTINUATIONS
    assert llm.n == MAX_CONTINUATIONS + 1


def test_tokens_from_continuations_are_counted():
    """续写的 token 必须计入总量 —— 否则成本被系统性低估。"""

    llm = _Chunked("一", "二", "三")
    agent = Agent(name="测试", role="将军", llm=llm)
    agent.execute("任务")
    assert agent.last_run_tokens == 300, "三次调用各 100，应累加"


# ══════════════════════════════════════════════════════════
#  裁判侧的截断防护
# ══════════════════════════════════════════════════════════

class _FinishReason:
    """按指定 finish_reason 返回一段**残缺**的 JSON。"""

    def __init__(self, reason):
        self.reason = reason

    def generate(self, *args, **kwargs):
        reason = self.reason

        class R:
            finish_reason = reason
            content = '{"success": true, "completion_score": 4'   # 少了右括号
            usage = {}

        return R()


def test_judge_rejects_truncated_response():
    """★ 与 Agent 那边是同一个 bug 的两个位置。

    裁判的回复是 JSON，max_tokens=1024。一旦被截断，
    JSON 就是残缺的 → 解析失败 → 落进 except Exception
    → **静默降级成启发式打分**（按输出长度给分）。

    于是报告上照样有一整套成功率数字，而实际用的评分方法
    与论文所述的 LLM-as-judge 不是一回事，
    且没有任何地方说明发生过降级。
    """

    from bingfu.experiment import LLMCallFailed, _generate_checked

    with pytest.raises(LLMCallFailed) as exc:
        _generate_checked(_FinishReason("length"))
    assert "截断" in str(exc.value)


def test_judge_still_rejects_error():
    from bingfu.experiment import LLMCallFailed, _generate_checked

    with pytest.raises(LLMCallFailed):
        _generate_checked(_FinishReason("error"))


@pytest.mark.parametrize("reason", ["stop", "tool_calls"])
def test_judge_allows_normal_finish_reasons(reason):
    """正常结束的不能误拦 —— 否则每次评分都变成基础设施故障。"""

    from bingfu.experiment import _generate_checked

    assert _generate_checked(_FinishReason(reason)) is not None


# ══════════════════════════════════════════════════════════
#  输出上限必须真的传到 provider
# ══════════════════════════════════════════════════════════

class _MaxTokensSpy:
    def __init__(self):
        self.seen = []
        self.n = 0

    def generate(self, messages, **kwargs):
        self.seen.append(kwargs.get("max_tokens"))
        self.n += 1
        n = self.n

        class R:
            content = "片段"
            finish_reason = "stop" if n >= 2 else "length"
            usage = {"total_tokens": 10}
            tool_calls = []
            has_tool_calls = False

        return R()


def test_output_limit_reaches_provider():
    """★ 这类"配置字段"最容易变成摆设：定义了、赋值了，就是没传下去。

    实测过一次工具 schema 的同类问题：定义构建成功、请求发送成功，
    唯独参数表是空的，而每一步都不报错。
    """

    spy = _MaxTokensSpy()
    Agent(name="测试", role="将军", llm=spy, max_output_tokens=8192).execute("任务")
    assert spy.seen[0] == 8192


def test_output_limit_applies_to_continuations_too():
    """续写也要带上限 —— 否则续写段又会被默认的 2048 截断。"""

    spy = _MaxTokensSpy()
    Agent(name="测试", role="将军", llm=spy, max_output_tokens=8192).execute("任务")
    assert spy.seen == [8192, 8192]


def test_unset_limit_defers_to_provider_default():
    """不设置时不该硬塞一个值 —— 让 provider 的配置生效。"""

    spy = _MaxTokensSpy()
    Agent(name="测试", role="将军", llm=spy).execute("任务")
    assert spy.seen[0] is None
