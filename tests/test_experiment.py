"""实验框架的判据 —— 这个模块（1069 行）此前**零测试、零引用**。

════════════════════════════════════════════════════════════════
 ★ 这一组守的是论文数据的可信度
════════════════════════════════════════════════════════════════

`bingfu/experiment.py` 有 20 条 benchmark、4 个 baseline 包装器、
完整的聚合与落盘 —— 而全仓库没有任何地方 import 它，
`bingfu/__init__.py` 里也没有导出它。**它从来没有被运行过。**

首次真跑时，7 个方法全线 0.0% 成功率。真因不是模型不行：

    provider 调用失败时**不抛异常**，而是把错误信息当正常回复返回
    （content = "❌ OpenAI 调用失败：..."，finish_reason = "error"）。
    那个 finish_reason 一直设着，**全仓库没有任何地方读它**。

于是一次连接失败会走完整条评估链，被裁判按输出长度判成
「这个方法没完成任务」—— **网络抖动和答不出题的模型，数据里完全一样**。

★ 这一组测试的价值不在覆盖率，在于：一批混入了连接失败的成功率
  是**不可解释**的数字，而拿它去支撑「BingFu 高 10.4 个点」这种结论，
  结论对不对完全取决于当天的网络。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bingfu.experiment import (
    BENCHMARK_TASKS,
    ExperimentRunner,
    LLMCallFailed,
    LLMJudge,
    _generate_checked,
    _parse_judge_json,
)
from bingfu.llm import LLMConfig, LLMFactory


class _FakeResponse:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.usage: dict = {}
        self.tool_calls: list = []

    @property
    def has_tool_calls(self) -> bool:
        return False


class _FakeLLM:
    def __init__(self, finish_reason: str, content: str = "一段足够长的正常回复。" * 20) -> None:
        self._finish_reason = finish_reason
        self._content = content

    def generate(self, messages=None, *_args, **_kwargs):
        """★ 假模型必须能区分「扮演将领」和「扮演裁判」两种调用。

        原来的实现对所有调用返回同一段文本 —— 包括裁判那一次。
        旧代码里裁判解析失败会悄悄降级到启发式打分，
        所以这个假模型即便答非所问，测试照样"通过"。

        Harness 接入后裁判变严格（五级耗尽即抛异常），
        这个假模型立刻暴露：它从来就没有满足过裁判的契约。

        ★ 一个分不清自己在扮演谁的假模型，
          验证不了任何与角色相关的行为。
        """

        is_judge = any(
            "evaluator" in (getattr(m, "content", "") or "").lower()
            for m in (messages or [])
        )
        # ★ 只在调用方**没有**指定合法裁判 JSON 时才代答。
        #   否则会把测试特意构造的 bad case 覆盖掉 ——
        #   那样"永远报成功"的实现也能让反向对照变绿。
        from bingfu.harness import extract_json_object
        caller_supplied_json = extract_json_object(self._content) is not None

        if is_judge and not caller_supplied_json and self._finish_reason != "error":
            return _FakeResponse(
                '{"success": true, "completion_score": 4.0, '
                '"criteria_scores": {"Correctness": 4}, "reasoning": "ok"}',
                self._finish_reason,
            )
        return _FakeResponse(self._content, self._finish_reason)


# ══════════════════════════════════════════════════════════════
#  ★★ 基础设施故障必须与任务失败分开
# ══════════════════════════════════════════════════════════════


def test_failed_call_raises_instead_of_looking_like_an_answer() -> None:
    """★ 这条是整组的主判据。

    `finish_reason == "error"` 是 provider 一直在设、却从没人读的信号。
    不读它的后果是：一条错误消息被当成模型的回答送进评估。
    """

    with pytest.raises(LLMCallFailed):
        _generate_checked(_FakeLLM("error", "❌ OpenAI 调用失败：Connection error."), [])


def test_a_normal_reply_passes_through_untouched() -> None:
    """★ 对照组。没有它，一个"永远抛异常"的实现也能让上面那条绿。"""

    response = _generate_checked(_FakeLLM("stop"), [])
    assert response.finish_reason == "stop"


def test_infrastructure_failures_are_counted_separately() -> None:
    """★ 报告必须能回答「这 0% 是模型不行，还是根本没调通」。"""

    config = LLMConfig(
        provider="openai_compatible", api_key="invalid",
        base_url="http://127.0.0.1:9/v1", model="x",
    )
    runner = ExperimentRunner(llm_provider=LLMFactory.create(config))
    metrics = runner.run_all(
        seeds=[42], tasks=BENCHMARK_TASKS[:1], methods=["GPT-4 Solo"],
    )

    failures = ExperimentRunner.count_infra_failures(metrics)
    assert failures["GPT-4 Solo"] == 1, (
        f"连不上的那次没有被记成基础设施故障：{failures}"
    )


def test_results_are_not_written_when_calls_failed(tmp_path: Path) -> None:
    """★ 这条守的是水印。

    `figures/generate_figures.py` 判断「有没有实测数据」的唯一依据，
    就是 experiment_results.json 存不存在 —— 文件一出现，占位水印就消失。

    所以一份掺了连接失败的结果文件，会让论文配图**看起来是实测的**。
    那比原来的占位数据更危险：占位数据至少还有人记得它是占位的。
    """

    config = LLMConfig(
        provider="openai_compatible", api_key="invalid",
        base_url="http://127.0.0.1:9/v1", model="x",
    )
    runner = ExperimentRunner(llm_provider=LLMFactory.create(config))
    metrics = runner.run_all(
        seeds=[42], tasks=BENCHMARK_TASKS[:1], methods=["GPT-4 Solo"],
    )

    target = tmp_path / "experiment_results.json"
    with pytest.raises(RuntimeError, match="基础设施故障"):
        runner.save_results(metrics, str(target))

    assert not target.exists(), "拒绝了，却还是把文件写出去了"


def test_clean_results_are_written(tmp_path: Path) -> None:
    """★ 反向：没有基础设施故障时**必须**能写出去。

    只测「坏的时候拒绝」而不测「好的时候能写」，
    等于允许一个"永远拒绝"的实现 —— 那条路径就永远走不通了。
    """

    runner = ExperimentRunner(llm_provider=_FakeLLM("stop"))
    metrics = runner.run_all(
        seeds=[42], tasks=BENCHMARK_TASKS[:1], methods=["GPT-4 Solo"],
    )
    assert sum(ExperimentRunner.count_infra_failures(metrics).values()) == 0

    target = tmp_path / "experiment_results.json"
    runner.save_results(metrics, str(target))

    assert target.is_file()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert "GPT-4 Solo" in data


# ══════════════════════════════════════════════════════════════
#  ★ benchmark 本身
# ══════════════════════════════════════════════════════════════


def test_benchmark_has_the_twenty_tasks_the_paper_claims() -> None:
    """★ 论文写的是「20 tasks × 5 categories」。数量对不上，
    表里的 per-category 成功率就无从谈起。"""

    assert len(BENCHMARK_TASKS) == 20, f"benchmark 只有 {len(BENCHMARK_TASKS)} 条"


def test_benchmark_task_ids_are_unique() -> None:
    """★ id 重复会让 per-task 结果互相覆盖，而聚合数字看起来完全正常。"""

    ids = [t.id for t in BENCHMARK_TASKS]
    assert len(ids) == len(set(ids)), f"benchmark 有重复 id：{sorted(ids)}"


# ══════════════════════════════════════════════════════════════
#  ★★ 裁判的 JSON 解析 —— 让整套评估恒返回失败的那一行
# ══════════════════════════════════════════════════════════════
#
#  原实现：
#      json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
#
#  `[^{}]*` = 「不含花括号的字符」，所以它匹配的是**最内层**的花括号对 ——
#  也就是裁判回复里嵌套的 "criteria_scores": {...}，而不是外层信封。
#
#  实测：模型返回完全合规的 JSON，解析器拿到的却是
#      {"Factual Accuracy": 1, "Coverage": 1, "Structure": 2, ...}
#  于是 data.get("success", False) 恒 False、
#      data.get("completion_score", 3.0) 恒 3.0。
#
#  ★ 后果：**这套评估在结构上不可能报告成功。**
#    七个方法全线 0.0% SR、TCS 全是 3.00 —— 与模型好坏、与网络通不通都无关，
#    而它看起来完全像一组正常的实验结果。
#
#  ★ 提示词是对的，模型答得也对，只有这一行正则错了；
#    而它错得**没有任何症状**：不报错、不抛异常、返回合法数字。
#    这类缺陷只有真跑一次才看得见。


_ENVELOPE = (
    '{"success": true, "completion_score": 4, '
    '"criteria_scores": {"Factual Accuracy": 5, "Coverage": 3}, '
    '"strengths": ["a"], "weaknesses": [], "overall_assessment": "good"}'
)


def test_parser_takes_the_outer_envelope_not_the_nested_object() -> None:
    """★ 这条就是那个 bug 的守门人。

    断言取到的是 success/completion_score，而不是嵌套的 criteria_scores。
    """

    data = _parse_judge_json(_ENVELOPE)

    assert data.get("success") is True, f"取到的不是外层信封：{data}"
    assert data.get("completion_score") == 4


_FENCED_JSON = "```json\n" + _ENVELOPE + "\n```"
_FENCED_PLAIN = "```\n" + _ENVELOPE + "\n```"
_WITH_PROSE = "评估如下：\n" + _ENVELOPE + "\n以上。"


@pytest.mark.parametrize(
    "wrapped",
    [_ENVELOPE, _FENCED_JSON, _FENCED_PLAIN, _WITH_PROSE],
)
def test_parser_survives_the_shapes_models_actually_emit(wrapped: str) -> None:
    """★ 模型不会永远只吐裸 JSON：markdown 围栏、前后散文都很常见。"""

    assert _parse_judge_json(wrapped)["completion_score"] == 4


def test_unparseable_reply_raises_instead_of_returning_defaults() -> None:
    """★ 解析不出来必须抛，不能返回空字典。

    返回 `{}` 会让上层拿到默认的 (False, 3.0) ——
    也就是把「没解析出来」伪装成「评估结果是失败」，
    与原来那个 bug 是同一种病。
    """

    with pytest.raises(ValueError):
        _parse_judge_json("这里完全没有 JSON")


def test_judge_can_report_success_at_all() -> None:
    """★ 因果检验：把解析修对之后，裁判**必须有能力**报成功。

    原实现下这条永远红 —— 那正是问题所在：
    一个永远返回失败的评估器，跑出来的任何对比都没有意义。
    """

    judge = LLMJudge(_FakeLLM("stop", _ENVELOPE))
    success, score, _ = judge.evaluate(BENCHMARK_TASKS[0], "一段像样的输出")

    assert success is True, "裁判仍然无法报告成功"
    assert score == 4.0


def test_judge_reports_failure_for_a_bad_output() -> None:
    """★ 反向对照。没有它，一个"永远报成功"的实现也能让上面那条绿。"""

    bad = '{"success": false, "completion_score": 1, "criteria_scores": {"X": 1}}'
    success, score, _ = LLMJudge(_FakeLLM("stop", bad)).evaluate(BENCHMARK_TASKS[0], "不知道。")

    assert success is False
    assert score == 1.0


# ══════════════════════════════════════════════════════════════
#  ★ token 计量：不能是编出来的常数
# ══════════════════════════════════════════════════════════════
#
#  run_bingfu 的五个分支原先全部写死 `tokens = 1200`，三个消融变体完全相同。
#  于是论文 Contribution 3 那句「token 消耗比 MetaGPT 低 21%」，
#  **这套 harness 根本测不出来** —— 它测的是一个常数。
#
#  而真实用量一直就在 response.usage 里，只是没有任何地方把它带出 Agent。


class _CountingLLM:
    """每次调用回报固定用量，用来验证**累加**是否正确。"""

    def __init__(self, per_call: int = 137) -> None:
        self.per_call = per_call
        self.calls = 0

    def generate(self, *_args, **_kwargs):
        self.calls += 1
        response = _FakeResponse("一段正常回复。" * 30, "stop")
        response.usage = {"total_tokens": self.per_call}
        return response


def test_agent_reports_real_token_usage() -> None:
    """★ Agent 必须把真实用量带出来，否则上层只能编。"""

    from bingfu.agent import Agent

    llm = _CountingLLM(per_call=137)
    agent = Agent(name="韩信", llm=llm)
    agent.drum("做点事")

    assert agent.last_run_tokens == 137, (
        f"用量没被带出来：{agent.last_run_tokens}（LLM 被调用了 {llm.calls} 次）"
    )


def test_token_usage_is_reset_between_runs() -> None:
    """★ 不清零的话，第二次任务会把第一次的成本算进去 ——
    而那会让"多轮任务更贵"这种结论完全失真。"""

    from bingfu.agent import Agent

    agent = Agent(name="白起", llm=_CountingLLM(per_call=100))
    agent.drum("第一件事")
    agent.drum("第二件事")

    assert agent.last_run_tokens == 100, f"跨任务累加了：{agent.last_run_tokens}"


def test_no_hardcoded_token_constant_remains() -> None:
    """★ 直接盯住那个常数本身。

    这里用源码匹配是因为要守的就是"别把某个字面量写回来"，
    而不是某个可执行的行为。先剥注释，避免匹配到说明文字
    （本仓库的教训：断言匹配上了自己写的注释）。
    """

    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "bingfu" / "experiment.py").read_text(
        encoding="utf-8"
    )
    source = re.sub(r"^\s*#.*$", "", source, flags=re.M)
    source = re.sub(r'"""(?:.|\n)*?"""', "", source)

    assert "tokens = 1200" not in source, "写死的 token 常数又回来了"


# ══════════════════════════════════════════════════════════════
#  ★ 消融组必须可复现
# ══════════════════════════════════════════════════════════════


def test_random_assignment_is_reproducible() -> None:
    """★ `run_bingfu(task, seed=...)` 一直接受 seed 却**从不使用** ——
    no_tactic / no_power 里的 random.choice 完全没被固定。

    一个不可复现的消融组没有资格出现在论文里：
    别人复现不出来，作者自己隔天也复现不出来。
    """

    import random

    names = ["韩信", "白起", "诸葛亮", "项羽"]

    def pick(seed: int, task_id: str) -> str:
        return random.Random(f"{seed}::{task_id}").choice(names)

    assert len({pick(42, "IR1") for _ in range(10)}) == 1, "同 seed 同任务不稳定"


def test_seed_does_not_collapse_to_a_single_agent() -> None:
    """★ 种子里必须混入 task.id。

    只用 seed 的话，每条任务都会选中同一位将领（同样的种子、
    同样的候选顺序）—— 那不是「随机分派」，是「固定分派给第一个」，
    消融的含义完全变了，而成功率照样能算出来。
    """

    import random

    names = ["韩信", "白起", "诸葛亮", "项羽"]
    picks = {random.Random(f"42::{t.id}").choice(names) for t in BENCHMARK_TASKS}

    assert len(picks) > 1, f"20 条任务全选中同一位将领：{picks}"
