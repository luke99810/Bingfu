r"""Loop —— Agent 的控制循环与决策策略。

════════════════════════════════════════════════════════════════
 这一层是什么（以及我先前把它理解成了什么）
════════════════════════════════════════════════════════════════

★ Loop 不是"一个验收门禁"。

  它是**决定下一步做什么**的那个东西：

      思考 → 行动 → 观察 → 验收 → 决策 { 继续 | 完成 | 重做 | 放弃 }

  验收只是决策所依据的**一种信号**，与"模型是否请求了工具"
  "预算还剩多少"并列。先前我把验收单独拎出来做成
  Agent 上的一个可选钩子 verify_fn —— 结果只有实验去填它，
  普通用户拿到的框架里那一层等于不存在。

★ 把决策集中到一处，还解决了另一个问题：

  先前"要不要继续"的逻辑散在三个地方 ——
  ReAct 内层看 has_tool_calls、外层看 verify 结果、
  再另有三处 break 检查预算。它们互相不知道对方的存在，
  于是"回炉两轮后预算耗尽"和"一次过"在结果上无法区分。

  现在所有出口都经由 Decision，且每个出口都带**为什么**。

════════════════════════════════════════════════════════════════
 为什么验收要默认开启
════════════════════════════════════════════════════════════════

一个默认关闭的验收层，与不存在的验收层，对绝大多数使用者是同一件事。
实测数据支持这个判断：基准里将领拿到的工具列表为空，
执行循环因此恒定只跑一轮 —— 而那正是"某个能力挂在那里但没人接"
的典型后果。

所以这里的默认值是**开**，由 RoutePlan 按任务类型下调，
而不是默认关、等着谁来开。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Protocol

from bingfu.runtime import Harness


class Outcome(str, Enum):
    """循环的终止方式。

    ★ 必须区分这些，而不是笼统的"成功/失败"：
      "验收通过"和"预算耗尽只好交付"都会返回产出，
      但它们对调用方意味着完全不同的事。
    """

    VERIFIED = "verified"          # 通过验收
    UNVERIFIED = "unverified"      # 没有验收策略，直接交付
    EXHAUSTED = "exhausted"        # 预算耗尽，交付当前最好结果
    FAILED = "failed"              # 无法产出任何结果


@dataclass
class LoopResult:
    output: str
    outcome: Outcome
    stopped_by: str = ""
    revisions: int = 0
    verify_reasons: List[str] = field(default_factory=list)
    verify_checks: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.outcome in (Outcome.VERIFIED, Outcome.UNVERIFIED)


class VerifyPolicy(Protocol):
    """验收策略：拿到产出，判定是否可交付。

    ★ 返回值必须能给出**可回炉的具体原因**。
      "质量较差"对下一轮没有任何指导作用；
      "第 12 行括号未闭合"才有。
    """

    def __call__(self, output: str) -> Any: ...


@dataclass
class LoopPolicy:
    """循环的行为策略。

    把这些参数集中在一个对象里，而不是散成 Agent 的一堆字段，
    是为了让"这一类任务该怎么跑"可以被整体传递、整体替换 ——
    Graph 层的路由表正是产出这样一个对象。
    """

    #: 验收策略。None 表示不验收（例如推理类：找不到机械可判的断言）
    verify: Optional[VerifyPolicy] = None
    #: 验收不通过时最多回炉几次
    max_revisions: int = 1
    #: 每轮是否允许工具调用
    allow_tools: bool = True


class AgentLoop:
    """思考 → 行动 → 观察 → 验收 → 决策。

    ★ 与 Harness 的分工：
        Harness 负责"把一次调用做对"（上下文、工具、截断、记账）
        Loop    负责"要不要再来一次"（终止条件、验收、回炉）

      分开的收益是可替换：换一种循环策略（例如加入反思、加入
      多候选投票）不必碰运行时；换一个模型供应商不必碰循环。
    """

    def __init__(self, harness: Harness, policy: Optional[LoopPolicy] = None) -> None:
        self.harness = harness
        self.policy = policy or LoopPolicy()

    # ── 主入口 ──────────────────────────────────────────

    def run(self, task: str, *, system_prompt: Optional[str] = None) -> LoopResult:
        h = self.harness
        h.reset(system_prompt)
        h.add_user(task)

        best = ""
        reasons: List[str] = []
        checks: List[str] = []

        for revision in range(self.policy.max_revisions + 1):
            output = self._react()
            best = output or best

            if self.policy.verify is None:
                h.trace.stopped_by = "无验收策略"
                return LoopResult(best, Outcome.UNVERIFIED, "无验收策略", revision)

            verdict = self.policy.verify(output)
            passed = bool(getattr(verdict, "passed", True))
            reasons = list(getattr(verdict, "reasons", []))
            checks = list(getattr(verdict, "checks_run", []))

            if passed:
                h.trace.stopped_by = "验收通过"
                return LoopResult(best, Outcome.VERIFIED, "验收通过", revision,
                                  reasons, checks)

            # 未通过：先看还能不能再来一次
            if revision >= self.policy.max_revisions:
                stopped = "回炉轮次用尽"
                break
            spent = h.exhausted()
            if spent:
                stopped = spent
                break

            # ★ 带**具体原因**回炉，而不是原样重试。
            #   对确定性失败（语法错、数字无出处），原样重试
            #   一百次还是同样的错，只是把成本乘以一百。
            feedback = getattr(verdict, "feedback", None)
            h.add_user(feedback() if callable(feedback) else str(verdict))
            h.trace.note(f"第 {revision + 1} 次回炉：{reasons[:1]}")
        else:
            stopped = "回炉轮次用尽"

        h.trace.stopped_by = stopped
        outcome = Outcome.EXHAUSTED if best else Outcome.FAILED
        return LoopResult(best, outcome, stopped,
                          self.policy.max_revisions, reasons, checks)

    # ── 内层：思考·行动·观察 ────────────────────────────

    def _react(self) -> str:
        """跑到模型给出文本回复，或预算耗尽。"""

        h = self.harness
        response = None

        while True:
            spent = h.exhausted()
            if spent:
                h.trace.note(f"内层停止：{spent}")
                return self._force_finish(response, spent)

            h.tick()
            response = h.generate(with_tools=self.policy.allow_tools)

            if getattr(response, "has_tool_calls", False):
                h.add_assistant(response)
                h.dispatch_tools(response)
                continue

            h.add_assistant(response)
            return response.content or ""

    def _force_finish(self, response: Any, reason: str) -> str:
        """预算耗尽时，逼模型把已有材料收成答案。

        ★ 不能直接返回最后一条回复。

          预算耗尽时模型多半正处在"还在调工具"的状态，
          那一轮的正文往往是空的或一句过渡话。实测：一条调研任务
          调用了 22 次工具、攒下 11.6 万字资料，然后轮次耗尽，
          **产出只有 88 个字符** —— 全部工具调用的成果连同成本一起作废。

        ★ 这使得"开工具"可能比"不开工具"结果更差：
          不开工具时模型一轮就给出完整回答；开了却收不了尾，
          就什么都没有。一个只在长任务上触发的静默失效。
        """

        h = self.harness
        if response is None or not getattr(response, "has_tool_calls", False):
            return (getattr(response, "content", "") or "") if response else ""

        h.add_user(
            "你已达到可用的资源上限，不能再调用任何工具。"
            "请立即基于**已经获取到的信息**给出完整的最终答案。"
            "不要再说'我将要'或'接下来'，直接输出成果本身。"
        )
        try:
            final = h.generate(with_tools=False)
            h.add_assistant(final)
            h.trace.note(f"因{reason}强制收束")
            return final.content or ""
        except Exception as exc:                  # noqa: BLE001
            h.trace.note(f"强制收束失败：{type(exc).__name__}")
            return getattr(response, "content", "") or ""
