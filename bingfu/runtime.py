r"""Harness —— Agent 的运行时。

════════════════════════════════════════════════════════════════
 这一层是什么（以及我先前把它理解成了什么）
════════════════════════════════════════════════════════════════

★ Harness 不是"一个 JSON 解析工具"。

  它是**包裹模型的运行时**：模型本身只有"给一段上下文、吐一段文本"
  这一个能力，其余全部由 Harness 提供 ——

    · 上下文的所有权     谁持有对话历史，何时截断，何时重置
    · 工具分发           模型说要调工具，由谁真的去调、结果怎么回灌
    · 结构化输出契约     要求 JSON 时，拿不到合法 JSON 该怎么办
    · 失败分级           瞬时故障重试、确定性失败带错回炉、基础设施故障上抛
    · 预算               token / 墙钟 / 轮次，谁来记账、谁来叫停
    · 可观测             这一次到底发生了什么、花了多少

  先前这些职责**散在 Agent 的四个私有方法里**：
  _run_react、_generate_complete、_execute_tool_call、_build_tool_definitions。
  它们确实存在，只是没有名字、没有边界、无法单独替换或测试。

  而我当时新建的 harness.py 只实现了上面第三条的一个切片，
  并且只有实验的评分裁判在用 —— 框架自身一处都没调。
  那是**实验脚手架**，不是运行时。

════════════════════════════════════════════════════════════════
 为什么要把它抽出来
════════════════════════════════════════════════════════════════

1. **可替换**：换一个模型供应商、换一种工具协议，只动 Harness
2. **可测量**：预算与用量有唯一的记账处，不会像先前那样被重复累加
3. **可测试**：不必构造一个完整 Agent 就能测重试、续写、预算
4. **强制在路径上**：Agent 不再自己发请求，只能经由 Harness ——
   于是"忘了接上某一层"这种事在结构上就不可能发生

★ 第 4 点是本次重建的核心。先前 verify 是 Agent 上的一个可选钩子，
  结果只有实验去填它，普通用户拿到的框架里那一层等于不存在。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from bingfu.llm.base import LLMMessage, LLMResponse, RoleType, ToolDefinition


# ══════════════════════════════════════════════════════════
#  预算
# ══════════════════════════════════════════════════════════

@dataclass
class Budget:
    """一次任务执行的资源上限。

    ★ 三条边界各自独立，且**耗尽时要能说明是哪一条** ——
      轮次用尽、token 用尽、超时，在结果上是不同的事，
      混成一个"失败"就查不出原因。
    """

    #: 一次 ReAct 循环里最多跑几轮。
    #:
    #: ★ 从 5 提到 25，是**对齐**：CrewAI 的 max_iter、LangGraph 的递归上限
    #:   都是 25，PydanticAI 不限。跨框架实测里 5 这个值在
    #:   safety-escape 上真的卡住过 —— 将领连试四次读不到的路径之后
    #:   预算就见底了，任务里那件做得到的事（写 ok.md）始终没轮到，
    #:   10 次里失手 1 次。**够不着不是能力问题，是额度问题。**
    max_iterations: int = 25
    max_tokens: int = 30000
    max_seconds: float = 180.0
    #: 单次生成的输出上限；None = 用供应商默认
    max_output_tokens: Optional[int] = None

    def start(self) -> "BudgetState":
        return BudgetState(budget=self, started_at=time.time())


@dataclass
class BudgetState:
    budget: Budget
    started_at: float
    tokens_used: int = 0
    iterations_used: int = 0

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    def exhausted(self) -> Optional[str]:
        """返回耗尽的那条边界的名字；未耗尽返回 None。"""

        # ★ 叫"思考轮次"而不是"轮次" —— Loop 那边还有一个
        #   "回炉轮次"，两者是不同的东西：
        #     思考轮次 = 内层 思考·行动·观察 的次数
        #     回炉轮次 = 外层 验收不通过后重做的次数
        #   都叫"轮次"的话，看到"轮次用尽"根本不知道是哪一个耗尽了，
        #   而这两种情况该采取的措施完全不同。
        if self.iterations_used >= self.budget.max_iterations:
            return "思考轮次用尽"
        if self.tokens_used >= self.budget.max_tokens:
            return "token 预算耗尽"
        if self.elapsed >= self.budget.max_seconds:
            return "墙钟预算耗尽"
        return None


# ══════════════════════════════════════════════════════════
#  可观测
# ══════════════════════════════════════════════════════════

@dataclass
class Trace:
    """一次执行发生了什么。

    ★ 这不是日志，是**数据**。
      没有它，"门禁触发了吗""工具被调用了吗""续写了几次"
      这些问题只能靠猜 —— 而它们恰恰是判断某一层有没有真的在工作的唯一依据。
    """

    generations: int = 0
    tool_calls: Dict[str, int] = field(default_factory=dict)
    continuations: int = 0
    compactions: int = 0
    tokens: int = 0
    stopped_by: str = ""
    events: List[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.events.append(msg)

    def count_tool(self, name: str) -> None:
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)


# ══════════════════════════════════════════════════════════
#  Harness
# ══════════════════════════════════════════════════════════

#: 续写上限。3 次 ≈ 4 倍单次输出上限，足以覆盖长脚本；
#: 再多通常说明任务本身该拆，而不是继续堆长度。
MAX_CONTINUATIONS = 3


#: 判定「同一种结果」时看的前缀长度。短到能忽略参数差异，
#: 长到不会把不同种类的结果混为一谈。
SHAPE_PREFIX = 6


def _result_shape(tool: str, result: Any) -> str:
    r"""把一次工具结果压成「它属于哪一类」。

    ════════════════════════════════════════════════════
     为什么不能靠错误前缀来判「失败」
    ════════════════════════════════════════════════════

    第一版的判据是 ``result.startswith("[工具执行出错]")``。
    看起来没问题，但跨框架基准里的工具本体返回的是

        "文件不存在：secret.txt"

    —— 一个普通字符串，**不带任何前缀**。于是撞墙检测
    在真实路径上<b>一次都没有触发过</b>，而冒烟测试照样通过
    （那是迭代额度从 5 提到 25 的功劳）。

    ★ 这是「能力存在但到不了」在本项目里的第五次，
      而且发生在**刚写好的代码**里：我给自己定了一个
      工具作者并不遵守的约定，然后按那个约定去检测。
      运行时不该假设工具怎么报错 —— 它管不着别人的返回格式。

    ════════════════════════════════════════════════════
     换成什么
    ════════════════════════════════════════════════════

    判「有没有拿到新信息」，而不是判「有没有报错」：
    把结果压成 (工具名, 结果前 %d 个字符)。
    连着几轮都是同一类结果，说明换参数没换来任何变化 ——
    无论那结果是不是「错误」。

    ★ 这是启发式，会误判两类：
      一是不同参数**碰巧**返回同样开头的成功结果
      （例如读到两个开头相同的文件）——代价只是多一句提示，
      模型可以忽略；二是同一种失败但措辞随参数变化 ——
      代价是没提示，退回到修复前的行为。
      <b>两个方向的代价都不严重，这是选它的理由。</b>
    """ % SHAPE_PREFIX

    text = str(result).strip()
    if not text:
        return ""
    return "%s|%s" % (tool, text[:SHAPE_PREFIX])


class Harness:
    """模型调用的运行时：持有上下文、分发工具、记账、留痕。

    ★ 它**不决定要不要继续**——那是 Loop 的职责。
      Harness 只负责"把一次调用做对"，并把发生的事如实记下来。
      两者分开，是为了让"循环策略"能被单独替换和测试。
    """

    def __init__(
        self,
        llm: Any,
        *,
        budget: Optional[Budget] = None,
        tool_functions: Optional[Dict[str, Callable]] = None,
        tool_definitions: Optional[Sequence[ToolDefinition]] = None,
        temperature: float = 0.7,
        context_threshold: int = 24000,
        on_step: Optional[Callable[[dict], None]] = None,
        ledger: Optional[Any] = None,
        stuck_after: int = 3,
    ) -> None:
        self.llm = llm
        # ★ 跨子任务共享的调用账本；None = 不做冗余熔断。
        #   共享是全部意义所在：实测冗余发生在**不同子任务之间**
        #   （两个子任务各读一遍同样的三个文件），
        #   每个子任务各有一本账等于没做。
        self.ledger = ledger
        # ★ 进度回调埋在**最底层**，不是在循环层。
        #
        #   循环层只看得见「又转了一轮」；只有这里知道调了哪个工具、
        #   参数是什么、返回了什么。而「监控进度」时真正有信息量的
        #   恰恰是后者 —— 一串「第 3 轮…第 4 轮…」等于没说。
        #
        #   回调失败不影响执行：观察者出错是观察者的事。
        self.on_step = on_step
        self.context_threshold = context_threshold
        self.budget = budget or Budget()
        self.tool_functions: Dict[str, Callable] = dict(tool_functions or {})
        self.tool_definitions: List[ToolDefinition] = list(tool_definitions or [])
        self.temperature = temperature

        #: 连着失败几次算「撞墙」。3 次是留出重试与换法的余地，
        #: 又不至于让整轮预算耗在一条走不通的路上。
        self.stuck_after = stuck_after
        #: 连着拿到同一类结果的轮数
        self.repeated_results = 0
        #: 上一次工具结果的「类别」，用于判断有没有拿到新信息
        self.last_shape = ""

        self.messages: List[LLMMessage] = []
        self.trace = Trace()
        self.state = self.budget.start()

    # ── 上下文 ──────────────────────────────────────────

    def reset(self, system_prompt: Optional[str] = None) -> None:
        """清空上下文并重新开始记账。"""

        self.messages = []
        # ★ 重置要连这两个一起，否则跨轮次累计
        self.repeated_results = 0
        self.last_shape = ""
        self.trace = Trace()
        self.state = self.budget.start()
        if system_prompt:
            self.messages.append(
                LLMMessage(role=RoleType.SYSTEM, content=system_prompt)
            )

    def add_user(self, content: str) -> None:
        self.messages.append(LLMMessage(role=RoleType.USER, content=content))

    def add_assistant(self, response: LLMResponse) -> None:
        self.messages.append(
            LLMMessage(
                role=RoleType.ASSISTANT,
                content=response.content or "",
                tool_calls=getattr(response, "tool_calls", None),
            )
        )

    def add_tool_result(self, tool_call: Any, result: str) -> None:
        self.messages.append(
            LLMMessage(
                role=RoleType.TOOL,
                content=result,
                tool_call_id=getattr(tool_call, "id", None),
                name=getattr(tool_call, "name", None),
            )
        )

    # ── 生成 ────────────────────────────────────────────

    def generate(self, *, with_tools: bool = True) -> LLMResponse:
        """发起一次生成，处理长度截断，并记账。

        ★ 截断必须在这里处理，不能交给调用方。

          API 会在响应里明确说"因长度而结束"，而这个信号
          先前全仓库没有任何消费方 —— 后果是模型写到一半的脚本
          被当成最终答案，且因为截断处在语句边界、语法完全合法，
          验收层也抓不到。实测一条任务因此 9 次运行全部失败。

          放在 Harness 里，是因为**任何调用方都可能撞到上限**，
          而没有人应该需要自己记得处理它。
        """

        # ★ 生成前先压缩上下文。
        #
        #   上下文原本是无界增长的：ReAct 每轮把模型回复与工具结果
        #   一路追加，从不回收。实测一次调研任务抓回 68 份资料
        #   共 11.6 万字，全部原样躺在上下文里。
        #
        #   后果是成本随轮次**平方级**增长（第 n 轮要把前 n-1 轮
        #   全部重发），而且没有任何现成信号会告诉你 ——
        #   与 finish_reason 那次不同，这次连一个可读的字段都没有，
        #   你只会观察到成本变高、答案变差。
        self._compact_if_needed()

        kwargs: Dict[str, Any] = {}
        if self.budget.max_output_tokens:
            kwargs["max_tokens"] = self.budget.max_output_tokens

        response = self.llm.generate(
            messages=self.messages,
            tools=(self.tool_definitions or None) if with_tools else None,
            temperature=self.temperature,
            **kwargs,
        )
        self.trace.generations += 1
        self._charge(response)

        # 有工具调用时不续写：工具调用本身是完整的结构化输出
        if getattr(response, "has_tool_calls", False):
            return response

        response = self._continue_if_truncated(response, kwargs)
        return response


    def _compact_if_needed(self) -> None:
        """上下文过长时折叠中段的工具结果。"""

        from bingfu.memory.working import compact

        result = compact(self.messages, threshold=self.context_threshold)
        if result.happened:
            self.messages = result.messages
            self.trace.compactions += 1
            self.trace.note(
                f"上下文压缩：折叠 {result.folded} 条工具结果，"
                f"{result.chars_before} → {result.chars_after} 字符"
            )

    def _continue_if_truncated(
        self, response: LLMResponse, kwargs: Dict[str, Any]
    ) -> LLMResponse:
        if getattr(response, "finish_reason", "") != "length":
            return response

        parts = [response.content or ""]
        while (
            getattr(response, "finish_reason", "") == "length"
            and self.trace.continuations < MAX_CONTINUATIONS
        ):
            self.trace.continuations += 1
            probe = list(self.messages)
            probe.append(
                LLMMessage(role=RoleType.ASSISTANT, content="".join(parts))
            )
            probe.append(
                LLMMessage(
                    role=RoleType.USER,
                    content=(
                        "你上面的回复因长度限制被截断了。"
                        "请**从中断处直接继续**，不要重复已经写过的内容，"
                        "也不要重新开头或加任何说明文字。"
                    ),
                )
            )
            try:
                response = self.llm.generate(
                    messages=probe, tools=None,
                    temperature=self.temperature, **kwargs,
                )
            except Exception as exc:              # noqa: BLE001
                self.trace.note(f"续写失败：{type(exc).__name__}")
                break
            self.trace.generations += 1
            self._charge(response)
            parts.append(response.content or "")

        merged = "".join(parts)
        if self.trace.continuations:
            self.trace.note(f"因长度截断续写 {self.trace.continuations} 次")
            try:
                response.content = merged
            except Exception:                     # noqa: BLE001
                pass
        return response

    def _charge(self, response: LLMResponse) -> None:
        """token 记账。

        ★ 唯一的记账处。先前这件事同时发生在两个地方，
          导致每轮把最后一次调用算两遍 —— 而 token 效率
          是这个框架唯一被实测证实的优势，把它的计量搞坏，
          等于毁掉唯一站得住的结论，且报出的数字只是偏大、不会报错。
        """

        usage = getattr(response, "usage", None) or {}
        n = int(usage.get("total_tokens", 0) or 0)
        self.state.tokens_used += n
        self.trace.tokens += n

    # ── 工具 ────────────────────────────────────────────

    def dispatch_tools(self, response: LLMResponse) -> int:
        """执行模型请求的全部工具调用，把结果回灌进上下文。

        返回执行了几个。工具本身抛异常时，**把异常文本当成观察结果
        回灌给模型**而不是中断执行 —— 报错信息正是模型改正所需的输入。
        """

        calls = getattr(response, "tool_calls", None) or []
        for call in calls:
            name = getattr(call, "name", "")
            fn = self.tool_functions.get(name)
            args = getattr(call, "arguments", None) or {}
            if fn is None:
                result = f"[工具不存在] {name}。可用工具：{sorted(self.tool_functions)}"
            else:
                # ★ 冗余熔断：同一次战役内完全相同的调用不再真的执行。
                #
                #   账本是**跨子任务共享**的 —— 这一点是全部意义所在。
                #   实测冗余全部发生在拆解出的不同子任务之间
                #   （两个子任务各读一遍同样的三个文件），
                #   每个子任务各有一本账等于没做。
                cached = None
                if self.ledger is not None:
                    cached = self.ledger.check(name, args)
                if cached is not None:
                    result = cached
                else:
                    try:
                        result = str(fn(**args) if isinstance(args, dict)
                                     else fn(args))
                    except Exception as exc:      # noqa: BLE001
                        result = f"[工具执行出错] {type(exc).__name__}: {exc}"
                    if self.ledger is not None and not str(result).startswith(
                            ("[工具执行出错]", "[工具不存在]")):
                        self.ledger.record(name, args, result)
            shape = _result_shape(name, result)
            # ★ 数的是「连着几轮拿到同一类结果」，不是「连续失败」。
            #
            #   第一版数的是失败次数，判据是 result 以 [工具执行出错] 开头。
            #   而基准里工具本体返回的是 "文件不存在：X" —— 没有前缀，
            #   于是它在真实路径上**一次都没触发过**。
            #   运行时管不着别人的返回格式，所以不能假设它。
            if shape and shape == self.last_shape:
                self.repeated_results += 1
            else:
                self.repeated_results = 1 if shape else 0
            self.last_shape = shape

            if self.repeated_results >= self.stuck_after:
                # 实测：safety-escape 要求先读工作区外的文件（读不到），
                # 再写一个 ok.md（做得到）。将领连试 ../secret.txt、
                # secret.txt、../、./ 全部失败，然后预算见底，
                # **那件做得到的事始终没轮到**。
                #
                # 它不知道自己在原地打转 —— 每一次结果在上下文里都只是
                # 一条独立信息，没有任何地方说「你已经连着拿到同样的东西了」。
                # 把这句话说出来，才是让它换方向的那个信号。
                #
                # ★ 措辞方向要紧：是「先做做得到的部分」，不是「放弃」。
                #   一个撞墙就放弃整个任务的 agent，比反复撞墙的更糟。
                result = (
                    "%s\n\n[连续 %d 次得到同样的结果] 换参数不会改变它 —— "
                    "这条路走不通。先去完成这个任务里**不依赖它**的部分，"
                    "并在产出中如实说明这一步没做成。"
                    % (result, self.repeated_results))
                # 说过一次就归零，不复读 —— 每轮都提示会把上下文刷爆
                self.repeated_results = 0
                self.last_shape = ""

            failed = str(result).startswith(
                ("[工具执行出错]", "[工具不存在]", "[错误]"))
            self.trace.count_tool(name)
            # ★ 参数与结果都截断后再发 —— 一个 read_file 的返回可能有
            #   两万字符，原样推给界面会把日志刷爆，而进度只需要知道
            #   「做了什么、成没成」。
            args_for_view = getattr(call, "arguments", None) or {}
            self._step(
                "tool", name=name,
                args=str(args_for_view)[:160],
                result=str(result)[:200],
                ok=not failed,
            )
            self.add_tool_result(call, result)
        return len(calls)

    # ── 预算 ────────────────────────────────────────────

    def _step(self, kind: str, **data: Any) -> None:
        if not self.on_step:
            return
        try:
            self.on_step({"kind": kind, **data})
        except Exception:
            pass

    def tick(self) -> None:
        self.state.iterations_used += 1
        self._step("think", iteration=self.state.iterations_used)

    def exhausted(self) -> Optional[str]:
        return self.state.exhausted()
