"""
Agent module (智能体模块)
Defines the Agent class with ancient Chinese warfare naming.

v0.4.0: 新增 LLM 驱动执行、ReAct 工具调用循环
"""

import json
import re
import time
from typing import Any, Callable, ClassVar, Dict, List, Optional
from pydantic import BaseModel, Field, PrivateAttr

from bingfu.llm.base import (
    LLMProvider, LLMResponse, LLMMessage, ToolDefinition,
    ToolCall, RoleType
)
from bingfu.profile import GeneralProfile


class Agent(BaseModel):
    """
    Agent (智能体 / 将领) — 代表一个可执行任务的智能体。

    每个 Agent 就像一员将领，可以：
    1. 接收军令（任务描述）
    2. 调用 LLM 理解任务并制定策略
    3. 使用兵器（工具）执行具体操作
    4. 多轮思考-行动循环（ReAct 模式）直到任务完成

    兵法云：将者，智、信、仁、勇、严也。
    """

    name: str = Field(..., description="Agent 名称（将领名号）")
    role: Optional[str] = Field(default=None, description="Agent 角色/职位")
    description: Optional[str] = Field(default=None, description="Agent 描述")
    profile: Optional[GeneralProfile] = Field(default=None,
                                              description="将军战力档案（专长/弱项/战力值）")

    # 内部状态 (内部状态)
    is_active: bool = Field(default=False, description="是否在线")
    memory: Optional[Any] = Field(default=None, description="Agent 记忆（军需库）")
    tools: list[Any] = Field(default_factory=list, description="可用工具（兵器谱）")

    # LLM 相关 (军师谋略)
    llm: Optional[Any] = Field(default=None, description="LLM Provider 实例（军师）")
    system_prompt: Optional[str] = Field(default=None, description="系统提示（将令）")
    #: ReAct 最大循环轮数。
    #:
    #: ★ 从 5 提到 25，与 CrewAI 的 max_iter、LangGraph 的递归上限对齐
    #:   （PydanticAI 不限）。跨框架实测里 5 真的卡住过：safety-escape
    #:   要求「先读一个读不到的文件，再写一个写得了的文件」，
    #:   将领连试四条路径失败后预算见底，那件做得到的事没轮到。
    #:
    #: ★ 这个值会**覆盖** Budget 的默认值 —— 只改 Budget 不改这里等于没改，
    #:   有一条测试盯着两者一致。
    max_iterations: int = Field(default=25, description="ReAct 最大循环轮数")

    # ── VERIFY 门禁与预算熔断 ──────────────────────────────
    #
    # ★ 为什么需要：benchmark 里将领拿到的 tools=None，
    #   于是 has_tool_calls 恒为 False，上面那个 max_iterations 循环
    #   **只跑一轮就 return** —— 让模型一次性写出带 JWT、WebSocket、
    #   异步数据库的全栈应用，不执行、不检查、不重试。
    #   实测 Code 类 12%、Write 类 38%。
    #
    #   这不是模型能力的问题，是没有验收环节的必然结果。
    verify_fn: Optional[Any] = Field(
        default=None,
        description="验收函数 (output) -> VerifyResult；None 表示不做门禁",
    )
    max_revisions: int = Field(
        default=2,
        description="验收不通过时最多回炉几次。0 = 只验不改（用于观测）",
    )
    max_wall_seconds: float = Field(
        default=180.0,
        description="单次 execute 的墙钟预算 —— 回炉不能无限拖下去",
    )
    max_total_tokens: int = Field(
        default=30000,
        description="单次 execute 的 token 预算，含所有回炉轮次",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        description="单次生成的输出上限；None = 用 provider 配置的默认值（2048）",
    )

    # ── 记忆 ──────────────────────────────────────────────
    #
    # ★ 原本 Agent 上有一个 memory 字段，**声明了却从未被读写** ——
    #   每次 execute 都清空上下文重来，将领打完一仗什么都不留下。
    #
    #   这两个字段是把记忆真正接进执行路径的入口。
    episodic: Optional[Any] = Field(
        default=None,
        description="战报库（EpisodicMemory）；None = 不记录",
    )
    knowledge: Optional[Any] = Field(
        default=None,
        description="知识库（KnowledgeBase）；None = 不跨任务复用知识",
    )
    category: str = Field(
        default="",
        description="任务类别，用于按类回放战报与选择策略",
    )
    recall_history: bool = Field(
        default=True,
        description="是否把同类任务的历史结果放进提示词",
    )
    ledger: Optional[Any] = Field(
        default=None,
        description="跨子任务共享的调用账本（CallLedger）；None = 不做冗余熔断",
    )

    # 内部对话历史
    _conversation: List[LLMMessage] = PrivateAttr(default_factory=list)
    # 最近一次 execute() 消耗的 token 总量
    _last_run_tokens: int = PrivateAttr(default=0)
    # 最近一次 execute() 的验收轨迹 —— 用了几轮、每轮为什么没过
    _last_verify_trace: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    # 工具函数映射
    _tool_functions: Dict[str, Callable] = PrivateAttr(default_factory=dict)
    # 随身兵器 —— 不随任务类别改变的常备工具
    #
    # ★ 为什么要和 _tool_functions 分开：
    #
    #   编排层 `_configure` 会按子任务类别**清空并重配**工具
    #   （给写作任务塞 Python 解释器确实没用，那个设计是对的）。
    #   但文件读写不是「某一类任务的附加能力」，而是每一类都要用的
    #   随身装备 —— 写作任务同样要读素材、要把稿子落盘。
    #
    #   此前没有这个区分，后果是：调用方给将领装的工具，
    #   一进编排就被卸干净，而外面看不出来 —— 将领照常回话，
    #   只是从未碰过任何文件。
    _base_tool_functions: Dict[str, Callable] = PrivateAttr(default_factory=dict)
    _base_tool_descriptions: Dict[str, str] = PrivateAttr(default_factory=dict)
    # 进度回调 —— 每一轮思考、每一次工具调用都会回调一次
    #
    # ★ 用私有属性而不是 Field：它是一个回调函数，不该进模型的
    #   序列化/校验路径，也不该出现在 dict() 里。
    _progress_cb: Optional[Callable] = PrivateAttr(default=None)
    # 最近一次执行的运行时（供调试与指标读取）
    _harness: Optional[Any] = PrivateAttr(default=None)
    _last_continuations: int = PrivateAttr(default=0)
    _last_finish_reason: str = PrivateAttr(default="")

    class Config:
        arbitrary_types_allowed = True

    def drum(self, task: str) -> str:
        """
        击鼓 — 启动 Agent 执行任务

        Args:
            task (str): 任务描述

        Returns:
            str: 执行结果
        """
        self.is_active = True
        if self.llm:
            result = self.execute(task)
            return result
        else:
            return f"🥁 Drum! Agent '{self.name}' 已激活（无 LLM，仅占位模式）。Task: {task}"

    def gong(self) -> str:
        """
        鸣金 — 停止 Agent

        Returns:
            str: 停止消息
        """
        self.is_active = False
        self._conversation.clear()
        return f"🔔 Gong! Agent '{self.name}' 已停止。"

    def add_tool(self, tool: Any, func: Optional[Callable] = None) -> None:
        """
        添加工具到兵器谱

        Args:
            tool: 工具对象（需要有 name 和 description 属性）
            func: 可选的执行函数，如果 tool 本身不可调用
        """
        self.tools.append(tool)
        # 注册工具函数
        if func:
            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
            self._tool_functions[tool_name] = func
        elif hasattr(tool, 'function') and callable(tool.function):
            self._tool_functions[tool.name] = tool.function
        elif callable(tool):
            name = getattr(tool, 'name', tool.__name__)
            self._tool_functions[name] = tool

    def remove_tool(self, tool: Any) -> None:
        """移除工具"""
        if tool in self.tools:
            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
            self.tools.remove(tool)
            self._tool_functions.pop(tool_name, None)

    def clear_tools(self) -> None:
        """清空兵器谱。

        ★ 补这个方法是因为 `examples/tool_usage.py` 一直在用它，
          而它从来不存在 —— 那个示例跑到第 210 行必然 AttributeError。
          示例代表的是使用者预期的 API：有 add_tool / remove_tool
          却没有 clear_tools，本来就是个缺口。

        ★ 工具列表与函数映射必须**一起**清 —— 只清 tools 会留下
          一批仍能被 LLM 调用、却不在兵器谱里的幽灵工具。
        """

        self.tools.clear()
        self._tool_functions.clear()

    def register_tool_function(self, name: str, func: Callable, description: str = "") -> None:
        """
        直接注册工具函数（更灵活的方式）

        Args:
            name: 工具名称
            func: 执行函数
            description: 工具描述
        """
        self._tool_functions[name] = func

    def register_base_tool(self, name: str, func: Callable, description: str = "") -> None:
        """注册**随身兵器** —— 不会被编排层按类别重配时卸掉的工具。

        普通 `register_tool_function` 注册的工具，在多智能体编排里会被
        `_configure` 按子任务类别清空重配；随身兵器则每次重配后都会
        重新装上。

        ★ 什么该放这里：文件读写、目录查找这类**每一类任务都要用**的能力。
          什么不该：Python 解释器、网页抓取这类与任务类别强相关的，
          交给类别路由表去决定 —— 一律给全会让上下文暴涨，
          实测检索类任务的 token 会涨十几倍。
        """

        self._base_tool_functions[name] = func
        if description:
            self._base_tool_descriptions[name] = description
        self._tool_functions[name] = func

    def set_progress_callback(self, cb: Optional[Callable]) -> None:
        """设置进度回调。

        回调收到的是字典：
            {"kind": "think", "iteration": 2}
            {"kind": "tool", "name": "read_file", "args": "...", "result": "...", "ok": True}

        ★ 回调会在**工作线程**里被调用（同层子任务并行执行）。
          要动界面的话，自己负责切回主线程 —— 这一层不替调用方做，
          因为它不知道调用方用的是哪套 UI。
        """

        self._progress_cb = cb

    def rearm_base_tools(self) -> None:
        """把随身兵器重新装上。

        供编排层在按类别重配工具之后调用 —— 顺序要紧：
        先配类别工具再装随身兵器，同名时以随身兵器为准。
        """

        for name, func in self._base_tool_functions.items():
            self._tool_functions[name] = func

    @property
    def base_tool_descriptions(self) -> Dict[str, str]:
        return dict(self._base_tool_descriptions)

    def execute(self, task: str, tools: Optional[list[Any]] = None) -> str:
        """
        执行任务（核心方法）

        如果绑定了 LLM，进入 ReAct 循环：
        1. LLM 思考 → 选择行动（可能调用工具）
        2. 执行工具 → 将结果反馈给 LLM
        3. 重复直到 LLM 给出最终回复或达到最大轮数

        如果没有 LLM，返回占位响应。

        Args:
            task: 任务描述
            tools: 额外工具列表

        Returns:
            str: 执行结果
        """
        if not self.llm:
            # 无 LLM，占位模式
            tools_to_use = tools if tools is not None else self.tools
            tool_names = [t.name if hasattr(t, 'name') else str(t) for t in tools_to_use]
            result = f"Agent '{self.name}' executing task: {task}\n"
            if tool_names:
                result += f"Using tools: {', '.join(tool_names)}\n"
            result += "⚠️ 无 LLM 绑定，任务未真正执行。请配置 LLM Provider 以启用智能执行。"
            return result

        # ══════════════════════════════════════════════════
        #  跑在 Harness + Loop 上
        # ══════════════════════════════════════════════════
        #
        # ★ 这里曾经是 Agent 自己实现的一整套运行时：
        #   自己拼上下文、自己发请求、自己分发工具、自己数轮次，
        #   而验收是一个可选钩子 verify_fn —— 结果只有实验去填它，
        #   普通用户拿到的框架里那一层等于不存在。
        #
        # ★ 现在 Agent 只负责**身份**（名号、画像、系统提示、工具清单），
        #   机制全部交给 Harness（运行时）与 Loop（决策）。
        #
        #   收益不只是分层好看：Agent 不再持有发请求的能力，
        #   于是"某一层忘了接上"这种事在结构上就不可能发生。
        from bingfu.runtime import Budget, Harness
        from bingfu.loop import AgentLoop, LoopPolicy

        harness = Harness(
            self.llm,
            budget=Budget(
                max_iterations=self.max_iterations,
                max_tokens=self.max_total_tokens,
                max_seconds=self.max_wall_seconds,
                max_output_tokens=self.max_output_tokens,
            ),
            ledger=self.ledger,
            tool_functions=dict(self._tool_functions),
            tool_definitions=self._build_tool_definitions(tools),
            on_step=self._wrap_progress(),
        )
        policy = LoopPolicy(
            verify=self.verify_fn,
            max_revisions=self.max_revisions,
            allow_tools=True,
        )

        # ★ 把同类任务的历史结果放进提示词。
        #
        #   只放**结果与教训**，不放完整产出：把上次的全文塞回去
        #   既贵，又会让模型倾向于复制上次的答案 ——
        #   而如果上次是错的，那正是最不该复制的东西。
        prompt = self.system_prompt or self._default_system_prompt()
        if self.episodic is not None and self.recall_history:
            from bingfu.memory import summarize
            recap = summarize(self.episodic, category=self.category)
            if recap:
                prompt = prompt + "\n\n" + recap

        import time as _time
        _t0 = _time.time()
        result = AgentLoop(harness, policy).run(task, system_prompt=prompt)

        # 把运行时的账本与轨迹映射回 Agent 的既有对外字段
        self._harness = harness
        self._last_run_tokens = harness.trace.tokens
        self._last_continuations = harness.trace.continuations
        self._last_finish_reason = ""
        self._conversation = harness.messages
        self._last_verify_trace = [{
            "revision": result.revisions,
            "passed": result.outcome.value == "verified",
            "reasons": list(result.verify_reasons),
            "checks": list(result.verify_checks),
            **({"stopped_by": result.stopped_by}
               if result.outcome.value in ("exhausted", "failed") else {}),
        }] if self.verify_fn else []

        # ★ 留一条战报。
        #
        #   记**结果与代价**，而不只是"做过"。只记做过的话，
        #   回放时无法区分"这条路走通了"和"这条路走死了" ——
        #   而那恰恰是历史唯一有用的地方。
        #
        #   success 留空（None）而不是猜一个：这里没有裁判，
        #   Loop 的 outcome 只能说明"有没有通过机械验收"，
        #   不能说明任务是否真的完成。把两者混同，
        #   会让历史分建立在一个它没有资格下的判断上。
        if self.episodic is not None:
            from bingfu.memory import Episode
            self.episodic.record(Episode(
                task=task[:200],
                agent_name=self.name,
                category=self.category,
                success=(True if result.outcome.value == "verified" else None),
                tokens=harness.trace.tokens,
                elapsed=round(_time.time() - _t0, 2),
                outcome=result.outcome.value,
                stopped_by=result.stopped_by,
                revisions=result.revisions,
                tool_calls=dict(harness.trace.tool_calls),
            ))

        return result.output

    # ★ 这里原本有 _generate_complete 与 _run_react 两个方法，
    #   共约 190 行，实现了一整套"Agent 自己的运行时"：
    #   拼上下文、发请求、处理截断、分发工具、数轮次。
    #
    #   它们已被 Harness（bingfu/runtime.py）与 Loop（bingfu/loop.py）取代。
    #   删掉而不是留着，是因为**两套执行路径共存比一套坏的更危险**：
    #   读者无法判断哪一套在生效，改错地方也不会有任何报错。

    def _default_system_prompt(self) -> str:
        """生成默认系统提示"""
        prompt = f"你是将领「{self.name}」"
        if self.role:
            prompt += f"，职位「{self.role}」"
        if self.description:
            prompt += f"。{self.description}"

        # 注入战力档案信息
        if self.profile:
            p = self.profile
            prompt += f"\n\n【战力档案】作战风格：{p.style.value}"
            if p.specialties:
                prompt += f"\n专长领域：{'、'.join(p.specialties)}"
            if p.weaknesses:
                prompt += f"\n弱项领域：{'、'.join(p.weaknesses)}"
            prompt += f"\n五维战力：{p.stats.summary()}"
            prompt += f"\n{self.name}应发挥专长、规避弱项，选择最适合的策略执行任务。"

        prompt += "。你是一个古代军事风格的智能体，用中文回复，风格简练有力如军令。"
        prompt += "\n\n你可以使用提供的工具来完成任务。思考-行动-观察循环执行，直到给出最终结论。"
        return prompt

    def _wrap_progress(self) -> Optional[Callable]:
        """把将领名字附到进度事件上。

        ★ 多个将领并行时，「调用了 read_file」这句话没有名字就没有意义 ——
          看不出是谁在做，也就无从判断进度。
        """

        cb = self._progress_cb
        if cb is None:
            return None

        def _wrapped(ev: dict) -> None:
            cb({**ev, "agent": self.name, "category": self.category or ""})

        return _wrapped

    def _build_tool_definitions(self, extra_tools: Optional[list] = None) -> List[ToolDefinition]:
        """构建 LLM 可见的工具定义列表"""
        definitions = []
        all_tools = list(self.tools)
        if extra_tools:
            all_tools.extend(extra_tools)

        for tool in all_tools:
            if isinstance(tool, ToolDefinition):
                definitions.append(tool)
            elif hasattr(tool, 'name') and hasattr(tool, 'description'):
                # 从 Tool 对象构建定义
                params = getattr(tool, 'parameters', {
                    "type": "object",
                    "properties": {},
                })
                definitions.append(ToolDefinition(
                    name=tool.name,
                    description=tool.description or f"工具: {tool.name}",
                    parameters=params if isinstance(params, dict) else {"type": "object", "properties": {}}
                ))

        # 注册的工具函数也需要定义
        for name, func in self._tool_functions.items():
            # 避免重复
            if any(d.name == name for d in definitions):
                continue
            doc = (func.__doc__ or f"工具: {name}").strip()
            definitions.append(ToolDefinition(
                name=name,
                description=doc.split('\n')[0],
                parameters=_schema_from_signature(func),
            ))

        return definitions

    def chat(self, message: str) -> str:
        """
        继续对话（多轮交互）

        Args:
            message: 用户消息

        Returns:
            str: Agent 回复
        """
        if not self.llm:
            return f"⚠️ Agent '{self.name}' 无 LLM 绑定，无法对话。"

        self._conversation.append(LLMMessage(
            role=RoleType.USER,
            content=message
        ))

        response = self.llm.generate(
            messages=self._conversation,
            temperature=0.7,
        )

        self._conversation.append(LLMMessage(
            role=RoleType.ASSISTANT,
            content=response.content,
        ))

        return response.content or ""

    @property
    def last_run_tokens(self) -> int:
        """最近一次 execute()/drum() 消耗的 token 总量（跨 ReAct 多轮累计）。

        ★ 0 表示**没有测到**，而不是"没消耗" —— 无 LLM 的占位模式、
          或 provider 没回 usage 时都会是 0。调用方要能区分这两件事，
          所以这里不做任何兜底猜测（比如 len(output)//4 那种估算）。
        """

        return self._last_run_tokens

    @property
    def tool_call_counts(self) -> Dict[str, int]:
        """最近一次执行里，每个工具被调用了几次。

        ★ 这是「这位将领到底动没动手」的唯一凭据。

          一次没有调用任何工具的执行，产出的东西全部来自模型的既有知识 ——
          它可能仍然有用，但它没有触碰过这台电脑。而这两种情况在最终
          那段文本上看起来完全一样，所以必须有一个地方能把它们分开。

        ★ 空字典表示「这次没调用工具」，也可能表示「还没执行过」。
          两者的区别由调用方自己掌握（有没有调过 execute），
          这里不猜。
        """

        h = getattr(self, "_harness", None)
        trace = getattr(h, "trace", None)
        counts = getattr(trace, "tool_calls", None)
        return dict(counts) if isinstance(counts, dict) else {}

    @property
    def has_tools(self) -> bool:
        """有没有配上兵器。

        ★ 单独暴露，是因为「将领没有工具」是这个框架最容易发生、
          且最难看出来的失效：ReAct 循环照常跑，只是恒定一轮就返回，
          外面看到的仍是一段像模像样的文本。
        """

        return bool(self._tool_functions)

    def reset_conversation(self) -> None:
        """重置对话历史"""
        self._conversation.clear()

    def get_conversation_summary(self) -> str:
        """获取对话摘要"""
        if not self._conversation:
            return f"将领 {self.name} 暂无对话记录"
        return f"将领 {self.name} 对话轮数: {len(self._conversation)}"

    def get_profile_summary(self) -> str:
        """获取战力档案摘要（用于 UI 展示）"""
        if not self.profile:
            return "无档案"
        p = self.profile
        return (
            f"{p.style.value} | "
            f"专长: {'、'.join(p.specialties[:2])} | "
            f"战力: {p.stats.summary()}"
        )

    def __str__(self) -> str:
        status = "🟢 Active" if self.is_active else "⚫ Inactive"
        llm_status = f"🧠 {self.llm}" if self.llm else "⚠️ No LLM"
        return f"Agent(name='{self.name}', role='{self.role}', status={status}, llm={llm_status})"

    def __repr__(self) -> str:
        return self.__str__()


# ══════════════════════════════════════════════════════════
#  从函数签名生成 JSON Schema
# ══════════════════════════════════════════════════════════

_PY_TO_JSON = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object",
}


def _schema_from_signature(func: Callable) -> Dict[str, Any]:
    r"""内省函数签名，生成工具的参数 schema。

    ════════════════════════════════════════════════════════
     ★ 这里曾经硬编码为 {"type": "object", "properties": {}}
    ════════════════════════════════════════════════════════

    后果是**整个工具层静默失效**：模型看到的是一个
    "不接受任何参数"的 execute_python —— 调用它无法传入代码，
    所以模型理性地选择不调用。

    实测：注册了 execute_python 与 run_tests，提示里明确要求
    "必须用工具实际运行验证"，模型一次都没调，
    却在回答里写下"经实测，fib(10) 得 55"。

    ★ 危险在于它**完全不报错**：工具注册成功、定义构建成功、
      请求发送成功、模型正常回复 —— 每一步都"正常"，
      只有"工具调用次数为 0"这一个观察点能暴露它，
      而没有人会去看那个数。

    ★ 更糟的是它诱导模型**编造**执行结果。
      给一个用不了的工具，比不给工具更坏：
      不给的话模型会说"建议你运行验证"；
      给了却用不了，它就说"我已经验证过了"。

    参数说明优先取 docstring 里的 "参数名: 说明" 行，
    取不到就退回一个占位描述。
    """

    import inspect

    props: Dict[str, Any] = {}
    required: List[str] = []
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {"type": "object", "properties": {}}

    doc = inspect.getdoc(func) or ""
    doc_params: Dict[str, str] = {}
    for line in doc.splitlines():
        m = re.match(r"\s*(\w+)\s*[:：]\s*(.+)", line)
        if m:
            doc_params[m.group(1)] = m.group(2).strip()

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        props[pname] = {
            "type": _PY_TO_JSON.get(param.annotation, "string"),
            "description": doc_params.get(pname, f"参数 {pname}"),
        }
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    schema: Dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema
