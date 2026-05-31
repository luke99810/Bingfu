"""
Agent module (智能体模块)
Defines the Agent class with ancient Chinese warfare naming.

v0.4.0: 新增 LLM 驱动执行、ReAct 工具调用循环
"""

import json
from typing import Any, Callable, Dict, List, Optional
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
    max_iterations: int = Field(default=5, description="ReAct 最大循环轮数")

    # 内部对话历史
    _conversation: List[LLMMessage] = PrivateAttr(default_factory=list)
    # 工具函数映射
    _tool_functions: Dict[str, Callable] = PrivateAttr(default_factory=dict)

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

    def register_tool_function(self, name: str, func: Callable, description: str = "") -> None:
        """
        直接注册工具函数（更灵活的方式）

        Args:
            name: 工具名称
            func: 执行函数
            description: 工具描述
        """
        self._tool_functions[name] = func

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

        # === LLM 驱动执行 ===
        # 初始化对话
        self._conversation.clear()
        if self.system_prompt:
            self._conversation.append(LLMMessage(
                role=RoleType.SYSTEM,
                content=self.system_prompt
            ))
        else:
            self._conversation.append(LLMMessage(
                role=RoleType.SYSTEM,
                content=self._default_system_prompt()
            ))

        # 用户任务
        self._conversation.append(LLMMessage(
            role=RoleType.USER,
            content=task
        ))

        # ReAct 循环
        for i in range(self.max_iterations):
            # 构建 LLM 可用的工具定义
            tool_defs = self._build_tool_definitions(tools)

            # 调用 LLM
            response: LLMResponse = self.llm.generate(
                messages=self._conversation,
                tools=tool_defs if tool_defs else None,
                temperature=0.7,
            )

            # 将 assistant 回复加入历史
            self._conversation.append(LLMMessage(
                role=RoleType.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
            ))

            # 如果没有工具调用，说明 LLM 已给出最终回复
            if not response.has_tool_calls:
                return response.content or f"Agent '{self.name}' 已完成任务（无文本输出）"

            # 执行工具调用
            for tool_call in response.tool_calls:
                tool_result = self._execute_tool_call(tool_call)

                # 将工具结果反馈给 LLM
                self._conversation.append(LLMMessage(
                    role=RoleType.TOOL,
                    content=tool_result,
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                ))

        # 达到最大轮数
        return response.content or f"Agent '{self.name}' 达到最大思考轮数({self.max_iterations})，任务可能未完成。"

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
            doc = func.__doc__ or f"工具: {name}"
            definitions.append(ToolDefinition(
                name=name,
                description=doc.strip().split('\n')[0],
                parameters={"type": "object", "properties": {}}
            ))

        return definitions

    def _execute_tool_call(self, tool_call: ToolCall) -> str:
        """
        执行工具调用

        Args:
            tool_call: 工具调用请求

        Returns:
            str: 工具执行结果（JSON 字符串）
        """
        func = self._tool_functions.get(tool_call.name)

        if func is None:
            # 尝试在 tools 列表中查找
            for tool in self.tools:
                tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                if tool_name == tool_call.name:
                    if hasattr(tool, 'function') and callable(tool.function):
                        func = tool.function
                    elif hasattr(tool, 'use'):
                        func = tool.use
                    break

        if func is None:
            return json.dumps({
                "error": f"工具 '{tool_call.name}' 未找到",
                "available": list(self._tool_functions.keys())
            }, ensure_ascii=False)

        try:
            # 调用工具
            if isinstance(tool_call.arguments, dict):
                result = func(**tool_call.arguments)
            else:
                result = func(tool_call.arguments)

            # 确保结果是字符串
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, default=str)

            return result

        except Exception as e:
            return json.dumps({
                "error": f"工具 '{tool_call.name}' 执行失败: {e}"
            }, ensure_ascii=False)

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
