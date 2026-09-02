"""
Commander module (指挥系统模块)
Implements multi‑agent coordination, inspired by ancient Chinese warfare command.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, PrivateAttr

from bingfu.agent import Agent
from bingfu.signal import Drum, Gong
from bingfu.matcher import TaskMatcher, MatchResult
from bingfu.assessment import TaskAssessment, TaskAssessor


class Commander(BaseModel):
    """
    Commander (指挥系统) — coordinates multiple agents.
    Inspired by ancient Chinese warfare command structures.
    """
    
    name: str = Field(default="Marshal", description="Commander name (e.g., 'Marshal', 'General')")
    agents: Dict[str, Agent] = Field(
        default_factory=dict,
        description="Dictionary of agents under command: {agent_name: Agent}"
    )
    strategy: str = Field(
        default="round_robin",
        description="Coordination strategy: 'round_robin', 'priority', 'smart'"
    )

    # 点兵台（延迟初始化，避免循环导入）
    __matcher = None  # 类型: Optional[TaskMatcher]
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the command.
        
        Args:
            agent (Agent): The agent to add.
        """
        self.agents[agent.name] = agent
    
    def remove_agent(self, agent_name: str) -> bool:
        """
        Remove an agent from the command.
        
        Args:
            agent_name (str): Name of the agent to remove.
            
        Returns:
            bool: True if removed, False if not found.
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
            return True
        return False
    
    def drum_all(self, task: str) -> Dict[str, str]:
        """
        击鼓 — 向所有将领下达任务

        Args:
            task: 任务描述

        Returns:
            Dict[str, str]: 每个将领的执行结果
        """
        results = {}
        for name, agent in self.agents.items():
            try:
                result = agent.drum(task)
                results[name] = result
            except Exception as e:
                results[name] = f"❌ 执行失败: {e}"
        return results

    def gong_all(self) -> Dict[str, str]:
        """
        鸣金 — 让所有将领停止

        Returns:
            Dict[str, str]: 每个将领的停止结果
        """
        results = {}
        for name, agent in self.agents.items():
            try:
                result = agent.gong()
                results[name] = result
            except Exception as e:
                results[name] = f"❌ 停止失败: {e}"
        return results

    def drum_one(self, agent_name: str, task: str) -> str:
        """
        击鼓 — 向指定将领下达任务

        Args:
            agent_name: 目标将领名称
            task: 任务描述

        Returns:
            str: 执行结果

        Raises:
            KeyError: 若将领不存在
        """
        if agent_name not in self.agents:
            return f"❌ 未找到将领 '{agent_name}'"

        agent = self.agents[agent_name]
        try:
            return agent.drum(task)
        except Exception as e:
            return f"❌ 执行失败: {e}"

    def gong_one(self, agent_name: str) -> str:
        """
        鸣金 — 让指定将领停止

        Args:
            agent_name: 目标将领名称

        Returns:
            str: 停止结果
        """
        if agent_name not in self.agents:
            return f"❌ 未找到将领 '{agent_name}'"

        agent = self.agents[agent_name]
        try:
            return agent.gong()
        except Exception as e:
            return f"❌ 停止失败: {e}"

    # ========== 智能调度 (点兵台) ==========

    def set_llm(self, llm_provider) -> None:
        """设置LLM Provider（军师），用于智能任务评估"""
        self._llm = llm_provider
        if self.__matcher is not None:
            self.__matcher.assessor = TaskAssessor(llm_provider=llm_provider)

    @property
    def matcher(self) -> TaskMatcher:
        """获取点兵台（延迟初始化）"""
        if self.__matcher is None:
            llm = getattr(self, '_llm', None)
            self.__matcher = TaskMatcher(assessor=TaskAssessor(llm_provider=llm))
        return self.__matcher

    def assess_task(self, task: str) -> TaskAssessment:
        """
        评估任务 — 分析复杂度、所需能力、敌方战力

        Args:
            task: 任务描述

        Returns:
            TaskAssessment 评估结果
        """
        return self.matcher.assessor.assess(task)

    def match_task(self, task: str) -> List[MatchResult]:
        """
        点兵 — 评估任务并为所有将领评分排序

        Args:
            task: 任务描述

        Returns:
            按评分降序排列的匹配结果列表
        """
        return self.matcher.match(task, self.agents)

    def smart_drum(self, task: str) -> str:
        """
        智能击鼓 — 自动选择最适合的将领执行任务

        流程：
        1. 评估任务复杂度
        2. 为所有将领评分
        3. 选择最优将领执行
        4. 返回执行结果

        Args:
            task: 任务描述

        Returns:
            str: 执行结果（包含派兵理由）
        """
        # 评估任务
        assessment = self.assess_task(task)

        # 匹配将领
        results = self.match_task(task)
        if not results:
            return "❌ 无可用将领"

        best = results[0]

        # 检查是否有档案的将领
        best_agent = self.agents.get(best.agent_name)
        if best_agent is None:
            return "❌ 未找到最优将领"

        # 构建派兵理由
        reasoning_lines = [
            f"📋 任务评估：{assessment.complexity.value}({assessment.complexity_score}/10)",
            f"⚔️  敌方战力：≈{assessment.enemy_power}",
            f"🎖️  推荐将领：{best.agent_name}（{best.total_score:.2f}分）",
            f"📝 派兵理由：{best.reasoning}",
            "",
            f"🥁 开始执行...",
        ]
        header = "\n".join(reasoning_lines)

        # 执行任务
        try:
            result = best_agent.drum(task)
            return f"{header}\n\n📋 执行结果：\n{result}"
        except Exception as e:
            return f"{header}\n\n❌ 执行失败: {e}"

    def coordinate(self, task: str, strategy: Optional[str] = None) -> Dict[str, str]:
        """
        协调指挥 — 按指定策略分配任务

        Args:
            task: 任务描述
            strategy: 协调策略（round_robin/priority/smart）

        Returns:
            Dict[str, str]: 协调结果
        """
        use_strategy = strategy if strategy else self.strategy

        if use_strategy == "smart":
            # 智能策略 → 点兵台选一将执行
            return {"smart_dispatch": self.smart_drum(task)}

        if use_strategy == "graph":
            # ★ 真正的多智能体：拆解 → 点将 → 按依赖执行 → 汇总
            result = self.orchestrate(task)
            out = {"output": result.output}
            for tid, agent_name in result.assignments.items():
                node = result.graph.nodes.get(tid)
                out[f"{tid}({agent_name})"] = (
                    str(node.output) if node and node.output else f"[{node.status.value if node else '未执行'}]"
                )
            return out

        # ── broadcast：把同一个任务发给每个将领 ──────────
        #
        # ★ 这**不是协作**，是广播。保留它是因为某些场景确实需要
        #   多个视角对同一问题各自作答（例如做多样性对比），
        #   但名字必须诚实。
        #
        #   原实现里它叫 round_robin，另有一个叫 priority 的分支
        #   与它**行为完全相同**（多建了个没用的 priority_list，
        #   enumerate 的 i 从未使用）—— 三个策略只有两种行为。
        #   一个名字暗示了顺序语义、实际没有任何顺序的策略，
        #   比没有这个策略更糟：它让调用方以为自己在调优。
        results = {}
        for name in self.agents:
            results[name] = self.drum_one(name, task)
        return results

    def orchestrate(self, task: str, max_workers: int = 4):
        """多智能体编排：拆解 → 点将 → 按依赖执行 → 汇总。

        ★ 这是 Commander 真正该做的事，也是先前完全缺失的一环。

          返回的 OrchestrationResult 带一个 is_real_collaboration 属性 ——
          用来自查这次执行到底有没有在协作（多于一个子任务、
          且至少存在一条依赖边），而不是退化成了并行广播。
        """

        from bingfu.orchestration import orchestrate as _orchestrate

        return _orchestrate(
            task,
            dict(self.agents),
            llm=self.llm_provider,
            matcher=self.matcher,
            max_workers=max_workers,
        )
    
    def status(self) -> Dict[str, Any]:
        """
        获取所有将领的状态

        Returns:
            Dict[str, Any]: 状态信息
        """
        agents_info = {}
        for name, a in self.agents.items():
            profile_summary = a.get_profile_summary() if hasattr(a, 'get_profile_summary') else "无档案"
            agents_info[name] = {
                "status": "🟢 Active" if a.is_active else "⚫ Inactive",
                "role": a.role or "",
                "profile": profile_summary,
                "has_llm": a.llm is not None,
            }
        return {
            "commander": self.name,
            "strategy": self.strategy,
            "agent_count": len(self.agents),
            "agents": agents_info
        }
    
    def __str__(self) -> str:
        return f"Commander(name='{self.name}', agents={len(self.agents)}, strategy='{self.strategy}')"
    
    def __repr__(self) -> str:
        return self.__str__()
