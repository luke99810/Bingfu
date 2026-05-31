"""
点兵台 (兵符 · 智能任务匹配引擎)
根据任务特点和将军战力，智能推荐最优执行者

兵法云：善将者，择人任势——选对将领，战事已半。
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .profile import GeneralProfile, CombatStyle, CombatStats
from .assessment import TaskAssessment, TaskAssessor


class MatchResult(BaseModel):
    """
    匹配结果 — 一位将军 vs 一个任务的评估报告

    Attributes:
        agent_name: 将军名称
        total_score: 综合评分 (0-1)
        specialty_match: 专长匹配分 (0-1)
        power_advantage: 战力优势分 (0-2, 1为持平)
        style_bonus: 作风加成 (0-1)
        weakness_penalty: 弱项扣分 (0-1)
        reasoning: 派兵理由（供人阅读）
    """
    agent_name: str = Field(..., description="将军名称")
    total_score: float = Field(..., ge=0.0, le=2.0, description="综合评分")
    specialty_match: float = Field(default=0.0, ge=0.0, le=1.0,
                                    description="专长匹配分")
    power_advantage: float = Field(default=1.0, ge=0.0, le=2.0,
                                    description="战力优势分")
    style_bonus: float = Field(default=0.0, ge=0.0, le=1.0,
                                 description="作风加成")
    weakness_penalty: float = Field(default=0.0, ge=0.0, le=1.0,
                                     description="弱项扣分")
    reasoning: str = Field(default="", description="派兵理由")

    def __str__(self) -> str:
        score_display = f"{self.total_score:.2f}"
        return (
            f"{self.agent_name}: 综合{score_display}分 "
            f"(专长{self.specialty_match:.0%} | 战力{self.power_advantage:.0%} "
            f"| 作风{self.style_bonus:.0%} | 弱项扣{self.weakness_penalty:.0%})"
        )

    def recommendation(self) -> str:
        """友好的推荐理由"""
        if self.total_score >= 1.0:
            return f"⭐ 推荐 {self.agent_name} — {self.reasoning}"
        elif self.total_score >= 0.7:
            return f"✓ 可选 {self.agent_name} — {self.reasoning}"
        elif self.total_score >= 0.5:
            return f"△ 备选 {self.agent_name} — {self.reasoning}"
        return f"✗ 不推荐 {self.agent_name} — {self.reasoning}"


class TaskMatcher:
    """
    点兵台 — 智能任务匹配引擎

    对每个候选将军打分，排序后返回推荐列表。
    评分维度：
    - 专长匹配 40%：任务需求 ∩ 将军专长
    - 战力优势 30%：将军主属性 vs 敌方战力
    - 作风加成 20%：作战风格与任务类型的契合度
    - 弱项扣分 10%：弱项与任务需求的冲突

    Attributes:
        assessor: 任务评估器（用于先评估再匹配）
    """

    # 作战风格 → 任务类型加成表
    # (风格, 能力关键词元组) → 加成分数 (0-1)
    # 格式：(CombatStyle, (kw1, kw2, ...)) → bonus
    _STYLE_BONUS_RAW = [
        ((CombatStyle.STRATEGIC, ("分析", "规划", "评估", "战略", "推演", "研究")), 0.9),
        ((CombatStyle.STRATEGIC, ("数据", "统计", "报告")), 0.8),

        ((CombatStyle.ASSAULT, ("实现", "代码", "开发", "创建", "生成", "写")), 0.9),
        ((CombatStyle.ASSAULT, ("优化", "改进", "重构", "攻坚")), 0.85),

        ((CombatStyle.COMMAND, ("协调", "综合", "调度", "全局", "管理")), 0.9),
        ((CombatStyle.COMMAND, ("分析", "决策", "规划")), 0.75),

        ((CombatStyle.BRAVE, ("紧急", "快速", "突击", "立即", "突破")), 0.95),
        ((CombatStyle.BRAVE, ("执行", "实现", "完成")), 0.75),

        ((CombatStyle.SCOUT, ("搜索", "查找", "收集", "调研", "侦察", "初步")), 0.9),
        ((CombatStyle.SCOUT, ("探索", "发现", "尝试")), 0.8),
    ]

    # 风格主属性（用于战力计算）
    STYLE_DOMINANT_ATTR = {
        CombatStyle.STRATEGIC: "strategy",
        CombatStyle.ASSAULT: "attack",
        CombatStyle.COMMAND: "intelligence",
        CombatStyle.BRAVE: "speed",
        CombatStyle.SCOUT: "speed",
    }

    def __init__(self, assessor: Optional[TaskAssessor] = None):
        """
        初始化点兵台

        Args:
            assessor: 可选的任务评估器；不传则自动创建
        """
        self.assessor = assessor or TaskAssessor()

    def match(
        self,
        task: str,
        agents: Dict[str, "Agent"],
        assessment: Optional[TaskAssessment] = None
    ) -> List[MatchResult]:
        """
        匹配任务与将军 — 核心入口

        Args:
            task: 任务描述文本
            agents: 可用将军字典 {name: Agent}
            assessment: 可选，预评估结果；不传则自动评估

        Returns:
            按评分降序排列的匹配结果列表
        """
        # 1. 评估任务
        if assessment is None:
            assessment = self.assessor.assess(task)

        # 2. 对每个有档案的将军评分
        results = []
        for name, agent in agents.items():
            profile = getattr(agent, "profile", None)
            if profile is None:
                # 无档案的将军给一个中等偏低的基础分
                results.append(MatchResult(
                    agent_name=name,
                    total_score=0.5,
                    specialty_match=0.0,
                    power_advantage=1.0,
                    style_bonus=0.5,
                    weakness_penalty=0.0,
                    reasoning="无档案，按通用标准评估"
                ))
                continue

            result = self._score_agent(profile, assessment)
            result.agent_name = name
            results.append(result)

        # 3. 按总分降序排列
        results.sort(key=lambda r: r.total_score, reverse=True)
        return results

    def best_match(
        self,
        task: str,
        agents: Dict[str, "Agent"],
        assessment: Optional[TaskAssessment] = None
    ) -> Optional[MatchResult]:
        """
        返回最优匹配的将军

        Returns:
            评分最高的 MatchResult，若无可用将军则返回 None
        """
        results = self.match(task, agents, assessment)
        return results[0] if results else None

    def _score_agent(
        self,
        profile: GeneralProfile,
        assessment: TaskAssessment
    ) -> MatchResult:
        """
        对单将军评分

        评分公式：
        total = specialty_match * 0.40
              + power_advantage * 0.30
              + style_bonus * 0.20
              - weakness_penalty * 0.10
        """
        # ── 1. 专长匹配分 (0-1) ─────────────────────────────────────
        if assessment.required_capabilities:
            match_count = profile.specialty_match_count(assessment.required_capabilities)
            specialty_match = match_count / len(assessment.required_capabilities)
        else:
            specialty_match = 0.5

        # ── 2. 战力优势分 (0-2) ─────────────────────────────────────
        dominant_attr_name = self.STYLE_DOMINANT_ATTR.get(profile.style, "intelligence")
        dominant_value = getattr(profile.stats, dominant_attr_name, 50)
        # 战力比 = 将军主属性 / 敌方战力（clamp 到 0-2）
        power_ratio = dominant_value / max(assessment.enemy_power, 1)
        power_advantage = max(0.0, min(2.0, power_ratio))

        # ── 3. 作风加成 (0-1) ─────────────────────────────────────
        style_bonus = self._get_style_bonus(profile.style, assessment)

        # ── 4. 弱项扣分 (0-1) ─────────────────────────────────────
        if assessment.required_capabilities:
            weak_count = profile.weakness_match_count(assessment.required_capabilities)
            weakness_penalty = weak_count / len(assessment.required_capabilities)
        else:
            weakness_penalty = 0.0

        # ── 5. 综合评分 ───────────────────────────────────────────
        total = (
            specialty_match * 0.40
            + power_advantage * 0.30
            + style_bonus * 0.20
            - weakness_penalty * 0.10
        )
        total = max(0.0, min(2.0, total))

        # ── 6. 生成理由 ───────────────────────────────────────────
        reasoning = self._generate_reasoning(
            profile, assessment,
            specialty_match, power_advantage, style_bonus, weakness_penalty
        )

        return MatchResult(
            agent_name="",
            total_score=total,
            specialty_match=specialty_match,
            power_advantage=power_advantage,
            style_bonus=style_bonus,
            weakness_penalty=weakness_penalty,
            reasoning=reasoning
        )

    def _get_style_bonus(self, style: CombatStyle, assessment: TaskAssessment) -> float:
        """查作风-任务加成表"""
        task_text = " ".join(assessment.required_capabilities)
        best_bonus = 0.3  # 最低保底加成

        for (s, keywords), bonus in self._STYLE_BONUS_RAW:
            if s == style and any(kw in task_text for kw in keywords):
                best_bonus = max(best_bonus, bonus)

        return best_bonus

    def _generate_reasoning(
        self,
        profile: GeneralProfile,
        assessment: TaskAssessment,
        specialty_match: float,
        power_advantage: float,
        style_bonus: float,
        weakness_penalty: float,
    ) -> str:
        """生成友好的推荐理由"""
        reasons = []

        if specialty_match >= 0.7:
            matched = [c for c in assessment.required_capabilities
                       if profile.has_specialty(c)]
            if matched:
                reasons.append(f"专长契合：{'、'.join(matched[:2])}")
        elif specialty_match < 0.3:
            reasons.append("专长覆盖不足")

        if power_advantage >= 1.5:
            reasons.append("战力绰绰有余")
        elif power_advantage >= 1.0:
            reasons.append("战力足以应对")
        elif power_advantage >= 0.5:
            reasons.append("战力略显不足")

        if style_bonus >= 0.8:
            reasons.append(f"{profile.style.value}最擅长此类任务")
        elif style_bonus >= 0.5:
            reasons.append(f"作风匹配良好")

        if weakness_penalty >= 0.3:
            reasons.append("需注意其弱项领域")

        if not reasons:
            reasons.append("综合能力均衡")

        return "；".join(reasons)
