"""
军情评估 (兵符 · 任务难度评估)
分析任务的复杂度、所需能力，推导敌方战力

兵法云：多算胜，少算不胜——评估清楚，方能派兵有据。
"""

import re
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from .llm.base import LLMMessage, RoleType


class TaskComplexity(str, Enum):
    """任务难度等级"""
    TRIVIAL = "易"       # 1-3：简单查询、格式化、快速问答
    MODERATE = "中"      # 4-6：分析报告、代码实现、多步骤任务
    DIFFICULT = "难"     # 7-8：复杂推理、系统设计、跨领域问题
    EXTREME = "极难"     # 9-10：未知领域、长期项目、战略性规划


class TaskAssessment(BaseModel):
    """
    军情评估结果 — 一次任务评估的完整报告

    Attributes:
        complexity_score: 复杂度评分 (1-10)
        complexity: 复杂度等级枚举
        required_capabilities: 任务所需的能力/技能列表
        enemy_power: 敌方战力值 (derived from complexity)
        reasoning: 评估理由（供人阅读）
    """
    complexity_score: int = Field(..., ge=1, le=10, description="复杂度评分 1-10")
    complexity: TaskComplexity = Field(..., description="复杂度等级")
    required_capabilities: List[str] = Field(
        default_factory=list,
        description="任务所需能力列表"
    )
    enemy_power: int = Field(..., ge=1, description="敌方战力值")
    reasoning: str = Field(default="", description="评估理由")

    def __str__(self) -> str:
        caps = "、".join(self.required_capabilities[:4])
        return (
            f"【军情评估】难度 {self.complexity.value}({self.complexity_score}/10) | "
            f"敌方战力 ≈ {self.enemy_power} | "
            f"需 {caps or '综合能力'}"
        )


class TaskAssessor:
    """
    军情评估器

    评估输入任务的质量/复杂度，推导出敌方战力。
    采用 LLM 优先 + 规则降级的策略：

    - 有 LLM 时：让 LLM 分析任务文本，返回结构化评估
    - 无 LLM 时：基于关键词规则快速估算

    Attributes:
        llm_provider: LLM Provider 实例（有则用，无则规则降级）
    """

    # 复杂度关键词映射（用于规则降级）
    COMPLEXITY_KEYWORDS: Dict[int, List[str]] = {
        1: ["查", "找", "告诉我", "什么是", "解释"],
        2: ["总结", "翻译", "格式", "列表"],
        3: ["分析", "对比", "评估", "报告"],
        4: ["写", "生成", "创建", "实现", "代码"],
        5: ["设计", "规划", "方案", "策略"],
        6: ["优化", "改进", "重构", "重构"],
        7: ["架构", "系统", "全面", "复杂"],
        8: ["研究", "探索", "前沿", "未知"],
        9: ["战略", "长期", "全局", "变革"],
        10: ["开创", "革命", "颠覆", "全新领域"],
    }

    # 能力关键词映射（用于规则降级）
    CAPABILITY_KEYWORDS: Dict[str, List[str]] = {
        "数据分析": ["分析", "数据", "统计", "图表", "趋势", "报告", "营销", "销售", "调研"],
        "代码开发": ["代码", "编程", "开发", "实现", "函数", "算法", "bug", "修复", "功能"],
        "文案写作": ["写", "文案", "文章", "内容", "创作", "编辑", "撰写"],
        "设计创意": ["设计", "创意", "界面", "视觉", "UX", "交互"],
        "战略规划": ["战略", "规划", "方案", "策略", "季度", "营销策略", "商业计划", "全局"],
        "快速执行": ["快速", "立即", "紧急", "突击", "马上", "紧急修复", "生产环境"],
        "信息收集": ["搜索", "查找", "调研", "收集", "侦察", "竞品", "情报", "探索"],
        "沟通协调": ["协调", "沟通", "团队", "合作", "调度", "分配"],
        "系统架构": ["架构", "系统设计", "高并发", "分布式", "微服务", "架构设计"],
        "歼灭执行": ["实现", "完成", "攻坚", "批量", "大规模"],
    }

    def __init__(self, llm_provider=None):
        """
        初始化评估器

        Args:
            llm_provider: LLM Provider 实例，可选
        """
        self.llm_provider = llm_provider

    def assess(self, task: str) -> TaskAssessment:
        """
        评估任务 — 入口方法

        优先使用 LLM，无 LLM 时降级为规则匹配。

        Args:
            task: 任务描述文本

        Returns:
            TaskAssessment 评估结果
        """
        if self.llm_provider:
            return self._assess_with_llm(task)
        return self._assess_with_rules(task)

    def _assess_with_llm(self, task: str) -> TaskAssessment:
        """
        LLM 驱动的任务评估

        让 LLM 分析任务文本，提取复杂度、所需能力、评估理由。
        """
        system_prompt = (
            "你是一位任务分析专家。请分析以下任务，"
            "评估其复杂度并识别所需能力。\n\n"
            "请以JSON格式返回评估结果：\n"
            "{\n"
            "  \"complexity_score\": 5,  // 1-10整数，复杂度评分\n"
            "  \"required_capabilities\": [\"代码开发\", \"数据分析\"],  // 所需能力列表\n"
            "  \"reasoning\": \"任务涉及多步骤逻辑...\"  // 评估理由\n"
            "}\n\n"
            "评分参考：\n"
            "  1-3: 简单查询、格式化、快速问答\n"
            "  4-6: 分析报告、代码实现、多步骤任务\n"
            "  7-8: 复杂推理、系统设计、跨领域问题\n"
            "  9-10: 未知领域、长期项目、战略性规划\n\n"
            "能力分类：数据分析/代码开发/文案写作/设计创意/战略规划/快速执行/信息收集/沟通协调"
        )

        try:
            messages = [
                LLMMessage(role=RoleType.SYSTEM, content=system_prompt),
                LLMMessage(role=RoleType.USER, content=task),
            ]
            response = self.llm_provider.generate(
                messages,
                temperature=0.3,
                max_tokens=512
            )
            content = response.content or ""

            # 解析 JSON
            import json
            json_match = re.search(r'\{[^{}]*"complexity_score"[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # 尝试直接解析
                data = json.loads(content.strip())

            complexity_score = max(1, min(10, int(data.get("complexity_score", 5))))
            required_caps = data.get("required_capabilities", [])
            reasoning = data.get("reasoning", "")

        except Exception:
            # LLM 解析失败，降级到规则
            return self._assess_with_rules(task)

        # 推导敌方战力
        enemy_power = self._derive_enemy_power(complexity_score, required_caps)

        complexity = self._score_to_complexity(complexity_score)

        return TaskAssessment(
            complexity_score=complexity_score,
            complexity=complexity,
            required_capabilities=required_caps,
            enemy_power=enemy_power,
            reasoning=reasoning
        )

    def _assess_with_rules(self, task: str) -> TaskAssessment:
        """
        规则驱动的任务评估（无 LLM 时的降级方案）

        基于关键词匹配估算复杂度。
        """
        text = task.lower()

        # 1. 估算复杂度
        score = 5  # 默认中等

        # 任务长度加成
        if len(task) > 500:
            score += 1
        elif len(task) > 1000:
            score += 2

        # 关键词匹配
        for lvl, keywords in sorted(self.COMPLEXITY_KEYWORDS.items()):
            if any(kw in text for kw in keywords):
                # 使用加权平均，避免被多个关键词过度叠加
                score = (score + lvl) // 2

        # 多步骤指示词
        multi_indicators = ["然后", "接下来", "首先", "最后", "step", "步骤", "流程"]
        if sum(1 for ind in multi_indicators if ind in text) >= 2:
            score += 1

        score = max(1, min(10, score))

        # 2. 识别所需能力
        required_caps = []
        for cap_name, cap_keywords in self.CAPABILITY_KEYWORDS.items():
            if any(kw in text for kw in cap_keywords):
                required_caps.append(cap_name)

        # 如果没有匹配到任何能力，给一个默认值
        if not required_caps:
            required_caps = ["综合分析"]

        complexity = self._score_to_complexity(score)
        enemy_power = self._derive_enemy_power(score, required_caps)

        reasoning = (
            f"【规则评估】基于关键词匹配。"
            f"识别到 {len(required_caps)} 项能力需求。"
            f"任务长度 {len(task)} 字。"
        )

        return TaskAssessment(
            complexity_score=score,
            complexity=complexity,
            required_capabilities=required_caps,
            enemy_power=enemy_power,
            reasoning=reasoning
        )

    def _score_to_complexity(self, score: int) -> TaskComplexity:
        """将数值评分转为复杂度枚举"""
        if score <= 3:
            return TaskComplexity.TRIVIAL
        elif score <= 6:
            return TaskComplexity.MODERATE
        elif score <= 8:
            return TaskComplexity.DIFFICULT
        return TaskComplexity.EXTREME

    def _derive_enemy_power(self, complexity: int, capabilities: List[str]) -> int:
        """
        推导敌方战力

        敌方战力 = 复杂度基准 × 能力数量加成 × 难度系数
        """
        # 复杂度基准: 10-100
        base = complexity * 10

        # 能力数量加成: 每多一项能力，+10
        cap_bonus = len(capabilities) * 10

        # 难度系数: 高复杂度时指数增长
        if complexity >= 8:
            factor = 1.5
        elif complexity >= 5:
            factor = 1.2
        else:
            factor = 1.0

        enemy_power = int((base + cap_bonus) * factor)
        return max(10, min(200, enemy_power))
