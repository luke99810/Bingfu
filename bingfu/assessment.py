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

    def assess_fast(self, task: str) -> TaskAssessment:
        """不调模型的庙算 —— 只走规则。

        ★ 用途：**子任务级**的点将。

          战役级的庙算值得花一次调用；但拆解出的每个子任务再各庙算一次
          就是纯浪费 —— 实测一次「写三个文件」的战役里庙算跑了 4 次：
          战役级 1 次 + 每个子任务 1 次，后三次合计 587 tokens，
          而它们只是为了给点兵台一个打分依据。

          点兵台的评分本身是本地向量运算，它需要的只是
          「所需能力」与「复杂度」两个粗粒度输入 —— 规则路径足够。

        ★ 附带的好处是确定性：同一段描述永远得到同一份评估，
          于是「为什么点了这位将领」可复算。
        """

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


# ── 交付物计数 ──────────────────────────────────────────

#: 「写/产出」类动词，后面跟的文件名才算**交付物**而不是素材
_PRODUCE_VERB = (
    "写", "写入", "写进", "写到", "生成", "产出", "输出", "创建", "新建", "保存",
    "write", "create", "generate", "output", "save", "produce",
)

_FILENAME = re.compile(r"[\w\-.]+\.[A-Za-z0-9]{1,5}")


#: 「需要不止一种将领」的能力数门槛。
#:
#: ★ 这个数是**标定出来的，不是推导出来的**，必须说清楚：
#:   跨框架基准那 12 道题，规则版抽出的能力数是 1–2；
#:   几个明显该拆的任务（检索+分析+编码、分析+建模）是 3–5。
#:   门槛取 3，落在这条缝里。
#:
#: ★ 标定样本很小，而且**这套基准无法验证它** —— 基准里没有
#:   任何一道需要拆解的题（120 次运行计划全部单节点）。
#:   所以这是一个**判断**，不是一个测量结果。
#:   要验证它，得先有一类拆解真能赢的任务。
MULTI_ROLE_CAPABILITIES = 3

#: 「产物多到一个人做不划算」的交付物数门槛（需与 ≥2 种能力同时成立）
MANY_DELIVERABLES = 3


def needs_orchestration(task: str, assessment: Any = None) -> bool:
    """这活要不要动用不止一位将领 —— **不调模型**。

    ════════════════════════════════════════════════════
     为什么闸门不该花一次调用
    ════════════════════════════════════════════════════

    原先的闸门是「LLM 给一个 1–10 的复杂度分，低于阈值就单干」。
    实测下来这个设计有三处问题：

    ★ 它**有噪声**。同一道 agg-pick 连问两次，一次得 4、一次得 3。
      花一次网络往返买一个抖动的数，再拿它去过一道阈值线。

    ★ 它**几乎从不改变结论**。12 道题里规则版与 LLM 版唯一一次分歧是
      robust-missing（规则 4、LLM 5），而那一次 LLM 是**错的** ——
      越过闸门之后「读一个文件、写一个文件」被拆成两个子任务，
      13 次调用、14 311 token，对手是 3 次、1 750。

    ★ 它问错了问题。「这题有多难」是个连续量，而拆解是个**结构决定**：
      这活要不要不止一种将领？多难都不重要 ——
      一件很难但只需要一位将领的事，拆开只会多付几套 ReAct 开销。

    ════════════════════════════════════════════════════
     换成什么
    ════════════════════════════════════════════════════

    两个都从任务文本上直接读、零成本、且**确定性**
    （同一段描述永远得到同一个结论，于是「为什么没拆」可复算）：

      一、所需能力种类 ≥ 3        —— 需要不同类型的将领
      二、交付物 ≥ 3 且能力 ≥ 2   —— 产物多到一个人做不划算

    ★ 保守方向是**倾向不拆**：少拆的代价是少一点并行，
      多拆的代价是每个子任务各付一套 ReAct、还可能互相覆盖产物。
      实测后者贵得多。

    ★ 想强制拆解的调用方仍然可以：`orchestrate(..., force_decompose=True)`。
      闸门是默认策略，不是不可绕过的规则。
    """

    from bingfu.assessment import TaskAssessor  # 局部导入，避免循环

    if assessment is None:
        assessment = TaskAssessor()._assess_with_rules(task)
    caps = len(getattr(assessment, "required_capabilities", ()) or ())
    if caps >= MULTI_ROLE_CAPABILITIES:
        return True
    return deliverable_count(task) >= MANY_DELIVERABLES and caps >= 2


def deliverable_count(task: str) -> int:
    """数出任务里有几件**独立交付物**。

    ★ 只数被「写/生成/输出」这类动词带出来的文件名。

      素材与交付物必须分开数：`chain-sum` 里 data.csv 是输入、
      total.md 是产出，全都算上就成了 2 件，
      于是一件事被当成可拆的两件。

    ★ 这是启发式，会**低估**（没点名文件的任务数出 0）。
      所以调用方只用它来决定「要不要多花一次调用去问」，
      不用它单独决定拆不拆 —— 低估的代价必须是少花钱，
      不能是少干活。
    """

    text = str(task or "")
    found = set()
    for m in _FILENAME.finditer(text):
        head = text[max(0, m.start() - 14):m.start()]
        if any(v in head.lower() for v in _PRODUCE_VERB):
            found.add(m.group(0))
    return len(found)
