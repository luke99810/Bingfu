"""
将军档案 (兵符 · 将军战力体系)
定义将领的作战风格、战力值、专长与弱项

兵法云：知彼知己，百战不殆——先知其能，方能用其所长。
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class CombatStyle(str, Enum):
    """
    作战风格枚举

    每种风格对应不同的任务执行策略和思维模式：
    - 谋略型：分析、规划、推演、评估
    - 突击型：强力执行、攻坚、规模化
    - 统帅型：综合调度、全局把控、协调
    - 勇武型：快速行动、正面突破、士气激励
    - 侦察型：信息收集、快速反馈、初步探索
    """
    STRATEGIC = "谋略型"    # 诸葛亮：分析、规划、推演
    ASSAULT = "突击型"      # 白起：强力执行、攻坚克难
    COMMAND = "统帅型"      # 韩信：综合调度、全局把控
    BRAVE = "勇武型"       # 项羽：快速行动、正面突破
    SCOUT = "侦察型"       # 斥候：信息收集、快速反馈


class CombatStats(BaseModel):
    """
    五维战力值 (1-100)

    每位将领在五个维度上有不同的战力分配，
    决定了他们擅长什么类型的任务。

    Attributes:
        attack: 攻击力 — 执行力、攻坚能力、快速交付
        defense: 防御力 — 稳定性、容错能力、持久性
        strategy: 谋略值 — 分析、推理、规划、策略制定
        speed: 速度值 — 响应速度、效率、迭代频率
        intelligence: 智力值 — 综合理解、创新、复杂问题处理
    """
    attack: int = Field(default=50, ge=1, le=100,
                        description="攻击力 — 执行力、攻坚能力")
    defense: int = Field(default=50, ge=1, le=100,
                         description="防御力 — 稳定性、容错能力")
    strategy: int = Field(default=50, ge=1, le=100,
                          description="谋略值 — 分析、推理、规划能力")
    speed: int = Field(default=50, ge=1, le=100,
                       description="速度值 — 响应速度、效率")
    intelligence: int = Field(default=50, ge=1, le=100,
                               description="智力值 — 综合理解、创新力")

    def dominant_stat(self) -> str:
        """返回最高属性名称"""
        stats = {
            "attack": self.attack,
            "defense": self.defense,
            "strategy": self.strategy,
            "speed": self.speed,
            "intelligence": self.intelligence,
        }
        return max(stats, key=stats.get)

    def dominant_value(self) -> int:
        """返回最高属性值"""
        return max(self.attack, self.defense, self.strategy, self.speed, self.intelligence)

    def total_power(self) -> int:
        """返回总战力（用于快速比较）"""
        return self.attack + self.defense + self.strategy + self.speed + self.intelligence

    def summary(self) -> str:
        """战力摘要，用于UI展示"""
        stat_names = {
            "attack": "攻",
            "defense": "防",
            "strategy": "谋",
            "speed": "速",
            "intelligence": "智",
        }
        parts = [f"{stat_names[k]}:{v}" for k, v in {
            "attack": self.attack,
            "defense": self.defense,
            "strategy": self.strategy,
            "speed": self.speed,
            "intelligence": self.intelligence,
        }.items()]
        return " ".join(parts)


class GeneralProfile(BaseModel):
    """
    将军档案 — 定义一位将领的核心能力画像

    Attributes:
        style: 作战风格，决定任务执行的整体策略
        specialties: 专长列表，最擅长的任务类型/领域
        weaknesses: 弱项列表，不擅长或容易失误的领域
        stats: 五维战力值
        bio: 简短人物描述（用于日志展示）
    """
    style: CombatStyle = Field(..., description="作战风格")
    specialties: List[str] = Field(
        default_factory=list,
        description="专长领域列表，如 ['数据分析', '歼灭战', '风险评估']"
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="弱项列表，如 ['耐心等待', '防守固守']"
    )
    stats: CombatStats = Field(
        default_factory=CombatStats,
        description="五维战力值"
    )
    bio: str = Field(default="", description="简短人物描述")

    def has_specialty(self, keyword: str) -> bool:
        """检查是否具有某项专长（模糊匹配）"""
        kw = keyword.lower()
        return any(kw in s.lower() for s in self.specialties)

    def has_weakness(self, keyword: str) -> bool:
        """检查是否具有某项弱项（模糊匹配）"""
        kw = keyword.lower()
        return any(kw in w.lower() for w in self.weaknesses)

    def specialty_match_count(self, keywords: List[str]) -> int:
        """返回匹配的专长数量"""
        count = 0
        for kw in keywords:
            if self.has_specialty(kw):
                count += 1
        return count

    def weakness_match_count(self, keywords: List[str]) -> int:
        """返回冲突的弱项数量"""
        count = 0
        for kw in keywords:
            if self.has_weakness(kw):
                count += 1
        return count

    def __str__(self) -> str:
        spec_str = "、".join(self.specialties[:3])
        return f"{self.style.value} | 专长: {spec_str} | {self.stats.summary()}"
