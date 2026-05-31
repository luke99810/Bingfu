"""
名将档案预设 (兵符 · 将军预设档案)
提供历史上著名将领的预设战力档案，即装即用

兵法云：知己知彼——先了解将领的优劣，方能用其所长。
"""

from typing import Dict

from .profile import GeneralProfile, CombatStyle, CombatStats


# ============================================================
# 名将档案预设库
# ============================================================

PRESET_GENERALS: Dict[str, GeneralProfile] = {
    "韩信": GeneralProfile(
        style=CombatStyle.COMMAND,
        specialties=[
            "全局调度", "兵力分配", "出奇制胜",
            "综合分析", "多线协调", "战略决策",
            "数据分析", "战略规划", "沟通协调",
        ],
        weaknesses=["防守固守", "缓慢推进", "后勤规划"],
        stats=CombatStats(
            attack=80,
            defense=65,
            strategy=95,
            speed=85,
            intelligence=90
        ),
        bio="汉初三杰，兵仙。善于统兵作战，出奇制胜，以少胜多。"
    ),

    "白起": GeneralProfile(
        style=CombatStyle.ASSAULT,
        specialties=[
            "歼灭战", "攻坚战", "大规模作战",
            "精确执行", "兵团指挥", "消耗战",
            "代码开发", "功能", "紧急修复", "快速执行",
        ],
        weaknesses=["柔性策略", "谈判斡旋", "外交斡旋"],
        stats=CombatStats(
            attack=98,
            defense=75,
            strategy=85,
            speed=80,
            intelligence=78
        ),
        bio="战国四大名将之首。长平之战坑杀赵军四十万，善于大规模歼灭战。"
    ),

    "诸葛亮": GeneralProfile(
        style=CombatStyle.STRATEGIC,
        specialties=[
            "谋略推演", "情报分析", "长期规划",
            "风险评估", "制度设计", "内政管理",
            "数据分析", "战略规划", "信息收集", "系统架构", "设计创意",
        ],
        weaknesses=["快速突击", "正面硬刚", "临阵应变"],
        stats=CombatStats(
            attack=55,
            defense=80,
            strategy=99,
            speed=60,
            intelligence=98
        ),
        bio="卧龙先生。辅佐刘备三分天下，善于谋略推演，未出茅庐已知天下三分。"
    ),

    "项羽": GeneralProfile(
        style=CombatStyle.BRAVE,
        specialties=[
            "快速突击", "正面突破", "士气激励",
            "紧急行动", "个人战斗", "骑兵战术",
            "紧急", "立即", "突击", "生产环境",
        ],
        weaknesses=["谋略规划", "持久战", "复杂分析", "后勤管理"],
        stats=CombatStats(
            attack=99,
            defense=60,
            strategy=50,
            speed=95,
            intelligence=55
        ),
        bio="西楚霸王。力能扛鼎，巨鹿之战破釜沉舟，勇冠三军，无人可挡。"
    ),

    # ---- 通用预设 ----
    "斥候": GeneralProfile(
        style=CombatStyle.SCOUT,
        specialties=[
            "信息收集", "快速调研", "目标搜索",
            "初步探索", "数据抓取", "竞品分析"
        ],
        weaknesses=["深度分析", "复杂决策", "系统设计"],
        stats=CombatStats(
            attack=40,
            defense=30,
            strategy=50,
            speed=90,
            intelligence=70
        ),
        bio="行军打仗，斥候先行。负责情报收集、快速反馈。"
    ),

    "谋士": GeneralProfile(
        style=CombatStyle.STRATEGIC,
        specialties=[
            "方案设计", "策略规划", "风险评估",
            "竞品分析", "趋势预测", "架构规划"
        ],
        weaknesses=["快速执行", "紧急交付", "编码实现"],
        stats=CombatStats(
            attack=45,
            defense=70,
            strategy=95,
            speed=40,
            intelligence=92
        ),
        bio="运筹帷幄之中，决胜千里之外。善于策略规划与风险评估。"
    ),

    "猛将": GeneralProfile(
        style=CombatStyle.ASSAULT,
        specialties=[
            "代码实现", "功能开发", "问题修复",
            "性能优化", "紧急交付", "批量处理"
        ],
        weaknesses=["战略规划", "文档撰写", "用户沟通"],
        stats=CombatStats(
            attack=90,
            defense=55,
            strategy=45,
            speed=85,
            intelligence=60
        ),
        bio="摧城拔寨，所向披靡。执行力极强，善于快速交付可用成果。"
    ),
}


def get_preset(name: str) -> GeneralProfile:
    """
    根据名称获取预设档案

    Args:
        name: 将军名称（模糊匹配）

    Returns:
        对应的 GeneralProfile，若不存在则抛出 KeyError
    """
    # 精确匹配
    if name in PRESET_GENERALS:
        return PRESET_GENERALS[name]

    # 模糊匹配（包含关系）
    name_lower = name.lower()
    for preset_name, profile in PRESET_GENERALS.items():
        if name_lower in preset_name.lower() or preset_name.lower() in name_lower:
            return profile

    raise KeyError(f"未找到名为「{name}」的预设档案，可用：{list(PRESET_GENERALS.keys())}")


def list_presets() -> Dict[str, str]:
    """列出所有可用预设（名称 → 描述摘要）"""
    return {
        name: f"{profile.style.value} | 专长: {'、'.join(profile.specialties[:2])}"
        for name, profile in PRESET_GENERALS.items()
    }
