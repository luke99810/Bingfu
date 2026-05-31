"""
兵符 · 中军帐 启动入口 (v0.5.0)
一键启动 LLM 驱动的可视化控制台 + 智能派兵系统

兵法云：将听吾计，用之必胜——军师就位，方能运筹帷幄。
知人善任，点兵有据——选对将领，战事已半。
"""

import os
import sys

# 确保可以导入 bingfu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bingfu import BingFu, Agent
from bingfu.llm import LLMFactory, LLMConfig
from bingfu.profile import GeneralProfile, CombatStats, CombatStyle
from bingfu.presets import get_preset
from bingfu.visual.console import MilitaryCommandConsole


def main():
    print("=" * 60)
    print("  兵符 · 中军帐  Multi-Agent 战役指挥系统 v0.5.0")
    print("  新增功能：将军战力体系 + 智能派兵")
    print("=" * 60)
    print()

    # ===== 1. 创建 BingFu 框架 =====
    master = BingFu(name="兵符", version="0.5.0")

    # ===== 2. 初始化 LLM（军师） =====
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    if not api_key:
        try:
            import tomli as tomllib
        except ImportError:
            tomllib = None

        if tomllib:
            config_path = os.path.expanduser("~/.openharness/config.toml")
            if os.path.exists(config_path):
                with open(config_path, "rb") as f:
                    toml_data = tomllib.load(f)
                api_key = toml_data.get("api_key", "")

    if api_key:
        print(f"🧠 军师初始化中... (DeepSeek)")
        llm_config = LLMConfig(
            provider="deepseek",
            api_key=api_key,
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2048,
        )
        llm = LLMFactory.create(llm_config)
        master.set_llm(llm)
        print(f"🧠 军师就位：DeepSeek (deepseek-chat)")
    else:
        print("⚠️  未检测到 DEEPSEEK_API_KEY，将领将以关键词匹配模式运作")
        llm = None

    # ===== 3. 创建将领 Agent（带预设战力档案）=====
    # 从预设库加载档案
    generals_setup = [
        ("韩信", "统帅", "汉初三杰之一，善于统兵作战，出奇制胜", "online"),
        ("白起", "主将", "战国四大名将之首，善于歼灭战", "busy"),
        ("诸葛亮", "军师", "卧龙先生，善于谋略推演", "online"),
        ("项羽", "虎将", "西楚霸王，勇冠三军", "idle"),
    ]

    for name, role, desc, _ in generals_setup:
        try:
            # 尝试从预设库获取档案
            profile = get_preset(name)
        except KeyError:
            # 无预设则使用通用档案
            profile = GeneralProfile(
                style=CombatStyle.COMMAND,
                specialties=["综合分析"],
                weaknesses=[],
                stats=CombatStats(),
                bio=desc,
            )

        agent = Agent(
            name=name,
            role=role,
            description=desc,
            profile=profile,
            llm=llm,
            system_prompt=(
                f"你是{name}，{desc}。"
                f"作战风格：{profile.style.value}。"
                f"专长：{'、'.join(profile.specialties)}。"
                f"弱项：{'、'.join(profile.weaknesses)}。"
                f"用简练有力的军令风格回复。在分析局势时要条理清晰，给出的建议要具有可操作性。"
            ),
        )
        master.add_agent(agent)
        print(f"  🎖️  {name}（{role}）已就位 — {profile}")

    # 启用指挥系统（解锁智能派兵）
    master.enable_commander(name="中军帐")
    print(f"  📯 指挥系统已启用：{master.commander}")

    # ===== 4. 创建可视化控制台 =====
    console = MilitaryCommandConsole(
        title="兵符 · 中军帐 v0.5.0",
        width=1200,
        height=800,
        llm_provider=llm,
        bingfu_instance=master,
    )

    # 同步 Agent 状态到控制台 UI
    status_map = {
        "online": "online",
        "busy": "busy",
        "idle": "idle",
        "offline": "offline",
    }
    for name, role, desc, status_text in generals_setup:
        console.add_general(name, status_map.get(status_text, "online"), role, desc)

    # 添加初始军情
    console.add_report("系统就绪", "兵符框架 v0.5.0 初始化完成，所有将领已就位（含战力档案）", "success")
    console.add_report("侦察回报", "敌军粮草队已过乌江，预计明日午时抵达", "success")
    console.add_report("斥候急报", "发现敌军增援部队约三万人，从北面逼近", "warning")
    console.add_report("后勤报告", "我军粮草尚可支撑七日", "info")

    # 更新战役态势
    console.update_battle_status(30000, 80000, "敌众我寡，宜用奇兵")

    # 添加战术建议
    console.clear_tactics()
    console.add_tactics("以逸待劳，后发制人")
    console.add_tactics("诱敌深入，设伏聚歼")
    console.add_tactics("断其粮道，围而不攻")

    # ===== 5. 启动！ =====
    print()
    print("🥁 中军帐已升起，输入指令开始指挥！")
    print()
    print("  【新功能：将军战力档案 + 智能派兵】")
    print("  /match <任务>  — 点兵，展示所有将领匹配评分")
    print("  /smart <任务>  — 智能派兵，自动选最优将领执行")
    print()
    print("  自然语言示例：")
    print("    让系统自动选择合适的将领去完成任务")
    print("    /smart 分析这份数据报告并给出建议")
    print()
    print("=" * 60)

    console.run()


if __name__ == "__main__":
    main()
