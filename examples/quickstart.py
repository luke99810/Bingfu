"""
BingFu 快速入门指南 (Quick Start Guide)
最简单的开始方式，5分钟上手兵符框架

运行方式:
    python examples/quickstart.py
"""

# ★ 让这个示例在**未安装包**的情况下也能直接跑（见 _bootstrap.py）。
#   没有这一行，clone 下来第一条 `python examples/xxx.py` 就是
#   ModuleNotFoundError —— 六个示例此前无一例外。
import _bootstrap  # noqa: F401

from bingfu import BingFu, Agent, Tool, Memory
from bingfu.commander import Commander
from bingfu.signal import drum, gong


def quickstart_1_single_agent():
    """
    快速入门 1: 单智能体使用
    最基础的使用方式
    """
    print("\n" + "=" * 60)
    print("🚀 快速入门 1: 单智能体使用")
    print("=" * 60)

    # 1. 创建兵符主对象
    master = BingFu(name="我的指挥官")

    # 2. 创建智能体
    scout = Agent(name="侦察兵", role="侦察")

    # 3. 添加到兵符
    master.add_agent(scout)

    # 4. 发送击鼓信号启动任务
    result = master.drum("侦察兵", "侦察敌方营地")
    print(f"\n📍 任务: {result}")

    # 5. 发送鸣金信号停止
    result = master.gong("侦察兵")
    print(f"📍 停止: {result}")

    print("\n✅ 完成! 学会了创建和使用单个智能体。")


def quickstart_2_multi_agent():
    """
    快速入门 2: 多智能体协作
    多个智能体协同工作
    """
    print("\n" + "=" * 60)
    print("🚀 快速入门 2: 多智能体协作")
    print("=" * 60)

    # 创建兵符
    master = BingFu(name="战役指挥部")

    # 创建多个智能体
    agents = [
        Agent(name="先锋营", role="先锋"),
        Agent(name="主力军", role="主攻"),
        Agent(name="后勤司", role="后勤"),
    ]

    # 添加所有智能体
    for agent in agents:
        master.add_agent(agent)

    print(f"\n📍 已创建 {len(master.agents)} 个智能体")

    # 启用指挥官模式
    master.enable_commander(name="元帅")

    # 击鼓全军
    print("\n🥁 击鼓!全军出击!")
    results = master.drum_all("发起进攻")
    for name, result in results.items():
        print(f"  - {name}: 收到")

    # 鸣金收兵
    print("\n🔔 鸣金!收兵!")
    results = master.gong_all()
    for name, result in results.items():
        print(f"  - {name}: 收到")

    print("\n✅ 完成! 学会了多智能体协作。")


def quickstart_3_with_tools():
    """
    快速入门 3: 智能体 + 工具
    给智能体添加工具扩展能力
    """
    print("\n" + "=" * 60)
    print("🚀 快速入门 3: 智能体 + 工具")
    print("=" * 60)

    # 定义一个工具函数
    def scout_location(location: str) -> str:
        """侦察工具 - 侦察指定位置"""
        return f"📍 {location} 侦察完成: 发现敌军3人"

    # 创建工具
    scout_tool = Tool.create(
        name="侦察",
        func=scout_location,
        description="侦察指定地点并返回结果"
    )

    # 创建智能体
    scout = Agent(name="精锐斥候", role="侦察兵")

    # 给智能体添加工具
    scout.add_tool(scout_tool)

    # 使用工具
    print("\n🔧 使用工具:")
    result = scout_tool.use("敌方营地北侧")
    print(f"  {result}")

    print("\n✅ 完成! 学会了给智能体添加工具。")


def quickstart_4_with_memory():
    """
    快速入门 4: 智能体 + 记忆
    智能体具有持久记忆能力
    """
    print("\n" + "=" * 60)
    print("🚀 快速入门 4: 智能体 + 记忆")
    print("=" * 60)

    # 创建记忆 (dict模式，数据存储在内存中)
    memory = Memory(name="战场记忆", memory_type="dict")

    # 存储信息
    memory.store("敌将", "项羽")
    memory.store("我军士气", "高昂")
    memory.store("建议", "速战速决")

    print("\n📝 已存储记忆:")
    for key, value in memory.all().items():
        print(f"  - {key}: {value}")

    # 读取信息
    print("\n📖 读取记忆:")
    print(f"  敌将: {memory.retrieve('敌将')}")
    print(f"  建议: {memory.retrieve('建议')}")

    print("\n✅ 完成! 学会了使用记忆系统。")


def quickstart_5_tactics_engine():
    """
    快速入门 5: 孙子兵法战术引擎
    使用古代智慧分析战场态势
    """
    print("\n" + "=" * 60)
    print("🚀 快速入门 5: 孙子兵法战术引擎")
    print("=" * 60)

    # ★ 这一段原先用的是 `TacticsEngine` 和 `engine.analyze(context)` ——
    #   两者都已不存在（引擎改名为 TacticEngine，接口换成了 select_tactic）。
    #   于是**快速入门**，新人跑的第一条命令，直接 ImportError。
    #   它能一直坏着，唯一的原因是没有任何东西跑过 examples/。
    from bingfu.presets import PRESET_GENERALS, get_preset
    from bingfu.tactics import TacticEngine

    # 备好将领 —— 战术是「选给谁用」的，没有将领就无从谈起
    generals = {
        name: Agent(name=name, profile=get_preset(name))
        for name in list(PRESET_GENERALS)[:3]
    }
    print("\n⚔️ 可用将领: " + "、".join(generals))

    engine = TacticEngine()
    task = "我军三万对敌八万，需在山地拖住对方主力"
    print(f"📜 任务: {task}")

    # 核心接口：为「任务 + 将领池」选出最优的（战术, 将领）配对
    result = engine.select_tactic(task, generals)

    ctx = result.tactical_context
    print(
        f"\n🗺️ 战场态势: {ctx.regime}"
        f"（我方 {ctx.self_strength:.0f} / 敌方 {ctx.enemy_strength:.0f}，地形 {ctx.terrain}）"
    )
    print(f"🎯 推荐战术: {result.selected_tactic.name}")
    print(f"👤 建议派遣: {result.selected_agent_name}")
    print(
        f"📊 综合得分: {result.combined_score:.3f}"
        f"（对齐 {result.alignment_score:.3f} · 战力 {result.power_score:.3f}）"
    )
    print(f"\n💡 理由: {result.explanation}")

    print("\n✅ 完成! 学会了使用战术引擎。")


def quickstart_6_visual_console():
    """
    快速入门 6: 中军帐可视化
    打开可视化控制台监控智能体
    """
    print("\n" + "=" * 60)
    print("🚀 快速入门 6: 中军帐可视化")
    print("=" * 60)

    from bingfu.visual import MilitaryCommandConsole

    # 创建可视化控制台
    console = MilitaryCommandConsole(title="兵符 · 中军帐")

    # 添加将领
    console.add_general("韩信", "online", "统帅", "正在分析战场")
    console.add_general("白起", "busy", "主将", "率领先锋突击")

    # 添加军情
    console.add_report("侦察回报", "发现敌军粮草运输队", "success")
    console.add_report("紧急军情", "敌军增援约三万", "warning")

    # 更新战役态势
    console.update_battle_status(30000, 80000, "敌众我寡，宜用奇兵")

    print("\n📍 可视化控制台已创建!")
    print("   运行 python examples/console_demo.py 启动完整演示")
    print("\n✅ 完成! 学会了使用可视化控制台。")


def quickstart_7_famous_generals():
    """
    快速入门 7: 古代名将
    使用预置的著名将领智能体
    """
    print("\n" + "=" * 60)
    print("🚀 快速入门 7: 古代名将")
    print("=" * 60)

    # ★ 第二处旧 API：`sunbin.analyze_battlefield(context)` 也已不存在，
    #   现在的接口是 `analyze_and_recommend(task, agents)` ——
    #   输入从「战场态势对象」变成了「任务 + 将领池」，态势由引擎自己推导。
    from bingfu.presets import PRESET_GENERALS, get_preset
    from bingfu.tactics import SunTzuAgent

    sunbin = SunTzuAgent(name="孙子")
    print(f"\n📍 创建名将: {sunbin.name}")

    generals = {
        name: Agent(name=name, profile=get_preset(name))
        for name in list(PRESET_GENERALS)[:3]
    }

    print("\n⚔️ 孙子曰:")
    result = sunbin.analyze_and_recommend("我军三万对敌八万，需在山地拖住对方主力", generals)
    print(f"  战术：{result.selected_tactic.name}")
    print(f"  派遣：{result.selected_agent_name}")
    print(f"  {result.explanation}")

    # 获取兵法智慧
    wisdom = sunbin.get_wisdom()
    print(f"\n📜 古训: {wisdom}")

    print("\n✅ 完成! 学会了使用古代名将。")


def run_all():
    """运行所有快速入门教程"""
    print("\n" + "=" * 60)
    print("🎯 BINGFU 兵符框架 - 5分钟快速入门")
    print("=" * 60)
    print("\n本教程将带你快速上手兵符框架的所有核心功能。\n")

    quickstart_1_single_agent()
    quickstart_2_multi_agent()
    quickstart_3_with_tools()
    quickstart_4_with_memory()
    quickstart_5_tactics_engine()
    quickstart_6_visual_console()
    quickstart_7_famous_generals()

    print("\n" + "=" * 60)
    print("🎉 恭喜! 你已掌握兵符框架的所有基础功能!")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 查看 examples/ 目录下的完整示例")
    print("  2. 阅读 README.md 了解详细文档")
    print("  3. 运行 python examples/console_demo.py 体验可视化")
    print("  4. 开始构建你的智能体系统!")
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 运行指定的入门教程
        num = sys.argv[1]
        {
            "1": quickstart_1_single_agent,
            "2": quickstart_2_multi_agent,
            "3": quickstart_3_with_tools,
            "4": quickstart_4_with_memory,
            "5": quickstart_5_tactics_engine,
            "6": quickstart_6_visual_console,
            "7": quickstart_7_famous_generals,
        }.get(num, run_all)()
    else:
        # 运行所有
        run_all()
