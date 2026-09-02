"""
BingFu 中军帐可视化示例
演示如何使用 MilitaryCommandConsole 可视化控制台

运行方式:
    python examples/console_demo.py
"""

# ★ 让这个示例在**未安装包**的情况下也能直接跑（见 _bootstrap.py）。
#   没有这一行，clone 下来第一条 `python examples/xxx.py` 就是
#   ModuleNotFoundError —— 六个示例此前无一例外。
import _bootstrap  # noqa: F401

from bingfu.visual import MilitaryCommandConsole
from bingfu.visual.styles import COLORS
import time


def basic_demo():
    """基础演示"""
    print("正在启动兵符中军帐...")

    # 创建控制台实例
    console = MilitaryCommandConsole(
        title="兵符 · 中军帐 - 演示模式",
        width=1200,
        height=800
    )

    # 添加自定义将领
    console.add_general("岳飞", "online", "主将", "精忠报国，誓守中原")
    console.add_general("戚继光", "busy", "名将", "鸳鸯阵已布，倭寇闻风丧胆")

    # 添加军情报告
    console.add_report("战况汇报", "岳家军连战连捷，收复建康", "success")
    console.add_report("紧急军情", "金兵主力蠢蠢欲动，恐有异动", "warning")
    console.add_report("后勤情报", "粮草充足，可支撑月余", "info")

    # 更新战役态势
    console.update_battle_status(
        own_strength=50000,
        enemy_strength=70000,
        strategy="敌进我退，敌退我追，敌疲我扰"
    )

    # 添加战术建议
    console.clear_tactics()
    console.add_tactics("兵贵神速，出其不意")
    console.add_tactics("十则围之，五则攻之")
    console.add_tactics("知彼知己，百战不殆")

    # 启动控制台
    console.run()


def integrate_with_bingfu():
    """与 BingFu 框架集成的演示"""
    from bingfu import BingFu, Agent
    from bingfu.memory import Memory
    from bingfu.visual.console import MilitaryCommandConsole

    # 初始化框架组件
    memory = Memory()
    bingfu = BingFu(memory=memory)

    # 创建将领 Agent
    hanxin = Agent(
        name="韩信",
        role="统帅",
        system_prompt="你是韩信，西汉开国名将，兵仙。"
    )

    baiqi = Agent(
        name="白起",
        role="主将",
        system_prompt="你是白起，战国名将，嗜战如命。"
    )

    # 注册 Agent
    bingfu.register_agent(hanxin)
    bingfu.register_agent(baiqi)

    # 创建可视化控制台
    console = MilitaryCommandConsole(
        title="兵符 · 中军帐 - 框架集成",
        commander=bingfu.commander,
        memory=memory
    )

    # 同步 Agent 状态到控制台
    def sync_agent_status():
        for agent in bingfu.commander.agents.values():
            console.add_general(
                name=agent.name,
                status="online",
                role=agent.role or "",
                message=f"任务数: {len(agent.tasks)}"
            )

    sync_agent_status()

    # 添加示例军情
    console.add_report(
        "系统就绪",
        "框架初始化完成，将领已就位",
        "success"
    )

    # 启动控制台
    console.run()


def real_time_update_demo():
    """实时更新演示"""
    from bingfu.visual import MilitaryCommandConsole

    console = MilitaryCommandConsole(
        title="兵符 · 中军帐 - 实时更新",
        width=1200,
        height=800
    )

    # 启动非阻塞模式
    console.run(blocking=False)

    # 模拟实时更新
    print("开始模拟实时更新...")

    for i in range(10):
        # 更新战役态势
        own = 30000 + i * 1000
        enemy = 80000 - i * 500
        console.update_battle_status(
            own,
            enemy,
            f"第{i+1}回合战况"
        )

        # 动态添加军情
        if i % 3 == 0:
            console.add_report(
                f"第{i+1}回合战报",
                f"我军攻势顺利，歼敌{(i+1)*500}人",
                "success"
            )

        # 更新将领状态
        console.add_general(
            "韩信",
            status=["online", "busy", "idle"][i % 3],
            role="统帅",
            message=f"回合 {i+1} 指挥中"
        )

        time.sleep(2)

    print("实时更新演示完成")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "basic":
            basic_demo()
        elif mode == "integrate":
            integrate_with_bingfu()
        elif mode == "realtime":
            real_time_update_demo()
        else:
            print(f"未知模式: {mode}")
            print("可用模式: basic, integrate, realtime")
    else:
        # 默认运行基础演示
        basic_demo()
