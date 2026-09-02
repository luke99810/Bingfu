"""
BingFu Framework - 工具使用示例
Demonstrates how to use Tools with Agents.

本示例展示：
1. 创建带函数的基础工具
2. 创建复杂工具（模拟古代兵器）
3. 为Agent添加工具并执行
4. 工具组合使用
"""

# ★ 让这个示例在**未安装包**的情况下也能直接跑（见 _bootstrap.py）。
#   没有这一行，clone 下来第一条 `python examples/xxx.py` 就是
#   ModuleNotFoundError —— 六个示例此前无一例外。
import _bootstrap  # noqa: F401

from bingfu import Agent, Tool, BingFu


# ==================== 基础工具示例 ====================

def sword_attack(enemy: str, damage: int = 10) -> str:
    """模拟剑的攻击"""
    return f"⚔️ 剑挥向 {enemy}，造成 {damage} 点伤害！"


def shield_block(attack_power: int) -> str:
    """模拟盾的防御"""
    blocked = min(attack_power, 8)
    remaining = attack_power - blocked
    return f"🛡️ 盾牌抵挡了 {blocked} 点伤害，剩余 {remaining} 点伤害需要承受。"


def scout_reconnaissance(area: str) -> str:
    """侦察兵情报收集"""
    return f"🔍 侦察兵报告：{area} 区域发现敌军动向，疑似伏兵。"


def archer_volley(target: str, arrows: int = 5) -> str:
    """弓箭齐射"""
    hits = arrows - 1  # 假设有1支箭射偏
    return f"🏹 弓箭手齐射 {arrows} 支箭，命中 {hits} 支，目标 {target} 遭受重创！"


def cavalry_charge(enemy_camp: str) -> str:
    """骑兵冲锋"""
    return f"🐎 骑兵连夜奔袭，直捣 {enemy_camp} 大营！"


# ==================== Agent示例 ====================

def demo_basic_tools():
    """基础工具使用演示"""
    print("=" * 60)
    print("演示1：基础工具创建和使用")
    print("=" * 60)

    # 创建各种工具
    sword = Tool.create(
        name="青虹剑",
        func=sword_attack,
        description="刘备的佩剑，削铁如泥"
    )

    shield = Tool.create(
        name="玄铁盾",
        func=shield_block,
        description="重甲步兵的防御利器"
    )

    # 使用工具
    print(f"\n工具信息: {sword}")
    print(f"工具信息: {shield}")

    print(f"\n{sword.use('曹操大军')}")
    print(f"{shield.use(15)}")


def demo_agent_with_tools():
    """Agent使用工具演示"""
    print("\n" + "=" * 60)
    print("演示2：Agent使用工具")
    print("=" * 60)

    # 创建武器工具
    sword = Tool.create(name="Sword", func=sword_attack)
    bow = Tool.create(name="Bow", func=archer_volley)
    scout = Tool.create(name="Scout", func=scout_reconnaissance)

    # 创建Agent并添加工具
    warrior = Agent(name=" Warriors")
    warrior.add_tool(sword)
    warrior.add_tool(bow)

    scout_agent = Agent(name="Scout")
    scout_agent.add_tool(scout)

    print(f"\n战士Agent: {warrior}")
    print(f"侦察Agent: {scout_agent}")

    # Agent执行任务（使用工具）
    print("\n--- 战斗开始 ---")
    result1 = warrior.execute("Attack the enemy camp with sword")
    print(f"战士执行: {result1}")

    result2 = scout_agent.execute("Scout the forest area")
    print(f"侦察执行: {result2}")


def demo_ancient_weapons_scenario():
    """古代兵器场景演示"""
    print("\n" + "=" * 60)
    print("演示3：古代兵器完整场景")
    print("=" * 60)

    # 创建各种古代兵器工具
    weapons = {
        "sword": Tool.create(name="长剑", func=sword_attack),
        "shield": Tool.create(name="盾牌", func=shield_block),
        "bow": Tool.create(name="弓箭", func=archer_volley),
        "cavalry": Tool.create(name="骑兵", func=cavalry_charge),
        "scout": Tool.create(name="斥候", func=scout_reconnaissance),
    }

    # 创建军队编制
    army = BingFu()

    # 步兵
    infantry = Agent(name="步兵营")
    infantry.add_tool(weapons["sword"])
    infantry.add_tool(weapons["shield"])

    # 弓箭手
    archer = Agent(name="弓箭手方阵")
    archer.add_tool(weapons["bow"])

    # 骑兵
    cavalry = Agent(name="骑兵队")
    cavalry.add_tool(weapons["cavalry"])

    # 斥候
    scout_unit = Agent(name="斥候班")
    scout_unit.add_tool(weapons["scout"])

    # 添加到军队
    army.add_agent(infantry)
    army.add_agent(archer)
    army.add_agent(cavalry)
    army.add_agent(scout_unit)

    # 战役流程
    print("\n📜 战役开始！\n")

    # 1. 侦察阶段
    print("【第一阶段】侦察部署")
    scout_result = army.drum("斥候班", "Scout the enemy position")
    print(f"斥候回报: {scout_result}")

    # 2. 远程攻击
    print("\n【第二阶段】弓箭压制")
    archer_result = army.drum("弓箭手方阵", "Fire arrows at the front line")
    print(f"弓箭手: {archer_result}")

    # 3. 骑兵突击
    print("\n【第三阶段】骑兵突击")
    cavalry_result = army.drum("骑兵队", "Charge the enemy flank")
    print(f"骑兵: {cavalry_result}")

    # 4. 步兵推进
    print("\n【第四阶段】步兵推进")
    infantry_result = army.drum("步兵营", "Attack the weakened enemy")
    print(f"步兵: {infantry_result}")

    # 鸣金收兵
    print("\n【鸣金收兵】")
    army.gong_all()
    print("战斗结束，各部撤回营地休整。")


def demo_tool_management():
    """工具管理演示"""
    print("\n" + "=" * 60)
    print("演示4：工具动态管理")
    print("=" * 60)

    # 创建Agent
    agent = Agent(name="将军")

    # 初始无武器
    print(f"初始武器列表: {agent.tools}")

    # 添加工具
    sword = Tool.create(name="剑", func=sword_attack)
    bow = Tool.create(name="弓", func=archer_volley)

    agent.add_tool(sword)
    print(f"添加剑后: {[t.name for t in agent.tools]}")

    agent.add_tool(bow)
    print(f"添加弓后: {[t.name for t in agent.tools]}")

    # 移除工具
    agent.remove_tool("弓")
    print(f"移除弓后: {[t.name for t in agent.tools]}")

    # 执行任务
    print(f"\n{agent.execute('Attack enemy')}")

    # 清空工具
    agent.clear_tools()
    print(f"清空工具后: {agent.tools}")


def demo_combined_tools():
    """组合工具使用演示"""
    print("\n" + "=" * 60)
    print("演示5：组合工具战术")
    print("=" * 60)

    def calculate_damage(base: int, multiplier: float, defense: int) -> int:
        """计算伤害"""
        damage = int(base * multiplier)
        final_damage = max(0, damage - defense)
        return final_damage

    def apply_buff(buff_name: str, target_hp: int) -> str:
        """施加增益"""
        return f"✨ {buff_name}生效！目标HP从 {target_hp} 恢复至 {target_hp + 20}"

    damage_calc = Tool.create(name="伤害计算", func=calculate_damage)
    heal_buff = Tool.create(name="治疗术", func=apply_buff)

    calculator = Agent(name="谋士")
    calculator.add_tool(damage_calc)
    calculator.add_tool(heal_buff)

    # 使用伤害计算
    result = calculator.tools[0].use(100, 1.5, 30)
    print(f"谋士计算：基础伤害100，倍率1.5，敌人防御30")
    print(f"计算结果：造成 {result} 点伤害")

    # 使用治疗
    heal_result = calculator.tools[1].use("回春术", 80)
    print(f"治疗结果: {heal_result}")


if __name__ == "__main__":
    # 运行所有演示
    demo_basic_tools()
    demo_agent_with_tools()
    demo_ancient_weapons_scenario()
    demo_tool_management()
    demo_combined_tools()

    print("\n" + "=" * 60)
    print("✨ 所有演示完成！")
    print("=" * 60)
