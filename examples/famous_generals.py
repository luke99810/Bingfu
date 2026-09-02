"""
BingFu Framework - 古代名将示例
Demonstrates Agent implementations inspired by famous Chinese generals.

本示例展示：
1. 十大古代名将的Agent实现
2. 名将独特战术能力
3. 战役模拟场景
"""

# ★ 让这个示例在**未安装包**的情况下也能直接跑（见 _bootstrap.py）。
#   没有这一行，clone 下来第一条 `python examples/xxx.py` 就是
#   ModuleNotFoundError —— 六个示例此前无一例外。
import _bootstrap  # noqa: F401

from bingfu import Agent, Tool, Memory, BingFu


# ==================== 名将工具定义 ====================

def bai_qi_strategy(enemy_strength: int) -> str:
    """白起战术 - 歼灭战"""
    if enemy_strength > 50000:
        return "全军出击，务必全歼敌军！不留活口。"
    return "诱敌深入，包围歼灭。"


def han_xin_tactics(terrain: str) -> str:
    """韩信战术 - 奇袭"""
    tactics = {
        "山地": "明修栈道，暗度陈仓",
        "平原": "背水一战，置之死地",
        "河流": "水淹七军，顺势而动",
        "森林": "草木皆兵，虚张声势"
    }
    return f"韩信献策：{tactics.get(terrain, '因地制宜')}"


def xiang_yu_assault(distance: int) -> str:
    """项羽战术 - 闪电战"""
    if distance < 100:
        return f"率三千江东子弟，千里奔袭，一战定乾坤！"
    return "破釜沉舟，不留退路！" if distance < 500 else "霸王别姬，时不我待"


def zhu_ge_strategy(enemy_count: int, ally_count: int) -> str:
    """诸葛亮战术 - 谋略"""
    ratio = ally_count / max(enemy_count, 1)
    if ratio < 0.3:
        return "敌众我寡，可用空城计。"
    elif ratio < 0.6:
        return "火烧新野，以少胜多。"
    elif ratio < 1.0:
        return "联吴抗曹，方为上策。"
    return "安居平五路，七擒孟获。"


def li_jing_mobility(weather: str) -> str:
    """李靖战术 - 机动战"""
    if weather == "雨天":
        return "雨天利于突袭，骑兵可速进速退。"
    elif weather == "雪天":
        return "踏雪无痕，奇兵突出。"
    return "兵贵神速，不给敌军喘息之机。"


def yue_fei_leadership(morale: int) -> str:
    """岳飞战术 - 统帅之道"""
    if morale > 80:
        return "精忠报国，身先士卒！直捣黄龙府！"
    elif morale > 50:
        return "撼山易，撼岳家军难！"
    return "激励士气，背嵬军出击！" if morale > 30 else "撤退保存实力，以待时机"


def qi_jiguang_formation() -> str:
    """戚继光战术 - 阵法"""
    return "鸳鸯阵！以十二人为一队，相互配合，刀牌手在前，长枪手在后。"


def cao_cao_scheme() -> str:
    """曹操权谋 - 挟天子以令诸侯"""
    import random
    strategies = [
        "官渡之战，以少胜多，断袁绍粮草。",
        "青梅煮酒论英雄，试探刘备心机。",
        "望梅止渴，激励士兵行军。"
    ]
    return f"曹操献策：{random.choice(strategies)}"


def guo_ziying_defense() -> str:
    """郭子仪战术 - 镇守"""
    return "安史之乱，再造唐室。单骑退回纥，不战而屈人之兵。"


def wang_jian_siege(duration: int) -> str:
    """王翦战术 - 持久战"""
    if duration < 30:
        return "深沟高垒，以逸待劳。"
    elif duration < 90:
        return "轮番进攻，消耗敌军。"
    return "养精蓄锐，一战而定天下。"


# ==================== 名将Agent ====================

def create_bai_qi() -> Agent:
    """创建白起Agent - 歼灭战专家"""
    tool = Tool.create(name="歼灭战", func=bai_qi_strategy)
    agent = Agent(name="白起", role="武安君")
    agent.add_tool(tool)
    return agent


def create_han_xin() -> Agent:
    """创建韩信Agent - 奇袭大师"""
    tool = Tool.create(name="奇袭", func=han_xin_tactics)
    memory = Memory(name="汉初三杰记忆", memory_type="dict")
    memory.store("title", "兵仙")
    memory.store("famous_battles", ["暗渡陈仓", "背水一战", "十面埋伏"])
    agent = Agent(name="韩信", role="齐王", memory=memory)
    agent.add_tool(tool)
    return agent


def create_xiang_yu() -> Agent:
    """创建项羽Agent - 闪电战之神"""
    tool = Tool.create(name="霸王突击", func=xiang_yu_assault)
    agent = Agent(name="项羽", role="西楚霸王")
    agent.add_tool(tool)
    return agent


def create_zhu_ge() -> Agent:
    """创建诸葛亮Agent - 智慧化身"""
    tool = Tool.create(name="锦囊妙计", func=zhu_ge_strategy)
    memory = Memory(name="卧龙记忆", memory_type="dict")
    memory.store("title", "卧龙先生")
    memory.store("tools", ["木牛流马", "诸葛连弩", "八卦阵"])
    agent = Agent(name="诸葛亮", role="蜀汉丞相", memory=memory)
    agent.add_tool(tool)
    return agent


def create_yue_fei() -> Agent:
    """创建岳飞Agent - 民族英雄"""
    tool = Tool.create(name="岳家军", func=yue_fei_leadership)
    agent = Agent(name="岳飞", role="宋鄂王")
    agent.add_tool(tool)
    return agent


# ==================== 战役场景 ====================

def demo_battle_annihilate():
    """战役演示：白起的歼灭战"""
    print("=" * 60)
    print("战役一：白起歼灭战")
    print("=" * 60)

    bai_qi = create_bai_qi()
    print(f"\n{bai_qi.name} ({bai_qi.role}) 出征！")

    # 面对不同规模敌军
    for strength in [30000, 80000, 100000]:
        tool = bai_qi.tools[0]
        tactic = tool.use(strength)
        print(f"  敌军 {strength} 人 → {tactic}")


def demo_battle_surprise():
    """战役演示：韩信的奇袭战"""
    print("\n" + "=" * 60)
    print("战役二：韩信奇袭战")
    print("=" * 60)

    han_xin = create_han_xin()

    print(f"\n{han_xin.name} ({han_xin.role}) 献策！")
    print(f"称号: {han_xin.memory.retrieve('title')}")
    print(f"经典战役: {han_xin.memory.retrieve('famous_battles')}")

    # 不同地形战术
    terrains = ["山地", "平原", "河流", "森林"]
    tool = han_xin.tools[0]
    for terrain in terrains:
        tactic = tool.use(terrain)
        print(f"  {terrain}地形 → {tactic}")


def demo_battle_zhuge():
    """战役演示：诸葛亮的谋略"""
    print("\n" + "=" * 60)
    print("战役三：诸葛亮谋略战")
    print("=" * 60)

    zhu_ge = create_zhu_ge()

    print(f"\n{zhu_ge.name} ({zhu_ge.role}) 用计！")
    print(f"称号: {zhu_ge.memory.retrieve('title')}")

    # 不同兵力对比
    scenarios = [
        (100000, 20000),  # 5:1
        (80000, 30000),   # 2.7:1
        (50000, 30000),   # 1.7:1
        (30000, 30000),   # 1:1
    ]
    tool = zhu_ge.tools[0]

    for enemy, ally in scenarios:
        ratio = ally / enemy * 100
        tactic = tool.use(enemy, ally)
        print(f"  敌军{enemy} vs 我军{ally} ({ratio:.0f}%兵力) → {tactic}")


def demo_famous_general_assembly():
    """名将集结演示"""
    print("\n" + "=" * 60)
    print("【名将集结】")
    print("=" * 60)

    # 创建所有名将
    generals = [
        create_bai_qi(),
        create_han_xin(),
        create_xiang_yu(),
        create_zhu_ge(),
        create_yue_fei(),
    ]

    # 创建兵符主控
    bingfu = BingFu()

    for general in generals:
        bingfu.add_agent(general)

    print("\n📜 名将录：")
    print("-" * 40)
    for agent in bingfu.list_agents():
        print(f"  {agent.name} ({agent.role if hasattr(agent, 'role') else '将领'})")

    # 模拟战役协调
    print("\n⚔️ 战役协调开始：")
    print("-" * 40)

    bingfu.drum("韩信", "制定作战计划")
    bingfu.drum("白起", "准备歼灭敌军")
    bingfu.drum("项羽", "等待突击命令")
    bingfu.drum("诸葛亮", "运筹帷幄")
    bingfu.drum("岳飞", "整军待发")

    print("\n各路兵马已就位，等待主帅号令...")


def demo_battle_simulation():
    """完整战役模拟"""
    print("\n" + "=" * 60)
    print("【巨鹿之战模拟】")
    print("=" * 60)

    # 楚霸王项羽
    xiang_yu = create_xiang_yu()

    # 谋士范增
    def fan_zeng_advice() -> str:
        return "项羽：破釜沉舟，只带三日粮草，直捣章邯大营！"

    fan_tool = Tool.create(name="范增献计", func=fan_zeng_advice)
    fan_zeng = Agent(name="范增", role="谋士")
    fan_zeng.add_tool(fan_tool)

    # 宋义
    song_yi = Agent(name="宋义", role="上将军")

    # 章邯
    def zhang_han_defense() -> str:
        return "章邯：王离，速速迎敌！"

    zhang_tool = Tool.create(name="章邯军令", func=zhang_han_defense)
    zhang_han = Agent(name="章邯", role="秦将")
    zhang_han.add_tool(zhang_tool)

    # 创建兵符
    bingfu = BingFu()
    bingfu.add_agent(xiang_yu)
    bingfu.add_agent(fan_zeng)
    bingfu.add_agent(song_yi)
    bingfu.add_agent(zhang_han)

    print("\n📜 战役背景：")
    print("  秦将章邯率军围攻赵国，诸侯援军皆不敢前。")
    print("  项羽斩杀上将军宋义，率楚军渡河救赵。\n")

    # 战役流程
    print("⚔️ 战役开始：")
    print("-" * 40)

    # 范增献计
    print(f"【谋士】{fan_zeng.name}: {fan_zeng.tools[0].use()}")

    # 项羽下令
    tool = xiang_yu.tools[0]
    print(f"【项羽】下令: {tool.use(1000)}")

    # 章邯应战
    print(f"【章邯】应战: {zhang_han.tools[0].use()}")

    # 战役结果
    print("\n📜 战役结果：")
    print("  项羽九战九捷，大破秦军，坑杀降卒二十万！")
    print("  诸侯将领入辕门，皆膝行而前，莫敢仰视。")
    print("  从此项羽威震天下，成为诸侯上将军。")


def demo_modern_application():
    """现代应用场景"""
    print("\n" + "=" * 60)
    print("【现代商业应用】")
    print("=" * 60)

    # 模拟商业竞争

    def baiqi_market(strategy: str) -> str:
        """白起式市场歼灭"""
        return f"全力投入，不留余地，目标是{strategy}市场份额第一！"

    def hanxin_penetration() -> str:
        """韩信式市场渗透"""
        return "农村包围城市，迂回包抄，差异化竞争！"

    def zhuge_strategy() -> str:
        """诸葛亮式战略规划"""
        return "隆中对：三分天下，联吴抗曹！"

    # 创建商业Agent团队
    company = BingFu()

    market_tool = Tool.create(name="市场分析", func=baiqi_market)
    ceo = Agent(name="CEO")
    ceo.add_tool(market_tool)

    strategy_tool = Tool.create(name="战略规划", func=zhuge_strategy)
    cso = Agent(name="CSO")
    cso.add_tool(strategy_tool)

    growth_tool = Tool.create(name="增长黑客", func=hanxin_penetration)
    cmo = Agent(name="CMO")
    cmo.add_tool(growth_tool)

    company.add_agent(ceo)
    company.add_agent(cso)
    company.add_agent(cmo)

    print("\n📈 商业策略会议：")
    print("-" * 40)

    print(f"CEO: {company.drum('CEO', '主航道战略')}")
    print(f"CSO: {company.drum('CSO', '战略规划')}")
    print(f"CMO: {company.drum('CMO', '市场增长')}")

    print("\n✨ 战略决策完成！")


if __name__ == "__main__":
    demo_battle_annihilate()
    demo_battle_surprise()
    demo_battle_zhuge()
    demo_famous_general_assembly()
    demo_battle_simulation()
    demo_modern_application()

    print("\n" + "=" * 60)
    print("✨ 古代名将示例完成！")
    print("=" * 60)
