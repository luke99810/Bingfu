"""
Custom Agent Example (自定义智能体示例)
Demonstrates how to create custom agents with tools and memory.
Inspired by ancient Chinese warfare: 练兵 (training soldiers).
"""

# ★ 让这个示例在**未安装包**的情况下也能直接跑（见 _bootstrap.py）。
#   没有这一行，clone 下来第一条 `python examples/xxx.py` 就是
#   ModuleNotFoundError —— 六个示例此前无一例外。
import _bootstrap  # noqa: F401

from bingfu import BingFu, Agent, Tool, Memory
from bingfu.signal import drum, gong


def custom_agent_basic():
    """
    Basic custom agent (基础自定义智能体).
    """
    print("=" * 50)
    print("Custom Agent Basic Example (自定义智能体基础示例)")
    print("=" * 50)
    
    # Create BingFu master
    master = BingFu(name="自定义智能体指挥官")
    
    # Create a custom agent with tools
    print("\n📜 Step 1: Create custom agent with tools...")
    
    # Define custom tools
    def scout_tool(location: str) -> str:
        """Scout tool - simulates scouting a location."""
        return f"📍 Scouting {location}... Found enemy patrol (2 units)."
    
    def report_tool() -> str:
        """Report tool - generates a scout report."""
        return "📋 Scout Report: Enemy forces estimated at 50 units. Ready for attack."
    
    # Create tools
    scout_tool_obj = Tool.create(
        name="Scout",
        func=scout_tool,
        description="Scout a location and report findings"
    )
    report_tool_obj = Tool.create(
        name="Report",
        func=report_tool,
        description="Generate a scout report"
    )
    
    # Create agent with tools
    scout = Agent(
        name="精锐斥候",
        role="Elite Scout",
        description="Highly trained scout with advanced tools"
    )
    scout.add_tool(scout_tool_obj)
    scout.add_tool(report_tool_obj)
    
    print(f"  ✅ Created agent: {scout.name}")
    print(f"  ✅ Added tools: {[t.name for t in scout.tools]}")
    
    # Add to master
    master.add_agent(scout)
    
    # Use tools
    print("\n📜 Step 2: Using tools...")
    result1 = scout_tool_obj.use("敌方营地")
    print(f"  📍 Scout tool result: {result1}")
    
    result2 = report_tool_obj.use()
    print(f"  📋 Report tool result: {result2}")
    
    print("\n" + "=" * 50)
    print("✅ Custom agent basic example completed!")
    print("=" * 50)


def custom_agent_with_memory():
    """
    Custom agent with memory (带记忆的自定义智能体).
    """
    print("\n" + "=" * 50)
    print("Custom Agent with Memory Example (带记忆的自定义智能体)")
    print("=" * 50)
    
    # Create BingFu master
    master = BingFu(name="记忆系统指挥官")
    
    # Create memory
    print("\n📜 Step 1: Create memory...")
    memory = Memory(
        name="战略记忆",
        memory_type="dict"  # Use in-memory dict (can also use "file")
    )
    print(f"  ✅ Created memory: {memory}")
    
    # Store some data
    print("\n📜 Step 2: Store data in memory...")
    memory.store("last_scout_location", "敌方营地北侧")
    memory.store("enemy_strength", "约50人")
    memory.store("recommendation", "建议发起夜袭")
    print(f"  ✅ Stored 3 items in memory")
    
    # Retrieve data
    print("\n📜 Step 3: Retrieve data from memory...")
    location = memory.retrieve("last_scout_location")
    strength = memory.retrieve("enemy_strength")
    recommendation = memory.retrieve("recommendation")
    print(f"  📍 Last location: {location}")
    print(f"  ⚔️ Enemy strength: {strength}")
    print(f"  💡 Recommendation: {recommendation}")
    
    # Show all memory
    print("\n📜 Step 4: Show all memory...")
    all_data = memory.all()
    for key, value in all_data.items():
        print(f"  - {key}: {value}")
    
    # Create agent with memory
    print("\n📜 Step 5: Create agent with memory...")
    analyst = Agent(
        name="军情分析师",
        role="Intelligence Analyst"
    )
    analyst.memory = memory
    master.add_agent(analyst)
    
    print(f"  ✅ Agent '{analyst.name}' has memory attached")
    
    print("\n" + "=" * 50)
    print("✅ Custom agent with memory example completed!")
    print("=" * 50)


def custom_agent_advanced():
    """
    Advanced custom agent (进阶自定义智能体).
    Demonstrates creating a specialized battle agent.
    """
    print("\n" + "=" * 50)
    print("Advanced Custom Agent Example (进阶自定义智能体)")
    print("=" * 50)
    
    # Define specialized tools
    def analyze_terrain(terrain: str) -> str:
        """Analyze terrain for tactical advantage."""
        advantages = {
            "山林": "🌲 树林遮蔽，适合伏击",
            "平原": "🌾 开阔地带，骑兵优势",
            "河流": "🌊 河流天堑，可用于防守",
            "山地": "⛰️ 高地优势，视野开阔"
        }
        return advantages.get(terrain, "未知地形")
    
    def calculate_strength(ally: int, enemy: int) -> str:
        """Calculate battle strength ratio."""
        ratio = ally / enemy if enemy > 0 else 0
        if ratio > 1.5:
            return f"⚔️ 我强敌弱 (比率: {ratio:.2f})"
        elif ratio > 1.0:
            return f"⚔️ 我占优势 (比率: {ratio:.2f})"
        elif ratio > 0.5:
            return f"⚔️ 势均力敌 (比率: {ratio:.2f})"
        else:
            return f"⚔️ 我弱敌强 (比率: {ratio:.2f})"
    
    def suggest_tactic(strength_ratio: float) -> str:
        """Suggest battle tactic based on strength ratio."""
        if strength_ratio > 1.5:
            return "🗡️ 建议: 全线进攻，速战速决"
        elif strength_ratio > 1.0:
            return "🗡️ 建议: 稳步推进，蚕食敌军"
        elif strength_ratio > 0.5:
            return "🗡️ 建议: 固守待援，寻机反击"
        else:
            return "🗡️ 建议: 且战且退，保存实力"
    
    # Create tools
    terrain_tool = Tool.create(name="分析地形", func=analyze_terrain)
    strength_tool = Tool.create(name="计算战力", func=calculate_strength)
    tactic_tool = Tool.create(name="建议战术", func=suggest_tactic)
    
    # Create battle commander agent
    commander = Agent(
        name="战术指挥官",
        role="Battle Commander",
        description="Specialized in tactical analysis and battle planning"
    )
    commander.add_tool(terrain_tool)
    commander.add_tool(strength_tool)
    commander.add_tool(tactic_tool)
    
    print(f"\n📜 Created battle commander: {commander.name}")
    print(f"  📋 Role: {commander.role}")
    print(f"  🔧 Tools: {[t.name for t in commander.tools]}")
    
    # Simulate battle planning
    print("\n⚔️ Battle Planning Simulation (战役推演)...")
    
    # Step 1: Analyze terrain
    print("\n  📍 Step 1: Analyze terrain...")
    terrain_result = terrain_tool.use("山林")
    print(f"    ✅ {terrain_result}")
    
    # Step 2: Calculate strength
    print("\n  ⚔️ Step 2: Calculate strength...")
    strength_result = strength_tool.use(ally=80, enemy=50)
    print(f"    ✅ {strength_result}")
    
    # Step 3: Suggest tactic
    print("\n  🗡️ Step 3: Suggest tactic...")
    tactic_result = tactic_tool.use(1.6)
    print(f"    ✅ {tactic_result}")
    
    print("\n" + "=" * 50)
    print("✅ Advanced custom agent example completed!")
    print("=" * 50)


if __name__ == "__main__":
    custom_agent_basic()
    custom_agent_with_memory()
    custom_agent_advanced()
