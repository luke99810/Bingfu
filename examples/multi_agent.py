"""
Multi‑Agent Example (多智能体示例)
Demonstrates multi‑agent coordination using the Commander.
Inspired by ancient Chinese warfare: 多路出兵 (multiple forces attacking).
"""

# ★ 让这个示例在**未安装包**的情况下也能直接跑（见 _bootstrap.py）。
#   没有这一行，clone 下来第一条 `python examples/xxx.py` 就是
#   ModuleNotFoundError —— 六个示例此前无一例外。
import _bootstrap  # noqa: F401

from bingfu import BingFu, Agent
from bingfu.commander import Commander


def multi_agent_basic():
    """
    Basic multi‑agent coordination (基础多智能体协作).
    """
    print("=" * 50)
    print("Multi‑Agent Basic Example (多智能体基础示例)")
    print("=" * 50)
    
    # Create BingFu master
    master = BingFu(name="多路指挥官")
    
    # Create multiple agents with different roles
    agents = [
        Agent(name="先锋营", role="Vanguard"),
        Agent(name="主力军", role="Main Force"),
        Agent(name="预备队", role="Reserve"),
        Agent(name="后勤司", role="Logistics"),
    ]
    
    # Add all agents
    print("\n📜 Creating and adding agents...")
    for agent in agents:
        master.add_agent(agent)
        print(f"  ✅ Added: {agent.name} ({agent.role})")
    
    print(f"\n📊 Total agents: {len(master.agents)}")
    
    # Enable commander
    print("\n📜 Enabling commander...")
    master.enable_commander(name="元帅")
    print(f"  ✅ Commander enabled: {master.commander}")
    
    # Drum all (击鼓全军)
    print("\n🥁 Drum signal to all (击鼓全军)...")
    results = master.drum_all("发起进攻")
    for name, result in results.items():
        print(f"  - {result}")
    
    # Gong all (鸣金收兵)
    print("\n🔔 Gong signal to all (鸣金收兵)...")
    results = master.gong_all()
    for name, result in results.items():
        print(f"  - {result}")
    
    # Status check
    print("\n📊 Final status:")
    status = master.status()
    for key, value in status.items():
        print(f"  - {key}: {value}")
    
    print("\n" + "=" * 50)
    print("✅ Multi‑agent basic example completed!")
    print("=" * 50)


def multi_agent_advanced():
    """
    Advanced multi‑agent coordination (进阶多智能体协作).
    Demonstrates using the Commander directly for complex coordination.
    """
    print("\n" + "=" * 50)
    print("Multi‑Agent Advanced Example (多智能体进阶示例)")
    print("=" * 50)
    
    # Create commander directly
    commander = Commander(name="前线总指挥", strategy="round_robin")
    
    # Create specialized agents
    agents = [
        Agent(name="侦察连", role="Scout Unit"),
        Agent(name="突击营", role="Assault Battalion"),
        Agent(name="防守团", role="Defense Regiment"),
        Agent(name="补给队", role="Supply Team"),
    ]
    
    # Add agents to commander
    print("\n📜 Building army...")
    for agent in agents:
        commander.add_agent(agent)
        print(f"  ✅ {agent.name} ({agent.role}) joined the army")
    
    # Show status
    print("\n📊 Army status:")
    status = commander.status()
    for key, value in status.items():
        print(f"  - {key}: {value}")
    
    # Coordinate (协调作战)
    print("\n⚔️ Coordinating attack (协调作战)...")
    results = commander.coordinate("发起总攻", strategy="round_robin")
    for name, result in results.items():
        print(f"  - {result}")
    
    # Drum specific agent (击鼓特定智能体)
    print("\n🥁 Drum to specific agent (击鼓特定智能体)...")
    result = commander.drum_one("突击营", "突袭敌军左翼")
    print(f"  ✅ {result}")
    
    # Gong specific agent (鸣金特定智能体)
    print("\n🔔 Gong to specific agent (鸣金特定智能体)...")
    result = commander.gong_one("突击营")
    print(f"  ✅ {result}")
    
    print("\n" + "=" * 50)
    print("✅ Multi‑agent advanced example completed!")
    print("=" * 50)


def multi_agent_battle_scenario():
    """
    Multi‑agent battle scenario (多智能体战役场景).
    Simulates a complete battle with multiple phases.
    """
    print("\n" + "=" * 50)
    print("Battle Scenario (战役场景)")
    print("=" * 50)
    
    # Create army
    master = BingFu(name="战役指挥部")
    master.enable_commander(name="战役总指挥")
    
    # Create battle units
    battle_units = [
        Agent(name="先锋营", role="Vanguard - Lead the attack"),
        Agent(name="左翼军", role="Left Wing - Flanking maneuver"),
        Agent(name="右翼军", role="Right Wing - Flanking maneuver"),
        Agent(name="中军", role="Center - Main force"),
        Agent(name="后勤司", role="Logistics - Support"),
        Agent(name="斥候队", role="Scouts - Intelligence"),
    ]
    
    for unit in battle_units:
        master.add_agent(unit)
    
    print(f"\n⚔️ Army assembled: {len(master.agents)} units")
    
    # Phase 1: Reconnaissance (阶段一：侦察)
    print("\n" + "-" * 40)
    print("⚔️ Phase 1: Reconnaissance (阶段一：侦察)")
    print("-" * 40)
    result = master.drum("斥候队", "侦察敌军部署")
    print(f"  🥁 {result}")
    
    # Phase 2: Deployment (阶段二：部署)
    print("\n" + "-" * 40)
    print("⚔️ Phase 2: Deployment (阶段二：部署)")
    print("-" * 40)
    print("  📜 Deploying forces...")
    print("  - 左翼军 → 敌军左翼")
    print("  - 右翼军 → 敌军右翼")
    print("  - 中军 → 正面战场")
    print("  - 后勤司 → 后方支援位置")
    
    # Phase 3: Attack (阶段三：进攻)
    print("\n" + "-" * 40)
    print("⚔️ Phase 3: Attack (阶段三：进攻)")
    print("-" * 40)
    print("  🥁 All units: 发起总攻！")
    results = master.drum_all("发起总攻！")
    for name, result in results.items():
        print(f"  - {name}: {result.split(': ')[-1]}")
    
    # Phase 4: Victory (阶段四：胜利)
    print("\n" + "-" * 40)
    print("⚔️ Phase 4: Victory (阶段四：胜利)")
    print("-" * 40)
    print("  🎉 敌军溃败！追击！")
    
    # Phase 5: Withdrawal (阶段五：撤退)
    print("\n" + "-" * 40)
    print("⚔️ Phase 5: Withdrawal (阶段五：鸣金收兵)")
    print("-" * 40)
    print("  🔔 鸣金收兵！")
    results = master.gong_all()
    for name, result in results.items():
        print(f"  - {name}: 收到指令")
    
    print("\n" + "=" * 50)
    print("✅ Battle scenario completed! (战役场景完成！)")
    print("=" * 50)


if __name__ == "__main__":
    multi_agent_basic()
    multi_agent_advanced()
    multi_agent_battle_scenario()
