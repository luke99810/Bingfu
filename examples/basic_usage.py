"""
Basic Usage Example (基础用法示例)
Demonstrates basic BingFu usage with a single agent.
Inspired by ancient Chinese warfare: 单骑闯营 (single rider charging the camp).
"""

# ★ 让这个示例在**未安装包**的情况下也能直接跑（见 _bootstrap.py）。
#   没有这一行，clone 下来第一条 `python examples/xxx.py` 就是
#   ModuleNotFoundError —— 六个示例此前无一例外。
import _bootstrap  # noqa: F401

from bingfu import BingFu, Agent


def basic_example():
    """
    Basic BingFu example (基础兵符示例).
    
    Demonstrates:
    1. Creating a BingFu master
    2. Creating and adding an agent
    3. Using drum (击鼓) and gong (鸣金) signals
    """
    print("=" * 50)
    print("BingFu Basic Usage Example (基础用法示例)")
    print("=" * 50)
    
    # Step 1: Create BingFu master (创建兵符主对象)
    print("\n📜 Step 1: Create BingFu master...")
    master = BingFu(name="战场指挥官")
    print(f"✅ Created: {master}")
    
    # Step 2: Create an agent (创建智能体)
    print("\n📜 Step 2: Create an agent...")
    scout = Agent(name="侦察兵", role="Scout")
    print(f"✅ Created: {scout}")
    
    # Step 3: Add agent to BingFu (添加智能体到兵符)
    print("\n📜 Step 3: Add agent to BingFu...")
    master.add_agent(scout)
    print(f"✅ Agent added. Current agents: {list(master.agents.keys())}")
    
    # Step 4: Send drum signal (击鼓信号)
    print("\n🥁 Step 4: Send drum signal (击鼓)...")
    result = master.drum("侦察兵", "侦察敌方营地")
    print(f"✅ {result}")
    
    # Step 5: Check agent status (检查智能体状态)
    print("\n📜 Step 5: Check agent status...")
    agent = master.get_agent("侦察兵")
    print(f"✅ Agent status: {agent.is_active}")
    
    # Step 6: Send gong signal (鸣金信号)
    print("\n🔔 Step 6: Send gong signal (鸣金)...")
    result = master.gong("侦察兵")
    print(f"✅ {result}")
    
    # Step 7: Final status check (最终状态检查)
    print("\n📜 Step 7: Final status check...")
    print(f"✅ Agent status: {master.get_agent('侦察兵').is_active}")
    
    print("\n" + "=" * 50)
    print("✅ Basic example completed! (基础示例完成！)")
    print("=" * 50)


def advanced_example():
    """
    Advanced BingFu example (进阶用法示例).
    
    Demonstrates:
    1. Using tools
    2. Using memory
    3. Using commander
    """
    print("\n" + "=" * 50)
    print("BingFu Advanced Example (进阶示例)")
    print("=" * 50)
    
    # Create BingFu master
    master = BingFu(name="高级战场指挥官")
    
    # Create multiple agents
    scout = Agent(name="侦察兵", role="Scout")
    soldier = Agent(name="士兵", role="Soldier")
    medic = Agent(name="军医", role="Medic")
    
    master.add_agent(scout)
    master.add_agent(soldier)
    master.add_agent(medic)
    
    print(f"\n✅ Added {len(master.agents)} agents")
    
    # Enable commander
    master.enable_commander(name="元帅")
    print(f"✅ Commander enabled: {master.commander}")
    
    # Send drum to all
    print("\n🥁 Sending drum to all agents...")
    results = master.drum_all("执行任务")
    for name, result in results.items():
        print(f"  - {result}")
    
    # Send gong to all
    print("\n🔔 Sending gong to all agents...")
    results = master.gong_all()
    for name, result in results.items():
        print(f"  - {result}")
    
    print("\n" + "=" * 50)
    print("✅ Advanced example completed! (进阶示例完成！)")
    print("=" * 50)


if __name__ == "__main__":
    basic_example()
    advanced_example()
