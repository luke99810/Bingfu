"""
BingFu Framework - 记忆系统示例
Demonstrates how to use Memory with Agents.

本示例展示：
1. 创建和管理Agent记忆
2. 文件持久化记忆
3. 记忆在多Agent间共享
4. 战役记忆系统
"""

import os
import tempfile
# ★ 让这个示例在**未安装包**的情况下也能直接跑（见 _bootstrap.py）。
#   没有这一行，clone 下来第一条 `python examples/xxx.py` 就是
#   ModuleNotFoundError —— 六个示例此前无一例外。
import _bootstrap  # noqa: F401

from bingfu import Agent, Memory, BingFu


def demo_basic_memory():
    """基础记忆使用演示"""
    print("=" * 60)
    print("演示1：基础记忆操作")
    print("=" * 60)

    # 创建记忆
    memory = Memory(name="士兵记忆", memory_type="dict")

    # 存储信息
    memory.store("name", "张飞")
    memory.store("rank", "校尉")
    memory.store("battles", 15)
    memory.store("skills", ["枪法", "骑马", "射箭"])

    # 检索信息
    print(f"\n士兵姓名: {memory.retrieve('name')}")
    print(f"军衔: {memory.retrieve('rank')}")
    print(f"战斗经验: {memory.retrieve('battles')} 场")
    print(f"技能: {memory.retrieve('skills')}")

    # 获取所有记忆
    print(f"\n所有记忆: {memory.all()}")
    print(f"记忆总数: {len(memory)}")


def demo_file_memory():
    """文件持久化记忆演示"""
    print("\n" + "=" * 60)
    print("演示2：文件持久化记忆")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "soldier_memory.json")

        # 创建文件记忆
        memory = Memory(
            name="士兵档案",
            memory_type="file",
            file_path=file_path
        )

        # 存储重要信息
        memory.store("姓名", "关羽")
        memory.store("字", "云长")
        memory.store("兵器", "青龙偃月刀")
        memory.store("坐骑", "赤兔马")

        print(f"记忆文件: {file_path}")
        print(f"当前记忆: {memory.all()}")

        # 模拟重启程序，重新加载记忆
        print("\n--- 模拟程序重启 ---")
        restored_memory = Memory(
            name="士兵档案",
            memory_type="file",
            file_path=file_path
        )
        print(f"恢复后的记忆: {restored_memory.all()}")


def demo_agent_memory():
    """Agent记忆使用演示"""
    print("\n" + "=" * 60)
    print("演示3：Agent使用记忆")
    print("=" * 60)

    # 创建Agent时指定记忆
    memory = Memory(name="谋臣记忆", memory_type="dict")
    strategist = Agent(
        name="诸葛亮",
        memory=memory
    )

    # 存储重要情报
    strategist.memory.store("敌方兵力", "十万大军")
    strategist.memory.store("天气", "大雾")
    strategist.memory.store("计策", "草船借箭")
    strategist.memory.store("预计收获", "十万支箭")

    print(f"Agent: {strategist.name}")
    print(f"记忆: {strategist.memory.all()}")

    # 执行任务并记录
    result = strategist.execute("草船借箭")
    print(f"\n执行结果: {result}")

    # 记录执行结果到记忆
    strategist.memory.store("执行状态", "成功")
    strategist.memory.store("执行时间", "建安十三年冬")
    print(f"更新后记忆: {strategist.memory.all()}")


def demo_shared_memory():
    """共享记忆演示 - 模拟情报共享"""
    print("\n" + "=" * 60)
    print("演示4：多Agent共享情报")
    print("=" * 60)

    # 创建共享情报库
    shared_intel = Memory(name="军情共享库", memory_type="dict")

    # 创建多个Agent，共享同一记忆
    scout1 = Agent(name="斥候甲")
    scout2 = Agent(name="斥候乙")
    commander = Agent(name="统帅")

    # 模拟情报收集
    scout1.memory = shared_intel
    scout2.memory = shared_intel
    commander.memory = shared_intel

    # 斥候甲汇报
    scout1.memory.store("情报_甲", {
        "位置": "东门",
        "敌军数量": "5000",
        "状态": "戒备中"
    })
    print(f"{scout1.name} 汇报: 东门发现敌军5000人")

    # 斥候乙汇报
    scout2.memory.store("情报_乙", {
        "位置": "西门",
        "敌军数量": "3000",
        "状态": "换防中"
    })
    print(f"{scout2.name} 汇报: 西门敌军3000人正在换防")

    # 统帅获取完整情报
    print(f"\n{commander.name} 获取完整情报:")
    all_intel = commander.memory.all()
    for key, value in all_intel.items():
        print(f"  {key}: {value}")


def demo_battle_memory_system():
    """战役记忆系统演示"""
    print("\n" + "=" * 60)
    print("演示5：战役记忆系统")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        battle_log = os.path.join(tmpdir, "赤壁之战.json")

        # 创建战役记忆
        memory = Memory(
            name="赤壁之战记录",
            memory_type="file",
            file_path=battle_log
        )

        # 战役阶段记录
        phases = [
            {
                "阶段": "准备阶段",
                "行动": "联合孙刘",
                "结果": "成功联盟"
            },
            {
                "阶段": "火攻准备",
                "行动": "黄盖诈降",
                "结果": "骗取信任"
            },
            {
                "阶段": "决战",
                "行动": "火烧曹营",
                "结果": "大获全胜"
            }
        ]

        for i, phase in enumerate(phases):
            memory.store(f"phase_{i+1}", phase)
            print(f"记录阶段{i+1}: {phase['阶段']} - {phase['行动']}")

        # 战役总结
        memory.store("战役名称", "赤壁之战")
        memory.store("时间", "建安十三年冬")
        memory.store("结果", "孙刘联军以少胜多，大败曹操")
        memory.store("关键因素", ["火攻", "东南风", "诈降计"])

        print(f"\n战役档案: {memory.all()}")

        # 读取战役日志
        print("\n--- 读取历史战役 ---")
        history = Memory(
            name="赤壁之战记录",
            memory_type="file",
            file_path=battle_log
        )
        print(f"战役结果: {history.retrieve('结果')}")
        print(f"关键因素: {history.retrieve('关键因素')}")


def demo_conversation_memory():
    """对话记忆演示"""
    print("\n" + "=" * 60)
    print("演示6：对话记忆系统")
    print("=" * 60)

    memory = Memory(name="对话记录", memory_type="dict")

    # 模拟多轮对话
    conversations = [
        {"speaker": "user", "content": "今天天气如何？"},
        {"speaker": "assistant", "content": "今天是晴天，适合出行。"},
        {"speaker": "user", "content": "有什么旅游推荐吗？"},
        {"speaker": "assistant", "content": "推荐去西湖游玩，风景优美。"},
        {"speaker": "user", "content": "需要准备什么？"},
        {"speaker": "assistant", "content": "建议带伞和防晒用品。"},
    ]

    # 存储对话
    for i, msg in enumerate(conversations):
        memory.store(f"msg_{i+1}", msg)

    # 检索最近对话
    print("最近3轮对话:")
    for i in range(len(conversations)-3, len(conversations)):
        msg = memory.retrieve(f"msg_{i+1}")
        print(f"  {msg['speaker']}: {msg['content']}")

    # 获取上下文摘要
    print(f"\n对话总数: {len(memory)} 条")


def demo_memory_tools():
    """记忆操作工具演示"""
    print("\n" + "=" * 60)
    print("演示7：记忆操作工具")
    print("=" * 60)

    memory = Memory(name="测试记忆", memory_type="dict")

    # 存储测试数据
    for i in range(5):
        memory.store(f"key_{i}", f"value_{i}")

    print(f"初始状态: {len(memory)} 条记录")
    print(f"数据: {memory.all()}")

    # 删除指定记录
    print(f"\n删除 key_2: {memory.delete('key_2')}")
    print(f"删除后: {len(memory)} 条记录")

    # 删除不存在的记录
    print(f"删除不存在的 key_100: {memory.delete('key_100')}")

    # 清空所有
    print("\n清空所有记忆...")
    memory.clear()
    print(f"清空后: {len(memory)} 条记录")


if __name__ == "__main__":
    demo_basic_memory()
    demo_file_memory()
    demo_agent_memory()
    demo_shared_memory()
    demo_battle_memory_system()
    demo_conversation_memory()
    demo_memory_tools()

    print("\n" + "=" * 60)
    print("✨ 所有演示完成！")
    print("=" * 60)
