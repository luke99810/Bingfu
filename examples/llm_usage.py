"""
BingFu LLM 集成使用示例
演示如何让将领（Agent）通过 LLM 理解军令、执行任务

兵法云：将听吾计，用之必胜——选对军师，胜算在握。
"""

import os
import sys

# 确保可以导入 bingfu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ★ 让这个示例在**未安装包**的情况下也能直接跑（见 _bootstrap.py）。
#   没有这一行，clone 下来第一条 `python examples/xxx.py` 就是
#   ModuleNotFoundError —— 六个示例此前无一例外。
import _bootstrap  # noqa: F401

from bingfu import BingFu, Agent, Tool
from bingfu.llm import LLMFactory, LLMConfig


def demo_basic_llm():
    """示例1：基础 LLM 对话"""
    print("=" * 60)
    print("【示例1】基础 LLM 对话")
    print("=" * 60)

    # 创建 DeepSeek LLM 配置
    # API Key 会自动从环境变量 DEEPSEEK_API_KEY 读取
    config = LLMConfig(
        provider="deepseek",
        model="deepseek-chat",
    )

    # 创建 LLM Provider
    llm = LLMFactory.create(config)

    # 简单对话
    response = llm.simple_generate(
        prompt="用孙子兵法的风格，简短评价一下'知己知彼'的重要性",
        system_prompt="你是孙子兵法的专家，用古代军事风格简短回复。"
    )
    print(f"\n军师曰：{response}")


def demo_agent_with_llm():
    """示例2：Agent 绑定 LLM 执行任务"""
    print("\n" + "=" * 60)
    print("【示例2】Agent 绑定 LLM 执行任务")
    print("=" * 60)

    # 创建 LLM
    config = LLMConfig(
        provider="deepseek",
        model="deepseek-chat",
    )
    llm = LLMFactory.create(config)

    # 创建 Agent 并绑定 LLM
    general = Agent(
        name="韩信",
        role="统帅",
        description="善于统兵作战、出奇制胜的名将",
        llm=llm,
        system_prompt="你是韩信，汉初三杰之一。你善于统兵作战，精通兵法。用简练有力的军令风格回复。"
    )

    # 击鼓！执行任务
    print(f"\n🥁 向韩信下达军令...")
    result = general.drum("分析当前形势：敌军十万驻守城池，我军三万，如何取胜？")
    print(f"\n📋 韩信回报：{result}")


def demo_multi_agent_with_tools():
    """示例3：多 Agent + 工具调用"""
    print("\n" + "=" * 60)
    print("【示例3】多 Agent + 工具调用（ReAct 模式）")
    print("=" * 60)

    # 创建 LLM
    config = LLMConfig(
        provider="deepseek",
        model="deepseek-chat",
    )
    llm = LLMFactory.create(config)

    # 定义工具函数
    def scout_enemy(location: str) -> str:
        """侦察敌情 - 派斥候侦察指定地点"""
        return f"斥候回报：{location}发现敌军先锋约五千人，粮草辎重在后队。"

    def calculate_forces(own: int, enemy: int) -> str:
        """计算兵力对比"""
        ratio = own / max(enemy, 1)
        if ratio < 0.5:
            return f"敌众我寡（{own}:{enemy}，1:{1/ratio:.1f}），需用奇谋"
        elif ratio < 1.0:
            return f"势均力敌（{own}:{enemy}，1:{1/ratio:.1f}），宜以正合以奇胜"
        else:
            return f"我军占优（{own}:{enemy}，{ratio:.1f}:1），可正面进攻"

    # 创建 Agent 并注册工具
    general = Agent(
        name="白起",
        role="主将",
        description="战国四大名将之首，善于歼灭战",
        llm=llm,
        system_prompt="你是白起，战国时期秦国名将。你用兵如神，善于歼灭战。用简练有力的风格回复。"
    )

    # 注册工具
    general.register_tool_function("scout_enemy", scout_enemy, "侦察敌情")
    general.register_tool_function("calculate_forces", calculate_forces, "计算兵力对比")

    # 执行任务（LLM 会自动选择工具）
    print(f"\n🥁 向白起下达军令...")
    result = general.drum("我军七万，敌军十五万据守长平。先侦察敌情，再计算兵力对比，最后给出作战建议。")
    print(f"\n📋 白起回报：{result}")


def demo_bingfu_with_llm():
    """示例4：BingFu 框架 + LLM 完整集成"""
    print("\n" + "=" * 60)
    print("【示例4】BingFu 框架 + LLM 完整集成")
    print("=" * 60)

    # 创建 BingFu 实例
    master = BingFu(name="兵符", version="0.4.0")

    # 方式1：通过配置文件自动初始化 LLM
    # master.load_config("config.yaml")

    # 方式2：手动创建 LLM 并设置
    config = LLMConfig(
        provider="deepseek",
        model="deepseek-chat",
    )
    llm = LLMFactory.create(config)
    master.set_llm(llm)

    # 创建 Agent（会自动绑定默认 LLM）
    hanxin = Agent(name="韩信", role="统帅", description="善于统兵作战")
    baiqi = Agent(name="白起", role="主将", description="善于歼灭战")
    zhuge = Agent(name="诸葛亮", role="军师", description="善于谋略")

    master.add_agent(hanxin)
    master.add_agent(baiqi)
    master.add_agent(zhuge)

    # 查看状态
    status = master.status()
    print(f"\n兵符状态：")
    print(f"  版本: {status['version']}")
    print(f"  将领: {status['agent_count']} 位")
    print(f"  军师: {status.get('llm', '未配置')}")

    # 指挥任务
    print(f"\n🥁 指挥韩信分析战局...")
    result = master.drum("韩信", "敌军以逸待劳据守坚城，我军远道而来，如何破敌？")
    print(f"📋 韩信回报：{result}")


def demo_openai_compatible():
    """示例5：OpenAI 兼容接口（通义千问/Ollama等）"""
    print("\n" + "=" * 60)
    print("【示例5】OpenAI 兼容接口")
    print("=" * 60)

    # 通义千问
    qwen_config = LLMConfig(
        provider="openai_compatible",
        api_key="sk-xxx",  # 替换为你的 API Key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
    )

    # Ollama 本地模型
    ollama_config = LLMConfig(
        provider="openai_compatible",
        api_key="ollama",  # Ollama 不需要真实 API Key
        base_url="http://localhost:11434/v1",
        model="qwen2.5:7b",
    )

    print("可用 Provider 类型：")
    print(f"  1. DeepSeek — 性价比极高，推荐默认使用")
    print(f"  2. OpenAI — GPT 系列，最强但最贵")
    print(f"  3. OpenAI 兼容 — 通义千问/智谱/Ollama 等")
    print(f"\n已注册 Provider：{LLMFactory.list_providers()}")


if __name__ == "__main__":
    # 检查 API Key
    has_key = bool(os.environ.get("DEEPSEEK_API_KEY"))
    if not has_key:
        print("=" * 60)
        print("⚠️ 未检测到 DEEPSEEK_API_KEY 环境变量")
        print("=" * 60)
        print("\n请先设置 API Key：")
        print("  Windows: set DEEPSEEK_API_KEY=sk-xxx")
        print("  Linux/Mac: export DEEPSEEK_API_KEY=sk-xxx")
        print("\n或者直接在代码中传入 api_key 参数：")
        print("  config = LLMConfig(provider='deepseek', api_key='sk-xxx', model='deepseek-chat')")
        print("\n" + "=" * 60)
        print("仅展示示例5（无需 API Key）...")
        demo_openai_compatible()
    else:
        # 运行所有示例
        demo_basic_llm()
        demo_agent_with_llm()
        demo_multi_agent_with_tools()
        demo_bingfu_with_llm()
        demo_openai_compatible()

    print("\n" + "=" * 60)
    print("✨ BingFu LLM 集成示例完成！")
    print("兵法云：将听吾计，用之必胜！")
    print("=" * 60)
