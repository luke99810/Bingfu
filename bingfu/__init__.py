"""
BingFu (兵符) — Lightweight Multi‑Agent Framework
Inspired by ancient Chinese warfare strategies.

兵符 · 轻量级多智能体框架
以中国古代军事智慧为灵感的多智能体协作框架

v0.6.0: 新增 LangChain 集成
        - LangChainAgent: 使用LangChain重构的Agent框架
        - LangChainMemory: 基于LangChain的记忆模块（支持buffer/summary/window/vector）
        - RAGRetriever: RAG检索增强功能
        - 支持多种向量存储（FAISS/Chroma）
v0.5.0: 新增将军战力体系 + 智能派兵系统
        - GeneralProfile 五维战力档案
        - TaskAssessor 军情评估（LLM优先+规则降级）
        - TaskMatcher 点兵台（加权评分匹配）
        - 修复 Commander/BingFu drum/gong 桩方法
        - 控制台新增 /match /smart 命令
"""

__version__ = "0.6.0"
__author__ = "SuXin (州哥)"
__email__ = "luke99810@example.com"

from bingfu.agent import Agent
from bingfu.tool import Tool
from bingfu.memory import Memory
from bingfu.signal import Drum, Gong
from bingfu.commander import Commander
from bingfu.bingfu import BingFu
from bingfu.tactic_definitions import TacticCategory, TacticDefinition, TacticalContext, TacticCategory as TacticType
from bingfu.tactics import TacticEngine, SunTzuAgent
from bingfu.tactic_library import (
    get_tactic_library, get_tactic, get_tactics_by_regime, get_tactics_by_style
)
from bingfu.visual import MilitaryCommandConsole

# LLM 模块
from bingfu.llm import LLMFactory, LLMConfig, LLMManager, LLMProvider

# 将军战力体系
from bingfu.profile import GeneralProfile, CombatStats, CombatStyle
from bingfu.assessment import TaskAssessment, TaskAssessor, TaskComplexity
from bingfu.matcher import TaskMatcher, MatchResult
from bingfu.presets import PRESET_GENERALS, get_preset, list_presets
from bingfu.campaign import Campaign, CampaignEvent, CampaignResult

# ══════════════════════════════════════════════════════════
#  LangChain 集成 —— 惰性导入
# ══════════════════════════════════════════════════════════
#
# ★ 这三个类**不在任何运行路径上**（见设计文档第 12 节），
#   但此前是在这里直接 import 的，于是每一次 `import bingfu`
#   都要把整条 LangChain 链拉起来。
#
#   实测：`import bingfu` 32.6 秒，其中 31.9 秒是这一行 ——
#   再往里 26 秒花在 transformers 与 sentence_transformers 上。
#   跨框架实测里兵符「41.8 秒」的中位耗时，其中 35 秒是它，
#   真正跑一次战役只要 6.2 秒。
#
#   一条从未被使用的可选路径，让每次启动都付 30 秒 ——
#   这是「能力到不了」的反面：**够不着，却照样收费**。
#
# ★ 用 PEP 562 的模块级 __getattr__ 做惰性：
#   `from bingfu import RAGRetriever` 仍然可用，
#   只是在真正取用时才付那 30 秒。
_LAZY_ATTRS = {
    "LangChainAgent": "bingfu.langchain_integration",
    "LangChainMemory": "bingfu.langchain_integration",
    "RAGRetriever": "bingfu.langchain_integration",
}


def __getattr__(name):
    """按需加载重型可选依赖。"""

    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    import importlib

    value = getattr(importlib.import_module(target), name)
    globals()[name] = value          # 只付一次
    return value

__all__ = [
    # 核心模块
    "Agent",
    "Campaign",
    "CampaignEvent",
    "CampaignResult",
    "Tool",
    "Memory",
    "Drum",
    "Gong",
    "Commander",
    "BingFu",
    # 战术引擎
    "TacticEngine",
    "SunTzuAgent",
    "TacticType",
    "TacticalContext",
    # 可视化控制台
    "MilitaryCommandConsole",
    # LLM 模块
    "LLMFactory",
    "LLMConfig",
    "LLMManager",
    "LLMProvider",
    # 将军战力体系
    "GeneralProfile",
    "CombatStats",
    "CombatStyle",
    "TaskAssessment",
    "TaskAssessor",
    "TaskComplexity",
    "TaskMatcher",
    "MatchResult",
    "PRESET_GENERALS",
    "get_preset",
    "list_presets",
    # 战术定义系统
    "TacticCategory",
    "TacticDefinition",
    "get_tactic_library",
    "get_tactic",
    "get_tactics_by_regime",
    "get_tactics_by_style",
    # LangChain 集成
    "LangChainAgent",
    "LangChainMemory",
    "RAGRetriever",
]
