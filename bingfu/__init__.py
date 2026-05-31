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
from bingfu.tactics import TacticsEngine, SunTzuAgent, TacticType, TacticalContext
from bingfu.visual import MilitaryCommandConsole

# LLM 模块
from bingfu.llm import LLMFactory, LLMConfig, LLMManager, LLMProvider

# 将军战力体系
from bingfu.profile import GeneralProfile, CombatStats, CombatStyle
from bingfu.assessment import TaskAssessment, TaskAssessor, TaskComplexity
from bingfu.matcher import TaskMatcher, MatchResult
from bingfu.presets import PRESET_GENERALS, get_preset, list_presets

# LangChain 集成模块
from bingfu.langchain_integration import LangChainAgent, LangChainMemory, RAGRetriever

__all__ = [
    # 核心模块
    "Agent",
    "Tool",
    "Memory",
    "Drum",
    "Gong",
    "Commander",
    "BingFu",
    # 战术引擎
    "TacticsEngine",
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
    # LangChain 集成
    "LangChainAgent",
    "LangChainMemory",
    "RAGRetriever",
]
