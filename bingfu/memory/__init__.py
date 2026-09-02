"""军需库 —— 分层记忆系统。

════════════════════════════════════════════════════════════════
 四层，各自回答一个不同的问题
════════════════════════════════════════════════════════════════

    Memory          键值存储        "把这个存起来"        （原有）
    working         工作记忆        "当前上下文太长了怎么办"
    episodic        情景记忆        "上次这类任务怎么样"
    semantic        语义记忆        "我以前读到过什么"

★ 原框架只有第一层，而且 Agent.memory 这个字段
  **声明了却从未被读写** —— 每次执行都清空上下文重来。
  将领打完一仗，什么都不会留下。

★ 补齐这几层不是为了"功能更多"，是为了闭合两个断掉的回路：

  1. TacticEngine 的打分含一项 history，权重 0.20，
     而 history_scores **从未被任何调用方传入** —— 恒为常数 0.5。
     一个"考虑历史表现"的公式，从来没拿到过历史。
     → episodic 提供它。

  2. 工具抓回的资料随任务结束即丢弃，同一领域的连续任务
     会把同一批页面反复重抓（实测一次调研 8 次抓取）。
     → semantic 让知识跨任务复用，同时**不让溯源证据跨任务串**。
"""

from bingfu.memory.store import Memory
from bingfu.memory.episodic import (
    Episode, EpisodicMemory, MIN_SAMPLES, history_scores, summarize,
)
from bingfu.memory.semantic import Fact, KnowledgeBase
from bingfu.memory.working import (
    CompactionResult, DEFAULT_THRESHOLD, compact, measure,
)

__all__ = [
    "Memory",
    "Episode", "EpisodicMemory", "history_scores", "summarize", "MIN_SAMPLES",
    "Fact", "KnowledgeBase",
    "compact", "measure", "CompactionResult", "DEFAULT_THRESHOLD",
]
