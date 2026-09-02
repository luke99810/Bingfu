r"""战报 —— 情景记忆（Episodic Memory）。

════════════════════════════════════════════════════════════════
 这一层补的是什么
════════════════════════════════════════════════════════════════

原框架的"记忆"只有一个带落盘的键值字典，而且 ``Agent.memory``
这个字段**声明了却从未被读写**：每次 execute 都清空上下文重来。
也就是说，将领打完一仗，什么都不会留下。

后果不只是"没有记忆"这么抽象，它造成了一个具体的死回路：

    TacticEngine.select_tactic 的打分是
        Q(t, a) = 0.35·对齐(t) + 0.45·战力(a) + 0.20·历史(a)

    而 ``history_scores`` 这个参数**没有任何调用方传过** ——
    历史项恒为常数 0.5。占 20% 权重的一整项是死重。

★ 一个"考虑历史表现"的公式，从来没拿到过历史。
  这与 tactic 那次同型：接口在、参数在、注释在，就是没人喂数据。
  而它不报错 —— 常数 0.5 是个完全合法的值。

════════════════════════════════════════════════════════════════
 记什么
════════════════════════════════════════════════════════════════

每次执行留一条战报：谁、用什么战术、干了什么、结果如何、花了多少。

★ 关键是记**结果**与**代价**，而不只是记"做过"。
  只记做过的话，回放时无法区分"这条路走通了"和"这条路走死了" ——
  而那恰恰是历史唯一有用的地方。
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Episode:
    """一次执行的战报。"""

    task: str
    agent_name: str
    #: 任务类型（Code / IR / Data / Write / Reason / 未知）
    category: str = ""
    tactic: str = ""
    #: 是否达成。None = 未评判（没有裁判时不要假装知道）
    success: Optional[bool] = None
    #: 完成度 1–5；None = 未评判
    score: Optional[float] = None
    tokens: int = 0
    elapsed: float = 0.0
    #: Loop 的终止方式：verified / unverified / exhausted / failed
    outcome: str = ""
    stopped_by: str = ""
    revisions: int = 0
    tool_calls: Dict[str, int] = field(default_factory=dict)
    #: 单调递增的序号，用于按时间排序（不用时间戳：便于复现实验）
    seq: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class EpisodicMemory:
    """战报库：按时间记录每次执行，可按将领/类别回放。

    ★ 线程安全 —— 编排层会并行执行子任务，多个将领可能同时写入。
      不加锁的话，索引会在某些交错下损坏，而症状是"偶尔少几条战报"，
      这种间歇性问题极难排查。
    """

    def __init__(self, path: Optional[str] = None, max_episodes: int = 2000) -> None:
        self.path = Path(path) if path else None
        self.max_episodes = max_episodes
        self._episodes: List[Episode] = []
        self._lock = threading.Lock()
        self._seq = 0
        if self.path and self.path.exists():
            self._load()

    # ── 读写 ────────────────────────────────────────────

    def record(self, episode: Episode) -> Episode:
        with self._lock:
            self._seq += 1
            episode.seq = self._seq
            self._episodes.append(episode)
            # ★ 有上界。无界增长的记忆最终会拖垮加载与检索，
            #   而"记忆越多越好"在有限上下文里并不成立。
            if len(self._episodes) > self.max_episodes:
                self._episodes = self._episodes[-self.max_episodes:]
        self._save()
        return episode

    def all(self) -> List[Episode]:
        with self._lock:
            return list(self._episodes)

    def by_agent(self, name: str) -> List[Episode]:
        return [e for e in self.all() if e.agent_name == name]

    def by_category(self, category: str) -> List[Episode]:
        return [e for e in self.all() if e.category == category]

    def recent(self, n: int = 5, *, category: str = "") -> List[Episode]:
        eps = self.by_category(category) if category else self.all()
        return eps[-n:]

    def __len__(self) -> int:
        with self._lock:
            return len(self._episodes)

    # ── 持久化 ──────────────────────────────────────────

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:                          # noqa: BLE001
            # ★ 记忆文件损坏时**从空开始**，而不是让整个框架起不来。
            #   但要留痕：静默丢弃历史，会让人以为从来没有过历史。
            self._episodes = []
            return
        self._episodes = [Episode.from_dict(d) for d in raw if isinstance(d, dict)]
        self._seq = max((e.seq for e in self._episodes), default=0)

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = [e.to_dict() for e in self._episodes]
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:                          # noqa: BLE001
            pass


# ══════════════════════════════════════════════════════════
#  从战报派生出可用于决策的历史分
# ══════════════════════════════════════════════════════════

#: 样本量太少时不参与决策的阈值。
#:
#: ★ 这个阈值是这一层最重要的一个设计。
#:
#:   本项目吃过一次大亏：在 n≈12 的格子里，把 1 个任务的差异
#:   读成了"战术注入有害 −26 点"，并据此关掉了一整类任务的战术注入。
#:   而 Fisher 精确检验给出的 p 值是 1.000。
#:
#:   历史分会**直接改变后续选择**，因此它比一份报表更危险：
#:   基于噪声的偏好会自我固化 —— 一旦某个将领因为偶然的一次失败
#:   被降权，它就更少被选中，也就更少有机会产生反驳那次失败的数据。
MIN_SAMPLES = 5


def history_scores(
    memory: EpisodicMemory, *, min_samples: int = MIN_SAMPLES
) -> Dict[str, Dict[str, float]]:
    """把战报聚合成 ``{将领: {任务类型: 分数}}``。

    分数取该将领在该类任务上的成功率，落在 [0, 1]。
    样本不足 ``min_samples`` 的格子**不出现在结果里** ——
    调用方会退回中性默认 0.5。

    ★ "没有足够证据"与"表现是 0.5"必须区分。
      前者应当不影响决策，后者是一个真实的中等评价。
      混为一谈的话，一个只跑过一次且失败的将领
      会被当成"确凿地不行"。
    """

    buckets: Dict[str, Dict[str, List[Episode]]] = {}
    for ep in memory.all():
        if ep.success is None or not ep.category:
            continue                                # 未评判的不进入统计
        buckets.setdefault(ep.agent_name, {}).setdefault(ep.category, []).append(ep)

    out: Dict[str, Dict[str, float]] = {}
    for agent, by_cat in buckets.items():
        for cat, eps in by_cat.items():
            if len(eps) < min_samples:
                continue
            rate = sum(1 for e in eps if e.success) / len(eps)
            out.setdefault(agent, {})[cat] = rate
    return out


def _tool_trace(tool_calls: Dict[str, int], top: int = 3) -> str:
    """把工具调用统计写成一小段人话。

    ★ 工具序列比结论有用。

      下一个任务的**结论**多半不同，回放它没有意义；
      但"上次这类任务是靠 read_file + execute_python 做成的"
      是可迁移的。这也是 Episode 一开始就记 tool_calls 的原因 ——
      记了却不回放，等于没记。
    """

    if not tool_calls:
        return ""
    items = sorted(tool_calls.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return "；用了 " + "、".join(f"{k}×{v}" for k, v in items)


def summarize(memory: EpisodicMemory, *, category: str = "",
              limit: int = 5, max_chars: int = 700) -> str:
    """把最近的战报写成一段可放进提示词的文字。

    ★ 只放**结果与教训**，不放完整产出。
      把上次的全文塞回上下文，既贵又会让模型倾向于复制上次的答案 ——
      而如果上次是错的，那正是最不该复制的东西。

    ★ 失败优先。

      "上次这么干没成"能改变下一次的选择；"上次成了"多半只是
      让模型照抄。回放名额有限（要占系统提示的预算），
      名额应该给信息量大的那些。

    ★ 有字数上限，且截断要说出来。

      这段文字每一轮都进上下文。无上限的话，跑得越久提示词越长，
      而变长的过程是无声的 —— 等发现时已经在为历史付一大笔 token。
    """

    # 先捞一批再筛：只取最后 5 条的话，5 条成功会把更早的失败挤掉
    pool = memory.recent(40, category=category)
    if not pool:
        return ""

    def _rank(e: Episode) -> int:
        if e.success is False:
            return 0                       # 明确失败：最有信息量
        if e.success is None:
            return 1                       # 未评判：其次
        return 2                           # 成功：最后

    picked = sorted(pool, key=lambda e: (_rank(e), -e.seq))[:limit]
    picked.sort(key=lambda e: e.seq)       # 展示时仍按时间正序

    lines = ["以下是最近同类任务的执行记录，供参考："]
    for e in picked:
        verdict = "成功" if e.success else ("失败" if e.success is False else "未评判")
        detail = f"（{e.stopped_by}）" if e.stopped_by and not e.success else ""
        lines.append(
            f"  · {e.agent_name} 用「{e.tactic or '无战术'}」→ "
            f"{verdict}{detail}{_tool_trace(e.tool_calls)}"
        )

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n  （历史过长，已截断；共 {len(picked)} 条）"
    return text
