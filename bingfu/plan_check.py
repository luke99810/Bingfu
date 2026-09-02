r"""拆解校验 —— 让「拆坏了」从不可见变成可见。

════════════════════════════════════════════════════════════════
 为什么需要这一层
════════════════════════════════════════════════════════════════

拆解由 LLM 完成，拆得好不好此前没有任何校验。拆坏了的表现是
**「跑完了，但做的不是那件事」** —— 不报错、有输出、语气也对。

实测三个真实任务的拆解结果，全部是链：

    t1 → t2 → t3 → t4

``parallel_width`` 恒为 1。也就是说这个框架最核心的「同层并行」
一次都没有被触发过 —— DAG 退化成了链表，而没有任何地方看得出来。
``graph.py`` 的模块头里早就写着这条自查，只是从来没人执行它。

还观察到一种更隐蔽的坏拆解：上游产物本来会**自动**作为输入喂给下游
（见 orchestration._node 里拼 context 的那段），但拆解提示词只强调
「子任务要自包含」，于是模型自作主张发明了文件中转：

    t1: 解析 data.csv，计算总销量
    t2: 读取 total.txt（由其他任务生成），获取总销量

**t1 从未被要求创建 total.txt。** t2 于是要么失败，要么编一个数出来。

════════════════════════════════════════════════════════════════
 这一层只做判定，不做修复
════════════════════════════════════════════════════════════════

★ 校验返回问题清单，由调用方决定重拆、告警还是照跑。
  自动"修好"一张拆错的图是危险的：修的依据同样来自猜测，
  而修完之后连"拆错过"这件事都看不见了。
"""

from __future__ import annotations

import hashlib
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

#: 严重程度。error = 必须重拆；warn = 可以跑，但要说出来
ERROR = "error"
WARN = "warn"


@dataclass
class PlanIssue:
    """拆解图的一个问题。"""

    level: str
    code: str
    message: str
    nodes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        tail = ("（%s）" % "、".join(self.nodes)) if self.nodes else ""
        return "[%s] %s%s" % (self.code, self.message, tail)


# ══════════════════════════════════════════════════════════
#  分层（与 GraphOrchestrator 同一套 Kahn，但不需要建图）
# ══════════════════════════════════════════════════════════

def layers_of(subtasks: Sequence[Any]) -> List[List[str]]:
    """按依赖分层。有环时返回已剥离的部分（判定环交给 validate_plan）。"""

    ids = [s.id for s in subtasks]
    known = set(ids)
    indeg: Dict[str, int] = {}
    adj: Dict[str, List[str]] = {i: [] for i in ids}
    for s in subtasks:
        deps = [d for d in (s.depends_on or []) if d in known and d != s.id]
        indeg[s.id] = len(deps)
        for d in deps:
            adj[d].append(s.id)

    queue = deque(sorted(i for i in ids if indeg[i] == 0))
    layers: List[List[str]] = []
    while queue:
        layer = sorted(queue)
        layers.append(layer)
        queue.clear()
        for name in layer:
            for nxt in adj[name]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        queue = deque(sorted(queue))
    return layers


def parallel_width(subtasks: Sequence[Any]) -> int:
    """最宽的一层有几个节点。1 表示彻底退化成链表。"""

    return max((len(l) for l in layers_of(subtasks)), default=0)


# ══════════════════════════════════════════════════════════
#  稳定 id
# ══════════════════════════════════════════════════════════

def stable_subtask_id(description: str, index: int = 0) -> str:
    """按**内容**生成 id，而不是按顺序编号。

    ★ 这是为断点服务的。

      断点签名由「节点名 + 依赖」构成，而节点名此前是模型随口给的
      t1/t2/step-1。同一道军令重跑一次，模型换一批 id，签名就对不上，
      断点直接作废 —— 表现是「存了但从来没续上过」。

      按描述哈希之后，只要拆解结果实质相同，id 就相同，断点能续。
      拆解真的变了，id 也跟着变，续跑被正确拒绝。两种情况都对。
    """

    text = re.sub(r"\s+", " ", (description or "").strip())
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return "t%s" % digest


# ══════════════════════════════════════════════════════════
#  校验
# ══════════════════════════════════════════════════════════

#: 「由其他任务生成的文件」这类臆造中转的痕迹
_INVENTED_HANDOFF = re.compile(
    r"[\w./-]+\.(txt|json|csv|md|yaml|yml)[^\n]{0,12}"
    r"(由|from)[^\n]{0,10}(其他|上|前置|另一|other|previous)"
)


def validate_plan(plan: Any, *, min_subtasks_for_width: int = 3) -> List[PlanIssue]:
    """检查一张拆解图，返回问题清单（空 = 没发现问题）。"""

    subs = list(getattr(plan, "subtasks", []) or [])
    issues: List[PlanIssue] = []
    if not subs:
        return [PlanIssue(ERROR, "empty", "拆解结果为空")]

    ids = [s.id for s in subs]
    known = set(ids)

    # ── 结构性错误 ──────────────────────────────────────
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        issues.append(PlanIssue(ERROR, "duplicate_id", "子任务 id 重复", dup))

    unknown = sorted({d for s in subs for d in (s.depends_on or [])
                      if d not in known})
    if unknown:
        issues.append(PlanIssue(
            ERROR, "unknown_dep",
            "依赖了不存在的子任务 —— 这条边等于断的，图会悄悄退化", unknown))

    selfdep = sorted({s.id for s in subs if s.id in (s.depends_on or [])})
    if selfdep:
        issues.append(PlanIssue(ERROR, "self_dep", "子任务依赖了自己", selfdep))

    layers = layers_of(subs)
    placed = {n for l in layers for n in l}
    stuck = sorted(set(ids) - placed)
    if stuck:
        issues.append(PlanIssue(
            ERROR, "cycle",
            "依赖成环，这些子任务永远排不进任何一层", stuck))

    # ── 退化：能跑，但跑出来没有意义 ────────────────────
    if len(subs) > 1:
        edges = sum(len(s.depends_on or []) for s in subs)
        if edges == 0:
            issues.append(PlanIssue(
                WARN, "no_edges",
                "多个子任务却一条依赖都没有 —— 这是广播，不是协作"))
        width = max((len(l) for l in layers), default=0)
        if width == 1 and len(subs) >= min_subtasks_for_width:
            issues.append(PlanIssue(
                WARN, "degenerate_chain",
                "拆成了一条链（并行宽度 1），同层并行完全没有被用上"))

    # ── 臆造的文件中转 ──────────────────────────────────
    invented = sorted({s.id for s in subs
                       if _INVENTED_HANDOFF.search(s.description or "")})
    if invented:
        issues.append(PlanIssue(
            WARN, "invented_handoff",
            "描述里让下游去读一个「由其他任务生成」的文件 —— "
            "上游产物本来就会自动作为输入传下去，凭空约定的中间文件多半不存在",
            invented))

    return issues


def issues_as_feedback(issues: Sequence[PlanIssue]) -> str:
    """把问题清单写成给模型看的重拆提示。"""

    if not issues:
        return ""
    lines = ["上一次拆解有以下问题，请重新拆解并避免它们："]
    for i in issues:
        lines.append("  · " + str(i))
    return "\n".join(lines)


def has_error(issues: Sequence[PlanIssue]) -> bool:
    return any(i.level == ERROR for i in issues)
