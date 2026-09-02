r"""工作流编排层（Graph）—— 按任务类型分流，层内并行，失败隔离。

════════════════════════════════════════════════════════════════
 这一层解决的是一个**被实测指出来的**问题
════════════════════════════════════════════════════════════════

战术注入的分类别效应，用 Fisher 精确检验重算（成功数/样本数）：

    类别      full    no_tactic    差(任务)     p        结论
    Code      3/12      4/12         −1      1.000    无效应
    Write      3/9       5/9         −2      0.637    无效应
    Data      9/12      3/12         +6      0.039    显著有益
    IR       12/12      5/12         +7      0.005    显著有益
    Reason     7/8       6/8         +1      1.000    无效应

★ 这张表曾经被读成"战术注入对 Code −26 点、Write −12 点，即有害"，
  并据此在路由表里关掉了这两类的注入。**那个读法是错的。**

  n≈12 的格子里，一个任务翻转就是 8 个百分点。
  Code 的"−26 点"实际是 3/12 对 4/12 —— **差一个任务**，p = 1.000。
  在这种样本量下把百分比差当成效应，是在噪声里读故事。

★ 更危险的是这个错误会**自我固化**：一旦据此关掉 Code 的注入，
  之后所有 Code 运行都不带战术，再也不会产生能反驳它的数据。
  一个基于噪声的决定，从此看起来像是被数据支持的。

因此当前策略：**只在有显著证据时才偏离默认**。
战术注入默认全开（唯二显著的效应都是正向的）；
门禁只在有提升空间且断言机械可判时开启 —— 那条依据独立成立，
与战术注入无关。

════════════════════════════════════════════════════════════════
 为什么是 DAG 而不是线性管道
════════════════════════════════════════════════════════════════

1. 无依赖的节点可以并行 —— 拆解出的子任务互不依赖时能省一半墙钟
2. 依赖显式化 —— 下游需要上游的哪个产物，写在边上，不靠命名约定
3. 节点是纯函数，可以单独 mock 输入测试，不必跑全流程
4. 失败隔离 —— 「跳过而非崩溃」，部分结果优于无结果

★ 反模式自查：如果所有节点都串行，DAG 就退化成了链表。
  ``topological_sort`` 返回的层里至少要有一层含多个节点，
  这一层才算真的在做事。
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence


class NodeStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"       # 上游失败导致本节点没有输入


@dataclass
class NodeResult:
    name: str
    status: NodeStatus
    output: Any = None
    error: str = ""
    elapsed: float = 0.0
    #: True = 产物取自断点，本次并未执行 fn。
    #: 这个标记是必要的：耗时 0.0 秒的成功节点，
    #: 与"跑得飞快"从统计上分不开。
    from_checkpoint: bool = False


@dataclass
class GraphResult:
    """整张图的执行结果。

    ★ ``partial`` 的存在是刻意的：一个节点失败不该让整次执行归零。
      但"部分完成"必须**显式可见** —— 悄悄返回半份结果，
      比直接失败更危险。
    """

    nodes: Dict[str, NodeResult] = field(default_factory=dict)
    layers: List[List[str]] = field(default_factory=list)
    elapsed: float = 0.0
    #: 断点相关的说明。**不可续时这里必须有话说** ——
    #: 否则使用者只看到"它又从头跑了一遍"，却不知道为什么。
    resume_note: str = ""
    #: 本次从断点直接取用、没有真正执行的节点
    resumed: List[str] = field(default_factory=list)
    #: 产物无法序列化、因而下次仍需重跑的节点
    unresumable: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(n.status is NodeStatus.OK for n in self.nodes.values())

    @property
    def partial(self) -> bool:
        return (not self.ok) and any(
            n.status is NodeStatus.OK for n in self.nodes.values()
        )

    @property
    def parallel_width(self) -> int:
        """最宽的一层有几个节点 —— 用来自查 DAG 有没有退化成链表。"""

        return max((len(layer) for layer in self.layers), default=0)

    def output_of(self, name: str) -> Any:
        node = self.nodes.get(name)
        return node.output if node and node.status is NodeStatus.OK else None


@dataclass
class Node:
    """一个节点。

    ``fn`` 接收 ``{上游节点名: 上游输出}``，返回本节点的输出。
    保持纯函数形态 —— 不读全局状态，才能单独测试和并行执行。
    """

    name: str
    fn: Callable[[Dict[str, Any]], Any]
    depends_on: Sequence[str] = ()


class CycleError(ValueError):
    """图里有环 —— 拓扑排序无解。"""


class GraphOrchestrator:
    """Kahn 拓扑分层 + 层内并行。"""

    def __init__(self, max_workers: int = 4) -> None:
        self.nodes: Dict[str, Node] = {}
        self.max_workers = max_workers

    def add(self, name: str, fn: Callable[[Dict[str, Any]], Any],
            depends_on: Sequence[str] = ()) -> "GraphOrchestrator":
        if name in self.nodes:
            raise ValueError(f"节点名重复：{name}")
        self.nodes[name] = Node(name=name, fn=fn, depends_on=tuple(depends_on))
        return self

    def topological_sort(self) -> List[List[str]]:
        """Kahn 算法逐层剥离；同一层内的节点互不依赖，可并行。

        ★ 环必须**抛异常**而不是静默丢弃剩余节点。
          静默的话，一个循环依赖会表现为"某些节点莫名其妙没跑"，
          而整体看起来是成功的 —— 又一个不报错的坏结果。
        """

        unknown = {
            dep
            for node in self.nodes.values()
            for dep in node.depends_on
            if dep not in self.nodes
        }
        if unknown:
            raise ValueError(f"依赖了不存在的节点：{sorted(unknown)}")

        in_degree = {n: len(node.depends_on) for n, node in self.nodes.items()}
        adjacency: Dict[str, List[str]] = {n: [] for n in self.nodes}
        for name, node in self.nodes.items():
            for dep in node.depends_on:
                adjacency[dep].append(name)

        queue = deque(sorted(n for n, d in in_degree.items() if d == 0))
        layers: List[List[str]] = []
        seen = 0
        while queue:
            layer = list(queue)
            layers.append(layer)
            queue.clear()
            for name in layer:
                seen += 1
                for nxt in adjacency[name]:
                    in_degree[nxt] -= 1
                    if in_degree[nxt] == 0:
                        queue.append(nxt)
            queue = deque(sorted(queue))

        if seen != len(self.nodes):
            stuck = sorted(n for n, d in in_degree.items() if d > 0)
            raise CycleError(f"图中存在环，无法拓扑排序。涉及节点：{stuck}")
        return layers

    def run(self, initial: Optional[Dict[str, Any]] = None, *,
            checkpointer: Optional[Any] = None,
            thread_id: str = "default") -> GraphResult:
        """执行整张图。

        失败策略是**跳过而非崩溃**：某节点抛异常 → 标 FAILED，
        其所有下游标 SKIPPED，其余分支照常执行。

        ★ 但"部分完成"通过 ``GraphResult.partial`` 显式暴露 ——
          调用方必须自己决定半份结果能不能用，
          而不是由编排层替它假装一切正常。

        ════════════════════════════════════════════════════
         断点续跑（checkpointer 非 None 时启用）
        ════════════════════════════════════════════════════

        每执行完一层就把产物存一次；再次以同一个 ``thread_id`` 运行时，
        已有产物的节点**不再执行 fn**，直接取用。

        ★ 图变了就不续。续跑的依据是"这些产物已经算好了"，
          而它们对应的是旧图。形状对不上时从头跑，并在
          ``GraphResult.resume_note`` 里说明原因。

        ★ 重跑会让副作用再发生一次。产物存不下（不可 JSON 序列化）的节点
          下次一定重跑 —— 如果那个节点会写文件、发请求，它会再做一遍。
          这类节点的名字记在 ``GraphResult.unresumable`` 里，看得见。
        """

        from bingfu.checkpoint import graph_signature, make_state, validate

        started = time.time()
        result = GraphResult(layers=self.topological_sort())
        pool: Dict[str, Any] = dict(initial or {})
        dead: set[str] = set()
        done: set[str] = set()          # 取自断点、无需再跑的节点

        sig = graph_signature(self.nodes)
        if checkpointer is not None:
            usable, note = validate(checkpointer.get(thread_id), sig)
            result.resume_note = note
            if usable:
                saved = checkpointer.get(thread_id)
                for name, value in (saved.get("pool") or {}).items():
                    if name in self.nodes:
                        pool[name] = value
                        done.add(name)
                        result.nodes[name] = NodeResult(
                            name, NodeStatus.OK, value, from_checkpoint=True)
                        result.resumed.append(name)
                result.unresumable = list(saved.get("unresumable") or [])

        for layer_index, layer in enumerate(result.layers):
            runnable, skipped = [], []
            for name in layer:
                if name in done:
                    # 产物已在断点里，本层不必执行它
                    continue
                if any(dep in dead for dep in self.nodes[name].depends_on):
                    skipped.append(name)
                else:
                    runnable.append(name)

            for name in skipped:
                dead.add(name)
                result.nodes[name] = NodeResult(
                    name=name, status=NodeStatus.SKIPPED,
                    error="上游失败，本节点缺少输入",
                )

            def _exec(name: str) -> NodeResult:
                node = self.nodes[name]
                t0 = time.time()
                inputs = {d: pool.get(d) for d in node.depends_on}
                inputs.update({k: v for k, v in pool.items() if k not in inputs})
                try:
                    out = node.fn(inputs)
                    return NodeResult(name, NodeStatus.OK, out,
                                      elapsed=time.time() - t0)
                except Exception as exc:                # noqa: BLE001
                    return NodeResult(name, NodeStatus.FAILED, None,
                                      error=f"{type(exc).__name__}: {exc}",
                                      elapsed=time.time() - t0)

            if len(runnable) == 1:
                outcomes = [_exec(runnable[0])]
            elif runnable:
                with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                    outcomes = list(ex.map(_exec, runnable))
            else:
                outcomes = []

            for outcome in outcomes:
                result.nodes[outcome.name] = outcome
                if outcome.status is NodeStatus.OK:
                    pool[outcome.name] = outcome.output
                else:
                    dead.add(outcome.name)

            # ★ 每层存一次，而不是最后存一次。
            #   最后才存的话，崩在中途等于没存 —— 而"崩在中途"
            #   正是这套机制存在的唯一理由。
            if checkpointer is not None:
                statuses = {n: r.status.value for n, r in result.nodes.items()}
                state = make_state(sig, layer_index, pool, statuses,
                                   result.unresumable)
                checkpointer.put(thread_id, state)
                result.unresumable = list(state["unresumable"])

        result.elapsed = time.time() - started
        return result


# ══════════════════════════════════════════════════════════
#  按任务类型分流 —— 每一项都由实测数据决定
# ══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RoutePlan:
    """某一类任务该怎么跑。

    每个字段旁边的数字都是实测出来的，不是设想。
    """

    category: str
    inject_tactic: bool
    verify_enabled: bool
    max_revisions: int
    rationale: str

    #: 是否给这一类任务配发工具。
    #:
    #: ★ 默认 True —— "无工具"是被测的旧行为，不该是默认。
    #:
    #:   但工具的成本很高：实测有工具时 token 是无工具的 3–4 倍
    #:   （约 13K → 37–51K）。因此这个字段大概率会像 verify_enabled
    #:   一样按类别分化，而不是全开或全关。
    #:
    #:   具体取值等对比实验的数据 —— 在拿到数据之前按类别改它，
    #:   就又是一次"顺着直觉偏离默认"。
    tools_enabled: bool = True

    #: 单次生成的输出 token 上限。
    #:
    #: ★ 全局默认是 2048，对代码类任务**不够**。
    #:
    #:   实测：基准任务 C3（含读取、清洗、统计、可视化、导出五个环节
    #:   的数据管线脚本）在 9 次运行中零成功，分数每次都是 2.0。
    #:   裁判的理由是"可视化函数中途截断、JSON 导出完全缺失"。
    #:   对照：max_tokens=2048 → finish_reason='length'；
    #:        max_tokens=8192 → 25731 字符，完整。
    #:
    #: ★ 为什么按类别设而不是全局调大：
    #:
    #:   输出上限只是**上限**，不产生成本 —— 模型写多少算多少。
    #:   但调大它会让模型倾向于写得更长（更多铺垫、更多解释），
    #:   而检索类任务并不需要更长的答案。
    #:
    #: ★ 为什么这是主路径、续写只是安全网：
    #:
    #:   续写要把上下文和已生成内容一起重发。静态估算，
    #:   写同样 6000 token 的脚本，"2048+续写两次"要 21144 token，
    #:   "8192 一次生成"只要 9000 —— 贵 2.3 倍。
    #:   续写解决的是"撞到任何上限时不静默截断"，不该当常规手段。
    max_output_tokens: int = 2048


#: 实测 n≈39/组：
#:   分类别成功率      Code 12% / Data 75% / IR 100% / Reason 86% / Write 38%
#:   战术注入的效应    Data +50、IR +50；Code −26、Write −12
#:   门禁的 token 代价 约 4–8 倍
ROUTES: Dict[str, RoutePlan] = {
    "Code": RoutePlan(
        category="Code",
        # ★ 这里曾经是 False，依据是「实测 −26 点」。那个依据是错的。
        #
        #   重新做 Fisher 精确检验：Code 类 full 3/12 vs no_tactic 4/12，
        #   **p = 1.000**。所谓的 −26 点换算成任务数是**差 1 个任务**。
        #
        #   n≈12 的格子里，一个任务的翻转就是 8 个百分点。
        #   把它读成「战术注入有害」，是在噪声里读故事。
        #
        #   ★ 更糟的是这个错误会自我固化：关掉注入之后，
        #     Code 类再也不会产生反驳它的数据。
        inject_tactic=True,
        verify_enabled=True,      # 门禁的依据独立成立：语法错是确定性可修的
        max_revisions=2,
        max_output_tokens=8192,   # 完整脚本写不下 2048
        rationale="战术无显著效应(p=1.000)故不关闭；门禁开启因语法错可机械修复",
    ),
    "Write": RoutePlan(
        category="Write",
        # ★ 同上：full 3/9 vs no_tactic 5/9，**p = 0.637**，差 2 个任务。
        inject_tactic=True,
        verify_enabled=True,
        max_revisions=1,          # 只捞空洞产出，不做深度回炉
        max_output_tokens=4096,   # 长文比脚本短，但 2048 仍偏紧
        rationale="战术无显著效应(p=0.637)故不关闭；轻门禁只拦空洞产出",
    ),
    "Data": RoutePlan(
        category="Data",
        inject_tactic=True,       # 9/12 vs 3/12，p=0.039 —— 真实有益
        verify_enabled=True,
        max_revisions=1,
        max_output_tokens=4096,   # 含代码与分析，居中
        rationale="战术注入显著有益(+6 任务, p=0.039) → 开战术、轻门禁",
    ),
    "IR": RoutePlan(
        category="IR",
        inject_tactic=True,       # 12/12 vs 5/12，p=0.005 —— 全组最强的真实效应
        verify_enabled=False,     # 已达 12/12，开门禁只增成本
        max_revisions=0,
        rationale="战术注入显著有益(+7 任务, p=0.005)；已满分故关门禁",
    ),
    "Reason": RoutePlan(
        category="Reason",
        inject_tactic=True,       # 7/8 vs 6/8，p=1.000 —— 无差异，按默认开启
        verify_enabled=False,     # 推理类没有机械可判的断言
        max_revisions=0,
        rationale="战术无显著效应(p=1.000)；无机械可判断言故关门禁",
    ),
}

_DEFAULT_ROUTE = RoutePlan(
    category="*",
    inject_tactic=True,
    verify_enabled=True,
    max_revisions=1,
    rationale="未知类别：按中庸配置走，两边都开但回炉从简",
)


def route_for(category: str) -> RoutePlan:
    """按任务类别取执行计划。

    ★ 这里不做"智能判断"，就是一张查表。

      分流依据是**已经测出来的**分类别数据；
      让模型在运行时临时决定用哪条流水线，等于把一个
      已知答案的问题重新变成不确定的 —— 而且不可复现。
    """

    return ROUTES.get(category, _DEFAULT_ROUTE)
