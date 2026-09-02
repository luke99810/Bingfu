r"""拆解质量基准 —— 把「拆解变好了」从印象变成可复现的数字。

════════════════════════════════════════════════════════════════
 为什么单独做一层，而不是复用 experiment.py
════════════════════════════════════════════════════════════════

``experiment.py`` 量的是**端到端成功率**：跑完整流程 + LLM 裁判，
一个样本要几十次模型调用。而这里要量的是**拆解这一步的形状** ——
一个样本只需要一次调用。便宜十几倍，因此可以跑够重复次数，
而重复次数正是上一次栽跟头的地方。

════════════════════════════════════════════════════════════════
 上一次是怎么栽的
════════════════════════════════════════════════════════════════

``graph.py`` 的模块头里记着：一张 n≈12 的表被读成
「战术注入对 Code 有害 −26 点」，据此关掉了那一类的注入。
实际是 3/12 对 4/12 —— **差一个任务**，p = 1.000。

更要命的是那个错误会自我固化：关掉之后，再也不会产生能反驳它的数据。

★ 所以这个模块有三条硬规矩：

  一、任务集**带标注**。哪些任务本来就该串行、哪些应该能并行，
      事先写死。不然「宽度 1」到底是拆坏了还是拆对了，无从判断。

  二、每个任务重复多次。单次结果只是一个样本，不是结论。

  三、**报告必须带 n 与 p**，不显著就说不显著。
      「改完之后看起来好多了」不是结论，是印象。
"""

from __future__ import annotations

import json
import math
import os
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .plan_check import has_error, validate_plan

#: 期望形状
SEQUENTIAL = "sequential"      # 本来就该是链，宽度 1 是正确的
PARALLEL = "parallel"          # 存在互不依赖的部分，宽度应当 ≥ 2


@dataclass
class BenchTask:
    """一道带标注的拆解任务。"""

    id: str
    text: str
    #: 期望形状：SEQUENTIAL / PARALLEL
    shape: str
    note: str = ""


#: 任务集。
#:
#: ★ 标注依据写在 note 里，好让后来者能反驳它 ——
#:   「这个任务其实可以并行」是一个可以讨论的问题，
#:   而一个没有依据的标注没法讨论。
BENCH_TASKS: List[BenchTask] = [
    BenchTask("seq-1", "读取 数据.csv，算出总销量，把结果写进 结果.md",
              SEQUENTIAL, "算完才能写，只有一条链"),
    BenchTask("seq-2", "把 config.yaml 里的端口号改成 8080，然后重启说明写进 README",
              SEQUENTIAL, "改完才能写说明"),
    BenchTask("seq-3", "统计 logs/ 目录下各类日志的条数，按数量排序后输出一张表",
              SEQUENTIAL, "统计→排序→输出，严格顺序"),

    BenchTask("par-1",
              "调研当前主流的多智能体框架，对比它们的编排能力，"
              "写一份带代码示例的评估报告，并给出选型建议",
              PARALLEL, "调研之后，「定评估维度」与「写代码示例」互不依赖"),
    BenchTask("par-2",
              "为这个项目补一套 CI：检查测试、检查代码风格、"
              "检查文档与代码是否一致，并写好说明文档",
              PARALLEL, "三项检查彼此独立，可并列设计"),
    BenchTask("par-3",
              "为一个待上线的 Web 应用准备发布材料：写用户手册、"
              "写发布说明、准备一份 FAQ",
              PARALLEL, "三份材料互不依赖"),
    BenchTask("par-4",
              "分析一份销售数据：分别按地区、按品类、按月份做三张统计表，"
              "最后写一段综合结论",
              PARALLEL, "三张表并列，结论汇总"),
    BenchTask("par-5",
              "调研 Python 的三个 HTTP 客户端库（requests / httpx / aiohttp），"
              "各写一段最小示例，再横向对比它们的优缺点",
              PARALLEL, "三个库的调研与示例互不依赖"),
]


@dataclass
class Sample:
    """一次拆解的观测。"""

    task_id: str
    shape: str
    variant: str
    repeat: int
    n_subtasks: int = 0
    edges: int = 0
    width: int = 0
    codes: List[str] = field(default_factory=list)
    failed: bool = False
    error: str = ""

    @property
    def degenerate(self) -> bool:
        """该并行却拆成了链。"""

        return self.shape == PARALLEL and self.width < 2

    @property
    def structurally_broken(self) -> bool:
        return any(c in self.codes
                   for c in ("cycle", "unknown_dep", "self_dep",
                             "duplicate_id", "empty"))

    @property
    def invented_handoff(self) -> bool:
        return "invented_handoff" in self.codes


# ══════════════════════════════════════════════════════════
#  Fisher 精确检验（2×2）
# ══════════════════════════════════════════════════════════

def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """双尾 Fisher 精确检验，返回 p 值。

        表格          成功   失败
        变体 A          a      b
        变体 B          c      d

    ★ 自己算而不是拉 scipy：这个函数只有几行，而基准脚本
      不该因为一个统计依赖装不上就跑不了。数值上与 scipy 一致
      （见 tests/test_plan_bench.py 里的对照）。
    """

    n = a + b + c + d
    if n == 0:
        return 1.0

    def _p(x: int) -> float:
        r1, r2 = a + b, c + d
        c1 = a + c
        y = r1 - x
        z = c1 - x
        w = r2 - z
        if min(x, y, z, w) < 0:
            return 0.0
        return math.exp(
            math.lgamma(r1 + 1) + math.lgamma(r2 + 1)
            + math.lgamma(c1 + 1) + math.lgamma(n - c1 + 1)
            - math.lgamma(n + 1)
            - math.lgamma(x + 1) - math.lgamma(y + 1)
            - math.lgamma(z + 1) - math.lgamma(w + 1)
        )

    observed = _p(a)
    lo = max(0, (a + c) - (c + d))
    hi = min(a + b, a + c)
    total = 0.0
    for x in range(lo, hi + 1):
        p = _p(x)
        if p <= observed * (1 + 1e-9):
            total += p
    return min(1.0, total)


# ══════════════════════════════════════════════════════════
#  跑基准
# ══════════════════════════════════════════════════════════

def run_variant(llm: Any, agents: Dict[str, Any], *, variant: str,
                prompt_template: Optional[str] = None,
                tasks: Sequence[BenchTask] = (),
                repeats: int = 3, max_workers: int = 4,
                on_sample: Optional[Any] = None) -> List[Sample]:
    """在一组任务上重复跑拆解，收集观测。

    ★ 失败也是观测。

      拆解调用崩了就记 ``failed=True``，不要丢掉 ——
      丢掉失败样本会让成功率凭空变高，而这是最常见的一种自欺。
    """

    from .orchestration import decompose

    tasks = list(tasks or BENCH_TASKS)
    jobs = [(t, r) for t in tasks for r in range(repeats)]
    out: List[Sample] = []
    lock = threading.Lock()

    def _one(job):
        task, rep = job
        s = Sample(task_id=task.id, shape=task.shape, variant=variant, repeat=rep)
        try:
            plan = decompose(task.text, agents, llm,
                             prompt_template=prompt_template)
            issues = list(getattr(plan, "issues", None) or validate_plan(plan))
            s.n_subtasks = len(plan.subtasks)
            s.edges = plan.edge_count
            s.width = plan.parallel_width
            s.codes = [i.code for i in issues]
        except Exception as exc:                # noqa: BLE001
            s.failed = True
            s.error = "%s: %s" % (type(exc).__name__, exc)
        with lock:
            out.append(s)
            if on_sample:
                try:
                    on_sample(s)
                except Exception:               # noqa: BLE001
                    pass
        return s

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_one, jobs))
    return out


def summarize(samples: Sequence[Sample]) -> Dict[str, Any]:
    """把观测汇总成可读的数字。**每一项都带 n。**"""

    ok = [s for s in samples if not s.failed]
    par = [s for s in ok if s.shape == PARALLEL]
    seq = [s for s in ok if s.shape == SEQUENTIAL]

    def _rate(hits, total):
        return (hits / total) if total else 0.0

    widths = [s.width for s in par]
    return {
        "n_total": len(samples),
        "n_failed": sum(1 for s in samples if s.failed),
        "n_parallel_tasks": len(par),
        "n_sequential_tasks": len(seq),
        # 该并行的任务里，真的拆出并行的比例
        "parallel_hit": sum(1 for s in par if s.width >= 2),
        "parallel_rate": _rate(sum(1 for s in par if s.width >= 2), len(par)),
        "median_width_on_parallel": statistics.median(widths) if widths else 0,
        # 该串行的任务里，被硬拆出并行的比例（过度拆解）
        "over_split": sum(1 for s in seq if s.width >= 2),
        "over_split_rate": _rate(sum(1 for s in seq if s.width >= 2), len(seq)),
        "invented_handoff": sum(1 for s in ok if s.invented_handoff),
        "invented_rate": _rate(sum(1 for s in ok if s.invented_handoff), len(ok)),
        "structurally_broken": sum(1 for s in ok if s.structurally_broken),
    }


def compare(a: Sequence[Sample], b: Sequence[Sample], *,
            label_a: str = "A", label_b: str = "B") -> Dict[str, Any]:
    """A/B 对照。返回两侧汇总 + 关键指标的 Fisher 检验。"""

    sa, sb = summarize(a), summarize(b)
    hit_a, n_a = sa["parallel_hit"], sa["n_parallel_tasks"]
    hit_b, n_b = sb["parallel_hit"], sb["n_parallel_tasks"]
    p_parallel = fisher_exact_2x2(hit_a, n_a - hit_a, hit_b, n_b - hit_b)

    inv_a, na_ok = sa["invented_handoff"], sa["n_total"] - sa["n_failed"]
    inv_b, nb_ok = sb["invented_handoff"], sb["n_total"] - sb["n_failed"]
    p_invented = fisher_exact_2x2(inv_a, na_ok - inv_a, inv_b, nb_ok - inv_b)

    return {
        "label_a": label_a, "label_b": label_b,
        "a": sa, "b": sb,
        "p_parallel_rate": p_parallel,
        "p_invented_rate": p_invented,
        # ★ 显著性判定写死在这里，免得每次报告时临时改口径
        "parallel_significant": p_parallel < 0.05,
        "invented_significant": p_invented < 0.05,
    }


def format_report(cmp: Dict[str, Any]) -> str:
    """把对照结果写成一段可以直接贴进文档的文字。

    ★ 不显著就明说不显著 —— 这是这个模块存在的全部理由。
    """

    a, b = cmp["a"], cmp["b"]
    la, lb = cmp["label_a"], cmp["label_b"]
    lines = [
        "拆解质量对照（每格 n 已标注）",
        "",
        "%-26s %-18s %-18s" % ("指标", la, lb),
        "-" * 64,
        "%-26s %-18s %-18s" % (
            "该并行→真并行",
            "%d/%d" % (a["parallel_hit"], a["n_parallel_tasks"]),
            "%d/%d" % (b["parallel_hit"], b["n_parallel_tasks"])),
        "%-26s %-18s %-18s" % (
            "并行任务的宽度中位数",
            a["median_width_on_parallel"], b["median_width_on_parallel"]),
        "%-26s %-18s %-18s" % (
            "臆造文件中转",
            "%d/%d" % (a["invented_handoff"], a["n_total"] - a["n_failed"]),
            "%d/%d" % (b["invented_handoff"], b["n_total"] - b["n_failed"])),
        "%-26s %-18s %-18s" % (
            "该串行→被过度拆解",
            "%d/%d" % (a["over_split"], a["n_sequential_tasks"]),
            "%d/%d" % (b["over_split"], b["n_sequential_tasks"])),
        "%-26s %-18s %-18s" % (
            "结构性错误", a["structurally_broken"], b["structurally_broken"]),
        "%-26s %-18s %-18s" % ("调用失败", a["n_failed"], b["n_failed"]),
        "",
        "并行率 Fisher 精确检验：p = %.4f  → %s" % (
            cmp["p_parallel_rate"],
            "显著" if cmp["parallel_significant"] else "**不显著**"),
        "臆造中转 Fisher 精确检验：p = %.4f  → %s" % (
            cmp["p_invented_rate"],
            "显著" if cmp["invented_significant"] else "**不显著**"),
    ]
    if not cmp["parallel_significant"]:
        lines += [
            "",
            "★ 并行率的差异未达显著。样本量不足以支撑「改好了」这个结论 ——",
            "  在这个量级上把百分比差当成效应，就是在噪声里读故事。",
        ]
    return "\n".join(lines)


def save(samples: Sequence[Sample], path: str) -> None:
    """存下原始观测，便于复算与反驳。"""

    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(s) for s in samples], fh, ensure_ascii=False, indent=2)


def load(path: str) -> List[Sample]:
    with open(path, "r", encoding="utf-8") as fh:
        return [Sample(**d) for d in json.load(fh)]


# ══════════════════════════════════════════════════════════
#  对照用的旧提示词
# ══════════════════════════════════════════════════════════

#: 改造前的「要求」段落。
#:
#: ★ 只替换这一段，其余（任务、将领名册、JSON 格式说明）保持逐字相同。
#:   否则比出来的差异说不清是哪一处带来的 —— 那就不是受控对照。
_LEGACY_REQUIREMENTS = """要求：
1. 拆成 2-5 个子任务。如果任务本身很简单、不值得拆，就只给 1 个
2. 用 depends_on 标明依赖：某个子任务需要另一个的产出才能开始
3. 没有依赖关系的子任务放在同一"层"，它们会被并行执行
4. 子任务描述要**自包含**——执行它的将领看不到原始任务
5. 给每个子任务标注类型 category，取值只能是以下之一：
   Code（写代码）/ IR（查资料）/ Data（数据分析）/ Write（写文章）/ Reason（推理论证）"""


def legacy_prompt() -> str:
    """还原改造前的拆解提示词，用于 A/B。"""

    from .orchestration import _DECOMPOSE_PROMPT

    head, sep, tail = _DECOMPOSE_PROMPT.partition("要求：")
    if not sep:
        raise RuntimeError("当前提示词里找不到「要求：」段落，无法还原旧版")
    # 从「要求：」到「只输出 JSON」之间整段换掉
    marker = "只输出 JSON"
    idx = tail.find(marker)
    if idx < 0:
        raise RuntimeError("当前提示词里找不到「只输出 JSON」，无法还原旧版")
    return head + _LEGACY_REQUIREMENTS + chr(10) * 2 + tail[idx:]
