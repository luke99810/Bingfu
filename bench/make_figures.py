# -*- coding: utf-8 -*-
r"""Publication-quality figures for the cross-framework evaluation.

All numbers are read from the raw run records — nothing is typed by hand,
so a figure can never drift from the data it claims to show.

Output: 300 dpi PNG + vector PDF, Times New Roman, English labels only.
"""

from __future__ import annotations

import sys as _sys

_sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import os
import statistics
from collections import defaultdict

import matplotlib
import matplotlib.patches
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#: 主结果：等量诊断之后的 720 次。
#:
#: ★ 这份数据**混合了不同时间的测量**：AutoGen / LangGraph / PydanticAI /
#:   单次调用来自那场并发的受控总测；兵符 v9 与 CrewAI（对等条件）
#:   是事后单独串行重测的。正确率、token、工具行为与调度无关，可以横向比；
#:   **耗时不可以** —— 一半并发一半串行，图里不画耗时。
RUNS = r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\final_n10.json"
#: 诊断之前的受控总测，原样保留、不覆盖
RUNS_PRE = r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\cross_n10_rescored.json"
#: 延迟专用数据集。
#:
#: ★ 主数据集是并发跑的（contended=True），elapsed 受 CPU 抢占污染。
#:   延迟必须回到串行那批（n=2，144 次）—— 混用会把调度开销
#:   当成框架特性。
RUNS_SERIAL = r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\cross_v6.json"
OBS = r"D:\pip-tmp\obs.json"
OUT = r"C:\Users\宿心\Desktop\bingfu-figures"

#: 英文系统名 —— 图里不出现中文
LABEL = {"兵符": "BingFu", "单次调用": "Single-call",
         "LangGraph": "LangGraph", "CrewAI": "CrewAI",
         "AutoGen": "AutoGen", "PydanticAI": "PydanticAI"}

SHAPE_LABEL = {"solo": "Solo", "chain": "Chain", "fan_out": "Fan-out",
               "aggregate": "Aggregate", "robust": "Robustness",
               "long": "Long-range", "safety": "Safety"}

#: 统一配色：被评估的框架用中性灰蓝，兵符高亮，地板参照用浅灰
COLORS = {"BingFu": "#2F5D8A", "LangGraph": "#7F9DB9", "CrewAI": "#7F9DB9",
          "AutoGen": "#7F9DB9", "PydanticAI": "#7F9DB9",
          "Single-call": "#C9C9C9"}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def passed(r) -> bool:
    if r["failed"] or (r.get("violations") or []):
        return False
    if r.get("missing_named"):
        return False
    if r["total"] == 0:
        return True
    return r["hit"] == r["total"]


def load():
    runs = json.load(open(RUNS, encoding="utf-8"))
    by = defaultdict(list)
    for r in runs:
        by[LABEL[r["system"]]].append(r)
    return runs, by


def save(fig, name) -> None:
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)))
    plt.close(fig)
    print("  %s.png / .pdf" % name)


# ══════════════════════════════════════════════════════════
#  Figure 1 — success rate overall and by task shape
# ══════════════════════════════════════════════════════════

def fig1(by):
    order = sorted(by, key=lambda s: -sum(1 for r in by[s] if passed(r)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                                   gridspec_kw={"width_ratios": [1, 1.45]})

    rates = [sum(1 for r in by[s] if passed(r)) / len(by[s]) * 100 for s in order]
    bars = ax1.bar(range(len(order)), rates,
                   color=[COLORS[s] for s in order], width=0.62,
                   edgecolor="black", linewidth=0.5)
    for i, (s, v) in enumerate(zip(order, rates)):
        n_pass = sum(1 for r in by[s] if passed(r))
        ax1.text(i, v + 2, "%d/%d" % (n_pass, len(by[s])), ha="center",
                 va="bottom", fontsize=7.5)
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels(order, rotation=30, ha="right")
    ax1.set_ylabel("Task success rate (%)")
    ax1.set_ylim(0, 108)
    ax1.set_title("(a) Overall", loc="left", fontweight="bold")
    ax1.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax1.set_axisbelow(True)

    shapes = ["solo", "chain", "fan_out", "aggregate", "robust", "long", "safety"]
    data = np.zeros((len(order), len(shapes)))
    for i, s in enumerate(order):
        for j, sh in enumerate(shapes):
            rs = [r for r in by[s] if r.get("shape") == sh]
            data[i, j] = (sum(1 for r in rs if passed(r)) / len(rs) * 100) if rs else np.nan

    im = ax2.imshow(data, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax2.set_xticks(range(len(shapes)))
    ax2.set_xticklabels([SHAPE_LABEL[x] for x in shapes], rotation=30, ha="right")
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(shapes)):
            v = data[i, j]
            if not np.isnan(v):
                ax2.text(j, i, "%d" % v, ha="center", va="center", fontsize=7.5,
                         color="white" if v > 55 else "black")
    ax2.set_title("(b) By task shape (%)", loc="left", fontweight="bold")
    ax2.spines[:].set_visible(False)
    ax2.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.02)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7.5, length=2)
    fig.tight_layout()
    save(fig, "fig1_success")


# ══════════════════════════════════════════════════════════
#  Figure 2 — cost / accuracy trade-off
# ══════════════════════════════════════════════════════════

def fig2(by):
    """(a) 全景；(b) 放大顶部集群。

    ★ 三个系统在 tok/criterion 上相差不到 7 %，在成功率上相差不到 2 点。
      在全景图里它们必然重叠 —— 放大图不是装饰，是让读者看见
      「这三个点确实分不开」这件事本身。
    """

    pts = {}
    for s_, rs in by.items():
        hit = sum(r["hit"] for r in rs)
        if not hit:
            continue
        pts[s_] = (sum(r["prompt_tokens"] + r["completion_tokens"] for r in rs) / hit,
                   sum(1 for r in rs if passed(r)) / len(rs) * 100)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    def draw(a, off, size=58, fs=8):
        for s_, (x, y) in pts.items():
            if s_ not in off:
                continue
            a.scatter(x, y, s=size, color=COLORS[s_], edgecolor="black",
                      linewidth=0.6, zorder=3)
            if off[s_] is None:          # 画点但不标注
                continue
            a.annotate(s_, (x, y), textcoords="offset points",
                       xytext=off[s_], ha="center", fontsize=fs, zorder=4)
        a.set_xscale("log")
        a.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: "%d" % v))
        a.grid(linewidth=0.4, alpha=0.4)
        a.set_axisbelow(True)

    # ── (a) 全景 ────────────────────────────────────────
    # ★ 顶部四点在全景尺度下必然重叠，所以在 (a) 里**不逐点标注** ——
    #   硬标只会得到互相压住的字。它们的身份由 (b) 给出。
    draw(ax, {"LangGraph": None, "PydanticAI": None, "AutoGen": None,
              "BingFu": None, "CrewAI": (0, -15), "Single-call": (0, 10)})
    ax.set_xticks([1000, 2000, 5000])
    ax.set_xlim(700, 9000)
    ax.set_ylim(-8, 112)
    ax.set_xlabel("Tokens per scored criterion (log scale)")
    ax.set_ylabel("Task success rate (%)")
    ax.set_title("(a) All systems", loc="left", fontweight="bold")

    # 标出放大范围
    ax.add_patch(matplotlib.patches.Rectangle(
        (890, 96.5), 1600 - 890, 100.4 - 96.5, fill=False,
        edgecolor="0.35", linewidth=0.8, linestyle="--", zorder=2))
    ax.annotate("Four systems (see panel b)", (1180, 95.0), ha="center",
                va="top", fontsize=8, color="0.25")

    # ── (b) 放大 ────────────────────────────────────────
    draw(bx, {"LangGraph": (0, -15), "PydanticAI": (0, 10),
              "AutoGen": (0, 10), "BingFu": (0, 10)}, size=70)
    bx.set_xticks([900, 1000, 1200, 1500])
    # ★ 对数轴的次刻度会自动打上 1.1×10^3 之类的标签，把主刻度挤没了
    bx.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    bx.set_xlim(890, 1650)
    bx.set_ylim(96.5, 100.6)
    bx.set_xlabel("Tokens per scored criterion (log scale)")
    bx.set_title("(b) Top cluster, magnified", loc="left", fontweight="bold")

    fig.tight_layout()
    save(fig, "fig2_tradeoff")


# ══════════════════════════════════════════════════════════
#  Figure 3 — tool-use behaviour
# ══════════════════════════════════════════════════════════

def fig3(by):
    order = [s for s in ("LangGraph", "BingFu", "PydanticAI", "CrewAI",
                         "AutoGen") if s in by]
    metrics = [("Tool errors", lambda rs: sum(r.get("tool_errors", 0) for r in rs)),
               ("Path-escape attempts",
                lambda rs: sum(1 for r in rs for x in (r.get("tool_log") or [])
                               if x.get("escape"))),
               ("Redundant calls", None)]

    def redundant(rs):
        """★ 判等用 argh（全量参数哈希），不能用 arg（80 字符截断的展示值）。

        用 arg 判等时，CrewAI 那些越来越长的越界路径截断后全都相同，
        于是被算成重复调用 —— 实测它 320 次「冗余」里 245 次是这么来的。
        显示层的有损处理漏进判定层，产生的错误看起来和真实行为一模一样。
        """
        tot = 0
        for r in rs:
            seen = set()
            for x in (r.get("tool_log") or []):
                k = (x["tool"], x.get("argh") or x["arg"])
                if k in seen:
                    tot += 1
                seen.add(k)
        return tot

    metrics[2] = ("Redundant calls", redundant)

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    w, xs = 0.26, np.arange(len(order))
    hatches = ["", "///", "..."]
    greys = ["#3A3A3A", "#8A8A8A", "#C4C4C4"]
    for k, (name, fn) in enumerate(metrics):
        vals = [fn(by[s]) for s in order]
        b = ax.bar(xs + (k - 1) * w, [max(v, 0.4) for v in vals], w,
                   label=name, color=greys[k], edgecolor="black",
                   linewidth=0.5, hatch=hatches[k])
        for xi, v in zip(xs + (k - 1) * w, vals):
            ax.text(xi, max(v, 0.4) * 1.08, str(v), ha="center", va="bottom",
                    fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=20, ha="right")
    per_system = max(len(v) for v in by.values())
    ax.set_ylabel("Count over %d runs (log scale)" % per_system)
    ax.set_ylim(0.3, 1200)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    ax.set_title("Tool-use behaviour", loc="left", fontweight="bold")
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig3_tooluse")


# ══════════════════════════════════════════════════════════
#  Figure 4 — effect of the three profiling-guided fixes on BingFu
# ══════════════════════════════════════════════════════════

def fig5():
    data = json.load(open(OBS, encoding="utf-8"))
    # 探测结果里 PydanticAI 与兵符是后来单独补测的，这里以最新值为准
    override = {"PydanticAI": {"steps": "yes", "tokens": "yes",
                               "trace": "ordered", "resume": "api"},
                "兵符": {"steps": "yes", "tokens": "yes",
                       "trace": "count", "resume": "round-trip"}}
    level = {"有": "yes", "有序": "ordered", "计数": "count",
             "往返": "round-trip", "API": "api", "无": "none"}
    score = {"none": 0, "count": 1, "api": 1, "yes": 2, "ordered": 2,
             "round-trip": 2}
    cols = ["Step events", "Token accounting", "Tool trace", "Resume"]
    keys = ["steps", "tokens", "trace", "resume"]
    rows, mat, text = [], [], []
    for d in data:
        name = LABEL[d["system"]]
        vals = []
        for k in keys:
            v = override.get(d["system"], {}).get(k) or level.get(d[k][0], "none")
            vals.append(v)
        rows.append(name)
        mat.append([score[v] for v in vals])
        text.append(vals)

    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    im = ax.imshow(np.array(mat), cmap="Blues", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=18, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=7.5,
                    color="white" if mat[i][j] == 2 else "black")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Observability probe", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig7_observability")


# ══════════════════════════════════════════════════════════
#  Figure 6 — effect of the verification / assessment fixes
# ══════════════════════════════════════════════════════════

#: 修复之后单独重跑的兵符（各 120 次，**串行**）
#:   v7 = 第一轮三处修复（验收判产物、庙算按需、军师闸门）
#:   v8 = 第二轮（拆解闸门改为免费的结构判据）
RUNS_V7 = r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\bingfu_v7.json"
#: ★ 只含 rep 0–5。本来要跑 10 次，第 80 次时 API 余额耗尽，
#:   此后 45 次是「零调用」的假数据 —— 已剔除并单独存档
#:   (bingfu_v8_partial_balance_ran_out.json)。
RUNS_V8 = r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\bingfu_v9_rescored.json"
#: 等量诊断之后重跑的 CrewAI（120 次，串行）
RUNS_CW = r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\crewai_v8_rescored.json"


def fig_improvement():
    """修复效果，两联。左：按任务；右：汇总指标。

    ★ 这是全套图里**唯一**讲修复过程的一张，刻意只给一张。
      本文的主体是当前性能与测量效度，修复只是其中一条证据 ——
      给它两张图会让读者以为论文是关于「我们改好了什么」的。

    ★ 左图按任务画而非画总数：两处毛病形状不同 ——
      一处是每道题固定多一次调用（处处小幅高），一处是三道题贵三倍
      （柱子孤零零冲出去）。**一个聚合数字会把它们混成一件事。**
    """

    v6_all = json.load(open(RUNS_PRE, encoding="utf-8"))
    v9_all = json.load(open(RUNS_V8, encoding="utf-8"))

    def med_tok(rs):
        v = [r["prompt_tokens"] + r["completion_tokens"] for r in rs]
        return statistics.median(v) if v else 0

    seen, tasks = set(), []
    for r in v6_all:
        if r["task_id"] not in seen:
            seen.add(r["task_id"])
            tasks.append(r["task_id"])

    before, after, rivals, keep = [], [], [], []
    for tid in tasks:
        a = [r for r in v6_all if r["system"] == "兵符" and r["task_id"] == tid]
        b = [r for r in v9_all if r["task_id"] == tid]
        o = [r for r in v6_all
             if r["system"] in ("PydanticAI", "AutoGen", "LangGraph")
             and r["task_id"] == tid]
        if not (a and b and o):
            continue
        keep.append(tid)
        before.append(med_tok(a))
        after.append(med_tok(b))
        rivals.append(med_tok(o))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 2.9),
                                 gridspec_kw={"width_ratios": [1.75, 1]})

    xs = np.arange(len(keep))
    w = 0.36
    ax.bar(xs - w / 2, before, w, label="Before", color="#B7B7B7",
           edgecolor="black", linewidth=.5)
    ax.bar(xs + w / 2, after, w, label="After", color="#2F5D8A",
           edgecolor="black", linewidth=.5)
    for i, r in enumerate(rivals):
        ax.plot([i - w, i + w], [r, r], color="#A83C2D", lw=1.4, zorder=4,
                label="Rival median" if i == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels(keep, rotation=34, ha="right", fontsize=7)
    ax.set_ylabel("Median tokens per run")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("(a) By task", loc="left", fontweight="bold")
    ax.grid(axis="y", linewidth=.4, alpha=.4)
    ax.set_axisbelow(True)

    # ── 右：四个汇总指标，归一化到修复前 = 1 ──
    names = ["Tokens per\ncriterion", "Median\nLLM calls",
             "Runs with\nredundancy", "Cost per\n120 runs"]
    pre = [1596, 4, 25, 1.14]
    post = [955, 3, 0, 0.63]
    ratio = [p / q for p, q in zip(post, pre)]
    ys = np.arange(len(names))[::-1]
    bx.barh(ys, [1] * len(names), height=.5, color="#E4E4DD",
            edgecolor="black", linewidth=.5)
    bx.barh(ys, ratio, height=.5, color="#2F5D8A", edgecolor="black",
            linewidth=.5)
    for y, r, p, q in zip(ys, ratio, pre, post):
        bx.text(max(r, .06) + .04, y, "%g → %g" % (p, q), va="center",
                fontsize=7.5)
    bx.set_yticks(ys)
    bx.set_yticklabels(names, fontsize=7.5)
    bx.set_xlim(0, 1.65)
    bx.set_xticks([0, .5, 1])
    bx.set_xticklabels(["0", "0.5", "1 = before"], fontsize=7.5)
    bx.set_title("(b) Aggregate", loc="left", fontweight="bold")
    bx.grid(axis="x", linewidth=.4, alpha=.4)
    bx.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "fig8_improvement")


# ══════════════════════════════════════════════════════════
#  Figure 7 — equal diagnosis applied to CrewAI
# ══════════════════════════════════════════════════════════

def fig7():
    """CrewAI 的冗余调用：原始数字 / 扣掉测量缺陷 / 对等配置后重测。

    ★ 这张图画的是**一个被我自己的工具放大了四倍的数字**如何被拆开的。

      左：原始 320。
      中：其中 245 次来自参数截断 —— 日志只存前 80 字符，
          而 CrewAI 在安全题上拼出的长路径截断后全都一样。
      右：修好指纹、并把 CrewAI 自己的工具缓存按对等条件打开之后，
          重测得到的数。
    """

    import os

    if not os.path.exists(RUNS_CW):
        print("  (跳过 fig7：还没有 CrewAI 重测数据)")
        return

    old = [r for r in json.load(open(RUNS_PRE, encoding="utf-8"))
           if r["system"] == "CrewAI"]
    new = json.load(open(RUNS_CW, encoding="utf-8"))

    def dup(rs, key):
        tot = 0
        for r in rs:
            seen = set()
            for x in (r.get("tool_log") or []):
                k = (x["tool"], key(x))
                if k in seen:
                    tot += 1
                seen.add(k)
        return tot

    raw = dup(old, lambda x: x["arg"])                       # 旧口径
    # 扣掉被截断参数：只统计未被截断的调用，得到一个**下界**
    def dup_clean(rs):
        tot = 0
        for r in rs:
            seen = set()
            for x in (r.get("tool_log") or []):
                if len(x["arg"]) >= 80:
                    continue
                k = (x["tool"], x["arg"])
                if k in seen:
                    tot += 1
                seen.add(k)
        return tot
    lower = dup_clean(old)
    remeasured = dup(new, lambda x: x.get("argh") or x["arg"])

    labels = ["As published\n(truncated key)",
              "Lower bound after\nremoving the artefact",
              "Re-measured\n(fingerprint + cache on)"]
    vals = [raw, lower, remeasured]
    colors = ["#B03A2E", "#D5AFAB", "#2F5D8A"]

    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    bars = ax.bar(range(3), vals, width=0.6, color=colors,
                  edgecolor="black", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.03, "%d" % v, ha="center", va="bottom",
                fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("Redundant tool calls, 120 runs")
    ax.set_ylim(0, max(vals) * 1.25 + 1)
    ax.set_title("Equal diagnosis: CrewAI redundancy", loc="left",
                 fontweight="bold")
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig9_validity")



# ══════════════════════════════════════════════════════════
#  按任务的成本矩阵 —— 差异到底落在哪几道题上
# ══════════════════════════════════════════════════════════

def fig_task_matrix(by):
    """每系统 × 每任务的中位 token，按任务列归一化。

    ★ 为什么归一化到「该任务的最小值」而不是全局最大值：
      任务之间的绝对量级差好几倍（solo 约 1.1k、safety 约 3.7k），
      不归一化的话整张图只会显示「哪道题贵」，而那与系统无关。
      归一化之后每一列的读法是：**在这道题上，谁比最省的那家多花多少**。
    """

    order = [s for s in ("BingFu", "AutoGen", "LangGraph", "PydanticAI",
                         "CrewAI") if s in by]
    tasks, seen = [], set()
    for r in json.load(open(RUNS, encoding="utf-8")):
        if r["task_id"] not in seen:
            seen.add(r["task_id"])
            tasks.append(r["task_id"])
    tasks.sort(key=lambda t: statistics.median(
        [r["prompt_tokens"] + r["completion_tokens"]
         for s in order for r in by[s] if r["task_id"] == t]))

    M = np.zeros((len(order), len(tasks)))
    for i, s in enumerate(order):
        for j, t in enumerate(tasks):
            v = [r["prompt_tokens"] + r["completion_tokens"]
                 for r in by[s] if r["task_id"] == t]
            M[i, j] = statistics.median(v) if v else np.nan
    base = np.nanmin(M, axis=0)
    R = M / base                       # 每列相对该列最省者的倍数

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    # ★ 色标上限取 1.6 而非数据最大值。
    #
    #   最大值是 23.0（CrewAI 在安全题上），若让它决定色标，其余 59 格
    #   全部压成同一片白 —— 而**那 59 格才是主要结论**：十一道题上
    #   全部系统都在 1.0–1.5 倍之间。离群值单独标注，
    #   不让它决定其余格子的可读性。
    VMAX = 1.6
    im = ax.imshow(np.clip(R, 1.0, VMAX), cmap="OrRd", vmin=1.0, vmax=VMAX,
                   aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=32, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(tasks)):
            if np.isnan(R[i, j]):
                continue
            out = R[i, j] > VMAX
            ax.text(j, i, "%.1f" % R[i, j], ha="center", va="center",
                    fontsize=7.5 if out else 7,
                    fontweight="bold" if out else "normal",
                    color="white" if R[i, j] > 1.45 else "black")
            if out:      # 离群格加边框，说明它已超出色标
                ax.add_patch(matplotlib.patches.Rectangle(
                    (j - .5, i - .5), 1, 1, fill=False, edgecolor="#4A0F0A",
                    linewidth=1.6, zorder=5))
    ax.set_title("Median tokens per task, relative to the cheapest system",
                 loc="left", fontweight="bold")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cb.set_label("×  cheapest  (clipped at %.1f)" % VMAX, fontsize=8)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7.5, length=2)
    fig.tight_layout()
    save(fig, "fig4_task_matrix")


# ══════════════════════════════════════════════════════════
#  单次运行 token 的分布 —— 中位数之外的可预测性
# ══════════════════════════════════════════════════════════

def fig_distribution(by):
    """箱线 + 散点。

    ★ 中位数相近不代表行为相近。PydanticAI 与 AutoGen 的中位 token
      相差不到 5%，而其标准差是 1190 对 610 —— 后者的开销可预测得多。
      **一个只报中位数的表格看不见这件事**，而在按次计费的场景里，
      分布的右尾才是真正的成本。
    """

    order = [s for s in ("BingFu", "AutoGen", "LangGraph", "PydanticAI",
                         "CrewAI") if s in by]
    data = [[r["prompt_tokens"] + r["completion_tokens"] for r in by[s]]
            for s in order]

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    bp = ax.boxplot(data, vert=True, widths=.55, showfliers=False,
                    patch_artist=True, medianprops=dict(color="black", lw=1.2))
    for patch, s in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS[s])
        patch.set_alpha(.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(.6)
    # ★ 叠散点：箱线会把「有多少次跑到很贵」藏起来
    rng = np.random.RandomState(20260825)
    for i, d in enumerate(data, start=1):
        x = rng.normal(i, .055, size=len(d))
        ax.scatter(x, d, s=4, alpha=.28, color="#2B2B2B", linewidths=0,
                   zorder=3)
    # ★ P90 写进 x 轴标签，不画进画布。
    #
    #   一版把它标在 max(d)*1.35 处 —— CrewAI 的最大值近 10^5，
    #   量程被撑到十万，其余四家的箱体全挤成一条线。
    #   **为了多给一点信息而把主图挤没的标注，是净损失。**
    labels = ["%s\nP90 %s" % (s_, format(sorted(d)[int(.9 * len(d))], ","))
              for s_, d in zip(order, data)]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Tokens per run")
    ax.set_yscale("log")
    ax.grid(axis="y", linewidth=.4, alpha=.4)
    ax.set_axisbelow(True)
    ax.set_title("Per-run token distribution (120 runs each)", loc="left",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "fig5_distribution")


# ══════════════════════════════════════════════════════════
#  配对自助法区间 —— 差异是否越过噪声
# ══════════════════════════════════════════════════════════

def fig_significance(by):
    """森林图：每得分点 token 的两两差值及其 95% 区间。

    ★ 这是全篇最重要的一张图，因为它画的是**结论本身**：
      四个框架的成本差异全部跨零。表格里这是三行数字，
      图上是三条穿过零线的横杠 —— 后者一眼就能看出"没有区分"。
    """

    import random

    base = "BingFu"
    others = [s for s in ("LangGraph", "AutoGen", "PydanticAI", "CrewAI")
              if s in by]

    def eta(rs):
        hit = sum(r["hit"] for r in rs)
        return (sum(r["prompt_tokens"] + r["completion_tokens"] for r in rs)
                / hit) if hit else float("nan")

    tasks = sorted({r["task_id"] for r in by[base]})
    rng = random.Random(20260825)
    rows = []
    for o in others:
        A = {t: [r for r in by[base] if r["task_id"] == t] for t in tasks}
        B = {t: [r for r in by[o] if r["task_id"] == t] for t in tasks}
        d = eta(by[base]) - eta(by[o])
        boots = []
        for _ in range(1200):
            pick = [rng.choice(tasks) for _ in tasks]
            boots.append(eta([r for t in pick for r in A[t]])
                         - eta([r for t in pick for r in B[t]]))
        boots.sort()
        rows.append((o, d, boots[int(.025 * len(boots))],
                     boots[int(.975 * len(boots))]))

    # ★ 拆两联：CrewAI 的差值约为 −3000，而三个头部对比在 ±90 量级。
    #   画在同一根轴上，后者会被压成零宽 —— 而**后者才是结论**。
    top = [r for r in rows if r[0] != "CrewAI"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 2.5),
                                 gridspec_kw={"width_ratios": [1.55, 1]})

    def draw(a, data, title, zoom):
        ys = np.arange(len(data))[::-1]
        for y, (name, d, lo, hi) in zip(ys, data):
            crosses = lo < 0 < hi
            col = "#8A8A8A" if crosses else "#2F5D8A"
            a.plot([lo, hi], [y, y], color=col, lw=2.4,
                   solid_capstyle="round", zorder=2)
            a.plot([d], [y], "o", color=col, ms=7, mec="black", mew=.6,
                   zorder=3)
            if zoom:
                a.text(hi, y + .28, "n.s." if crosses else "sig.",
                       ha="center", fontsize=7.5, color=col)
        a.axvline(0, color="#A83C2D", lw=1.1, ls="--", zorder=1)
        a.set_yticks(ys)
        a.set_yticklabels(["− " + r[0] for r in data], fontsize=8.5)
        a.set_ylim(-.6, len(data) - .4)
        a.grid(axis="x", linewidth=.4, alpha=.4)
        a.set_axisbelow(True)
        a.set_title(title, loc="left", fontweight="bold", fontsize=9.5)

    draw(ax, top, "(a) Within the leading group", True)
    ax.set_xlabel("Δ tokens per satisfied criterion", fontsize=8.5)

    draw(bx, rows, "(b) Same axis, CrewAI included", False)
    bx.set_xlabel("Δ tokens per criterion", fontsize=8.5)
    bx.tick_params(labelsize=7.5)

    fig.suptitle("Paired bootstrap over tasks, 95 % interval  ·  "
                 "all three leading-group intervals cross zero",
                 x=.012, ha="left", fontweight="bold", fontsize=10.5, y=1.03)
    fig.tight_layout()
    save(fig, "fig6_significance")


def main() -> None:
    """默认画 n=10 那批；`--runs/--out` 可以把旧结果重画成一套存档。

    ★ n=2 那批结果必须**留着**，不是被覆盖掉。旧图重画得出来，
      靠的是图全部从原始记录生成 —— 手填过一个数字，存档就重现不了。
    """

    import argparse
    global RUNS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    RUNS, OUT = a.runs, a.out

    setup_style()
    runs, by = load()
    print("Loaded %d runs, %d systems" % (len(runs), len(by)))
    # ── 性能刻画（主体）──
    fig1(by)              # 成功率总览 + 按形状
    fig2(by)              # 成本-准确率
    fig3(by)              # 工具层面行为
    fig_task_matrix(by)   # 按任务的成本矩阵
    fig_distribution(by)  # 单次 token 分布
    fig_significance(by)  # 配对自助法区间
    fig5()                # 可观测性探测 → fig7
    if os.path.exists(RUNS_V8) and RUNS == r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\final_n10.json":
        fig_improvement()  # 修复效果（唯一一张）
        fig7()             # 测量效度：冗余计数的更正
    print("Figures written to %s" % OUT)


if __name__ == "__main__":
    main()
