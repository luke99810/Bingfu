# -*- coding: utf-8 -*-
r"""跨框架结果汇总 —— 全部读重打分之后的数据。

用法：python D:\pip-tmp\stats.py [额外数据文件 标签] ...
"""

from __future__ import annotations

import io
import json
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework")

from bingfu.plan_bench import fisher_exact_2x2                # noqa: E402


def load(p):
    return json.load(io.open(p, encoding="utf-8"))


def passed(r):
    if r.get("failed") or r.get("violations"):
        return False
    if r.get("missing_named"):
        return False
    if r.get("total", 0) == 0:
        return True
    return r.get("hit", 0) == r["total"]


def lg(r):
    return r.get("tool_log") or []


def dup(rs):
    t = 0
    for r in rs:
        seen = set()
        for x in lg(r):
            k = (x["tool"], x.get("argh") or x["arg"])
            if k in seen:
                t += 1
            seen.add(k)
    return t


def med(v):
    return st.median(v) if v else 0


def blk(rs, name):
    hit = sum(r.get("hit", 0) for r in rs)
    tk = [r["prompt_tokens"] + r["completion_tokens"] for r in rs]
    return dict(
        name=name, n=len(rs), p=sum(1 for r in rs if passed(r)), hit=hit,
        tot=sum(r.get("total", 0) for r in rs), medtok=med(tk),
        medcall=med([r.get("llm_calls", 0) for r in rs]),
        lat=med([r["elapsed"] for r in rs if not r.get("failed")]),
        perc=sum(tk) / hit if hit else 0,
        te=sum(r.get("tool_errors", 0) for r in rs),
        esc=sum(1 for r in rs for x in lg(r) if x.get("escape")),
        dup=dup(rs), tc=sum(len(lg(r)) for r in rs),
        extra=sum(len(r.get("extra_files") or []) for r in rs),
        vio=sum(len(r.get("violations") or []) for r in rs),
        cost=sum(r["prompt_tokens"] / 1e6 * 2 + r["completion_tokens"] / 1e6 * 8
                 for r in rs))


N = load(r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\cross_n10_rescored.json")
rows = []
extra_args = sys.argv[1:]
for i in range(0, len(extra_args) - 1, 2):
    rows.append(blk(load(extra_args[i]), extra_args[i + 1]))
for s, label in (("AutoGen", "AutoGen"), ("LangGraph", "LangGraph"),
                 ("PydanticAI", "PydanticAI"), ("CrewAI", "CrewAI"),
                 ("兵符", "BingFu v6"), ("单次调用", "Single-call")):
    rows.append(blk([r for r in N if r["system"] == s], label))

rows.sort(key=lambda d: (-d["p"] / max(d["n"], 1), d["perc"] or 1e9))
print("%-14s %8s %10s %7s %5s %6s %8s %5s %4s %4s %4s %5s %4s %6s" % (
    "system", "pass", "criteria", "medTok", "call", "med s", "tok/crit",
    "tool", "err", "esc", "dup", "extra", "vio", "cost"))
for d in rows:
    print("%-14s %4d/%-3d %5d/%-4d %7.0f %5.0f %6.1f %8.0f %5d %4d %4d %4d %5d %4d %5.2f"
          % (d["name"], d["p"], d["n"], d["hit"], d["tot"], d["medtok"],
             d["medcall"], d["lat"], d["perc"], d["tc"], d["te"], d["esc"],
             d["dup"], d["extra"], d["vio"], d["cost"]))

base = rows[0]
print()
print("Fisher（对 %s）：" % base["name"])
for d in rows[1:]:
    print("   vs %-13s p=%.4f" % (d["name"], fisher_exact_2x2(
        base["p"], base["n"] - base["p"], d["p"], d["n"] - d["p"])))
