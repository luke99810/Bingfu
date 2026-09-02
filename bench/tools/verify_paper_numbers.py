# -*- coding: utf-8 -*-
r"""把论文里的每个关键数字，与原始记录重算出来的值逐一核对。

★ 论文中的数字是手写的；原始记录是机器算的。二者必须一致，
  而"一致"这件事本身必须由机器检查 —— 人工核对一张七列的表，
  漏掉一格不会有任何地方报错。
"""

import io
import json
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework")


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


def stat(rs):
    hit = sum(r.get("hit", 0) for r in rs)
    tk = [r["prompt_tokens"] + r["completion_tokens"] for r in rs]
    return {
        "pass": sum(1 for r in rs if passed(r)), "n": len(rs), "hit": hit,
        "tot": sum(r.get("total", 0) for r in rs),
        "medtok": int(st.median(tk)), "medcall": int(st.median([r.get("llm_calls", 0) for r in rs])),
        "eta": round(sum(tk) / hit) if hit else None,
        "tools": sum(len(lg(r)) for r in rs),
        "err": sum(r.get("tool_errors", 0) for r in rs),
        "esc": sum(1 for r in rs for x in lg(r) if x.get("escape")),
        "dup": dup(rs),
        "dupruns": sum(1 for r in rs if dup([r]) > 0),
        "cost": round(sum(r["prompt_tokens"] / 1e6 * 2 + r["completion_tokens"] / 1e6 * 8
                          for r in rs), 2),
    }


PRE = load(r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\cross_n10_rescored.json")
V7 = load(r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\bingfu_v7_rescored.json")
V9 = load(r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\bingfu_v9_rescored.json")
CW = load(r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\crewai_v8_rescored.json")
S = lambda n: [r for r in PRE if r["system"] == n]

#: 论文中写下的值 —— 逐格核对
CLAIMS = [
    # 表 1：成功率与成本（最终配置）
    ("表1 兵符 通过", stat(V9)["pass"], 120),
    ("表1 兵符 eta", stat(V9)["eta"], 955),
    ("表1 兵符 中位tok", stat(V9)["medtok"], 1769),
    ("表1 兵符 花费", stat(V9)["cost"], 0.63),
    ("表1 LangGraph 通过", stat(S("LangGraph"))["pass"], 120),
    ("表1 LangGraph eta", stat(S("LangGraph"))["eta"], 960),
    ("表1 LangGraph 中位tok", stat(S("LangGraph"))["medtok"], 1716),
    ("表1 AutoGen 通过", stat(S("AutoGen"))["pass"], 120),
    ("表1 AutoGen eta", stat(S("AutoGen"))["eta"], 976),
    ("表1 AutoGen 中位tok", stat(S("AutoGen"))["medtok"], 1856),
    ("表1 PydanticAI 通过", stat(S("PydanticAI"))["pass"], 119),
    ("表1 PydanticAI eta", stat(S("PydanticAI"))["eta"], 1036),
    ("表1 CrewAI 通过", stat(CW)["pass"], 110),
    ("表1 CrewAI eta", stat(CW)["eta"], 3933),
    ("表1 CrewAI 中位tok", stat(CW)["medtok"], 1823),
    ("表1 CrewAI 花费", stat(CW)["cost"], 2.30),
    # 表 2：工具层面
    ("表2 AutoGen 工具调用", stat(S("AutoGen"))["tools"], 340),
    ("表2 AutoGen 报错", stat(S("AutoGen"))["err"], 29),
    ("表2 AutoGen 越界", stat(S("AutoGen"))["esc"], 16),
    ("表2 AutoGen 冗余", stat(S("AutoGen"))["dup"], 1),
    ("表2 兵符 工具调用", stat(V9)["tools"], 344),
    ("表2 兵符 报错", stat(V9)["err"], 36),
    ("表2 兵符 越界", stat(V9)["esc"], 21),
    ("表2 兵符 冗余", stat(V9)["dup"], 0),
    ("表2 LangGraph 工具调用", stat(S("LangGraph"))["tools"], 362),
    ("表2 LangGraph 报错", stat(S("LangGraph"))["err"], 41),
    ("表2 LangGraph 越界", stat(S("LangGraph"))["esc"], 32),
    ("表2 LangGraph 冗余", stat(S("LangGraph"))["dup"], 5),
    ("表2 PydanticAI 报错", stat(S("PydanticAI"))["err"], 63),
    ("表2 PydanticAI 越界", stat(S("PydanticAI"))["esc"], 38),
    ("表2 PydanticAI 冗余", stat(S("PydanticAI"))["dup"], 3),
    ("表2 CrewAI 工具调用", stat(CW)["tools"], 679),
    ("表2 CrewAI 报错", stat(CW)["err"], 366),
    ("表2 CrewAI 越界", stat(CW)["esc"], 351),
    ("表2 CrewAI 冗余", stat(CW)["dup"], 0),
    # 表 3 涉及的计数
    ("表3 兵符 含冗余运行数", stat(V9)["dupruns"], 0),
    ("表3 LangGraph 含冗余运行数", stat(S("LangGraph"))["dupruns"], 5),
    ("表3 AutoGen 含冗余运行数", stat(S("AutoGen"))["dupruns"], 1),
]


bad = 0
for name, measured, written in CLAIMS:
    ok = (abs(measured - written) < 1e-9 if isinstance(written, float)
          else measured == written)
    if not ok:
        bad += 1
        print("  x %-28s 实测 %-8s 论文写 %s" % (name, measured, written))

print("核对 %d 格，%s" % (len(CLAIMS), "全部一致" if bad == 0 else "%d 格不符" % bad))

# 单点检查：全部失败是否都在 safety-escape
final = {"兵符": V9, "CrewAI": CW, "AutoGen": S("AutoGen"),
         "LangGraph": S("LangGraph"), "PydanticAI": S("PydanticAI")}
allbad = [(k, r["task_id"]) for k, rs in final.items()
          for r in rs if not passed(r)]
tasks = {t for _, t in allbad}
print("受控总测的全部失败任务：%s（论文称仅 safety-escape）%s"
      % (tasks, "" if tasks <= {"safety-escape"} else "  ← 不符"))
sys.exit(0 if bad == 0 and tasks <= {"safety-escape"} else 1)
