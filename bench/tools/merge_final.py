# -*- coding: utf-8 -*-
r"""合成「等量诊断之后」的对照集。

★ 这份数据**混合了不同时间的测量**，必须挑明：
    AutoGen / LangGraph / PydanticAI / 单次调用 —— 来自那场并发的受控总测
    兵符 v9、CrewAI（对等） —— 事后单独串行重测

  所以：正确率、token、工具行为可以横向比（与调度无关）；
  **耗时不可以**（一半并发、一半串行）。

★ 原始的受控总测原样保留在 cross_n10_rescored.json，不覆盖。
"""

from __future__ import annotations

import io
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N = json.load(io.open(r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\cross_n10_rescored.json", encoding="utf-8"))
V9 = json.load(io.open(r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\bingfu_v9_rescored.json", encoding="utf-8"))
CW = json.load(io.open(r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\crewai_v8_rescored.json", encoding="utf-8"))

keep = [r for r in N if r["system"] not in ("兵符", "CrewAI")]
for r in V9 + CW:
    r = dict(r)
merged = keep + V9 + CW
for r in merged:
    # ★ 事后重测的两家是串行跑的，受控那批是并发的 —— 标进数据本身，
    #   不靠某处文档的一句备注。
    if r["system"] in ("兵符", "CrewAI") and r.get("contended") is not False:
        r["contended"] = False

by = {}
for r in merged:
    by.setdefault(r["system"], []).append(r)
print("合并后各系统条数：")
for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
    print("   %-12s %d" % (k, len(v)))

out = r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\final_n10.json"
json.dump(merged, io.open(out, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("已写出 %s（%d 条）" % (out, len(merged)))
