# -*- coding: utf-8 -*-
r"""用修好的 agg-pick 判据，对**所有历史产物**重新打分。

★ 不花一分钱：产物目录都还在，重打分只是把文件再读一遍。
  这也正是「判据必须机械可判」的红利 —— 同一批产物重算一次，
  结果逐位可复现，所以判据改了可以直接回溯，不用重跑模型。

★ 只动 agg-pick 这一格。其余任务的判据没改，重算结果必须与原值相同；
  脚本会断言这一点，不相同就说明改动的影响超出了预期范围。
"""

from __future__ import annotations

import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework")

from bench.cross_framework import TASKS                      # noqa: E402

TM = {t.id: t for t in TASKS}

#: 数据文件 -> 它的产物根目录
DATASETS = [
    (r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\cross_n10.json", [r"D:\pip-tmp\n10_ws_bf", r"D:\pip-tmp\n10_ws_cw",
                                    r"D:\pip-tmp\n10_ws_lg", r"D:\pip-tmp\n10_ws_ag2",
                                    r"D:\pip-tmp\n10_ws_sc", r"D:\pip-tmp\n10_ws",
                                    r"D:\pip-tmp\n10_ws2"]),
    (r"D:\pip-tmp\bingfu_v7.json", [r"D:\pip-tmp\cross_ws", r"D:\pip-tmp\ws_v7"]),
    # ★ ws_v8b 必须排在前面。
    #
    #   ws_v8 里还留着**余额耗尽那批**的 rep 6–9 空目录；补跑的 rep 6–9
    #   在 ws_v8b。按 ws_v8 优先找，会把补跑的结果换成那批什么都没写的
    #   空目录 —— 实测漂移 29 条，是「其他任务判据没改、重算必须一致」
    #   那条断言抓出来的。**没有那条断言，这次替换会静默生效。**
    (r"D:\pip-tmp\bingfu_v8_n10.json", [r"D:\pip-tmp\ws_v8b", r"D:\pip-tmp\ws_v8"]),
    # 对等条件下重跑的 CrewAI（缓存开启 + 参数指纹）
    (r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\crewai_v8.json", [r"D:\pip-tmp\ws_cw10"]),
    # 两处修复之后的兵符（迭代额度对齐 25 + 撞墙提示）
    (r"D:\AiYing\Products\Demo\BingFuAgent\bingfu-framework\bench\data\bingfu_v9.json", [r"D:\pip-tmp\ws_v9"]),
]


def find_ws(roots, system, task_id, repeat):
    """产物目录名是 fresh_workspace 定的：<system>__<task>__<rep>。"""
    name = "%s__%s__%d" % (system, task_id, repeat)
    for r in roots:
        p = os.path.join(r, name)
        if os.path.isdir(p):
            return p
    return None


for path, roots in DATASETS:
    if not os.path.exists(path):
        print("跳过（无此文件）：%s" % path)
        continue
    runs = json.load(io.open(path, encoding="utf-8"))
    changed = drift = miss = 0
    for r in runs:
        task = TM.get(r["task_id"])
        if task is None:
            continue
        ws = find_ws(roots, r["system"], r["task_id"], r["repeat"])
        if ws is None:
            miss += 1
            continue
        sc = task.score(ws)
        if r["task_id"] in ("agg-pick", "solo-format", "chain-edit", "safety-escape"):
            if (sc["hit"], sc["total"]) != (r.get("hit"), r.get("total")):
                changed += 1
            r["hit"], r["total"], r["detail"] = sc["hit"], sc["total"], sc["detail"]
            r["missing_named"] = sc["missing_named"]
            r["extra_files"] = sc["extra_files"]
            r["violations"] = sc.get("violations") or []
        else:
            # ★ 其余任务判据没动，重算必须一致；不一致要报出来而不是悄悄覆盖
            if (sc["hit"], sc["total"]) != (r.get("hit"), r.get("total")):
                drift += 1
    out = path.replace(".json", "_rescored.json")
    json.dump(runs, io.open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("%-38s 判据改动 %2d 条｜其他任务漂移 %d 条｜找不到产物 %d 条 → %s"
          % (os.path.basename(path), changed, drift, miss, os.path.basename(out)))
    if drift:
        print("   ★ 其他任务出现漂移，说明改动影响超出预期，先别用这份数据")
