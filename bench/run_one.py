# -*- coding: utf-8 -*-
r"""跑一次 (系统, 任务, 重复)，把观测以 JSON 打到 stdout 最后一行。

★ 为什么要拆成单次子进程。

  crewai 与 autogen 的依赖互相冲突（同一个 venv 里 pip 会在
  autogen 版本上无限回溯，最终失败）。分开装进各自的 venv 之后，
  就必须由各自的解释器来跑 —— 一个进程装不下所有框架。

  副作用是好的：每个框架在自己的依赖闭包里运行，
  不会出现「A 框架被 B 框架顶掉的某个包影响了表现」这种脏对比。
"""

from __future__ import annotations

import sys as _sys
# ★ 中文控制台默认 GBK，¥ 与 ✓ 都编码不了 —— 先把 stdout 定死成 UTF-8
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.cross_framework import (                        # noqa: E402
    BASE_URL, MODEL, TASKS, Meter, Run, fresh_workspace, install_meter,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--workspaces", required=True)
    ap.add_argument("--key", required=True)
    args = ap.parse_args()

    task = next(t for t in TASKS if t.id == args.task)
    from bench.adapters import ADAPTERS, TOOL_STATS, reset_tool_stats

    meter = Meter()
    uninstall = install_meter(meter)
    root = fresh_workspace(args.workspaces, args.system, task, args.repeat)
    r = Run(system=args.system, task_id=task.id, repeat=args.repeat,
            shape=task.shape)
    reset_tool_stats()

    t0 = time.time()
    try:
        ADAPTERS[args.system](task, root, api_key=args.key,
                              model=MODEL, base_url=BASE_URL)
    except Exception as exc:                                # noqa: BLE001
        r.failed = True
        r.error = "%s: %s" % (type(exc).__name__, str(exc)[:300])
        traceback.print_exc(limit=3, file=sys.stderr)
    finally:
        uninstall()

    r.elapsed = round(time.time() - t0, 1)
    r.__dict__.update(meter.snapshot())
    sc = task.score(root)
    r.hit, r.total, r.detail = sc["hit"], sc["total"], sc["detail"]
    r.missing_named = sc["missing_named"]
    r.extra_files = sc["extra_files"]
    r.tool_calls = TOOL_STATS["calls"]
    r.tool_errors = TOOL_STATS["errors"]
    r.tool_log = list(TOOL_STATS["log"])
    r.violations = sc.get("violations") or []

    # ★ 一次 LLM 调用都没发生 = 基础设施失败，不是「这道题没做好」。
    #
    #   实测踩过：跑到第 80 次时 DeepSeek 余额耗尽，此后每个请求都是
    #   HTTP 402。供应商把异常吞掉、战役返回空输出，而这里照常打分 ——
    #   于是 45 次「什么都没发生」被记成了 45 次任务失败，
    #   正确率从 118/120 掉到 74/120，**而且没有任何地方报错**。
    #
    #   两者必须分开：跑不起来要报「跑不起来」，不能报「跑了但没做对」。
    #   一个把停电记成考试不及格的记分板，比没有记分板更糟。
    #
    #   ★ 单次调用那一档天然只有 1 次调用、且必然 >0，所以这条不会误伤它。
    if not r.failed and r.llm_calls == 0:
        r.failed = True
        r.error = ("零次 LLM 调用 —— 模型侧没有任何请求成功。"
                   "多半是配额/余额/网络问题，不是被测系统的表现。")

    # ★ 结果走 stdout 的最后一行。框架们会往 stdout 打各种日志，
    #   所以父进程只认最后一行的 JSON。
    print("__RESULT__" + json.dumps(r.__dict__, ensure_ascii=False))


if __name__ == "__main__":
    main()
