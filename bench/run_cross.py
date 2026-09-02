# -*- coding: utf-8 -*-
r"""跨框架实测的调度器：每次运行开一个子进程，用该框架自己的解释器。

用法：
    python bench/run_cross.py --estimate                 # 只估成本，不调用模型
    python bench/run_cross.py --repeats 2                # 串行，墙钟可比
    python bench/run_cross.py --repeats 10 --workers 3   # 并发，墙钟**不可比**

★ --workers > 1 时每条记录会带 contended=True。

  并发只压缩墙钟，代价必须写进**数据本身**而不是某处文档的一句备注：
  多个子进程互相抢 CPU 会让 elapsed 膨胀且不可比。准确率、token、
  工具行为与调度无关，不受影响；延迟对比必须回到串行那批数据上看。
"""

from __future__ import annotations

import sys as _sys

_sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.cross_framework import TASKS                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _venv(name: str) -> str:
    return os.path.join("D:" + os.sep, name, "Scripts", "python.exe")


#: 每个系统用哪个解释器。
#:
#: ★ crewai 与 autogen 的依赖互相冲突，必须各在各的 venv 里。
#:   兵符 / LangGraph / 单次调用 共用主环境。
INTERPRETERS = {
    "兵符": sys.executable,
    "单次调用": sys.executable,
    "LangGraph": sys.executable,
    "CrewAI": _venv("venv-crewai"),
    "PydanticAI": _venv("venv-pydanticai"),
    "AutoGen": _venv("venv-autogen"),
}

#: DeepSeek deepseek-chat 标价（元 / 百万 token），注明日期以便日后复核
PRICE_IN_PER_M, PRICE_OUT_PER_M = 2.0, 8.0
PRICE_NOTE = "按 2026-08 deepseek-chat 标价：输入 ¥2/M（缓存未命中）、输出 ¥8/M"

#: 子进程没能产出结果时的占位记录 —— 字段要与正常记录一致，
#: 否则下游统计会因为缺键而静默漏算。
EMPTY = {"hit": 0, "total": 0, "prompt_tokens": 0, "completion_tokens": 0,
         "llm_calls": 0, "detail": {}, "missing_named": [], "extra_files": [],
         "tool_calls": 0, "tool_errors": 0, "tool_log": [], "violations": []}


def cost_of(pt: int, ct: int) -> float:
    return pt / 1e6 * PRICE_IN_PER_M + ct / 1e6 * PRICE_OUT_PER_M


def resolve_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    env = os.path.join(ROOT, ".env")
    if os.path.exists(env):
        for line in open(env, encoding="utf-8", errors="replace"):
            if line.strip().startswith("DEEPSEEK_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("找不到 DEEPSEEK_API_KEY")


def available(system: str) -> bool:
    return os.path.exists(INTERPRETERS[system])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=os.path.join("D:" + os.sep, "pip-tmp",
                                                  "cross_runs.json"))
    ap.add_argument("--workspaces", default=os.path.join("D:" + os.sep, "pip-tmp",
                                                         "cross_ws"))
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rep-offset", type=int, default=0,
                    help="重复编号的起点，便于分批补跑而不撞号")
    args = ap.parse_args()

    systems = [s for s in INTERPRETERS
               if (not args.only or s == args.only) and available(s)]
    missing = [s for s in INTERPRETERS if not available(s)]
    if missing:
        print("跳过（解释器不存在）：%s" % "、".join(missing))

    n = len(systems) * len(TASKS) * args.repeats
    print("系统：%s" % "、".join(systems))
    print("任务 %d 道 × 重复 %d 次 → 共 %d 次运行" % (len(TASKS), args.repeats, n))
    print(PRICE_NOTE)
    if args.estimate:
        print("粗估上限：约 ¥%.2f（按每次 8k 输入 + 1.5k 输出估）"
              % (n * cost_of(8000, 1500)))
        return

    contended = args.workers > 1
    if contended:
        print("★ --workers=%d：本批 elapsed 受并发污染，不可用于延迟对比"
              % args.workers)

    key = resolve_key()
    os.makedirs(args.workspaces, exist_ok=True)
    runs: list = []
    lock = threading.Lock()
    t_start = time.time()

    jobs = [(args.rep_offset + rep, task, system)
            for rep in range(args.repeats) for task in TASKS for system in systems]

    def _one(job):
        rep, task, system = job
        cmd = [INTERPRETERS[system], os.path.join(HERE, "run_one.py"),
               "--system", system, "--task", task.id, "--repeat", str(rep),
               "--workspaces", args.workspaces, "--key", key]
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = ROOT
        base = {"system": system, "task_id": task.id, "repeat": rep,
                "shape": task.shape, "elapsed": 0.0, "failed": True, "error": ""}
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=args.timeout, env=env)
            line = next((l for l in reversed(proc.stdout.splitlines())
                         if l.startswith("__RESULT__")), None)
            if line:
                rec = json.loads(line[len("__RESULT__"):])
            else:
                rec = dict(base, **EMPTY)
                rec["error"] = "子进程没有产出结果：%s" % (proc.stderr or "")[-200:]
        except subprocess.TimeoutExpired:
            rec = dict(base, **EMPTY)
            rec["error"] = "超时 %ds" % args.timeout
            rec["elapsed"] = float(args.timeout)

        rec["contended"] = contended
        with lock:
            runs.append(rec)
            spent = sum(cost_of(x["prompt_tokens"], x["completion_tokens"])
                        for x in runs)
            k = len(runs)
            if k % 20 == 0 or k == n:
                print("  [%3d/%3d] 累计 ¥%.3f  用时 %.0fs"
                      % (k, n, spent, time.time() - t_start), flush=True)
            # ★ 每次都落盘：中途中断时已花的钱不白花
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(runs, fh, ensure_ascii=False, indent=2)
        return rec

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(_one, jobs))
    else:
        for job in jobs:
            _one(job)

    total = sum(cost_of(x["prompt_tokens"], x["completion_tokens"]) for x in runs)
    fails = sum(1 for x in runs if x["failed"])
    print()
    print("完成 %d 次（跑不起来 %d 次），用时 %.0fs，实际花费 ¥%.3f"
          % (len(runs), fails, time.time() - t_start, total))
    print("原始记录：%s" % args.out)


if __name__ == "__main__":
    main()
