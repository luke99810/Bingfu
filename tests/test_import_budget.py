# -*- coding: utf-8 -*-
r"""`import bingfu` 必须在预算内完成。

════════════════════════════════════════════════════════════════
 为什么要有这条
════════════════════════════════════════════════════════════════

`import bingfu` 曾经要 **35.5 秒**，而四个对手是 1.1–6.1 秒。
`python -X importtime` 把 32.6 秒里的 31.9 秒归到了一条
**没有任何执行路径会走**的 LangChain 集成上（其中 26 秒是
transformers 与 sentence_transformers）。改成惰性导入后降到 0.7 秒，
跨框架基准的中位耗时从 41.8 秒降到 19.4 秒，整套测试从约 480 秒降到 74 秒。

★ 它潜伏了很久没人发现，**唯一的原因是没有任何东西在量它**。
  这类退化不会报错、不会失败、不会出现在任何断言里 ——
  它只是让每一次启动都慢一点，而「慢一点」没有阈值就永远不算问题。

★ 所以这条测试的价值不在于它现在通过，而在于
  **下一次有人往 `__init__.py` 里加重型依赖时它会立刻变红**。

════════════════════════════════════════════════════════════════
 阈值怎么定
════════════════════════════════════════════════════════════════

实测 0.7 秒。阈值取 6 秒 —— 留了将近十倍的余量，
因为这条测试要防的是**数量级的退化**（0.7 → 32），不是抖动。
定得太紧会让它在慢一点的机器上偶发变红，而一条会误报的测试
很快就会被加上 skip，然后就等于不存在了。
"""

import os
import subprocess
import sys
import time

import pytest

#: 秒。见模块文档：防的是数量级退化，不是抖动。
IMPORT_BUDGET_SECONDS = 6.0

#: 这些包一旦在 `import bingfu` 之后出现在 sys.modules 里，
#: 就说明又有人把重型依赖拉回了启动路径。
HEAVY = ("torch", "transformers", "sentence_transformers", "langchain",
         "langchain_community", "faiss", "numpy.f2py")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fresh_python(code: str):
    """在**全新解释器**里跑一段代码。

    ★ 必须新开进程：本测试进程早就 import 过 bingfu 了，
      在这里计时只会量到一次缓存命中 —— 那是个恒为真的断言，
      也就是一条永远不会变红的测试。
    """

    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          env=env, timeout=120)
    return proc, time.time() - t0


def test_import_bingfu_stays_within_budget():
    proc, elapsed = _fresh_python("import bingfu")
    assert proc.returncode == 0, "import bingfu 直接失败了：%s" % proc.stderr[-400:]
    assert elapsed < IMPORT_BUDGET_SECONDS, (
        "import bingfu 用了 %.1f 秒，预算是 %.1f 秒。\n"
        "多半是有重型依赖被加回了启动路径 —— 用\n"
        "    python -X importtime -c \"import bingfu\"\n"
        "看是谁，然后改成惰性导入（见 bingfu/__init__.py 的 __getattr__）。"
        % (elapsed, IMPORT_BUDGET_SECONDS))


def test_no_heavy_dependency_is_pulled_in_at_import():
    """★ 比计时更稳的一条：直接看谁被拉进来了。

    计时会受机器状态影响，而「transformers 在不在 sys.modules 里」
    是确定的。两条一起：一条说明代价，一条指出责任人。
    """

    proc, _ = _fresh_python(
        "import bingfu, sys, json;"
        "print(json.dumps([m for m in sys.modules if m in %r]))" % (HEAVY,))
    assert proc.returncode == 0, proc.stderr[-400:]
    import json

    loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    assert loaded == [], (
        "这些重型包在 import bingfu 时就被拉进来了：%s\n"
        "它们应当在真正用到时才导入。" % "、".join(loaded))


def test_lazy_attributes_point_at_modules_that_exist():
    """★ 这条是上面两条的对照 —— 省下的启动时间不能是靠删功能换的。

    惰性导入的失败方式是「访问时才发现根本导不进来」：
    模块改名或挪走之后，`_LAZY_ATTRS` 里的路径就成了一句谎话，
    而启动阶段永远不会发现，因为它压根不去解析。

    ★ 这里用 `find_spec` **只解析、不执行**。
      真去 `getattr` 一次会把 LangChain 整套拉起来 —— 实测 31.2 秒，
      正好把这个模块要防的那笔开销，原样加回到每一次跑测上。
      <b>一条为了证明启动快而让测试变慢 31 秒的测试，是自相矛盾的。</b>
    """

    proc, _ = _fresh_python(
        "import bingfu, importlib.util as u, json;"
        "m = getattr(bingfu, '_LAZY_ATTRS', {});"
        "print(json.dumps({'n': len(m),"
        " 'missing': sorted(k for k, v in m.items()"
        "                   if u.find_spec(v) is None)}))")
    assert proc.returncode == 0, proc.stderr[-400:]
    import json

    info = json.loads(proc.stdout.strip().splitlines()[-1])
    assert info["n"] > 0, "_LAZY_ATTRS 空了 —— 惰性导入的登记表没了"
    assert info["missing"] == [], (
        "这些惰性属性登记的模块根本不存在：%s\n"
        "启动阶段发现不了，只有用到时才会炸。" % "、".join(info["missing"]))
