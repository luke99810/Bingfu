r"""代码执行工具（兵器谱 · 试锋）。

════════════════════════════════════════════════════════════════
 为什么这是最重要的一个工具
════════════════════════════════════════════════════════════════

基准里 Code 类只有 12–25%，直接原因是将领拿到的 ``tools=None``：
``has_tool_calls`` 恒为假，ReAct 循环**只跑一轮就返回**。

也就是说，模型被要求一次性写出带鉴权、异步数据库的服务，
**不执行、不看报错、不修改**，第一段回复就是最终答案。

人写代码不是这样的。给它一个能真的跑起来的解释器，
才谈得上"迭代"。这个工具的存在本身就是那 12% 的解药 ——
比任何提示词技巧都直接。

════════════════════════════════════════════════════════════════
 安全边界（必须先说清楚）
════════════════════════════════════════════════════════════════

★ 这个工具执行的是**模型生成的代码**，风险是真实的。

采取的隔离措施：

  · 独立子进程 —— 崩溃不影响宿主，且可强制终止
  · 硬超时 —— 默认 10 秒，到点 kill 掉整个进程树
  · 独立临时工作目录 —— 每次执行一个新目录，结束即删
  · 输出封顶 —— 防止无限打印撑爆内存
  · 禁用用户站点包与环境继承的部分变量

★ 没有采取的措施，以及为什么要写出来：

  · **没有做系统调用级沙箱**。子进程仍能读写文件系统、发起网络请求。
    真正的沙箱需要容器或 seccomp，那是另一个量级的工程。

  · 因此：**只在你自己的机器上、对你自己的基准任务使用**。
    不要拿它跑来路不明的代码，也不要暴露成服务。

  写出限制而不是含糊带过，是因为"有沙箱"这三个字会让人
  放心地做出它承受不起的事。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

#: 默认超时。基准里的任务都是小程序，10 秒足够；
#: 死循环则必须在这里被截断，否则一次实验会永远卡住。
DEFAULT_TIMEOUT = 10.0

#: 输出上限。超出部分截断并标注 —— 而不是静默丢弃。
MAX_OUTPUT_CHARS = 4000


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: Optional[int]
    timed_out: bool
    elapsed: float

    def as_observation(self) -> str:
        """拼成给模型看的观察结果。

        ★ 失败时必须把**真实报错**原样带回去。
          "执行失败"这四个字对模型毫无用处，
          ``NameError: name 'reqeusts' is not defined`` 才能指导修改。
        """

        if self.timed_out:
            return (
                f"[执行超时] 代码运行超过 {DEFAULT_TIMEOUT:.0f} 秒被终止。"
                f"通常意味着存在死循环、或在等待永远不会到来的输入。\n"
                f"超时前的输出：\n{self.stdout or '(无)'}"
            )
        head = "[执行成功]" if self.ok else f"[执行失败 退出码={self.exit_code}]"
        parts = [head]
        if self.stdout:
            parts.append(f"标准输出：\n{self.stdout}")
        if self.stderr:
            parts.append(f"错误输出：\n{self.stderr}")
        if not self.stdout and not self.stderr:
            parts.append("（无输出。若期望看到结果，请确认代码里有 print。）")
        return "\n".join(parts)


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return (
        text[:MAX_OUTPUT_CHARS]
        + f"\n...[输出被截断，共 {len(text)} 字符，仅显示前 {MAX_OUTPUT_CHARS}]"
    )


def run_python(code: str, timeout: float = DEFAULT_TIMEOUT) -> ExecResult:
    """在隔离的子进程里执行一段 Python，返回结构化结果。"""

    import time

    workdir = tempfile.mkdtemp(prefix="bingfu_exec_")
    script = os.path.join(workdir, "main.py")
    try:
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(code)

        env = dict(os.environ)
        # 不继承 API key —— 生成的代码没有理由需要它们，
        # 而一旦泄漏进日志或网络请求就收不回来了
        for key in list(env):
            if any(m in key.upper() for m in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
                env.pop(key, None)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", script],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            elapsed = time.time() - t0
            return ExecResult(
                ok=proc.returncode == 0,
                stdout=_truncate(proc.stdout or ""),
                stderr=_truncate(proc.stderr or ""),
                exit_code=proc.returncode,
                timed_out=False,
                elapsed=elapsed,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                ok=False,
                stdout=_truncate(exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")),
                stderr="",
                exit_code=None,
                timed_out=True,
                elapsed=time.time() - t0,
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ── 暴露给 Agent 的工具函数 ──────────────────────────────

def execute_python(code: str) -> str:
    """执行一段 Python 代码并返回运行结果（标准输出与报错）。

    用于验证自己写的代码能否真的跑通。发现报错后应当修改代码再试。
    代码在隔离的临时目录中运行，超过 10 秒会被终止。
    """

    if not code or not code.strip():
        return "[未执行] 传入的代码为空。"
    return run_python(code).as_observation()


def run_tests(code: str, tests: str) -> str:
    """把实现代码与测试代码拼在一起执行，返回测试结果。

    实现写在前、测试写在后。测试用 assert 断言即可，
    全部通过则无输出并以退出码 0 结束。
    """

    if not code or not code.strip():
        return "[未执行] 实现代码为空。"
    combined = code + "\n\n# ── 测试 ──\n" + (tests or "")
    result = run_python(combined)
    if result.ok and not result.stderr:
        return "[全部测试通过]\n" + (result.stdout or "(测试无输出，说明断言均成立)")
    return result.as_observation()
