"""示例的判据 —— 这个文件此前**完全不存在**。

════════════════════════════════════════════════════════════════
 ★ 为什么示例需要被测试
════════════════════════════════════════════════════════════════

实测（2026-08-18）：clone 下来直接跑，**八个离线示例无一能跑通**。

    六个   ModuleNotFoundError: No module named 'bingfu'
    quickstart  ImportError: cannot import name 'TacticsEngine'
                AttributeError: 'SunTzuAgent' has no attribute 'analyze_battlefield'
    tool_usage  AttributeError: 'Agent' has no attribute 'clear_tools'
    famous_generals  AttributeError: 'BingFu' has no attribute 'list_agents'

★ 后三条是同一种病：**示例写的是旧 API**。
  `TacticsEngine → TacticEngine`、`analyze() → select_tactic()` 这些改名
  发生过，而示例（和 tests/test_tactics.py）都停在改名之前。

★ 示例是框架的第一接触面。新人跑的第一条命令 —— `quickstart` ——
  直接 ImportError，他不会去读 README 找原因，他会直接关掉。

★ 这一组的做法是**真的把每个示例执行一遍**。
  「import 得进来」证明不了它能跑完：上面三个 AttributeError
  全都发生在运行到一半的时候。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_EXAMPLES = _REPO / "examples"

# ★ 需要图形界面 / 真实模型 / 网络的示例不在这里跑 ——
#   它们的失败会指向环境，而不是代码，那种红灯只会训练人忽略它。
#   但**必须点名**：默默排除等于假装它们被覆盖了。
_NEEDS_LLM = {"llm_usage.py", "langchain_usage.py"}
_NEEDS_GUI = {"console_demo.py"}
_EXCLUDED = _NEEDS_LLM | _NEEDS_GUI


def _offline_examples() -> list[str]:
    return sorted(
        f.name
        for f in _EXAMPLES.glob("*.py")
        if not f.name.startswith("_") and f.name not in _EXCLUDED
    )


def test_the_exclusion_list_matches_reality() -> None:
    """★ 排除名单必须指向真实存在的文件。

    一个拼错的排除项不会报错，只会**永远不生效** ——
    那比没有排除更糟：名单看起来覆盖了，实际没有。
    """

    for name in _EXCLUDED:
        assert (_EXAMPLES / name).is_file(), f"排除名单里的 {name} 并不存在"


@pytest.mark.parametrize("example", _offline_examples())
def test_example_runs_to_completion(example: str) -> None:
    """★ 主判据：每个离线示例都必须**跑完并退出码 0**。

    起子进程而不是 import —— 示例是给人 `python examples/xxx.py` 跑的，
    那就按它被使用的方式验证。进程内 import 也重现不了
    「包根不在 sys.path 上」这个最常见的失败。
    """

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        [sys.executable, str(_EXAMPLES / example)],
        cwd=_EXAMPLES,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        stdin=subprocess.DEVNULL,
        env=env,
    )

    assert proc.returncode == 0, (
        f"`python examples/{example}` 退出码 {proc.returncode}\n"
        f"--- stdout 尾部 ---\n{proc.stdout[-1500:]}\n"
        f"--- stderr 尾部 ---\n{proc.stderr[-1500:]}"
    )


def test_every_example_can_find_the_package(tmp_path: Path) -> None:
    """★ 从**任意工作目录**运行都必须能 import 到 bingfu。

    此前六个示例是 ModuleNotFoundError：脚本所在目录会被自动加进
    sys.path，但**包根不会**，而这个包也没被安装
    （egg-info 是旧构建残留，不代表装过）。

    这里刻意 `cwd=tmp_path` —— 一个与项目无关的目录，
    正是别人 clone 下来之后的处境。
    """

    for example in _offline_examples():
        proc = subprocess.run(
            [sys.executable, "-c", f"import runpy; runpy.run_path(r'{_EXAMPLES / example}')"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        assert "No module named 'bingfu'" not in proc.stderr, (
            f"{example} 在别的目录下找不到包 —— bootstrap 没生效"
        )
