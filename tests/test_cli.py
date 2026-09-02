"""CLI 的判据 —— 这个文件此前**完全不存在**。

════════════════════════════════════════════════════════════════
 ★ 为什么单独一组测试盯 CLI
════════════════════════════════════════════════════════════════

pyproject.toml 把 `bingfu = "bingfu.cli:main"` 作为产品的正式入口对外声明，
而实测下来**一条命令都跑不通**：NameError、两处 UnicodeEncodeError
（连 `--help` 都打不出来）、还有一个键盘打不出来的子命令名。

三个缺陷都不需要复杂场景 —— 跑一次就现形。
它们能活下来的唯一原因是：**没有任何一条测试碰过 cli.py**。

所以这一组的原则是：**每条命令都真的执行一遍**，
而不是断言 parser 里注册了哪些名字。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from bingfu.cli import create_parser, main

_REPO = Path(__file__).resolve().parents[1]
_GUIDE = _REPO / "examples" / "cli_guide.py"


def _subcommands() -> list[str]:
    """从 parser 里取出所有已注册的子命令名。"""
    for action in create_parser()._actions:
        if getattr(action, "choices", None) and hasattr(action, "_name_parser_map"):
            return sorted(action.choices)
    raise AssertionError("parser 里找不到子命令 —— CLI 结构已变，这条测试要重写")


# ══════════════════════════════════════════════════════════════
#  ★ 子命令名必须是能敲出来的
# ══════════════════════════════════════════════════════════════


def test_subcommand_names_are_pure_ascii() -> None:
    """★ 这条抓的是最阴的那个缺陷。

    子命令曾写成 `add‑agent`，中间是 U+2011（非断行连字符）——
    **键盘打不出来的字符**。于是这条命令永远无法被调用，
    而 `--help` 里显示的又和正常的一模一样：
    看得见、打不出、报 invalid choice。

    断言字符集，而不是断言某个具体名字 —— 换了名字这条依然守得住。
    """

    for name in _subcommands():
        assert name.isascii(), f"子命令 {name!r} 含非 ASCII 字符，用户敲不出来"
        assert re.fullmatch(r"[a-z][a-z0-9-]*", name), f"子命令 {name!r} 不是常规 CLI 命名"


def test_no_lookalike_hyphen_anywhere_in_the_module() -> None:
    """★ 连同源码里的其它位置一起守：曾有 6 处 U+2011。"""

    source = (_REPO / "bingfu" / "cli.py").read_text(encoding="utf-8")
    for bad, label in (("‐", "U+2010"), ("‑", "U+2011"), ("‒", "U+2012")):
        assert bad not in source, f"cli.py 里出现了 {label}（看起来像连字符，其实不是）"


# ══════════════════════════════════════════════════════════════
#  ★ 每条命令都真的跑一遍
# ══════════════════════════════════════════════════════════════


def test_every_registered_command_actually_runs(tmp_path: Path, monkeypatch) -> None:
    """★ 这条是整组的主判据。

    它遍历**parser 里实际注册的每一条命令**并真的执行 ——
    新增一条命令却写坏了，这里立刻变红，不需要有人记得补测试。

    此前 version / status / --help 三条都会抛异常，
    而没有任何东西会因此报警。
    """

    monkeypatch.chdir(tmp_path)
    config = str(tmp_path / "bingfu.yaml")

    argv_for = {
        "version": ["version"],
        "status": ["--config", config, "status"],
        "add-agent": ["--config", config, "add-agent", "韩信", "--role", "大将军"],
        "drum": ["--config", config, "drum", "韩信", "分析市场"],
        "drum-all": ["--config", config, "drum-all", "全军推进"],
        "gong": ["--config", config, "gong", "韩信"],
        "gong-all": ["--config", config, "gong-all"],
        "load-config": ["--config", config, "load-config", config],
    }

    registered = set(_subcommands())
    assert registered == set(argv_for), (
        f"注册的命令与本测试覆盖的不一致：只在 CLI 里={registered - set(argv_for)}，"
        f"只在测试里={set(argv_for) - registered}"
    )

    # add-agent 必须先跑，后面几条要用到它写下的将领
    order = ["version", "add-agent", "status", "drum", "drum-all", "gong", "gong-all", "load-config"]
    for name in order:
        code = main(argv_for[name])
        assert code == 0, f"命令 `{name}` 退出码 {code}"


def test_help_does_not_crash() -> None:
    """★ `--help` 曾经也崩 —— 一个连帮助都打不出来的 CLI。"""

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


# ══════════════════════════════════════════════════════════════
#  ★ GBK 控制台 —— 只有子进程才能重现
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("command", [["version"], ["--help"], ["status"]])
def test_output_survives_a_legacy_console(command: list[str], tmp_path: Path) -> None:
    """★ 这条必须起**子进程**并强制 GBK，进程内测试重现不了。

    pytest 会把 stdout 换成自己的捕获对象（UTF-8），于是
    `bingfu status` 在测试里一切正常、在用户的 cmd 里直接
    UnicodeEncodeError —— 那正是这个缺陷活下来的方式。

    ★ 崩溃点在**命令做完事情之后**：退出码从 0 变成 1，
      现象与「功能失败」完全无法区分。
    """

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "gbk"
    env.pop("PYTHONUTF8", None)
    env["PYTHONPATH"] = str(_REPO)

    proc = subprocess.run(
        [sys.executable, "-m", "bingfu.cli", *command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )

    assert "UnicodeEncodeError" not in proc.stderr, (
        f"GBK 控制台上 `bingfu {' '.join(command)}` 崩了：\n{proc.stderr[-800:]}"
    )
    assert proc.returncode == 0, f"退出码 {proc.returncode}\n{proc.stderr[-800:]}"


# ══════════════════════════════════════════════════════════════
#  ★ 退出码必须能区分成败
# ══════════════════════════════════════════════════════════════


def test_failure_returns_nonzero(tmp_path: Path) -> None:
    """★ 此前每个分支都走到函数末尾返回 None ——
    「军中无此将领」和「任务执行成功」在退出码上完全一样。

    对脚本来说，那等于这个 CLI 从不失败。
    """

    config = str(tmp_path / "bingfu.yaml")
    assert main(["--config", config, "drum", "查无此人", "做点事"]) == 1
    assert main(["--config", config, "gong", "查无此人"]) == 1
    assert main(["--config", config, "load-config", str(tmp_path / "根本没有.yaml")]) == 1


# ══════════════════════════════════════════════════════════════
#  ★ 名册必须真的落盘（否则 add-agent 是句空话）
# ══════════════════════════════════════════════════════════════


def test_add_agent_survives_process_restart(tmp_path: Path) -> None:
    """★ `add-agent` 曾经打印「✅ 已添加」而**什么都没发生**：
    agents 从不落盘，进程一退就没了。

    断言的是「新造一个 BingFu 还能看见它」，
    不是「函数返回了成功」—— 返回值是自述，文件里有没有是事实。
    """

    config = str(tmp_path / "bingfu.yaml")
    assert main(["--config", config, "add-agent", "白起", "--role", "武安君"]) == 0

    from bingfu.bingfu import BingFu

    reloaded = BingFu()
    reloaded.load_config(config)

    assert "白起" in reloaded.agents, "将领没有被持久化 —— add-agent 是句空话"
    assert reloaded.agents["白起"].role == "武安君", "写回来了，但字段丢了"


def test_broken_roster_entry_does_not_kill_the_whole_config(tmp_path: Path) -> None:
    """★ 一条手写错的记录不该让整份名册都载不进来。"""

    config = tmp_path / "bingfu.yaml"
    config.write_text(
        "agents:\n- name: 孙膑\n- role: 没有名字的坏条目\n- name: 韩信\n",
        encoding="utf-8",
    )

    from bingfu.bingfu import BingFu

    master = BingFu()
    master.load_config(str(config))

    assert set(master.agents) == {"孙膑", "韩信"}


# ══════════════════════════════════════════════════════════════
#  ★ 文档与实现不许漂移
# ══════════════════════════════════════════════════════════════


def test_guide_teaches_exactly_the_commands_that_exist() -> None:
    """★ `cli_guide.py` 曾教了 8 条命令，而 CLI 只实现 5 条 ——
    其中 drum-all / gong-all / load-config **根本不存在**，
    尽管底层的 BingFu.drum_all / gong_all / load_config 一直都在。

    读的是指南**源码里出现的命令**，不在这里再抄一份名单：
    抄第三份只会让漂移多一个去处。
    """

    assert _GUIDE.is_file(), f"指南不见了：{_GUIDE}"
    text = _GUIDE.read_text(encoding="utf-8")
    taught = set(re.findall(r"\$ bingfu ([a-z][a-z0-9-]*)", text))

    assert taught, "指南里找不到任何 `$ bingfu <命令>` 示例 —— 格式已变，这条测试要重写"
    missing = taught - set(_subcommands())
    assert not missing, f"指南教的命令并不存在：{sorted(missing)}"
