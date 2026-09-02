"""
CLI module (命令行界面模块)

════════════════════════════════════════════════════════════════
 ★ 这个文件此前**一条命令都跑不通**
════════════════════════════════════════════════════════════════

实测（2026-08-18，Windows + GBK 控制台）：

    bingfu version   → NameError: name 'master' is not defined
    bingfu status    → UnicodeEncodeError: 'gbk' codec can't encode '⚫'
    bingfu --help    → UnicodeEncodeError —— **连帮助都打不出来**

而 pyproject.toml 把 `bingfu = "bingfu.cli:main"` 作为产品的正式入口对外声明。

★ 它能坏成这样，唯一的原因是**没有任何一条测试碰过 cli.py**。
  三个缺陷都不需要复杂场景，跑一次就现形。
  现在 tests/test_cli.py 会把每条命令真的执行一遍。

★ 子命令名曾经用的是 U+2011（非断行连字符）而不是 ASCII 的 `-`。
  那个字符键盘打不出来，于是 argparse 注册的那条命令**永远无法被调用**，
  而 `--help` 里显示的又长得和正常的一模一样 ——
  看得见、打不出、报「invalid choice」。全文件曾有 6 处。
"""

import argparse
import os
import sys
from typing import Any, Optional

from bingfu import __version__
from bingfu.agent import Agent
from bingfu.bingfu import BingFu

DEFAULT_CONFIG = "bingfu.yaml"
"""默认配置文件 —— `add-agent` 的落点，也是其它命令读将领的地方。

★ 没有它的时候，`bingfu add-agent 韩信` 打印「✅ 已添加」，
  进程一退什么都没留下，下一条 `bingfu drum 韩信 ...` 报「军中无此将领」。
  一条**报告成功而实际什么都没发生**的命令，比直接报错更难查。
"""


def _use_utf8_output() -> None:
    """让中文与 emoji 在 GBK 控制台上也能打出来。

    ★ 不做这件事的后果不是乱码，是**崩溃**：print 直接抛 UnicodeEncodeError，
      命令在真正做完事情**之后**才挂掉，退出码从 0 变成 1 ——
      现象与「功能失败」无法区分。
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def create_parser() -> argparse.ArgumentParser:
    """构造参数解析器。

    ★ 这里的子命令集合与 `examples/cli_guide.py` 教的**必须一致**。
      此前指南教了 8 条、实现只有 5 条，其中 3 条（drum-all / gong-all /
      load-config）根本不存在 —— 而底层的 BingFu.drum_all / gong_all /
      load_config 三个方法一直都在，只是没人把它们接到 CLI 上。
      由 tests/test_cli.py 的一条测试盯着两边不再漂移。
    """

    parser = argparse.ArgumentParser(
        prog="bingfu",
        description="BingFu (兵符) — 轻量级多智能体框架",
        epilog="示例: bingfu add-agent 韩信 --role 大将军 然后 bingfu drum 韩信 分析市场",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"配置文件路径（默认 {DEFAULT_CONFIG}），将领名册与 LLM 配置都存在这里",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    drum_parser = subparsers.add_parser("drum", help="击鼓 —— 令某位将领执行任务")
    drum_parser.add_argument("agent", help="将领名号")
    drum_parser.add_argument("task", help="任务描述")

    drum_all_parser = subparsers.add_parser("drum-all", help="击鼓 —— 令全军执行同一任务")
    drum_all_parser.add_argument("task", help="任务描述")

    gong_parser = subparsers.add_parser("gong", help="鸣金 —— 令某位将领收兵")
    gong_parser.add_argument("agent", help="将领名号")

    subparsers.add_parser("gong-all", help="鸣金 —— 全军收兵")

    add_parser = subparsers.add_parser("add-agent", help="点将 —— 新增一位将领并写入配置")
    add_parser.add_argument("name", help="将领名号")
    add_parser.add_argument("--role", default=None, help="职位（可选）")
    add_parser.add_argument("--description", default=None, help="描述（可选）")

    load_parser = subparsers.add_parser("load-config", help="载入配置文件")
    load_parser.add_argument("file", help="YAML 配置文件路径")

    subparsers.add_parser("status", help="查看兵符状态")
    subparsers.add_parser("version", help="查看版本")

    return parser


def _load(master: BingFu, path: str, *, required: bool) -> Optional[str]:
    """载入配置。返回 None 表示成功，返回字符串表示失败原因。

    ★ `required=False` 时「文件不存在」是正常状态（还没 add-agent 过），
      不是错误。把两者混成一个布尔，第一次使用的人会看到一条莫名的报错。
    """

    if not os.path.isfile(path):
        return f"配置文件不存在：{path}" if required else None
    try:
        master.load_config(path)
    except Exception as exc:  # noqa: BLE001 - 配置文件的错法太多，如实回传
        return f"载入配置失败（{path}）：{exc}"
    return None


def handle_drum(args: argparse.Namespace, master: BingFu) -> tuple[int, str]:
    if master.get_agent(args.agent) is None:
        known = "、".join(master.agents) or "（一位都没有）"
        return 1, (
            f"✗ 军中无此将领：{args.agent}\n"
            f"  现有将领：{known}\n"
            f"  先执行：bingfu add-agent {args.agent}"
        )
    return 0, master.drum(args.agent, args.task)


def handle_drum_all(args: argparse.Namespace, master: BingFu) -> tuple[int, str]:
    if not master.agents:
        return 1, "✗ 军中无将领，无法击鼓。先执行：bingfu add-agent <名号>"
    results = master.drum_all(args.task)
    return 0, "\n".join(f"【{name}】{text}" for name, text in results.items())


def handle_gong(args: argparse.Namespace, master: BingFu) -> tuple[int, str]:
    if master.get_agent(args.agent) is None:
        return 1, f"✗ 军中无此将领：{args.agent}"
    return 0, master.gong(args.agent)


def handle_gong_all(master: BingFu) -> tuple[int, str]:
    results = master.gong_all()
    if not results:
        return 0, "军中无将领，无需鸣金。"
    return 0, "\n".join(f"【{name}】{text}" for name, text in results.items())


def handle_add_agent(args: argparse.Namespace, master: BingFu, config_path: str) -> tuple[int, str]:
    """点将并**写入配置** —— 这条命令必须留下痕迹。"""

    agent = Agent(name=args.name, role=args.role, description=args.description)
    master.add_agent(agent)
    try:
        master.save_config(config_path)
    except OSError as exc:
        return 1, f"✗ 将领 '{args.name}' 未能写入 {config_path}：{exc}"
    return 0, f"✅ 将领 '{args.name}' 已入册，写入 {config_path}"


def handle_load_config(args: argparse.Namespace, master: BingFu) -> tuple[int, str]:
    error = _load(master, args.file, required=True)
    if error:
        return 1, f"✗ {error}"
    status = master.status()
    return 0, f"✅ 已载入 {args.file}：将领 {status['agent_count']} 位、兵器 {status['tool_count']} 件"


def handle_status(master: BingFu, config_path: str) -> tuple[int, str]:
    status = master.status()
    roster = f"（{'、'.join(master.agents)}）" if master.agents else ""
    lines = [
        "BingFu 兵符状态",
        "─" * 40,
        f"名称    : {status['name']}",
        f"版本    : {status['version']}",
        f"配置    : {config_path}",
        f"将领    : {status['agent_count']} 位{roster}",
        f"兵器    : {status['tool_count']} 件",
        f"军需库  : {status['memory_count']} 处",
        f"主帅    : {'已启用' if status['commander_enabled'] else '未启用'}",
    ]
    return 0, "\n".join(lines)


def handle_version() -> tuple[int, str]:
    """★ 此前这里引用了一个不存在的 `master`，`bingfu version` 必然 NameError。

    版本改从 `bingfu.__version__` 取 —— 那是包的单一来源。
    """

    return 0, f"BingFu (兵符) v{__version__}"


def main(argv: Optional[list[str]] = None) -> int:
    """CLI 主入口。**返回退出码**，不再靠异常决定成败。

    ★ 退出码是这条链路唯一能被脚本判读的信号：0 = 做成了，1 = 没做成。
      此前所有分支都走到函数末尾返回 None，于是「军中无此将领」
      和「任务执行成功」在退出码上完全一样。
    """

    _use_utf8_output()

    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # version 不需要读配置 —— 配置坏了也应该能问出版本号。
    if args.command == "version":
        code, result = handle_version()
        print(result)
        return code

    master = BingFu()
    # ★ 除 load-config 外，所有命令都先载入默认配置 ——
    #   否则 add-agent 写进去的将领，下一条命令看不见。
    if args.command != "load-config":
        error = _load(master, args.config, required=False)
        if error:
            print(f"✗ {error}", file=sys.stderr)
            return 1

    handlers: dict[str, Any] = {
        "drum": lambda: handle_drum(args, master),
        "drum-all": lambda: handle_drum_all(args, master),
        "gong": lambda: handle_gong(args, master),
        "gong-all": lambda: handle_gong_all(master),
        "add-agent": lambda: handle_add_agent(args, master, args.config),
        "load-config": lambda: handle_load_config(args, master),
        "status": lambda: handle_status(master, args.config),
    }

    handler = handlers.get(args.command)
    if handler is None:
        print(f"✗ 未知命令：{args.command}", file=sys.stderr)
        return 1

    code, result = handler()
    print(result, file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
