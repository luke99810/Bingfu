r"""工作区工具（兵器谱 · 器械）——  让将领真的能动这台电脑。

════════════════════════════════════════════════════════════════
 为什么 execute_python 不够
════════════════════════════════════════════════════════════════

``execute_python`` 理论上无所不能：读文件、列目录、调命令，写段 Python
都能做。实测也确实做得到（数了一个目录下的 .md 文件，结果正确）。

但把它当成唯一的手，有三处代价：

  · **每一次基本操作都要模型现写一段正确的代码。**
    读一个文件要写 open + encoding + 异常处理，写错一次就浪费一轮。
    而 ReAct 的轮数是有限的 —— 轮数花在样板代码上，就不在任务上。

  · **执行轨迹看不出它在干什么。**
    监控进度时，一串 ``execute_python(<40 行代码>)`` 和
    ``read_file("report/draft.md")`` 提供的信息量差着量级。
    前者要人去读代码才知道这一步做了什么。

  · **无法施加边界。**
    一段任意 Python 可以写到任何路径。而一个 ``write_file`` 工具
    可以被限制在工作区内 —— 边界写在工具里，不指望模型自觉。

所以这里把最常用的几件事显式化。``execute_python`` 仍然保留，
它负责这几件事覆盖不到的部分。

════════════════════════════════════════════════════════════════
 工作区边界
════════════════════════════════════════════════════════════════

★ 所有路径操作都限制在**工作区根目录**内。

  一个能写任意路径的 Agent，可以覆盖掉框架自己的源码 —— 而它
  多半不是恶意的，只是把相对路径算错了。这类错误的后果与意图
  不成比例，所以用结构挡住，而不是靠提示词叮嘱。

★ ``run_command`` 是唯一的例外，它执行的是任意命令。

  这一点必须写出来而不是含糊带过：命令的能力边界就是当前用户的
  权限边界。它默认关闭（``enable_shell=False``），要用得显式打开。

════════════════════════════════════════════════════════════════
 工具的错误必须是**返回值**，不是异常
════════════════════════════════════════════════════════════════

★ ReAct 循环把工具输出喂回模型。抛异常会中断循环，模型没有机会
  纠正；返回一句「文件不存在，当前目录下有这些文件：…」，
  它下一轮就能改对。

  这是工具层最容易写错的地方：按普通库的习惯抛异常，
  在 Agent 语境下等于让它失去一次自我修正的机会。
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

#: 单个文件读取上限。超出截断并标注 —— 而不是静默截断。
#:
#: ★ 静默截断最坏：模型基于半个文件下结论，而没有任何迹象表明
#:   它只看到了一半。
MAX_READ_CHARS = 20000

#: 目录列举上限。一个几万文件的目录会把上下文吃光。
MAX_ENTRIES = 300

#: 命令输出上限与默认超时
MAX_OUTPUT_CHARS = 4000
DEFAULT_CMD_TIMEOUT = 60.0


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


@dataclass
class Workspace:
    """一个受限的工作目录，以及在其中操作的工具。

    ★ 与 ToolBelt 一样按任务新建，但**根目录通常跨任务不变** ——
      工作区是「在哪干活」，不是「这次干了什么」。
    """

    root: Path
    #: 是否允许执行任意命令。默认关闭。
    enable_shell: bool = False
    #: 命令超时
    cmd_timeout: float = DEFAULT_CMD_TIMEOUT
    #: 每个工具被调用了几次 —— 监控进度时要靠它判断将领是否真的动了手
    call_counts: Dict[str, int] = field(default_factory=dict)
    #: 按顺序记录做过什么，供「监控进度」展示
    trace: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ── 内部 ────────────────────────────────────────────

    def _count(self, name: str, detail: str) -> None:
        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        self.trace.append(f"{name}({detail})")

    def _resolve(self, path: str) -> tuple[Optional[Path], str]:
        """把用户/模型给的路径解析到工作区内。

        返回 (路径, 错误说明)。越界或非法时路径为 None。

        ★ 用 resolve() 之后再比对前缀，而不是简单看字符串开头 ——
          ``../`` 与符号链接都能让一个「看起来在工作区内」的字符串
          指到外面去。
        """

        raw = (path or "").strip().strip('"').strip("'")
        if not raw:
            return None, "路径为空。请给出相对于工作区的路径，例如 report/draft.md"

        p = Path(raw)
        if not p.is_absolute():
            p = self.root / p
        try:
            p = p.expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            return None, f"路径无法解析：{raw}（{exc}）"

        try:
            p.relative_to(self.root)
        except ValueError:
            return None, (
                f"路径 {raw} 在工作区之外。\n"
                f"工作区是：{self.root}\n"
                f"只能操作这个目录里的文件 —— 这是硬边界，"
                f"不是可以商量的限制。")
        return p, ""

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root)).replace("\\", "/")
        except ValueError:
            return str(p)

    # ── 工具实现 ────────────────────────────────────────

    def list_dir(self, path: str = ".") -> str:
        """列出目录内容。"""

        p, err = self._resolve(path)
        self._count("list_dir", path)
        if err:
            return f"[错误] {err}"
        if not p.exists():
            return f"[错误] 目录不存在：{self._rel(p)}"
        if not p.is_dir():
            return f"[错误] 这不是目录，是文件：{self._rel(p)}"

        try:
            entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except OSError as exc:
            return f"[错误] 无法读取目录 {self._rel(p)}：{exc}"

        lines = [f"{self._rel(p) or '.'} 下共 {len(entries)} 项："]
        for e in entries[:MAX_ENTRIES]:
            if e.is_dir():
                lines.append(f"  {e.name}/")
            else:
                try:
                    lines.append(f"  {e.name}  {_fmt_size(e.stat().st_size)}")
                except OSError:
                    lines.append(f"  {e.name}")
        if len(entries) > MAX_ENTRIES:
            lines.append(f"  …还有 {len(entries) - MAX_ENTRIES} 项未列出")
        return "\n".join(lines)

    def read_file(self, path: str) -> str:
        """读取文本文件。"""

        p, err = self._resolve(path)
        self._count("read_file", path)
        if err:
            return f"[错误] {err}"
        if not p.exists():
            # ★ 不是只说「不存在」——把同目录下有什么一并给出，
            #   模型下一轮就能改对，而不是反复试。
            parent = p.parent
            hint = ""
            if parent.is_dir():
                try:
                    names = [x.name for x in sorted(parent.iterdir())][:20]
                    if names:
                        hint = f"\n{self._rel(parent) or '.'} 下有：{'、'.join(names)}"
                except OSError:
                    pass
            return f"[错误] 文件不存在：{self._rel(p)}{hint}"
        if p.is_dir():
            return f"[错误] 这是目录不是文件：{self._rel(p)}。用 list_dir 看它的内容。"

        for enc in ("utf-8", "utf-8-sig", "gbk"):
            try:
                text = p.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
            except OSError as exc:
                return f"[错误] 读取失败：{exc}"
        else:
            return (f"[错误] {self._rel(p)} 不是文本文件"
                    f"（utf-8 / gbk 都解不开）。它可能是二进制。")

        if len(text) > MAX_READ_CHARS:
            return (text[:MAX_READ_CHARS]
                    + f"\n\n…[已截断，全文 {len(text)} 字符，"
                      f"这里只给了前 {MAX_READ_CHARS} 字符]")
        return text

    def write_file(self, path: str, content: str) -> str:
        """写入文本文件（覆盖）。父目录自动创建。"""

        p, err = self._resolve(path)
        self._count("write_file", path)
        if err:
            return f"[错误] {err}"
        if p.is_dir():
            return f"[错误] {self._rel(p)} 是目录，不能当文件写。"

        existed = p.exists()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"[错误] 写入失败：{exc}"

        verb = "覆盖" if existed else "新建"
        return (f"已{verb} {self._rel(p)}"
                f"（{len(content)} 字符，{_fmt_size(p.stat().st_size)}）")

    def append_file(self, path: str, content: str) -> str:
        """追加写入文本文件。"""

        p, err = self._resolve(path)
        self._count("append_file", path)
        if err:
            return f"[错误] {err}"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            return f"[错误] 追加失败：{exc}"
        return f"已追加 {len(content)} 字符到 {self._rel(p)}"

    def find_files(self, pattern: str = "*", path: str = ".") -> str:
        """按通配符递归查找文件。"""

        p, err = self._resolve(path)
        self._count("find_files", f"{pattern} in {path}")
        if err:
            return f"[错误] {err}"
        if not p.is_dir():
            return f"[错误] 不是目录：{self._rel(p)}"

        pat = (pattern or "*").strip() or "*"
        try:
            hits = sorted(p.rglob(pat))
        except (OSError, ValueError) as exc:
            return f"[错误] 查找失败：{exc}"

        files = [h for h in hits if h.is_file()]
        if not files:
            return f"没有匹配 {pat} 的文件（在 {self._rel(p) or '.'} 下递归查找）"

        lines = [f"匹配 {pat} 的文件共 {len(files)} 个："]
        for f in files[:MAX_ENTRIES]:
            lines.append(f"  {self._rel(f)}")
        if len(files) > MAX_ENTRIES:
            lines.append(f"  …还有 {len(files) - MAX_ENTRIES} 个未列出")
        return "\n".join(lines)

    def run_command(self, command: str) -> str:
        """在工作区内执行一条命令。

        ★ 默认关闭。它的能力边界就是当前用户的权限边界 ——
          没有沙箱，写清楚比含糊带过重要。
        """

        self._count("run_command", command[:60])
        if not self.enable_shell:
            return ("[错误] 命令执行未启用。\n"
                    "这一项默认关闭，因为它的能力边界就是当前用户的权限边界。\n"
                    "如果这个任务确实需要，请让使用者在启动时开启 enable_shell。\n"
                    "多数文件操作用 list_dir / read_file / write_file / find_files 即可。")

        cmd = (command or "").strip()
        if not cmd:
            return "[错误] 命令为空"

        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=str(self.root),
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=self.cmd_timeout,
            )
        except subprocess.TimeoutExpired:
            return (f"[错误] 命令超过 {self.cmd_timeout:.0f} 秒未结束，已终止。\n"
                    f"如果它本来就耗时很长，把它拆成能快速返回的几步。")
        except OSError as exc:
            return f"[错误] 无法执行：{exc}"

        out = (proc.stdout or "")[:MAX_OUTPUT_CHARS]
        errtxt = (proc.stderr or "")[:MAX_OUTPUT_CHARS]
        parts = [f"退出码 {proc.returncode}"]
        if out.strip():
            parts.append(f"--- 标准输出 ---\n{out}")
        if errtxt.strip():
            parts.append(f"--- 标准错误 ---\n{errtxt}")
        if not out.strip() and not errtxt.strip():
            parts.append("(没有任何输出)")
        return "\n".join(parts)

    # ── 装配 ────────────────────────────────────────────

    def as_functions(self) -> Dict[str, Callable]:
        """交给 Agent 注册的工具函数表。

        ★ shell 关闭时**仍然注册** run_command。

          不注册的话，模型不知道有这个能力，会绕道用
          execute_python 去做同样的事 —— 而那条路没有边界。
          注册但返回一句「未启用，请让使用者开启」，
          它才知道该停下来说明，而不是自己想办法绕过去。
        """

        return {
            "list_dir": self.list_dir,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "append_file": self.append_file,
            "find_files": self.find_files,
            "run_command": self.run_command,
        }

    def descriptions(self) -> Dict[str, str]:
        return {
            "list_dir": "列出工作区内某个目录的内容。参数：path（相对路径，默认当前目录）",
            "read_file": "读取工作区内的文本文件。参数：path",
            "write_file": "写入文本文件（覆盖，父目录自动创建）。参数：path, content",
            "append_file": "在文本文件末尾追加内容。参数：path, content",
            "find_files": "按通配符递归查找文件。参数：pattern（如 *.md）, path",
            "run_command": (
                "在工作区内执行一条命令并返回输出。参数：command。"
                + ("" if self.enable_shell else "（当前未启用）")),
        }

    # ── 进度 ────────────────────────────────────────────

    def did_anything(self) -> bool:
        """将领是否真的动过手。

        ★ 「产出了一段文本」与「做了事」必须分开。

          一次执行如果一个工具都没调用，那它写出来的东西全部来自
          模型的既有知识 —— 那可能仍然有用，但它不是「完成了任务」。
        """

        return any(v > 0 for v in self.call_counts.values())

    def summary(self) -> str:
        if not self.call_counts:
            return "未调用任何工具"
        items = sorted(self.call_counts.items(), key=lambda kv: -kv[1])
        return "、".join(f"{k}×{v}" for k, v in items)


def workspace_at(root: str, *, enable_shell: bool = False) -> Workspace:
    """便捷构造。"""

    return Workspace(root=Path(root), enable_shell=enable_shell)
