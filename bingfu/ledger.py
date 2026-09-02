r"""调用账本 —— 冗余熔断与结果压缩。

════════════════════════════════════════════════════════════════
 要解决的是什么
════════════════════════════════════════════════════════════════

跨框架实测里兵符的冗余调用是 13–16 次，而 PydanticAI 是 0、LangGraph 是 2。
把日志摊开看，冗余**全部集中在会被拆解的 aggregate 任务**上：

    agg-merge   read(a.txt) read(b.txt) read(c.txt)  各被读了两遍
    agg-pick    write(facts.md) write(pick.md)       各被写了两遍

原因不是模型笨，是**结构性的**：拆解出的子任务各跑一套独立的 ReAct，
彼此不知道对方已经做过什么。第二个子任务没有任何途径得知
「a.txt 上一个子任务已经读过了」，于是它自己再读一遍。

════════════════════════════════════════════════════════════════
 三条对策，各自解决不同的一段
════════════════════════════════════════════════════════════════

★ 一、冗余熔断（本模块）

  同一次战役内，(工具, 参数) 完全相同的重复调用不再真的执行 ——
  直接返回上次的结果，并**明说这是上次的**。

  为什么要明说：不说的话模型会以为自己读了两遍都得到一样的东西，
  下一轮很可能再读第三遍。把「你已经读过了」写进观察结果，
  才是让它停下来的那个信号。

★ 二、结果压缩（本模块的 compress）

  长结果重复回灌是 token 的大头。熔断命中时只回一个简短摘要 +
  「内容与上次相同」的标记，而不是把两万字符再推一遍。

★ 三、微操与战略之分（在拆解提示词里，不在这里）

  真正的根因是拆解把「读三个文件」当成了可以分头做的两件事。
  一个子任务应当是一件**值得单独交付的事**，
  而不是一次文件读写。那一条写在 _DECOMPOSE_PROMPT 里。

════════════════════════════════════════════════════════════════
 熔断的边界：什么时候不能熔断
════════════════════════════════════════════════════════════════

★ 写操作只在**内容也相同**时才熔断。

  同一个文件用不同内容写第二次是覆盖，是合法且常见的动作
  （先写草稿再改）。把它熔断掉会静默丢掉真实的修改 ——
  那比多花几个 token 严重得多。

★ 读操作在内容可能已变时不能熔断。

  本模块只在**同一次战役**内熔断，并且一旦某个文件被写过，
  它的读缓存立即失效 —— 否则「写完再读回来确认」这个正当动作
  会拿到过期内容。
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

#: 结果超过这个长度时，熔断回灌只给摘要
COMPRESS_ABOVE = 400
#: 摘要保留的首尾字符数
COMPRESS_HEAD = 220
COMPRESS_TAIL = 80

#: 视为「写」的工具名前缀 —— 写会让同名文件的读缓存失效
WRITE_HINTS = ("write", "append", "save", "创建", "写")


def _digest(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()[:16]


def compress(text: str, *, above: int = COMPRESS_ABOVE) -> str:
    """把长结果压成首尾片段，并**标明中间省略了多少**。

    ★ 省略必须说出来。悄悄截断会让模型以为自己看到了全部内容，
      而它据此做的判断没有任何地方能看出是基于半份材料。
    """

    text = str(text)
    if len(text) <= above:
        return text
    omitted = len(text) - COMPRESS_HEAD - COMPRESS_TAIL
    return "%s\n…（此处省略 %d 字符）…\n%s" % (
        text[:COMPRESS_HEAD], omitted, text[-COMPRESS_TAIL:])


@dataclass
class CallLedger:
    """一次战役内的工具调用账本。跨子任务共享。"""

    #: (工具, 参数指纹) -> 上次的结果
    _results: Dict[Tuple[str, str], str] = field(default_factory=dict)
    #: 被写过的目标，用于让读缓存失效
    _dirty: set = field(default_factory=set)
    _lock: Any = field(default_factory=threading.Lock)

    #: 统计
    hits: int = 0
    saved_calls: int = 0
    #: 同一产出被**不同内容**写了多次的目标。
    #:
    #: ★ 这类不熔断 —— 覆盖写入是合法动作（先草稿再改）。
    #:   但它也可能是拆解重叠的症状：两个子任务在做同一件事，
    #:   后一个把前一个的成果盖掉了。实测熔断上线后剩余的冗余
    #:   **全部是这一类**。
    #:
    #: ★ 分不清合法覆盖与重复劳动，所以不拦，只**记下来让它可见** ——
    #:   隐形的重复劳动会一直存在，而看得见的至少能被讨论。
    overwrites: Dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _target(args: Any) -> str:
        """从参数里取出「操作对象」——通常是文件名。"""

        if isinstance(args, dict):
            for k in ("filename", "path", "file", "name"):
                if k in args:
                    return str(args[k])
        return ""

    @staticmethod
    def _is_write(tool: str) -> bool:
        low = str(tool).lower()
        return any(h in low for h in WRITE_HINTS)

    def check(self, tool: str, args: Any) -> Optional[str]:
        """命中熔断则返回要回灌的文本，否则返回 None。"""

        target = self._target(args)
        key = (tool, _digest(args))
        with self._lock:
            if self._is_write(tool):
                # 写：只有**参数完全相同**（含内容）才算重复
                if target and target in self._dirty and key not in self._results:
                    self.overwrites[target] = self.overwrites.get(target, 0) + 1
                if key in self._results:
                    self.hits += 1
                    self.saved_calls += 1
                    return ("[重复调用已跳过] %s 的这次写入与本次战役中先前的"
                            "一次内容完全相同，未重复执行。" % target or tool)
                return None

            # 读：目标被写过之后缓存失效
            if target and target in self._dirty:
                return None
            if key in self._results:
                self.hits += 1
                self.saved_calls += 1
                return ("[本次战役中已读过此内容，未重复读取]\n%s"
                        % compress(self._results[key]))
            return None

    def record(self, tool: str, args: Any, result: Any) -> None:
        key = (tool, _digest(args))
        target = self._target(args)
        with self._lock:
            self._results[key] = str(result)
            if self._is_write(tool) and target:
                self._dirty.add(target)

    @property
    def written(self) -> Tuple[str, ...]:
        """本次战役里**真正落盘**的产物名。

        ★ 验收要用它来判断「回复是产物本身，还是产物的回执」。
          只有非报错的写入才会进 record，所以这里天然不含失败的写。
        """

        with self._lock:
            return tuple(sorted(self._dirty))

    def stats(self) -> Dict[str, int]:
        return {"breaker_hits": self.hits, "saved_calls": self.saved_calls,
                "overwrites": sum(self.overwrites.values())}

    def overwrite_report(self) -> str:
        """把重复覆盖写成一句可以发给界面的话。没有则返回空串。"""

        if not self.overwrites:
            return ""
        items = sorted(self.overwrites.items(), key=lambda kv: -kv[1])
        return ("同一产出被反复覆盖：%s —— 多半是拆解把一件事分给了两个子任务"
                % "、".join("%s×%d" % (k, v + 1) for k, v in items))
