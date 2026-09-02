r"""断点 —— 让一次跑到一半的战役能接着跑。

════════════════════════════════════════════════════════════════
 这一层补的是什么
════════════════════════════════════════════════════════════════

``GraphOrchestrator`` 原本把节点产物放在一个内存字典里。进程一退，
那个字典就没了 —— 一次跑了十分钟的战役，在最后一个节点上崩掉，
只能整个从头再来，前面所有已经花掉的 token 一起作废。

更要紧的是：**没有断点就没有人工介入**。「停下来、改掉某个中间产物、
再从那儿继续」这件事的前提是中间产物能被取出来。

════════════════════════════════════════════════════════════════
 接口形状为什么长这样
════════════════════════════════════════════════════════════════

``get(thread_id)`` / ``put(thread_id, state)`` 是照着 LangGraph 的
checkpointer 取的形状 —— 这是这类东西的经典写法，没有必要另发明一套。
自己实现是因为：接进 LangGraph 就意味着编排内核整个换掉，
而那正是这个框架刻意自研的部分。用它的接口，不用它的实现。

════════════════════════════════════════════════════════════════
 三条必须守住的规矩
════════════════════════════════════════════════════════════════

★ 一、图变了就不能续。

  续跑的全部依据是「这些节点的产物已经算好了」。如果图的形状变了
  —— 加了节点、改了依赖 —— 那些产物对应的还是旧图，喂给新图会得到
  一个看起来正常、实际上错位的结果。所以存一份图签名，对不上就**不续**，
  并且把不续的原因说出来。

★ 二、存不下的产物不假装存下了。

  节点产物是任意 Python 对象，不一定能 JSON 序列化。存不下的，
  就不记进 pool —— 续跑时那个节点会重跑。这比存一个残缺的表示安全，
  但代价必须写明：**重跑意味着副作用会再发生一次**（比如又写一遍文件）。

★ 三、写盘要原子。

  断点文件是在"随时可能崩"的前提下写的。直接覆盖原文件的话，
  崩在写到一半就会留下一个半截的 JSON —— 而那个文件恰恰是用来
  从崩溃里恢复的。先写临时文件再 os.replace。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional

#: 断点格式版本。格式不兼容地变了就加一，旧断点直接判为不可续。
CHECKPOINT_VERSION = 1


def graph_signature(nodes: Dict[str, Any]) -> str:
    """给图的形状算一个签名：节点名 + 依赖关系。

    ★ 不含节点函数本身 —— 函数体改了但图没变时，续跑仍然合理
      （那正是"改一行再接着跑"的常见场景）。
      要连实现一起管住，那是另一个层次的问题，这里不假装解决。
    """

    shape = sorted(
        (name, tuple(sorted(node.depends_on)))
        for name, node in nodes.items()
    )
    raw = json.dumps(shape, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value, ensure_ascii=False)
        return True
    except (TypeError, ValueError):
        return False


class MemoryCheckpointer:
    """存在内存里。用于测试，以及"只想要重试不想要落盘"的场景。"""

    def __init__(self) -> None:
        self._data: Dict[str, dict] = {}
        self._lock = threading.Lock()
        #: 写入次数 —— 测试用它确认"每层都存了"，而不只是"最后存了一次"
        self.writes = 0

    def get(self, thread_id: str) -> Optional[dict]:
        with self._lock:
            state = self._data.get(thread_id)
            return json.loads(json.dumps(state)) if state is not None else None

    def put(self, thread_id: str, state: dict) -> None:
        with self._lock:
            self._data[thread_id] = json.loads(json.dumps(state))
            self.writes += 1

    def clear(self, thread_id: str) -> None:
        with self._lock:
            self._data.pop(thread_id, None)


class JSONCheckpointer:
    """存成一个 JSON 文件。

    一个文件装所有 thread_id。战役数量是个位数量级，
    没必要为此引入数据库。
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.writes = 0

    def _read_all(self) -> Dict[str, dict]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            # ★ 断点文件坏了不能让整次执行挂掉 —— 它是"锦上添花"，
            #   坏了就当没有断点，从头跑。但这与"静默吞掉"不同：
            #   调用方会在 GraphResult.resume_note 里看到原因。
            return {}

    def get(self, thread_id: str) -> Optional[dict]:
        with self._lock:
            return self._read_all().get(thread_id)

    def put(self, thread_id: str, state: dict) -> None:
        with self._lock:
            data = self._read_all()
            data[thread_id] = state
            d = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(d, exist_ok=True)
            # ★ 原子写：先落临时文件，再 replace
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=2)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.path)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
            self.writes += 1

    def clear(self, thread_id: str) -> None:
        with self._lock:
            data = self._read_all()
            if data.pop(thread_id, None) is not None:
                with open(self.path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=2)


def make_state(signature: str, layer_index: int,
               pool: Dict[str, Any],
               statuses: Dict[str, str],
               unresumable: List[str]) -> dict:
    """组装一份可落盘的断点。不可序列化的产物会被剔除并记名。"""

    keep, dropped = {}, list(unresumable)
    for k, v in pool.items():
        if _jsonable(v):
            keep[k] = v
        elif k not in dropped:
            dropped.append(k)
    return {
        "version": CHECKPOINT_VERSION,
        "signature": signature,
        "layer": layer_index,
        "pool": keep,
        "statuses": statuses,
        "unresumable": dropped,
    }


def validate(state: Optional[dict], signature: str) -> tuple:
    """判断一份断点能不能用于当前这张图。

    返回 ``(可用?, 说明)``。说明总是非空 —— **不可续的原因必须说得出来**，
    否则使用者只会看到"它又从头跑了一遍"而不知道为什么。
    """

    if state is None:
        return False, ""
    if not isinstance(state, dict):
        return False, "断点格式无法识别，已忽略"
    if state.get("version") != CHECKPOINT_VERSION:
        return False, "断点版本为 %r，当前为 %d，已忽略" % (
            state.get("version"), CHECKPOINT_VERSION)
    if state.get("signature") != signature:
        return False, "图的形状已改变（节点或依赖不同），断点不可续，已从头执行"
    return True, "自断点续跑：已跳过 %d 个已完成节点" % len(state.get("pool") or {})
