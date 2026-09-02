r"""军帐 —— 工作记忆（Working Memory）。

════════════════════════════════════════════════════════════════
 这一层补的是什么
════════════════════════════════════════════════════════════════

上下文目前是**无界增长**的：ReAct 每轮把模型回复与工具结果
一路追加进消息列表，从不回收。

实测这不是理论风险 —— 一次调研任务调用了 22 次工具、
抓回 68 份资料共 11.6 万字，全部原样躺在上下文里。
后果有三个，且都不报错：

  1. 成本随轮次**平方级**增长：第 n 轮要把前 n-1 轮全部重发一遍
  2. 撞上模型的上下文上限后，请求直接失败或被静默截断
  3. 关键信息被淹没在中段 —— 模型对长上下文中部的注意力最弱

★ 而"上下文太长"这件事没有任何现成信号会告诉你。
  与 finish_reason 那次不同，这次连一个可读的字段都没有：
  你只会观察到成本变高、答案变差。

════════════════════════════════════════════════════════════════
 压缩策略
════════════════════════════════════════════════════════════════

保头保尾，压中间：

  · **系统提示**永远保留 —— 它定义了角色与约束，丢了就变一个人
  · **最初的任务**永远保留 —— 丢了会开始答非所问
  · **最近若干轮**完整保留 —— 当前正在做的事需要细节
  · 中间的工具结果**折叠成摘要**，只留"调了什么、拿到什么要点"

★ 折叠的是工具结果，不是模型的推理。
  工具结果通常是长文本（网页正文、执行输出），信息密度低；
  而模型的推理是它自己的思路链，压掉会让它忘记为什么走到这一步。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from bingfu.llm.base import LLMMessage, RoleType


#: 触发压缩的字符数阈值。
#:
#: ★ 用字符数而不是 token 数，是因为精确算 token 需要 tokenizer，
#:   而不同供应商的 tokenizer 不同、且未必可用。
#:   中英混排下大约 1 token ≈ 1.5–2 字符，这个估算对
#:   "要不要压缩"这个二元决策足够了 —— 不必为一个阈值引入依赖。
DEFAULT_THRESHOLD = 24000

#: 尾部完整保留的消息条数
KEEP_RECENT = 6

#: 折叠后每条工具结果保留的字符数
FOLD_TO = 300


@dataclass
class CompactionResult:
    messages: List[LLMMessage]
    folded: int = 0
    chars_before: int = 0
    chars_after: int = 0

    @property
    def happened(self) -> bool:
        return self.folded > 0


def measure(messages: List[LLMMessage]) -> int:
    return sum(len(m.content or "") for m in messages)


def compact(
    messages: List[LLMMessage],
    *,
    threshold: int = DEFAULT_THRESHOLD,
    keep_recent: int = KEEP_RECENT,
    fold_to: int = FOLD_TO,
) -> CompactionResult:
    """把过长的上下文压缩到可控范围。

    ★ 不改变消息的**结构**，只缩短内容。

      删除消息会破坏 assistant→tool 的配对关系，
      多数供应商会直接拒绝这样的请求 ——
      而那个错误发生在压缩之后很远的地方，极难归因。
      所以这里只折叠正文，不删条目。
    """

    before = measure(messages)
    if before <= threshold or len(messages) <= keep_recent + 2:
        return CompactionResult(list(messages), 0, before, before)

    out: List[LLMMessage] = []
    folded = 0
    # 头部：系统提示 + 第一条用户消息
    head_count = 0
    for m in messages:
        if m.role is RoleType.SYSTEM or head_count == 0 and m.role is RoleType.USER:
            head_count += 1
        else:
            break

    tail_start = max(head_count, len(messages) - keep_recent)

    for i, m in enumerate(messages):
        if i < head_count or i >= tail_start:
            out.append(m)
            continue
        # 中段：只折叠工具结果
        if m.role is RoleType.TOOL and len(m.content or "") > fold_to:
            folded += 1
            out.append(LLMMessage(
                role=m.role,
                content=(
                    (m.content or "")[:fold_to]
                    + f"\n…[此工具结果已折叠，原长 {len(m.content or '')} 字符。"
                      f"如需完整内容请重新调用工具]"
                ),
                tool_call_id=getattr(m, "tool_call_id", None),
                name=getattr(m, "name", None),
            ))
        else:
            out.append(m)

    return CompactionResult(out, folded, before, measure(out))
