r"""断言层（VERIFY）—— 对产出做**可证伪**的检查。

════════════════════════════════════════════════════════════════
 为什么要有这一层
════════════════════════════════════════════════════════════════

实测：Code 类成功率 12%、Write 类 38%，而信息检索类是 100%。

原因不难找 —— benchmark 里将领拿到的 tools=None，
于是 has_tool_calls 恒为 False，
for i in range(max_iterations) **只跑一轮就 return**。

也就是说：让模型一次性写出一个带 JWT 鉴权、WebSocket、
异步数据库的全栈应用，**不执行、不检查、不重试**，
把第一段回复直接当成最终答案。

12% 不是模型能力的问题，是没有验收环节的必然结果。

════════════════════════════════════════════════════════════════
 什么样的检查才算数
════════════════════════════════════════════════════════════════

★ 不能是"再问模型一次这答得好不好"。

  模型对自己输出的评价与输出质量相关性很弱，而且它**几乎不会失败** ——
  一个永远回答"还不错"的检查器，和没有检查器是一回事，
  却会让报表上多出一列看起来很负责的"已验证"。

  这里的每个检查都必须能对**确实有问题的产出**返回 False，
  并且给出可回炉的具体原因。

三类断言，按任务类型启用：

  Code    ── 语法必须能过 ast.parse；检查是否被截断
  IR/Data ── 结论里的数字必须能在来源里找到出处
  Write   ── 要求的结构要素必须齐全

中间那类借自 MiniAgent-Demo 的 numbers_traceable 思路：
凭空生成的数字是 LLM 最常见也最难察觉的错误，
而它恰好是**可以机械检验**的。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class VerifyResult:
    """一次验收的结果。

    ``reasons`` 不是给人看的日志 —— 它会被拼进回炉提示词，
    所以必须具体到能指导修改。"很差"没有用，
    "第 12 行 SyntaxError: unexpected EOF" 才有用。
    """

    passed: bool
    reasons: List[str] = field(default_factory=list)
    checks_run: List[str] = field(default_factory=list)

    def feedback(self) -> str:
        """拼成可直接回炉的反馈文本。"""

        if self.passed:
            return ""
        lines = ["你上一版的产出未通过验收，具体问题："]
        lines += [f"  {i}. {r}" for i, r in enumerate(self.reasons, 1)]
        lines.append("请针对以上问题修改后重新给出**完整**产出，不要只给差异。")
        return "\n".join(lines)


# ── 代码块抽取 ────────────────────────────────────────────

# 围栏开头的语言标签（```python / ```bash / ```），只吃到行尾
_LANG_TAG = re.compile(r"^[a-zA-Z0-9_+.-]{0,12}[ \t]*\n")
_CODEISH = re.compile(r"^\s*(def |class |import |from \w+ import )", re.M)


def extract_code_blocks(text: str) -> List[str]:
    """取出 Markdown 围栏里的代码块。

    ★ 按围栏切分取奇数段，**不用正则配对**。

      第一版写的是 ``"```(?:python|py)?\\s*\\n(.*?)```"``。
      问题在于开围栏和闭围栏是**同一个字面量**，正则分不开它们；
      而语言标签一旦不是 python/py（bash、text、json 都很常见），
      开围栏就匹配不上，正则于是从**它的闭围栏**开始配对，
      整个配对就此失步。

      实测：一段含 7 个代码块的正常产出被抽成 7 个"代码块"，
      内容全是两块之间的中文散文（``**第二阵：布阵（项目结构）**``），
      然后每一个都报 ``invalid character '：'`` 语法错误。

    ★ 危害不在误判本身，而在于它把 7 条**言之凿凿却完全错误**的
      反馈喂回模型，烧掉三倍 token 去"修复"根本不是代码的文字 ——
      全程不报错、不崩溃，看起来就像门禁在尽职工作。

      这正是"返回一组貌似合理的结果"那类缺陷：
      一个空手而归的检查会被发现，一个自信地指错方向的不会。

    切分法没有这个问题：围栏必然成对，所以奇数下标的段必在围栏内，
    与语言标签是什么无关。
    """

    raw = text or ""
    if "```" in raw:
        segments = raw.split("```")
        blocks: List[str] = []
        # segments[0] 是首个围栏之前的文字，此后奇数下标都在围栏内
        for seg in segments[1::2]:
            body = _LANG_TAG.sub("", seg, count=1)
            body = body.strip()
            # 只保留看起来是代码的段 —— bash/json 块不该拿去做 Python 语法检查
            if body and _CODEISH.search(body):
                blocks.append(body)
        if blocks:
            return blocks

    stripped = raw.strip()
    if _CODEISH.search(stripped):
        return [stripped]
    return []


# ── Code：语法必须过 ─────────────────────────────────────

def check_python_syntax(output: str) -> VerifyResult:
    """每个代码块都必须能被 ast.parse 解析。

    ★ 这是**真正会失败**的检查：语法错误是客观的，
      模型说"我写得很好"改变不了 SyntaxError。

    只查语法不查语义，是刻意的取舍：语义正确性需要执行，
    而执行任意生成代码有安全代价。语法检查已经能捞回很大一部分 ——
    单次生成的长代码里未闭合的括号、截断的函数体非常常见。
    """

    result = VerifyResult(passed=True, checks_run=["python_syntax"])
    blocks = extract_code_blocks(output)

    if not blocks:
        result.passed = False
        result.reasons.append(
            "产出里找不到任何 Python 代码块 —— 任务要求的是可运行的代码，"
            "请用 Markdown 围栏给出完整实现。"
        )
        return result

    for idx, block in enumerate(blocks, 1):
        try:
            ast.parse(block)
        except SyntaxError as exc:
            result.passed = False
            result.reasons.append(
                f"第 {idx} 个代码块有语法错误：{exc.msg}"
                f"（第 {exc.lineno} 行附近）。这段代码无法运行。"
            )
    return result


_TRUNCATED_TAIL = (":", ",", "+", "-", "=", "(", "[", "{")


def check_code_not_truncated(output: str) -> VerifyResult:
    """检查代码是否被截断 —— 单次生成长代码时的高频故障。"""

    result = VerifyResult(passed=True, checks_run=["not_truncated"])
    for idx, block in enumerate(extract_code_blocks(output), 1):
        stripped = block.rstrip()
        if not stripped:
            continue
        last = stripped.splitlines()[-1].strip()
        if last.endswith(_TRUNCATED_TAIL):
            result.passed = False
            result.reasons.append(
                f"第 {idx} 个代码块像是被截断了（最后一行以 {last[-1]!r} 结尾）。"
                f"请给出完整的实现。"
            )
    return result


# ── IR/Data：数字必须可溯源 ──────────────────────────────

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers_in(text: str) -> List[str]:
    """抽出文本里的数字，归一化后便于比对。"""

    out = []
    for raw in _NUM.findall(text or ""):
        try:
            out.append(f"{float(raw):g}")
        except ValueError:
            continue
    return out


def _informative(token: str) -> bool:
    """年份和小数值不参与判定 —— 它们太常见，命中与否没有信息量。"""

    try:
        val = float(token)
    except ValueError:
        return False
    if 1900 <= val <= 2100 and val == int(val):
        return False
    return abs(val) > 10


def check_numbers_traceable(
    output: str, sources: Sequence[str], *, tolerance_ratio: float = 0.35
) -> VerifyResult:
    """结论里的数字应当能在来源文本里找到出处。

    ★ 这条针对的是 LLM 最难察觉的错误：**编造具体数字**。
      一句"2023 年市场规模达 47.8 亿美元"读起来完全可信，
      而它可能纯属生成。

    容忍度是必要的：模型会做合理的算术（求和、百分比），
    这些派生数字不在来源里却是正确的。
    所以判据不是"全部命中"，而是"**命中率不能太低**" ——
    低到某个程度就说明它在自由发挥，而不是在引用。

    tolerance_ratio = 允许多大比例的数字找不到出处。
    0.35 是个判断，不是定理。
    """

    result = VerifyResult(passed=True, checks_run=["numbers_traceable"])
    claimed = [n for n in _numbers_in(output) if _informative(n)]
    if len(claimed) < 3:          # 数字太少，这条检查没有判别力
        return result

    pool = set()
    for src in sources:
        pool.update(_numbers_in(src))

    missing = [n for n in claimed if n not in pool]
    ratio = len(missing) / len(claimed)
    if ratio > tolerance_ratio:
        result.passed = False
        result.reasons.append(
            f"产出里 {len(claimed)} 个具体数字中有 {len(missing)} 个"
            f"（{ratio:.0%}）无法在任务描述或已获取的资料里找到出处，"
            f"例如：{', '.join(missing[:5])}。"
            f"请只使用有据可依的数据，或明确标注为估算。"
        )
    return result


# ── Write：结构要素齐全 ──────────────────────────────────

MIN_SUBSTANTIVE_LENGTH = 200


def check_required_structure(
    output: str,
    criteria: Sequence[str],
    *,
    artifacts: Sequence[str] = (),
) -> VerifyResult:
    """按任务的 evaluation_criteria 检查结构要素。

    ════════════════════════════════════════════════════════
     artifacts：产出落在文件里时，回复只是回执
    ════════════════════════════════════════════════════════

    ★ 长度判据成立的前提是**回复本身就是产物**。

      没有工具的任务确实如此：模型回 104 个字，那 104 个字就是全部交付，
      判它空洞是对的。但将领带着 write_file 时，交付物在磁盘上，
      回复是一句「已完成：写了 facts.md 和 pick.md」——
      **它短是因为它该短**。

    ★ 实测这条边界的代价（agg-pick，兵符）：

        write(facts.md) write(pick.md)   ← 两个文件都已正确写好
        「已完成：……」133 字             ← 判为「明显过短」，打回重做
        write(facts.md) write(pick.md)   ← 整件事重做一遍

      三重代价：LLM 调用翻倍、token 翻倍，而且**产物变差了** ——
      重做那版把 facts.md 从 41 字灌成 775 字的注水文，
      只为了让回复够长。判据在奖励注水。

    ★ 一般化的说法：**门禁必须判在交付物上，不是判在关于交付物的话上。**
      判错对象的门禁不是宽或严的问题，它测的根本是另一样东西。
    ★ 只做**存在性**检查，不做质量判断。

      "这一节写得好不好"是主观的，交给 LLM 裁判；
      "要求的这一节在不在"是客观的，机械可判。
      两者混在一起，会让这层退化成又一个"总是说还行"的检查器。
    """

    result = VerifyResult(passed=True, checks_run=["required_structure"])
    text = (output or "").strip()

    if artifacts:
        # 产物落在磁盘上：回复是回执，长度不承载信息。
        # 这里要判的是**产物存不存在**，而这一点由 artifacts 非空本身给出。
        result.checks_run = ["artifacts_produced"]
        return result

    if len(text) < MIN_SUBSTANTIVE_LENGTH:
        result.passed = False
        result.reasons.append(
            f"产出只有 {len(text)} 个字符，对这个任务而言明显过短。"
        )
        return result

    # ★ 这里原本还有一条「criteria 关键词命中」检查，已删除。
    #
    #   benchmark 的 evaluation_criteria 是**英文**
    #   （"Factual Accuracy"、"Coverage"、"Logical Consistency"），
    #   而模型的产出是**中文**。跨语言的关键词匹配永远命中 0 个，
    #   于是这条检查对每一个中文产出都判失败。
    #
    #   实测：W1（2971 字的完整文章）、R1（815 字的推理）
    #   都被它拦下并回炉 —— 两次都是误判，纯烧 token。
    #
    #   ★ 一个恒为 False 的检查和一个恒为 True 的检查一样没用，
    #     而且更贵：后者只是不做事，前者会驱动无意义的重试。
    #
    #   留下的长度判据是语言无关的，能捞到真正空洞的产出
    #   （实测 IR1 只回了 104 字、D1 只回了 90 字 —— 那些是真该拦的）。
    return result


# ── 按任务类型组装 ──────────────────────────────────────

def verify_output(
    output: str,
    *,
    category: str,
    criteria: Sequence[str] = (),
    sources: Sequence[str] = (),
    artifacts: Sequence[str] = (),
) -> VerifyResult:
    """按任务类型选择该跑哪些断言。

    ★ 分类型是有实测依据的，不是设想：
      战术注入在 Data/IR 上 +50 点、在 Code/Write 上 −26/−12 点。
      一套统一流水线对这五类任务本来就是错的。

    ★ artifacts 是本次**真正落盘**的产物名。非空时，回复只是回执，
      长度类判据不适用 —— 详见 check_required_structure 的说明。
    """

    checks: List[VerifyResult] = []

    if category == "Code":
        checks.append(check_python_syntax(output))
        checks.append(check_code_not_truncated(output))
    elif category in ("IR", "Data"):
        # ★ 数字溯源只在**确有外部来源**时才有意义。
        #
        #   benchmark 里 agent 没有工具，研究类任务的数字只能出自
        #   模型自身知识 —— 此时 sources 里只有任务描述，
        #   要求"每个数字都能在来源里找到"等于要求一件不可能的事。
        #
        #   实测代价：IR1 从「通过、4775 token」变成
        #   「回炉两轮、未过、12852 token」—— 门禁在惩罚一个
        #   本来就做对了的产出，同时烧掉 2.7 倍成本。
        #
        #   ★ 一个在当前配置下不可能通过的检查，不是严格，是坏的。
        #     它带工具取证时才成立 —— 那时 sources 里是真的工具返回。
        has_real_sources = len([s for s in sources if s]) > 1
        if has_real_sources:
            checks.append(check_numbers_traceable(output, sources))
        checks.append(check_required_structure(output, criteria,
                                               artifacts=artifacts))
    else:                                   # Write / Reason
        checks.append(check_required_structure(output, criteria,
                                               artifacts=artifacts))

    merged = VerifyResult(passed=all(c.passed for c in checks))
    for c in checks:
        merged.reasons.extend(c.reasons)
        merged.checks_run.extend(c.checks_run)
    return merged
