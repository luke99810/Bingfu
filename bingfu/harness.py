r"""执行护栏层（Harness）—— 结构化 LLM 调用的分级降级。

════════════════════════════════════════════════════════════════
 为什么这一层要先于 Loop 和 Graph
════════════════════════════════════════════════════════════════

这个框架的评估裁判曾经有一行正则 ``re.search(r'\{[^{}]*\}', content)``。
``[^{}]*`` 匹配的是**不含花括号的内容**，所以它取到的是模型回复里
最内层的那对花括号 —— 嵌套的 ``"criteria_scores": {...}`` ——
而不是外层那个含 ``success`` / ``completion_score`` 的信封。

后果：``data.get("success", False)`` 恒为 False，
``data.get("completion_score", 3.0)`` 恒为 3.0。
**这套评估在结构上不可能报告成功。**

而它错得没有任何症状：不报错、不抛异常、返回一组完全合法的数字。
七个方法全线 0.0%、TCS 全是 3.00 —— 看起来就像一组正常的实验结果。

★ 这就是 Harness 存在的理由：LLM 调用的失败模式差异极大，
  「少一个右括号」和「完全跑题」不该用同一种方式处理，
  更不该都悄悄变成一个合法的默认值。

════════════════════════════════════════════════════════════════
 五级降级链
════════════════════════════════════════════════════════════════

    Level 0  输入/Schema 校验      不调模型   <1ms
    Level 1  纯字符串修复           不调模型   <5ms
    Level 2  低温重生成             **调模型**  3-10s
    Level 3  模板兜底               不调模型   <1ms
    Level 4  部分字段输出           不调模型   <1ms
    Level 5  硬失败 —— **抛异常**   不调模型   —

★ Level 5 必须抛异常，不能返回"看起来合理的默认值"。
  静默返回默认值正是上面那个 bug 之所以能存活的原因。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


class AgentHarnessFailure(RuntimeError):
    """五级全部耗尽 —— 调用方必须显式处理，不会拿到伪装成正常的结果。"""


@dataclass
class HarnessResult:
    """把「成功 / 降级成功 / 硬失败」固化成类型契约。

    调用方只需看 ``success`` 与 ``output``；
    但 ``degradation_level`` 让"这个结果是怎么来的"永远可追溯 ——
    一个 level=3 的模板兜底和一个 level=0 的正常输出，
    在数据分析里绝不该被同等对待。
    """

    output: Any = None
    success: bool = False
    degradation_level: int = 0
    attempts: int = 0
    errors: List[str] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def degraded(self) -> bool:
        return self.degradation_level >= 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "degradation_level": self.degradation_level,
            "attempts": self.attempts,
            "errors": list(self.errors),
            "elapsed": round(self.elapsed, 3),
        }


# ── Level 1：纯字符串修复 ────────────────────────────────────

def extract_json_object(text: str) -> Optional[dict]:
    r"""从模型回复里取出**最外层**的 JSON 对象。

    ★ 用括号配对扫描，不用正则。
      正则写不出"配对的嵌套括号" —— 这是形式语言层面的限制，
      不是正则没写好。任何 ``\{.*\}`` 变体要么贪婪吞掉多个对象，
      要么像 ``[^{}]*`` 那样只能匹配不含嵌套的最内层。

    同时处理字符串字面量里的花括号（``{"note": "用 {} 表示"}``），
    否则配对计数会被内容里的括号带偏。
    """

    if not text:
        return None

    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break            # 这个起点不成立，换下一个 '{'
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


def repair_json(text: str) -> Optional[dict]:
    """Level 1：不调模型，只做字符串层面的修复。

    覆盖实际见过的几种脏输出：Markdown 围栏、尾逗号、
    单引号、缺右括号、前后夹带解释性文字。
    """

    if not text:
        return None

    direct = extract_json_object(text)
    if direct is not None:
        return direct

    s = text.strip()
    # 去 Markdown 围栏
    if "```" in s:
        parts = s.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            got = extract_json_object(part)
            if got is not None:
                return got

    # 去尾逗号
    cleaned = s
    for pair in (",}", ",]"):
        while pair in cleaned:
            cleaned = cleaned.replace(pair, pair[1])
    got = extract_json_object(cleaned)
    if got is not None:
        return got

    # 缺右括号：按未闭合数补齐
    opens = cleaned.count("{") - cleaned.count("}")
    if 0 < opens <= 5:
        got = extract_json_object(cleaned + "}" * opens)
        if got is not None:
            return got

    return None


# ── Level 0：Schema 校验 ─────────────────────────────────────

def validate_schema(data: dict, required: Sequence[str]) -> List[str]:
    """返回缺失字段列表；空列表 = 通过。"""

    if not isinstance(data, dict):
        return [f"顶层不是对象，而是 {type(data).__name__}"]
    return [f"缺少必填字段 `{k}`" for k in required if k not in data]


# ── 主入口 ──────────────────────────────────────────────────

def call_structured(
    generate: Callable[..., Any],
    *,
    required: Sequence[str] = (),
    template: Optional[dict] = None,
    max_regenerate: int = 1,
    low_temperature: float = 0.1,
) -> HarnessResult:
    """按五级降级链执行一次结构化 LLM 调用。

    Args:
        generate: 可调用对象，签名 ``generate(temperature=...) -> str``，
                  返回模型的原始文本。**它自己抛的异常代表基础设施故障，
                  会被原样上抛** —— 连不上和回复难看是两回事，
                  前者降级只会制造出无法解释的数据。
        required: Level 0 要求的必填字段。
        template: Level 3 的兜底模板；None 表示跳过该级。
        max_regenerate: Level 2 最多重生成几次。

    Raises:
        AgentHarnessFailure: 五级耗尽。**不返回默认值。**
    """

    started = time.time()
    result = HarnessResult()

    def _finish(output, success, level):
        result.output = output
        result.success = success
        result.degradation_level = level
        result.elapsed = time.time() - started
        return result

    # ── Level 0/1：首次调用 + 字符串修复 ──
    result.attempts += 1
    raw = generate()
    parsed = repair_json(raw if isinstance(raw, str) else str(raw))

    if parsed is not None:
        missing = validate_schema(parsed, required)
        if not missing:
            level = 0 if extract_json_object(raw or "") is not None else 1
            return _finish(parsed, True, level)
        result.errors.append("；".join(missing))
    else:
        result.errors.append("Level 1 修复后仍不是合法 JSON")

    # ── Level 2：降温重生成 ──
    for _ in range(max_regenerate):
        result.attempts += 1
        raw2 = generate(temperature=low_temperature)
        parsed2 = repair_json(raw2 if isinstance(raw2, str) else str(raw2))
        if parsed2 is not None:
            missing = validate_schema(parsed2, required)
            if not missing:
                return _finish(parsed2, True, 2)
            result.errors.append(f"重生成后仍缺字段：{'；'.join(missing)}")
        else:
            result.errors.append("重生成后仍不是合法 JSON")

    # ── Level 3：模板兜底 ──
    if template is not None:
        merged = dict(template)
        if parsed:
            merged.update({k: v for k, v in parsed.items() if k in template})
        return _finish(merged, True, 3)

    # ── Level 4：部分字段 ──
    if parsed:
        return _finish(parsed, False, 4)

    # ── Level 5：硬失败 ──
    result.elapsed = time.time() - started
    result.degradation_level = 5
    raise AgentHarnessFailure(
        "结构化调用五级全部耗尽：" + "；".join(result.errors)
        + "\n  ★ 此处**不返回默认值** —— 静默的合法默认值正是"
          "裁判正则那个 bug 能存活至今的原因。"
    )
