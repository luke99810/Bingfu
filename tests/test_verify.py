"""VERIFY 断言层的测试。

★ 这组测试有两个方向，缺一不可：

  ① 坏产出必须被拦下 —— 否则这层等于没有
  ② **好产出必须放行** —— 否则它会驱动无意义的回炉，
     烧掉几倍 token 去"修复"本来就对的东西

  第二个方向更容易被忽略。开发这层时先后出过两次误判，
  两次都不报错、不崩溃，只是安静地把成本翻了几倍：
    · 围栏正则失步，把代码块之间的散文当成代码去做语法检查
    · 英文 criteria 匹配中文产出，恒判"未涉及任何要求"
"""

import pytest

from bingfu.verify import (
    check_code_not_truncated,
    check_numbers_traceable,
    check_python_syntax,
    check_required_structure,
    extract_code_blocks,
    verify_output,
)


# ══════════════════════════════════════════════════════════
#  围栏抽取 —— 那个把散文当代码的 bug
# ══════════════════════════════════════════════════════════

MIXED_FENCES = """安装依赖：

```bash
pip install fastapi
```

**第二阵：布阵（项目结构）**

```python
from fastapi import FastAPI
app = FastAPI()
```

**第三阵：主战场**

```python
def create_task(db, item):
    return db.add(item)
```

```json
{"ok": true}
```
"""


def test_non_python_fences_do_not_desync_extraction():
    """★ 这是整组最重要的一条。

    第一版用 ``r"```(?:python|py)?\\s*\\n(.*?)```"`` 配对。
    开围栏与闭围栏是同一个字面量，正则分不开；
    一旦出现 ```bash 这种匹配不上的开围栏，
    正则就从**它的闭围栏**开始配对，此后全部失步 ——
    捕获到的是代码块之间的说明文字。

    实测：7 个代码块的产出被抽成 7 段中文散文，
    每段都报 "invalid character '：'"，
    然后这 7 条错误被喂回模型，烧掉三倍 token。
    """

    blocks = extract_code_blocks(MIXED_FENCES)
    assert len(blocks) == 2, "只应抽出两个 Python 块"
    for b in blocks:
        assert "第二阵" not in b and "第三阵" not in b, "抽到了围栏之间的散文"
    assert "FastAPI" in blocks[0]
    assert "create_task" in blocks[1]


def test_non_python_blocks_are_not_syntax_checked():
    """bash / json 块不该拿去做 Python 语法检查。"""

    assert check_python_syntax(MIXED_FENCES).passed


def test_unfenced_code_is_still_recognised():
    """模型经常不加围栏 —— 整段是代码时也要认。"""

    assert extract_code_blocks("import os\n\ndef f():\n    return 1") != []


# ══════════════════════════════════════════════════════════
#  Code：真断的代码必须拦下
# ══════════════════════════════════════════════════════════

def test_broken_code_is_caught():
    broken = MIXED_FENCES.replace("return db.add(item)", "return db.add(")
    r = check_python_syntax(broken)
    assert not r.passed
    assert "语法错误" in r.reasons[0]


def test_truncated_code_is_caught():
    r = check_code_not_truncated("```python\ndef f(x):\n    return g(\n```")
    assert not r.passed


def test_missing_code_for_a_code_task_is_caught():
    r = check_python_syntax("我建议你用 FastAPI 来实现这个服务。")
    assert not r.passed


def test_feedback_is_specific_enough_to_act_on():
    """回炉反馈必须具体到能指导修改 —— "很差"没有用。"""

    r = verify_output("```python\ndef f(x):\n    return g(\n```", category="Code")
    fb = r.feedback()
    assert "语法错误" in fb and "完整" in fb


# ══════════════════════════════════════════════════════════
#  数字溯源 —— 以及它什么时候**不该**启用
# ══════════════════════════════════════════════════════════

SRC = ["行业报告：2023 年市场规模 47.8 亿美元，同比增长 23.6%，2027 年预计 112.4 亿美元。"]


def test_fabricated_numbers_are_caught():
    txt = "综合资料：市场规模为 63.1 亿美元，增长率 41.2%，预计将达 288.7 亿美元。"
    assert not check_numbers_traceable(txt, SRC).passed


def test_grounded_numbers_pass():
    txt = "综合资料：市场规模为 47.8 亿美元，增长率 23.6%，预计 2027 年达 112.4 亿美元。"
    assert check_numbers_traceable(txt, SRC).passed


def test_reasonable_derivation_is_tolerated():
    """模型做的合理算术不在来源里，但它是对的 —— 容忍度就是为它留的。"""

    txt = "市场规模 47.8 亿美元，增长 23.6%，据此推算次年约为 59.1 亿美元。"
    assert check_numbers_traceable(txt, SRC).passed


def test_traceability_is_skipped_without_real_sources():
    """★ 一个在当前配置下不可能通过的检查，不是严格，是坏的。

    benchmark 里 agent 没有工具，研究类任务的数字只能出自模型知识，
    此时 sources 里只有任务描述本身 —— 要求"每个数字都有出处"
    等于要求一件不可能的事。

    实测代价：IR1 从「通过、4775 token」变成
    「回炉两轮、未过、12852 token」，惩罚一个本来做对了的产出。
    """

    txt = "市场规模 63.1 亿美元，增长 41.2%，预计 288.7 亿。" + "补充说明。" * 40
    r = verify_output(txt, category="IR", criteria=["Coverage"], sources=["任务描述"])
    assert "numbers_traceable" not in r.checks_run
    assert r.passed


def test_traceability_runs_when_sources_exist():
    txt = "市场规模 63.1 亿美元，增长 41.2%，预计 288.7 亿。" + "补充说明。" * 40
    r = verify_output(txt, category="IR", criteria=["Coverage"],
                      sources=["任务描述", SRC[0]])
    assert "numbers_traceable" in r.checks_run
    assert not r.passed


# ══════════════════════════════════════════════════════════
#  结构检查 —— 不能因为语言不同就恒判失败
# ══════════════════════════════════════════════════════════

def test_chinese_output_with_english_criteria_passes():
    """★ 第二个误判：英文 criteria 匹配中文产出，恒命中 0 个。

    实测 W1（2971 字完整文章）、R1（815 字推理）都被拦下并回炉 ——
    两次都是纯烧 token。

    一个恒为 False 的检查和一个恒为 True 的检查一样没用，
    而且更贵：后者只是不做事，前者会驱动无意义的重试。
    """

    zh = "这是一份完整的中文分析报告，包含背景、方法与结论。" * 20
    r = check_required_structure(zh, ["Clarity", "Technical Accuracy", "Coverage"])
    assert r.passed


def test_empty_output_is_still_caught():
    """去掉关键词匹配之后，长度判据仍要能捞到真正空洞的产出。"""

    assert not check_required_structure("还行。", ["Clarity"]).passed


# ══════════════════════════════════════════════════════════
#  分派
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("category,expected", [
    ("Code", "python_syntax"),
    ("IR", "required_structure"),
    ("Write", "required_structure"),
])
def test_checks_are_selected_by_category(category, expected):
    r = verify_output("x" * 500, category=category, criteria=(), sources=())
    assert expected in r.checks_run
