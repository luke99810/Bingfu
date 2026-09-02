"""工具层的测试。

★ 这一层最容易出的不是崩溃，是**静默失效** ——
  工具挂上了、请求发出去了、模型正常回复了，
  而工具一次都没被调用。成功率不变，日志无异常。

  开发过程中连续踩到三个，每一个都不报错：

    ① 工具参数 schema 是空的 → 模型无法传参，理性地不调用，
       转而**编造**执行结果（"经实测 fib(10) 得 55"）
    ② 调用计数器漏掉了代码类工具 → 我据此判断"模型不调用工具"，
       又去查 schema、查请求体、查 provider，全是好的
    ③ 轮数耗尽时直接返回最后一条回复 → 22 次工具调用、
       11.6 万字证据，产出只有 88 个字符

  三条都在这里钉住。
"""

import inspect

import pytest

from bingfu.agent import Agent, _schema_from_signature
from bingfu.tools import BM25, Document, SourceStore, belt_for, tokenize
from bingfu.tools.code_exec import run_python


# ══════════════════════════════════════════════════════════
#  ① 工具 schema 必须带参数
# ══════════════════════════════════════════════════════════

def test_schema_includes_parameters():
    """★ 曾经硬编码为 {"type":"object","properties":{}}。

    模型看到一个"不接受任何参数"的 execute_python，
    调用它无法传入代码，于是不调用 —— 并声称自己已经运行过。

    给一个用不了的工具，比不给工具更坏：
    不给的话模型会说"建议你运行验证"；
    给了却用不了，它就说"我已经验证过了"。
    """

    def sample(code: str, timeout: int = 5) -> str:
        """执行代码。

        code: 要执行的源码
        timeout: 超时秒数
        """
        return code

    schema = _schema_from_signature(sample)
    assert "code" in schema["properties"]
    assert "timeout" in schema["properties"]
    assert schema["properties"]["code"]["type"] == "string"
    assert schema["properties"]["timeout"]["type"] == "integer"
    assert schema["required"] == ["code"], "有默认值的参数不该是必填"
    assert "源码" in schema["properties"]["code"]["description"]


def test_registered_tools_expose_their_parameters():
    """端到端：注册到 Agent 之后，送给 LLM 的定义必须带参数。"""

    agent = Agent(name="测试", role="将军")
    for name, fn in belt_for("Data").as_functions().items():
        agent.add_tool(name, fn)

    by_name = {d.name: d for d in agent._build_tool_definitions(None)}
    assert "code" in by_name["execute_python"].parameters["properties"]
    assert "query" in by_name["web_search"].parameters["properties"]
    assert "url" in by_name["fetch_page"].parameters["properties"]


def test_counting_wrapper_preserves_signature():
    """★ 计数包装不能吃掉签名。

    Agent 靠内省签名生成 schema；包装若丢了签名，
    schema 就退化回空参数表 —— 也就是绕一圈回到 bug ①。
    """

    fn = belt_for("Code").as_functions()["execute_python"]
    assert "code" in inspect.signature(fn).parameters
    assert "code" in _schema_from_signature(fn)["properties"]


# ══════════════════════════════════════════════════════════
#  ② 调用计数必须覆盖全部工具
# ══════════════════════════════════════════════════════════

def test_all_tools_are_counted():
    """★ 曾经只有检索类工具计数，代码类直接挂模块级函数。

    后果不是少了个统计数字，而是**仪表指向了错误的方向**：
    工具明明被调用了，计数却是空的，于是排查全跑偏。
    坏掉的仪表比没有仪表更费时间。
    """

    belt = belt_for("Code")
    assert belt.used_any() is False
    belt.as_functions()["execute_python"]("print(1)")
    assert belt.call_counts.get("execute_python") == 1
    assert belt.used_any() is True


def test_web_tools_are_counted_without_network():
    """检索工具即使拿不到结果也要计数 —— 调用发生过就是发生过。"""

    belt = belt_for("IR")
    belt.search_knowledge("任何查询")      # 知识库为空，但调用确实发生了
    assert belt.call_counts.get("search_knowledge") == 1


# ══════════════════════════════════════════════════════════
#  ③ 代码执行的安全边界
# ══════════════════════════════════════════════════════════

def test_execution_returns_real_errors():
    """报错必须原样带回 —— "执行失败"四个字对模型没有用。"""

    r = run_python("import nonexistent_module_xyz")
    assert not r.ok
    assert "ModuleNotFoundError" in r.stderr or "ImportError" in r.stderr


def test_infinite_loop_is_terminated():
    """死循环必须被超时截断，否则一次实验会永远卡住。"""

    r = run_python("while True: pass", timeout=2.0)
    assert r.timed_out is True


def test_secrets_are_not_leaked_into_subprocess(monkeypatch):
    """★ 生成的代码没有理由需要 API key。

    一旦泄漏进它的日志或网络请求，就收不回来了。
    """

    monkeypatch.setenv("DEEPSEEK_API_KEY", "SECRET_VALUE_123")
    r = run_python("import os; print(os.environ.get('DEEPSEEK_API_KEY'))")
    assert "SECRET_VALUE_123" not in r.stdout


def test_output_is_capped():
    """无限打印不能撑爆内存，且截断要**标注**而不是静默丢弃。"""

    r = run_python("print('x' * 100000)")
    assert len(r.stdout) < 10000
    assert "截断" in r.stdout


# ══════════════════════════════════════════════════════════
#  ④ BM25 检索
# ══════════════════════════════════════════════════════════

@pytest.fixture
def store():
    s = SourceStore()
    s.add("量子计算", "http://a", "IBM 发布了 1121 量子比特的 Condor 处理器，错误率下降。")
    s.add("烹饪", "http://b", "红烧肉需冰糖炒色，小火慢炖四十分钟。")
    s.add("超导路线", "http://c", "超导量子比特是最成熟的路线，Google 与 IBM 均采用。")
    return s


def test_retrieval_ranks_relevant_documents_first(store):
    hits = BM25().build(store.all()).search("超导 量子比特", top_k=2)
    assert hits, "应当有命中"
    assert all(d.title != "烹饪" for d, _ in hits), "无关文档不该进前二"


def test_retrieval_on_empty_corpus_is_safe():
    assert BM25().build([]).search("任何查询") == []


def test_tokenizer_handles_mixed_scripts():
    toks = tokenize("IBM 发布 1121 qubit")
    assert "ibm" in toks and "1121" in toks and "发" in toks


# ══════════════════════════════════════════════════════════
#  ⑤ 证据流向验收层
# ══════════════════════════════════════════════════════════

def test_sources_flow_to_verification(store):
    """★ 工具抓回的原文既是 RAG 语料，也是溯源检查的证据。

    numbers_traceable 曾因"没有可溯源的对象"被迫禁用；
    工具补上之后它重新成立 —— 但前提是证据真的被留存下来。
    """

    belt = belt_for("IR")
    belt.store = store
    sources = belt.sources()
    assert len(sources) == 3
    assert any("1121" in s for s in sources)


def test_empty_sources_do_not_trigger_false_accusation():
    """★ 模型一次工具都没调时，来源集为空。

    此时溯源检查必须**跳过**，而不是把"没有证据"
    误判成"数字全是编的"。
    """

    from bingfu.verify import verify_output

    belt = belt_for("IR")
    assert belt.sources() == []
    text = "报告：市场规模 63.1 亿美元，增长 41.2%，预计 288.7 亿。" + "补充。" * 60
    r = verify_output(text, category="IR", criteria=["Coverage"],
                      sources=belt.sources() or ["任务描述"])
    assert "numbers_traceable" not in r.checks_run


# ══════════════════════════════════════════════════════════
#  ⑥ 按类别装配
# ══════════════════════════════════════════════════════════

def test_code_tasks_get_no_web_tools():
    """★ 不是"多多益善"。

    给写作任务一个 Python 解释器不会有帮助，
    只会增加跑偏的机会，并且每一轮都要把工具定义
    塞进上下文，白白消耗 token。
    """

    assert set(belt_for("Code").as_functions()) == {"execute_python", "run_tests"}


def test_write_tasks_get_no_interpreter():
    assert "execute_python" not in belt_for("Write").as_functions()


def test_data_tasks_get_both():
    funcs = belt_for("Data").as_functions()
    assert "execute_python" in funcs and "web_search" in funcs
