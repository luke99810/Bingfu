"""分层记忆系统。

★ 原框架只有一个键值字典，且 Agent.memory 这个字段
  **声明了却从未被读写** —— 每次执行清空上下文重来，
  将领打完一仗什么都不留下。

  补齐这几层不是为了功能更多，是为了闭合两个断掉的回路：

    1. TacticEngine 打分含 0.20 权重的 history 项，
       而 history_scores 从未被任何调用方传入 —— 恒为常数 0.5
    2. 工具抓回的资料随任务结束即丢弃，同领域连续任务反复重抓
"""

import pytest

from bingfu.llm.base import LLMMessage, RoleType
from bingfu.memory import (
    MIN_SAMPLES,
    Episode,
    EpisodicMemory,
    KnowledgeBase,
    Memory,
    compact,
    history_scores,
    summarize,
)


# ══════════════════════════════════════════════════════════
#  ① 情景记忆：证据不足时不许影响决策
# ══════════════════════════════════════════════════════════

def _fill(mem, agent, category, n, success):
    for _ in range(n):
        mem.record(Episode(task="t", agent_name=agent,
                           category=category, success=success))


def test_insufficient_samples_do_not_enter_decisions():
    """★ 这是这一层最重要的一条约束。

    本项目吃过一次大亏：在 n≈12 的格子里把 1 个任务的差异读成
    "战术注入有害 −26 点"，据此关掉了一整类任务的战术注入，
    而 Fisher 精确检验给出的 p 值是 1.000。

    历史分会**直接改变后续选择**，因此比一份报表更危险：
    基于噪声的偏好会自我固化 —— 一旦某个将领因偶然一次失败被降权，
    它就更少被选中，也就更少有机会产生反驳那次失败的数据。
    """

    mem = EpisodicMemory()
    _fill(mem, "白起", "Code", MIN_SAMPLES - 1, False)
    assert history_scores(mem) == {}, "样本不足却影响了决策"

    _fill(mem, "白起", "Code", 2, False)
    assert history_scores(mem)["白起"]["Code"] == 0.0


def test_unjudged_episodes_are_excluded():
    """★ "没有裁判"与"判为失败"必须区分。

    Agent 记录战报时把 success 留空，因为 Loop 的 outcome
    只能说明"有没有通过机械验收"，不能说明任务是否真的完成。
    把两者混同，会让历史分建立在一个它没资格下的判断上。
    """

    mem = EpisodicMemory()
    for _ in range(MIN_SAMPLES + 2):
        mem.record(Episode(task="t", agent_name="韩信", category="Code", success=None))
    assert history_scores(mem) == {}


def test_success_rate_is_computed_per_category():
    mem = EpisodicMemory()
    _fill(mem, "韩信", "Code", MIN_SAMPLES, True)
    _fill(mem, "韩信", "IR", MIN_SAMPLES, False)
    scores = history_scores(mem)
    assert scores["韩信"]["Code"] == 1.0
    assert scores["韩信"]["IR"] == 0.0


def test_recall_carries_outcome_not_full_output():
    """★ 只放结果与教训，不放完整产出。

    把上次的全文塞回上下文既贵，又会让模型倾向于复制上次的答案 ——
    而如果上次是错的，那正是最不该复制的东西。
    """

    mem = EpisodicMemory()
    mem.record(Episode(task="写一个 API", agent_name="白起", category="Code",
                       success=False, stopped_by="回炉轮次用尽"))
    text = summarize(mem, category="Code")
    assert "白起" in text and "失败" in text
    assert "回炉轮次用尽" in text


def test_episodic_memory_is_bounded():
    """无界增长的记忆最终会拖垮加载与检索。"""

    mem = EpisodicMemory(max_episodes=10)
    _fill(mem, "韩信", "Code", 25, True)
    assert len(mem) == 10


def test_episodic_memory_survives_a_corrupt_file(tmp_path):
    """★ 记忆文件损坏时从空开始，而不是让整个框架起不来。"""

    path = tmp_path / "ep.json"
    path.write_text("这不是 JSON", encoding="utf-8")
    mem = EpisodicMemory(str(path))
    assert len(mem) == 0
    mem.record(Episode(task="t", agent_name="韩信"))
    assert len(EpisodicMemory(str(path))) == 1


# ══════════════════════════════════════════════════════════
#  ② 语义记忆：跨任务复用，但不污染溯源
# ══════════════════════════════════════════════════════════

LONG_FACT = "IBM 在 2024 年发布了 1121 量子比特的 Condor 处理器，错误率显著下降。" * 2
OTHER_FACT = "红烧肉需要冰糖炒色，小火慢炖四十分钟，火候是关键所在。" * 2


def test_knowledge_is_retrievable_across_tasks():
    kb = KnowledgeBase()
    kb.learn(LONG_FACT, source="http://a", title="量子硬件")
    kb.learn(OTHER_FACT, source="http://b", title="烹饪")
    hits = kb.search("量子比特 处理器", top_k=1)
    assert hits and hits[0][0].title == "量子硬件"


def test_recalled_knowledge_is_labelled_as_prior():
    """★ 必须标注"这是过去学到的"。

    不标注的话，模型会把旧知识与本次新抓的资料混同，
    而数字溯源检查只认本次抓回的原文 —— 于是一个引用了
    旧知识的正确结论，会被判成"数字无出处"。
    """

    kb = KnowledgeBase()
    kb.learn(LONG_FACT, source="http://a", title="量子硬件")
    assert "非本次抓取" in kb.recall("量子比特")


def test_short_fragments_are_not_stored():
    """太短的片段没有检索价值，只会稀释语料并带偏 BM25 的 idf。"""

    kb = KnowledgeBase()
    assert kb.learn("短", source="x") is None
    assert len(kb) == 0


def test_duplicate_facts_are_not_stored_twice():
    kb = KnowledgeBase()
    kb.learn(LONG_FACT, source="http://a")
    kb.learn(LONG_FACT, source="http://a")
    assert len(kb) == 1


def test_eviction_keeps_facts_that_get_used():
    """★ 不按纯 FIFO 淘汰。

    一条被反复用到的旧知识，比一条刚学到却从没用过的更值得留。
    """

    kb = KnowledgeBase(max_facts=2)
    kb.learn(LONG_FACT, source="a", title="常用")
    kb.learn(OTHER_FACT, source="b", title="没用过")
    kb.search("量子比特")                    # 让第一条命中一次
    kb.learn("超导量子比特是目前最成熟的技术路线，被多家厂商采用。" * 2,
             source="c", title="新来的")
    titles = {f.title for f in kb._facts}
    assert "常用" in titles, "被用过的知识不该先被淘汰"


# ══════════════════════════════════════════════════════════
#  ③ 工作记忆：上下文压缩
# ══════════════════════════════════════════════════════════

def _long_conversation(rounds=10):
    msgs = [
        LLMMessage(role=RoleType.SYSTEM, content="你是将领"),
        LLMMessage(role=RoleType.USER, content="调研任务"),
    ]
    for i in range(rounds):
        msgs.append(LLMMessage(role=RoleType.ASSISTANT, content=f"第{i}轮思考"))
        msgs.append(LLMMessage(role=RoleType.TOOL, content="网页正文" * 900,
                               name="fetch_page"))
    return msgs


def test_long_context_is_compacted():
    r = compact(_long_conversation(), threshold=20000)
    assert r.happened
    assert r.chars_after < r.chars_before / 2


def test_compaction_keeps_head_and_structure():
    """★ 只折叠内容，不删条目。

    删除消息会破坏 assistant→tool 的配对关系，多数供应商会
    直接拒绝这样的请求 —— 而那个错误发生在压缩之后很远的地方，
    极难归因。
    """

    msgs = _long_conversation()
    r = compact(msgs, threshold=20000)
    assert len(r.messages) == len(msgs), "消息条数变了，配对关系可能被破坏"
    assert r.messages[0].content == "你是将领", "系统提示丢了就变成另一个角色"
    assert r.messages[1].content == "调研任务", "原始任务丢了会开始答非所问"


def test_short_context_is_untouched():
    msgs = _long_conversation(rounds=1)
    r = compact(msgs, threshold=10 ** 7)
    assert not r.happened
    assert r.messages == msgs


def test_reasoning_is_not_folded():
    """折叠的是工具结果，不是模型的推理。

    推理是它自己的思路链，压掉会让它忘记为什么走到这一步。
    """

    msgs = _long_conversation()
    r = compact(msgs, threshold=20000)
    reasoning = [m for m in r.messages if m.role is RoleType.ASSISTANT]
    assert all("已折叠" not in (m.content or "") for m in reasoning)


# ══════════════════════════════════════════════════════════
#  ④ 接进执行路径
# ══════════════════════════════════════════════════════════

class _Echo:
    def generate(self, messages, **kwargs):
        self.last = messages[0].content or ""

        class R:
            content = "完成"
            finish_reason = "stop"
            usage = {"total_tokens": 50}
            tool_calls = []
            has_tool_calls = False

        return R()


def test_agent_records_an_episode():
    """★ 执行完必须留下战报，否则记忆层又是"声明了没人用"。"""

    from bingfu.agent import Agent

    mem = EpisodicMemory()
    agent = Agent(name="韩信", role="将军", llm=_Echo(),
                  episodic=mem, category="Code")
    agent.execute("写个函数")
    assert len(mem) == 1
    ep = mem.all()[0]
    assert ep.agent_name == "韩信" and ep.category == "Code"
    assert ep.tokens == 50, "代价必须记下来"


def test_agent_recalls_history_into_prompt():
    mem = EpisodicMemory()
    mem.record(Episode(task="上一个任务", agent_name="韩信",
                       category="Code", success=False,
                       stopped_by="回炉轮次用尽"))

    from bingfu.agent import Agent

    llm = _Echo()
    Agent(name="韩信", role="将军", llm=llm,
          episodic=mem, category="Code").execute("新任务")
    assert "执行记录" in llm.last, "历史没有进入系统提示"


def test_tactic_engine_derives_history_from_memory():
    """★ 闭合那个断了很久的回路。

    打分公式含 0.20 权重的 history 项，而 history_scores
    从未被任何调用方传入 —— 恒为常数 0.5。
    接口在、参数在、注释在，就是没人喂数据；
    而它不报错，因为 0.5 是个完全合法的值。
    """

    from bingfu.tactics import TacticEngine

    mem = EpisodicMemory()
    _fill(mem, "韩信", "Code", MIN_SAMPLES + 1, True)
    engine = TacticEngine(assessor=None, episodic=mem)
    assert engine.episodic is mem
    assert history_scores(mem)["韩信"]["Code"] == 1.0


def test_kv_memory_still_works():
    """原有的键值存储不能被这次重构破坏。"""

    m = Memory(name="军需库")
    m.store("k", {"v": 1})
    assert m.retrieve("k") == {"v": 1}


# ══════════════════════════════════════════════════════════
#  ⑤ 知识跨任务复用，但证据不跨任务串
# ══════════════════════════════════════════════════════════

def test_knowledge_persists_but_evidence_does_not():
    """★ 这两件事必须分开，是这个设计最核心的一条。

    · 证据跨任务串 → 上一条任务抓到的资料会让下一条的数字溯源检查
      把"来自别处的数字"判成有出处，验收层就失去了意义
    · 知识不跨任务留 → 同领域的连续任务反复重抓同一批页面
      （实测一次调研 8 次抓取，而工具让检索类 token 涨 19.6 倍）
    """

    from bingfu.tools import ToolBelt

    kb = KnowledgeBase()
    fact = "IBM 在 2024 年发布了 1121 量子比特的 Condor 处理器，错误率显著下降。" * 3

    first = ToolBelt(enable_web=True, enable_code=False, knowledge=kb)
    first.store.add("量子硬件", "http://a", fact)
    kb.learn(fact, source="http://a", title="量子硬件")

    second = ToolBelt(enable_web=True, enable_code=False, knowledge=kb)
    assert second.sources() == [], "证据跨任务串了"

    recalled = second.search_knowledge("量子比特 处理器")
    assert "非本次抓取" in recalled, "没有复用历史知识"
    assert second.sources() == [], "检索历史知识不该污染本次证据集"


def test_traceability_skips_when_no_evidence_this_run():
    """引用历史知识时，溯源检查必须跳过而不是判失败。"""

    from bingfu.tools import ToolBelt
    from bingfu.verify import verify_output

    belt = ToolBelt(enable_web=True, enable_code=False, knowledge=KnowledgeBase())
    text = "报告：Condor 处理器达到 1121 量子比特。" + "补充说明。" * 50
    r = verify_output(text, category="IR", sources=belt.sources() or ["任务描述"])
    assert "numbers_traceable" not in r.checks_run
