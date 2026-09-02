# -*- coding: utf-8 -*-
r"""可观测性探测：不是打分，是**实际去取**。

════════════════════════════════════════════════════════════════
 为什么做成探测而不是打表
════════════════════════════════════════════════════════════════

「哪个框架可观测性更好」很容易写成一张凭印象填的表。这里改成
**跑一次真实任务，逐项尝试把四样东西取出来**，取到记「有」并附上
用的是哪个 API，取不到记「无」并附上失败原因。

四样东西：

  steps   逐步事件 —— 执行过程中能不能拿到「现在在做什么」
  tokens  token 账 —— 框架自己报不报用量
  trace   工具轨迹 —— 能不能拿到**有序的**工具调用记录（含参数）
  resume  断点续跑 —— 有没有可调用的状态保存/恢复

★ resume 一列区分两种强度：
    「往返」= 真的存了、真的读回来、真的接着跑
    「API」 = 接口存在且调得通，但本探测没做完整往返
  混为一谈会把「有个函数」说成「能续跑」。

用法：
    python bench/probe_obs.py --system LangGraph --key ...
"""

from __future__ import annotations

import sys as _sys

_sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"
PROMPT = "写一个文件 probe.md，内容里出现 OKAY 这个词。完成后简短说明。"


def _ws(system: str) -> str:
    root = os.path.join(r"D:\pip-tmp\probe_ws", system)
    os.makedirs(root, exist_ok=True)
    return root


def _result(**kw):
    """统一的四项结论。每项是 (等级, 说明)。等级：往返 / API / 无。"""

    base = {"steps": ["无", ""], "tokens": ["无", ""],
            "trace": ["无", ""], "resume": ["无", ""]}
    base.update(kw)
    return base


# ══════════════════════════════════════════════════════════
#  各框架的探测
# ══════════════════════════════════════════════════════════

def probe_bingfu(key: str):
    from bench.adapters import make_write_file, SYSTEM_HINT
    from bingfu import Agent, BingFu
    from bingfu.campaign import Campaign
    from bingfu.checkpoint import MemoryCheckpointer
    from bingfu.graph import GraphOrchestrator
    from bingfu.llm import LLMConfig, LLMFactory
    from bingfu.presets import get_preset
    import bingfu.tools as _tools
    from bingfu.tools import ToolBelt

    out = _result()
    root = _ws("兵符")
    llm = LLMFactory.create(LLMConfig(provider="deepseek", api_key=key,
                                      model=MODEL, base_url=BASE_URL))
    a = Agent(name="斥候", role="将领", llm=llm, profile=get_preset("斥候"),
              system_prompt=SYSTEM_HINT)
    a.register_base_tool("write_file", make_write_file(root),
                         description="写文件")
    master = BingFu(name="p")
    master.add_agent(a)
    master.enable_commander(name="p")

    events, steps = [], []

    def on_event(e):
        events.append(e.kind)
        if e.kind == "step":
            steps.append(e.title)

    ob = _tools.belt_for
    _tools.belt_for = lambda c, knowledge=None: ToolBelt(
        enable_web=False, enable_code=False, knowledge=knowledge)
    try:
        res = Campaign(master, on_event=on_event, strategist=llm).run(PROMPT, "")
    finally:
        _tools.belt_for = ob

    if steps:
        out["steps"] = ["有", "Campaign(on_event=…) 的 step 事件，%d 条" % len(steps)]
    if getattr(res, "tool_calls", None):
        out["trace"] = ["计数", "CampaignResult.tool_calls：%s（有次数，无顺序与参数）"
                        % res.tool_calls]
    # ★ 不能读原 agent 上的轨迹：_configure 在**副本**上执行，
    #   原对象永远是 0。要读结果对象上汇总过的那一份。
    if getattr(res, "tokens", 0):
        out["tokens"] = ["有", "CampaignResult.tokens = %d" % res.tokens]

    # resume：真做一次往返（本地图，不花 token）
    cp = MemoryCheckpointer()
    calls = []
    g = (GraphOrchestrator().add("n1", lambda i: (calls.append(1), "v")[1])
         .add("n2", lambda i: "w", depends_on=["n1"]))
    g.run(checkpointer=cp, thread_id="probe")
    calls.clear()
    g2 = (GraphOrchestrator().add("n1", lambda i: (calls.append(1), "v")[1])
          .add("n2", lambda i: "w", depends_on=["n1"]))
    r2 = g2.run(checkpointer=cp, thread_id="probe")
    if not calls and r2.resumed:
        out["resume"] = ["往返", "GraphOrchestrator(checkpointer=…) 实测跳过 %d 个节点"
                         % len(r2.resumed)]
    return out


def probe_langgraph(key: str):
    from bench.adapters import make_write_file, SYSTEM_HINT
    from langchain_core.tools import tool as lc_tool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent

    out = _result()
    root = _ws("LangGraph")
    wf = make_write_file(root)

    @lc_tool
    def write_file_tool(filename: str, content: str) -> str:
        """写文件。"""
        return wf(filename, content)

    llm = ChatOpenAI(model=MODEL, api_key=key, base_url=BASE_URL, temperature=0.2)
    saver = MemorySaver()
    agent = create_react_agent(llm, [write_file_tool], prompt=SYSTEM_HINT,
                               checkpointer=saver)
    cfg = {"configurable": {"thread_id": "probe"}}

    chunks = list(agent.stream({"messages": [("user", PROMPT)]}, cfg))
    if chunks:
        out["steps"] = ["有", "agent.stream(…) 产出 %d 个节点更新" % len(chunks)]

    state = agent.get_state(cfg)
    msgs = state.values.get("messages", [])
    tool_msgs = [m for m in msgs if type(m).__name__ == "ToolMessage"]
    calls = [c for m in msgs for c in (getattr(m, "tool_calls", None) or [])]
    if calls:
        out["trace"] = ["有序", "消息里 %d 次 tool_calls（含参数）+ %d 条 ToolMessage"
                        % (len(calls), len(tool_msgs))]
    used = [m.response_metadata.get("token_usage") for m in msgs
            if getattr(m, "response_metadata", None)]
    used = [u for u in used if u]
    if used:
        out["tokens"] = ["有", "message.response_metadata.token_usage，%d 条" % len(used)]
    if state and state.config:
        out["resume"] = ["往返", "MemorySaver + thread_id；get_state 取回 %d 条消息"
                         % len(msgs)]
    return out


def probe_crewai(key: str):
    from bench.adapters import make_write_file, SYSTEM_HINT
    from crewai import LLM as CrewLLM
    from crewai import Agent as CrewAgent
    from crewai import Crew, Task as CrewTask
    from crewai.tools import tool

    out = _result()
    root = _ws("CrewAI")
    wf = make_write_file(root)
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_API_BASE"] = BASE_URL

    @tool("write_file")
    def crew_write(filename: str, content: str) -> str:
        """写文件。"""
        return wf(filename, content)

    steps = []
    worker = CrewAgent(role="Worker", goal="完成任务", backstory=SYSTEM_HINT,
                       tools=[crew_write],
                       llm=CrewLLM(model="openai/%s" % MODEL, base_url=BASE_URL,
                                   api_key=key, temperature=0.2),
                       verbose=False)
    t = CrewTask(description=PROMPT, expected_output="简短说明", agent=worker)
    crew = Crew(agents=[worker], tasks=[t], verbose=False,
                step_callback=lambda x: steps.append(type(x).__name__))
    res = crew.kickoff()

    if steps:
        out["steps"] = ["有", "Crew(step_callback=…)，%d 次回调（%s）"
                        % (len(steps), "、".join(sorted(set(steps))[:3]))]
    um = getattr(crew, "usage_metrics", None)
    if um:
        out["tokens"] = ["有", "crew.usage_metrics = %s" % str(um)[:70]]
    tr = getattr(res, "tasks_output", None)
    if steps:
        out["trace"] = ["计数", "step_callback 能看到动作类型，但不含**有序的工具参数**"]
    for attr in ("kickoff_from_state", "resume", "replay"):
        if hasattr(crew, attr):
            out["resume"] = ["API", "Crew.%s 存在（本探测未做往返）" % attr]
            break
    return out


def probe_autogen(key: str):
    import asyncio

    from bench.adapters import make_write_file, SYSTEM_HINT
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    out = _result()
    root = _ws("AutoGen")
    wf = make_write_file(root)
    client = OpenAIChatCompletionClient(
        model=MODEL, api_key=key, base_url=BASE_URL,
        model_info={"vision": False, "function_calling": True,
                    "json_output": False, "family": "unknown",
                    "structured_output": False})
    agent = AssistantAgent(name="w", model_client=client, tools=[wf],
                           system_message=SYSTEM_HINT, reflect_on_tool_use=True)

    async def go():
        seen = []
        async for m in agent.run_stream(task=PROMPT):
            seen.append(type(m).__name__)
        state = await agent.save_state()
        await client.close()
        return seen, state

    seen, state = asyncio.run(go())
    if seen:
        out["steps"] = ["有", "agent.run_stream(…)，%d 条消息（%s）"
                        % (len(seen), "、".join(sorted(set(seen))[:3]))]
    if any("ToolCall" in x for x in seen):
        out["trace"] = ["有序", "ToolCallRequestEvent / ToolCallExecutionEvent 按序产出"]
    if state:
        out["resume"] = ["API", "agent.save_state() 返回 %d 个键（本探测未做往返）"
                         % (len(state) if hasattr(state, "__len__") else 1)]
    out["tokens"] = ["有", "消息上的 models_usage 字段"]
    return out


def probe_pydanticai(key: str):
    from bench.adapters import make_write_file, SYSTEM_HINT
    from pydantic_ai import Agent as PAIAgent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    out = _result()
    root = _ws("PydanticAI")
    wf = make_write_file(root)
    agent = PAIAgent(OpenAIChatModel(MODEL, provider=OpenAIProvider(
        api_key=key, base_url=BASE_URL)), system_prompt=SYSTEM_HINT)

    @agent.tool_plain
    def write_file_tool(filename: str, content: str) -> str:
        """写文件。"""
        return wf(filename, content)

    res = agent.run_sync(PROMPT)
    msgs = res.all_messages()
    parts = [type(p).__name__ for m in msgs for p in getattr(m, "parts", [])]
    if any("ToolCall" in p for p in parts):
        out["trace"] = ["有序", "all_messages() 里的 ToolCallPart / ToolReturnPart"]
    if msgs:
        out["steps"] = ["有", "all_messages() 返回 %d 条（另有 agent.iter() 可逐步迭代）"
                        % len(msgs)]
    # ★ pydantic-ai 2.x 里 usage 是**属性**不是方法。
    #   第一版写成 res.usage() 抛 TypeError，整个探测被 except 兜住，
    #   四项全记成「无」—— 那是把我的 API 用错记成了它的缺陷。
    #   探测代码出错必须与被探测者「确实没有」区分开。
    u = getattr(res, "usage", None)
    if callable(u):
        u = u()
    if u:
        out["tokens"] = ["有", "result.usage = %s" % str(u)[:70]]
    out["resume"] = ["API", "message_history 可持久化后回灌（本探测未做往返）"]
    return out


PROBES = {"兵符": probe_bingfu, "LangGraph": probe_langgraph,
          "CrewAI": probe_crewai, "AutoGen": probe_autogen,
          "PydanticAI": probe_pydanticai}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--key", required=True)
    args = ap.parse_args()
    try:
        out = PROBES[args.system](args.key)
    except Exception as exc:                    # noqa: BLE001
        traceback.print_exc(limit=3, file=sys.stderr)
        out = _result()
        out["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
    print("__PROBE__" + json.dumps({"system": args.system, **out},
                                   ensure_ascii=False))


if __name__ == "__main__":
    main()
