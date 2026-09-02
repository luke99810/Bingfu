r"""战役 —— 一道军令从下达到复命的完整链路。

════════════════════════════════════════════════════════════════
 这一层补的是什么
════════════════════════════════════════════════════════════════

评估引擎、点兵台、编排图、运行时，这些零件此前都在，但**没有一处
把它们串成一条可以从外面看见的链路**。使用者输入一句话，等一会儿，
拿到一段文本 —— 中间发生了什么、谁在做、做到哪一步，全都不可见。

而这个框架的主张恰恰是「庙算」：先量敌我，再决定怎么打。量了不给人看，
等于没量。

因此这里定义一条有明确阶段的流程，每个阶段都发出事件：

    军令 → 庙算(敌方战力) → 点将(我方战力) → 交兵 → 复命
                                              ↑
                                          军师随时进言

★ 事件是**回调**，不是打印。

  控制台要把它渲染成界面，命令行要把它打成文字，实验要把它收集成数据。
  在这里 print 会让后两者没法用。

════════════════════════════════════════════════════════════════
 三条不肯让步的地方
════════════════════════════════════════════════════════════════

★ **我方战力取自真实档案，不是现编的。**

  `CombatStats.total_power` 由五维属性算出，属性写在预设里。
  如果这个数字可以为了「好看」而调整，整套庙算就退化成一场表演。

★ **「有没有真的动手」必须单独可见。**

  一次执行如果一个工具都没调用，它产出的东西全部来自模型的既有知识。
  那可能仍然有用，但它不是「完成了任务」——
  这两者在最终那段文本上看起来一模一样，所以必须在别处标出来。

★ **军师只进言，不改结果。**

  辅佐是给使用者看的判断依据，不是偷偷修改执行输出。
  一个会悄悄改写结果的「建议者」，会让人分不清哪句话是谁说的。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .i18n import t as tr
from .assessment import TaskAssessment, TaskAssessor
from .matcher import MatchResult, TaskMatcher


# ══════════════════════════════════════════════════════════
#  事件
# ══════════════════════════════════════════════════════════

@dataclass
class CampaignEvent:
    """战役过程中的一个可观测节点。

    ``kind`` 取值：
        order    军令已受理
        assess   庙算完毕，敌方战力已估
        muster   点将完毕，我方战力已定
        verdict  敌我对比的结论
        march    某个子任务开始
        report   某个子任务复命
        step     执行中的一步（思考轮次 / 工具调用）
        advise   军师进言
        done     全军复命
        fail     战役中止
    """

    kind: str
    title: str
    detail: str = ""
    #: 结构化载荷，供界面取数（敌方战力、我方战力、进度等）
    data: Dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)


EventSink = Callable[[CampaignEvent], None]


# ══════════════════════════════════════════════════════════
#  结果
# ══════════════════════════════════════════════════════════

@dataclass
class CampaignResult:
    """一次战役的完整结果。"""

    task: str
    output: str
    assessment: Optional[TaskAssessment] = None
    #: 子任务 id -> 将领名
    assignments: Dict[str, str] = field(default_factory=dict)
    #: 参战将领的战力合计
    our_power: int = 0
    #: 各将领的匹配评分
    matches: List[MatchResult] = field(default_factory=list)
    #: 工具调用统计（谁动了手、动了几次）
    tool_calls: Dict[str, int] = field(default_factory=dict)
    #: 本次战役累计消耗的 token（含庙算、拆解与各子任务执行）
    tokens: int = 0
    events: List[CampaignEvent] = field(default_factory=list)
    elapsed: float = 0.0
    ok: bool = True
    error: str = ""

    @property
    def enemy_power(self) -> int:
        return self.assessment.enemy_power if self.assessment else 0

    @property
    def took_action(self) -> bool:
        """有没有真的动过手（调用过工具）。

        ★ 与 ``ok`` 是两件事：一次没调用任何工具的执行也可能
          「成功」返回一段文本 —— 但那段文本没有触碰过这台电脑。
        """

        return any(v > 0 for v in self.tool_calls.values())

    @property
    def power_ratio(self) -> float:
        e = self.enemy_power
        return (self.our_power / e) if e else 0.0

    def summary(self) -> str:
        lines = [
            f"敌方战力 ≈{self.enemy_power}｜我方战力 {self.our_power}"
            f"（{self.power_ratio:.2f}×）",
            f"参战：{'、'.join(sorted(set(self.assignments.values()))) or '无'}",
            f"用时 {self.elapsed:.1f}s",
        ]
        if self.tool_calls:
            items = sorted(self.tool_calls.items(), key=lambda kv: -kv[1])
            lines.append("动手：" + "、".join(f"{k}×{v}" for k, v in items))
        else:
            lines.append("动手：未调用任何工具（输出全部来自模型既有知识）")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════
#  战役
# ══════════════════════════════════════════════════════════

class Campaign:
    """把一道军令走完全程。

    Args:
        bingfu: BingFu 实例（持有将领与 commander）
        on_event: 每个阶段的回调。界面据此实时渲染。
        strategist: 军师用的 LLM。留空则复用 bingfu 的 LLM。
        checkpointer: 断点存储。留空则不存 —— 崩了只能从头再来。
    """

    def __init__(self, bingfu: Any, *, on_event: Optional[EventSink] = None,
                 strategist: Optional[Any] = None,
                 checkpointer: Optional[Any] = None) -> None:
        self.bingfu = bingfu
        self.on_event = on_event
        self.strategist = strategist or getattr(bingfu, "llm", None)
        self.checkpointer = checkpointer
        self.events: List[CampaignEvent] = []

    @staticmethod
    def thread_id_for(task: str) -> str:
        """同一道军令对应同一个断点槽位。

        ★ 用任务文本的哈希，不用时间戳 —— 时间戳每次都不同，
          那样每次都是新槽位，断点永远命中不了自己。
        """

        import hashlib
        return hashlib.sha256((task or "").strip().encode("utf-8")).hexdigest()[:16]

    # ── 事件 ────────────────────────────────────────────

    def _emit(self, kind: str, title: str, detail: str = "", **data: Any) -> None:
        ev = CampaignEvent(kind=kind, title=title, detail=detail, data=data)
        self.events.append(ev)
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                # ★ 界面回调出错不该让战役中止 —— 渲染失败是渲染的事。
                pass

    # ── 军师 ────────────────────────────────────────────

    def _advise(self, stage: str, context: str) -> str:
        """军师进言。失败时静默跳过。

        ★ 军师不可用不影响战役。

          它提供的是判断依据，不是执行能力。让一个辅助角色的故障
          阻断主流程，是把可选件变成了单点。
        """

        # ★ 没人听就不说 —— 也就不花那次调用。
        #
        #   进言是**给界面看的旁白**，不参与任何决策。没有挂事件槽时
        #   （无头运行、跑基准）它唯一的消费者不存在，而一个到不了
        #   任何人那里的产出不该计费。这与本项目里另外两处同形：
        #   一条没人走的 import 收了 32 秒启动费；
        #   一条从未真正执行过的 DAG 分支在测试里显示一切正常。
        if not self.strategist or self.on_event is None:
            return ""
        try:
            from bingfu.llm.base import LLMMessage, RoleType

            # ★ 这里原先调的是 self.strategist.chat(...)，
            #   而 LLM 供应商上根本没有 chat —— 抛出的 AttributeError
            #   被下面的 except 吞掉，于是**军师从来没有开过口**。
            #   界面上只是少了一段话，没有任何地方报错。
            #   静默降级把「坏了」演成了「这一步没什么可说的」。
            resp = self.strategist.generate([
                LLMMessage(role=RoleType.SYSTEM, content=(
                    "你是军师，用兵法口吻给出简短进言。"
                    "只讲这一步的关键判断与风险，两三句话，不要复述已知信息。")),
                LLMMessage(role=RoleType.USER,
                           content="【%s】\n%s" % (stage, context)),
            ])
            text = getattr(resp, "content", None) or str(resp)
            return text.strip()
        except Exception:
            return ""

    # ── 主流程 ──────────────────────────────────────────

    def run(self, task: str, requirements: str = "",
            *, max_workers: int = 4) -> CampaignResult:
        """走完一道军令。"""

        t0 = time.time()
        self.events = []

        full_task = task if not requirements.strip() else (
            f"{task}\n\n【要求】\n{requirements.strip()}")

        result = CampaignResult(task=full_task, output="")

        self._emit("order", tr("campaign.order"), full_task[:400],
                   task=task, requirements=requirements)

        # ── 一、庙算：敌方战力 ──────────────────────────
        try:
            assessor = TaskAssessor(llm_provider=self.strategist)
            # ★ 庙算走规则版，不花调用。
            #
            #   它唯一的行为后果是「拆不拆」这个闸门，而闸门现在读的是
            #   任务结构，不是复杂度分（见 assessment.needs_orchestration）。
            #   界面上要显示的敌方战力与理由，规则版一样给得出，
            #   而且**确定性**：同一段军令永远得到同一份庙算，
            #   于是「为什么这么打」可复算。
            assessment = assessor.assess_fast(full_task)
        except Exception as exc:
            result.ok = False
            result.error = f"庙算失败：{exc}"
            self._emit("fail", tr("campaign.assess_failed"), str(exc))
            result.elapsed = time.time() - t0
            result.events = list(self.events)
            return result

        result.assessment = assessment
        self._emit(
            "assess", f"庙算：敌方战力 ≈{assessment.enemy_power}",
            assessment.reasoning,
            enemy_power=assessment.enemy_power,
            complexity=assessment.complexity_score,
            capabilities=list(assessment.required_capabilities),
        )

        adv = self._advise(
            "庙算",
            f"任务：{task}\n难度 {assessment.complexity_score}/10，"
            f"敌方战力 ≈{assessment.enemy_power}，"
            f"需要的能力：{'、'.join(assessment.required_capabilities)}")
        if adv:
            self._emit("advise", tr("campaign.advise_assess"), adv)

        # ── 二、点将：我方战力 ──────────────────────────
        agents = dict(getattr(self.bingfu, "agents", {}) or {})
        if not agents:
            result.ok = False
            result.error = "帐下无将"
            self._emit("fail", tr("campaign.muster_failed"), "帐下无将，无法出兵")
            result.elapsed = time.time() - t0
            result.events = list(self.events)
            return result

        matcher = TaskMatcher(assessor=assessor)
        try:
            # match() 收的是 {name: Agent} 字典，不是列表
            matches = matcher.match(full_task, agents, assessment=assessment)
        except Exception as exc:
            # ★ 不静默吞掉。
            #
            #   第一版写的是 `except Exception: matches = []`——
            #   于是点将阶段整个消失，而流程照常往下走，
            #   外面看到的是「这一步好像没有发生」，
            #   而不是「这一步失败了，原因是 X」。
            #   这正是这个框架其余地方反复在防的那种失效。
            matches = []
            self._emit("advise", tr("campaign.muster_score_failed"),
                       f"{type(exc).__name__}: {exc}\n"
                       f"将改由编排层按任务类型直接点将。")
        result.matches = matches

        if matches:
            self._emit(
                "muster", tr("campaign.muster"),
                "\n".join(f"{m.agent_name}  {m.total_score:.2f}  {m.reasoning}"
                          for m in matches[:6]),
                ranking=[{"name": m.agent_name, "score": round(m.total_score, 3)}
                         for m in matches],
            )

        # ── 三、交兵：拆解 → 分派 → 执行 ────────────────
        self._emit("march", tr("campaign.march"), "")

        try:
            from .orchestration import orchestrate

            def _on_step(ev: Dict[str, Any]) -> None:
                who = ev.get("agent", "")
                sid = ev.get("subtask", "")
                # ★ ev 里也有一个 "kind"（think/tool），会和 _emit 的
                #   位置参数撞名 —— 直接 **ev 会抛 TypeError，
                #   而它发生在工作线程里，最外层一个 except 就吞了，
                #   表现是「进度一条都不出来」而不是报错。
                payload = {k: v for k, v in ev.items() if k != "kind"}
                payload["step_kind"] = ev.get("kind", "")
                if ev.get("kind") == "tool":
                    ok = ev.get("ok", True)
                    self._emit(
                        "step",
                        f"{'🔧' if ok else '⚠️'} {who} · {ev.get('name','')}",
                        f"{ev.get('args','')}\n→ {ev.get('result','')}",
                        **payload)
                elif ev.get("kind") == "think":
                    # ★ 思考轮次单独一类，界面可以选择不展开。
                    #   「第 3 轮」本身信息量很低，但轮数持续上涨
                    #   而工具调用为零，是「在原地打转」的唯一信号。
                    self._emit("step", f"💭 {who} 第 {ev.get('iteration','?')} 轮"
                                       + (f"（{sid}）" if sid else ""), "", **payload)

            orch = orchestrate(
                full_task, agents,
                llm=self.strategist, matcher=matcher, max_workers=max_workers,
                on_step=_on_step,
                assessment=assessment,
                checkpointer=self.checkpointer,
                thread_id=self.thread_id_for(full_task),
            )
            # ★ 续跑与否要说出来。
            #   悄悄跳过几个节点，与"这次跑得特别快"从外面分不清。
            ow = getattr(orch, "overwrite_note", "")
            if ow:
                self._emit("advise", tr("campaign.overwrite"), ow)
            note = getattr(getattr(orch, "graph", None), "resume_note", "")
            if note:
                self._emit("advise", tr("campaign.checkpoint"), note)
            result.output = orch.output
            result.assignments = dict(orch.assignments)
            result.tokens = getattr(orch, "tokens", 0)

            # ★ 拆解质量要说出来。
            #
            #   实测三个真实任务全被拆成一条链，parallel_width 恒为 1 ——
            #   也就是「同层并行」这个核心能力一次都没被用上，
            #   而从结果上完全看不出来。不报出来的话，
            #   这个框架可以永远宣称自己在做并行编排。
            plan = getattr(orch, "plan", None)
            if plan is not None:
                width = getattr(plan, "parallel_width", 0)
                n = len(getattr(plan, "subtasks", []) or [])
                self._emit(
                    "advise", tr("campaign.plan_shape"),
                    tr("campaign.plan_shape_detail",
                       n=n, edges=getattr(plan, "edge_count", 0), width=width),
                    subtasks=n, parallel_width=width,
                    layers=getattr(plan, "layers", []))
                for issue in list(getattr(plan, "issues", []) or []):
                    self._emit("advise", tr("campaign.plan_issue"), str(issue))
        except Exception as exc:
            result.ok = False
            result.error = f"执行失败：{exc}"
            self._emit("fail", tr("campaign.march_failed"), str(exc))
            result.elapsed = time.time() - t0
            result.events = list(self.events)
            return result

        # 我方战力 = 参战将领档案战力之和
        #
        # ★ 只算**真正参战**的，不是帐下所有人。
        #   把没出兵的将领也算进去，我方战力就永远碾压敌方，
        #   这个数字随即失去意义。
        engaged = sorted(set(result.assignments.values()))
        our = 0
        for name in engaged:
            ag = agents.get(name)
            prof = getattr(ag, "profile", None)
            stats = getattr(prof, "stats", None)
            if stats is None:
                continue
            # ★ total_power 是**方法**不是属性。
            #   写成属性读到的是 bound method，int() 一转就炸；
            #   而如果外面包了 try 兜住，它会变成「我方战力恒为 0」——
            #   一个不报错但永远错的数字。
            tp = getattr(stats, "total_power", None)
            try:
                our += int(tp() if callable(tp) else (tp or 0))
            except (TypeError, ValueError):
                pass
        result.our_power = our

        # 工具调用统计
        #
        # ★ 从编排结果取，不从将领对象取。
        #
        #   编排层执行的是将领的**副本**（同层并行时不复制会互相覆盖配置），
        #   轨迹留在副本上。去读原对象只会读到空 —— 那是个假阴性：
        #   明明写出了文件，却报告「未调用任何工具」。
        #   我第一版就是这么写的，实测时它对着一份真实生成的报告
        #   说「没有触碰过这台电脑」。
        result.tool_calls = dict(getattr(orch, "tool_calls", {}) or {})

        for sub_id, who in result.assignments.items():
            self._emit("report", f"{who} 复命", f"子任务 {sub_id}",
                       subtask=sub_id, agent=who)

        # ── 四、复命 ────────────────────────────────────
        ratio = result.power_ratio
        if ratio >= 1.5:
            verdict = "我众敌寡，可正面强攻"
        elif ratio >= 1.0:
            verdict = "势均力敌，宜稳扎稳打"
        elif ratio > 0:
            verdict = "敌众我寡，宜用奇兵、分而击之"
        else:
            verdict = "敌情未明"

        self._emit(
            "verdict", f"庙算结论：{verdict}",
            f"敌 ≈{result.enemy_power}｜我 {result.our_power}"
            f"（{ratio:.2f}×）",
            enemy_power=result.enemy_power, our_power=result.our_power,
            ratio=round(ratio, 3), verdict=verdict,
        )

        # ★ 「没动手」要单独说出来，不能混在成功里。
        if not result.took_action:
            self._emit(
                "advise", tr("campaign.no_tools"),
                "输出全部来自模型的既有知识，没有触碰过这台电脑上的文件。"
                "如果这道军令本应产生实际改动，说明将领没有可用的兵器 —— "
                "检查启动时是否给将领装配了工具。")

        adv = self._advise(
            "复命",
            f"任务：{task}\n结论：{verdict}\n"
            f"参战：{'、'.join(engaged)}\n"
            f"是否动手：{'是' if result.took_action else '否'}\n"
            f"产出摘要：{result.output[:600]}")
        if adv:
            self._emit("advise", tr("campaign.advise_report"), adv)

        result.elapsed = time.time() - t0
        result.events = list(self.events)
        self._emit("done", tr("campaign.done"), result.summary(),
                   elapsed=round(result.elapsed, 2))
        return result
