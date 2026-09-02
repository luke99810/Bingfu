r"""Graph 编排 —— 多智能体的拓扑执行。

════════════════════════════════════════════════════════════════
 这一层是什么（以及先前它有多假）
════════════════════════════════════════════════════════════════

★ Graph 不是"一个没人用的 DAG 执行器加一张路由查表"。

  它是**多智能体真正发生的地方**：把一个任务拆成若干子任务、
  为每个子任务点将、按依赖关系执行、把结果汇总成一份交付。

★ 先前 Commander 的"多智能体协作"实际是：

      round_robin  把**同一个任务**原样广播给每一个将领
      priority     与 round_robin **行为完全相同**
                   （多建了个没用的 priority_list，enumerate 的 i 从未使用）
      smart        选一个将领单独跑

  三个策略只有两种行为，且都不含拆解、依赖、交接、汇总中的任何一项。
  多个将领各自对着同一段文字重复劳动一遍，然后把 N 份互不相关的
  答案放进一个字典 —— 这是广播，不是协作。

★ 与 tactic 那次的教训相同：**一个名字叫"协调"的方法，
  不会因为名字而真的在协调。** 判据是它产出的结构 ——
  有没有子任务、有没有边、有没有把上游产物喂给下游。

════════════════════════════════════════════════════════════════
 拆解从哪来
════════════════════════════════════════════════════════════════

两条路径，按是否有 LLM 分流：

  · 有 LLM：让主帅把任务拆成带依赖的子任务清单（结构化输出，
    经 Harness 的五级降级保证拿到合法结构，拿不到就**抛错**而不是
    退回一个编造的默认拆解）
  · 无 LLM：退回单节点图 —— 即"不拆"，而不是假装拆了

★ 第二条很重要：拆不了就诚实地不拆。
  先前那种"退回 round_robin 广播"的写法，会让调用方以为
  发生了协作，而实际只是同一个任务跑了 N 遍、成本翻 N 倍。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from bingfu.graph import GraphOrchestrator, GraphResult, NodeStatus


# ══════════════════════════════════════════════════════════
#  子任务
# ══════════════════════════════════════════════════════════

@dataclass
class SubTask:
    """一个可以指派给某个将领的子任务。

    ``depends_on`` 是**这一层全部价值的来源**：没有边的图就是一堆
    互不相干的任务，那和广播没有区别。
    """

    id: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    #: 指定将领；留空则由点兵台按任务匹配
    agent_name: Optional[str] = None
    #: 任务类型（Code / IR / Data / Write / Reason）。
    #:
    #: ★ 它决定这个子任务用哪条流水线：验收开不开、给什么工具、
    #:   输出上限多少。原本这张路由表只有实验在用，
    #:   框架自身跑任务时一律走同一套配置 ——
    #:   而实测分类别差异极大（检索类 12/12、代码类 3/12）。
    category: str = ""


@dataclass
class Plan:
    """一次拆解的结果。"""

    subtasks: List[SubTask]
    #: 拆解由谁产生：llm / single / manual
    source: str = "manual"
    #: 校验发现的问题（PlanIssue 列表）。
    #:
    #: ★ 空列表与"没校验过"是两回事，但这里不区分 —— 因为
    #:   decompose 现在总会校验。手工构造的 Plan 留空是合理的。
    issues: List[Any] = field(default_factory=list)

    @property
    def is_trivial(self) -> bool:
        """只有一个节点、没有边 —— 也就是"没有真的拆"。"""

        return len(self.subtasks) <= 1

    @property
    def edge_count(self) -> int:
        return sum(len(t.depends_on) for t in self.subtasks)

    @property
    def layers(self) -> List[List[str]]:
        """按依赖分层 —— 同一层可以并行。"""

        from bingfu.plan_check import layers_of
        return layers_of(self.subtasks)

    @property
    def parallel_width(self) -> int:
        """最宽的一层有几个节点。

        ★ 1 表示**彻底退化成链表**：拆是拆了，但同层并行一次都用不上。
          实测三个真实任务的拆解全是链 —— 这个属性把它变成可观察的。
        """

        from bingfu.plan_check import parallel_width
        return parallel_width(self.subtasks)


#: 拆解任务时给主帅的结构约定
DECOMPOSE_SCHEMA = ("subtasks",)

_DECOMPOSE_PROMPT = """你是主帅，需要把一个任务拆成若干可以分派给不同将领的子任务。

任务：{task}

可用将领及其擅长：
{agents}

要求：
1. 拆成 2-5 个子任务。如果任务本身很简单、不值得拆，就只给 1 个

2. 用 depends_on 标明依赖：某个子任务**必须**拿到另一个的产出才能开始。
   只有真正的先后约束才写依赖，"顺手放在后面"不算。

3. **上游的产出会由系统自动作为输入交给下游**，你不需要、也不应该
   约定任何中间文件来传递结果。
   ✗ 错误示范：t2 写「读取 total.txt（由 t1 生成）」—— 那个文件不存在，
     t1 从来没有被要求创建它，t2 只能失败或者编一个数字出来。
   ✓ 正确写法：t2 写「根据上游给出的总销量，写一份说明」。

4. **尽量让互不依赖的子任务并列**，它们会被同时执行。
   全部串成一条链（t1→t2→t3→t4）是最差的拆法：那等于没有并行，
   只是把一件事切成了几段顺序做。
   ✓ 好的形状：t1 先做调研；t2、t3 各自基于 t1 并列展开；t4 汇总 t2 与 t3。

5. **一个子任务要是一件值得单独交付的事，不是一次文件读写。**
   ✗ 错误示范：t1「读取 a.txt」、t2「读取 b.txt」、t3「汇总」——
     这是把微操当成了战略。三个子任务各跑一套独立流程，
     而且 t3 多半会把 a、b 再读一遍，做了三遍活。
   ✓ 正确写法：t1「读取全部素材文件并整理出清单」、t2「基于清单撰写结论」。

6. 子任务描述要**自包含**——执行它的将领看不到原始任务，
   但**会看到上游的产出**。**上游读过的内容会随产出一起交给你，
   不需要重新读一遍。**

7. 给每个子任务标注类型 category，取值只能是以下之一：
   Code（写代码）/ IR（查资料）/ Data（数据分析）/ Write（写文章）/ Reason（推理论证）

只输出 JSON：
{{"subtasks": [
  {{"id": "t1", "description": "……", "depends_on": [], "agent_name": "韩信", "category": "IR"}},
  {{"id": "t2", "description": "……", "depends_on": ["t1"], "agent_name": null, "category": "Write"}}
]}}"""


#: 关键词 → 类型。模型没标注或标错时的兜底。
#:
#: ★ 这是**兜底**，不是主路径。写成一张表而不是让模型再判一次，
#:   是因为再问一次模型既慢又不可复现 —— 而这里只需要一个
#:   合理的默认值，不需要精确。
_CATEGORY_HINTS = (
    ("Code", ("代码", "函数", "脚本", "实现", "接口", "api", "程序", "调试", "重构")),
    # ★ 这里原本有一个关键词"计算"，它在"量子计算"里造成假阳性 ——
    #   一条明确写着"调研…现状"的检索任务被判成了数据分析。
    #   过泛的词在关键词匹配里是负资产：它带来的误判
    #   远多于它能捞回的漏判。
    ("Data", ("数据", "统计", "指标", "图表", "可视化", "算一下", "求和", "均值")),
    ("IR", ("调研", "检索", "搜索", "查找", "收集", "资料", "情报", "现状")),
    ("Write", ("撰写", "文章", "报告", "文案", "总结", "说明")),
    ("Reason", ("推理", "论证", "原因", "为什么", "评估", "判断", "权衡")),
)

_VALID_CATEGORIES = {"Code", "IR", "Data", "Write", "Reason"}


def infer_category(text: str) -> str:
    """从子任务描述猜类型。命中不了返回空字符串。

    ★ 猜不出来时返回空，而不是硬塞一个默认类型。

      空字符串会让 route_for 走"未知类别"的中庸配置，
      那是一个诚实的"不知道"；而硬塞成 Write 之类，
      会让一个写代码的子任务拿不到解释器，
      且没有任何地方能看出发生了误判。
    """

    # ★ 按命中数计分，而不是先到先得。
    #
    #   先到先得意味着**表的顺序决定结果** —— 一条同时含
    #   "调研"和"计算"的描述，会因为 Data 恰好排在 IR 前面
    #   被判成数据分析。顺序是实现细节，不该有语义。
    low = (text or "").lower()
    scored = [
        (sum(1 for k in keywords if k in low), category)
        for category, keywords in _CATEGORY_HINTS
    ]
    best_score, best_category = max(scored)
    return best_category if best_score else ""


def decompose(
    task: str,
    agents: Dict[str, Any],
    llm: Optional[Any] = None,
    *,
    prompt_template: Optional[str] = None,
) -> Plan:
    """把任务拆成带依赖的子任务图。

    ★ 拿不到合法拆解时**退回单节点**，而不是编一个。

      一个编造的拆解会产生看起来很像协作的执行轨迹
      （多个节点、多份产出），而那些子任务与原任务的关系
      是随机的 —— 比不拆更糟，因为它同时浪费了成本又污染了结果。
    """

    if llm is None or not agents:
        return Plan([SubTask(id="t1", description=task,
                         category=infer_category(task))], source="single")

    from bingfu.harness import AgentHarnessFailure, call_structured
    from bingfu.llm.base import LLMMessage, RoleType

    roster = "\n".join(
        f"- {name}：{getattr(getattr(a, 'profile', None), 'specialties', None) or a.role or '通用'}"
        for name, a in agents.items()
    )
    # ★ 允许覆盖模板，是为了能做**提示词的 A/B 对照**。
    #
    #   改了提示词就宣称「拆解变好了」，与在噪声里读故事没有区别 ——
    #   本项目已经犯过一次那种错（见 graph.py 里 Fisher 检验那段）。
    #   要说变好，得能把新旧两版跑在同一批任务上比。
    prompt = (prompt_template or _DECOMPOSE_PROMPT).format(
        task=task, agents=roster)

    #: 校验发现的问题（调用方可以读，界面据此告警）
    issues_found: List[Any] = []

    def _gen(extra: str = "") -> str:
        resp = llm.generate(
            messages=[LLMMessage(role=RoleType.USER, content=prompt + extra)],
            temperature=0.2,
        )
        return resp.content or ""

    def _single() -> Plan:
        return Plan([SubTask(id="t1", description=task,
                             category=infer_category(task))], source="single")

    def _attempt(extra: str = ""):
        """跑一次拆解，返回 (Plan 或 None)。"""

        try:
            result = call_structured(lambda **kw: _gen(extra),
                                     required=DECOMPOSE_SCHEMA,
                                     max_regenerate=1)
        except AgentHarnessFailure:
            return None
        return _build(result)

    def _build(result):
        # ★ 是 .output 不是 .data。
        #
        #   call_structured 返回 HarnessResult，解析结果放在 output 里，
        #   根本没有 data 这个属性。于是**拆解成功时这行必抛 AttributeError**，
        #   而异常被上层 except 兜住、变成一句"执行失败：…object has no
        #   attribute 'data'"。
        #
        #   它能长期不被发现，是因为拆解**失败**时会走上面的 fallback 返回
        #   单子任务 —— 那条路不碰这一行。也就是说：结构化拆解越成功，
        #   崩得越准；而单元测试里 LLM 是打桩的，桩多半返回不了合规 JSON，
        #   于是永远走 fallback，永远绿。
        #
        #   后果是多子任务的 DAG 并行在战役路径上**从未真正执行过**。
        raw = ((result.output if isinstance(result.output, dict) else {})
               .get("subtasks") or [])
        subtasks: List[SubTask] = []
        seen = set()
        for i, item in enumerate(raw, 1):
            if not isinstance(item, dict):
                continue
            tid = str(item.get("id") or f"t{i}")
            if tid in seen:
                continue
            seen.add(tid)
            desc = str(item.get("description") or "").strip() or task
            raw_cat = str(item.get("category") or "").strip()
            category = raw_cat if raw_cat in _VALID_CATEGORIES else infer_category(desc)
            subtasks.append(SubTask(
                id=tid,
                description=desc,
                depends_on=[str(d) for d in (item.get("depends_on") or [])],
                agent_name=item.get("agent_name") or None,
                category=category,
            ))

        if not subtasks:
            return None

        # ★ 稳定 id：按描述内容哈希，不用模型随口给的 t1/t2。
        #
        #   断点签名由「节点名 + 依赖」构成。模型每次换一批 id 的话，
        #   同一道军令重跑时签名必然对不上，断点直接作废 ——
        #   表现是「存了但从来没续上过」，看不出哪里错。
        from bingfu.plan_check import stable_subtask_id
        remap = {t.id: stable_subtask_id(t.description) for t in subtasks}
        if len(set(remap.values())) == len(remap):      # 描述重复时不改，免得撞 id
            for t in subtasks:
                t.depends_on = [remap.get(d, d) for d in t.depends_on]
            for t in subtasks:
                t.id = remap[t.id]

        # 丢掉指向不存在节点的依赖 —— 否则拓扑排序会直接抛错，
        # 而模型偶尔会引用一个它没有生成的 id
        valid = {t.id for t in subtasks}
        for t in subtasks:
            dropped = [d for d in t.depends_on if d not in valid]
            t.depends_on = [d for d in t.depends_on if d in valid]
            if dropped:
                t.description += f"\n（注：原计划依赖 {dropped}，该依赖不存在，已忽略）"

        return Plan(subtasks, source="llm")

    # ══════════════════════════════════════════════════════
    #  拆 → 校验 → 有结构性错误就带着问题重拆一次
    # ══════════════════════════════════════════════════════
    #
    # ★ 只对 ERROR 重拆，WARN 照跑但记下来。
    #
    #   成环、依赖不存在这类是硬错误，跑下去只会得到一个
    #   悄悄少了几个节点的结果。而「拆成一条链」是坏味道不是错误 ——
    #   重拆未必更好，把它说出来比替模型做决定更诚实。
    from bingfu.plan_check import has_error, issues_as_feedback, validate_plan

    plan = _attempt()
    if plan is None:
        return _single()

    issues = validate_plan(plan)
    if has_error(issues):
        retry = _attempt(chr(10) * 2 + issues_as_feedback(issues))
        if retry is not None:
            retry_issues = validate_plan(retry)
            # 重拆之后仍有硬错误就用重拆的结果（两害相权，至少是最新一次）
            plan, issues = retry, retry_issues

    plan.issues = list(issues)
    return plan



# ══════════════════════════════════════════════════════════
#  执行
# ══════════════════════════════════════════════════════════

@dataclass
class OrchestrationResult:
    """一次多智能体执行的完整结果。"""

    output: str
    plan: Plan
    graph: GraphResult
    assignments: Dict[str, str] = field(default_factory=dict)
    #: 本次执行里每个工具被调用了几次（跨全部子任务累计）
    #:
    #: ★ 必须在这一层收集。
    #:
    #:   `_configure` 会把将领**复制**一份再执行（同层并行时不复制就会
    #:   互相覆盖配置）。于是轨迹留在副本上，调用方去读原对象只会读到空 ——
    #:   那是个假阴性：明明动了手，却报告「未调用任何工具」。
    #:   一个会误报的护栏比没有护栏更糟，因为人会照着它下结论。
    tool_calls: Dict[str, int] = field(default_factory=dict)
    #: 本次执行累计消耗的 token。
    #:
    #: ★ 这个字段是可观测性探测逼出来的：跨框架对比时
    #:   LangGraph / CrewAI / AutoGen / PydanticAI 都能报出自己的用量，
    #:   唯独兵符报不出 —— 一个把「可审计」当主张的框架，
    #:   却让调用方无从知道这次花了多少。
    #:
    #: ★ 与 tool_calls 同一处、同一把锁汇总：轨迹留在 _configure 的
    #:   副本上，函数返回后副本就没人引用了，事后去读原对象只会读到 0。
    tokens: int = 0
    #: 冗余熔断命中次数 —— 省下的调用要看得见，否则无从判断它有没有用
    breaker_hits: int = 0
    #: 同一产出被反复覆盖的次数 —— 拆解重叠的信号
    overwrites: int = 0
    #: 覆盖情况的人话说明，空串表示没有
    overwrite_note: str = ""

    @property
    def took_action(self) -> bool:
        """这次执行有没有真的调用过工具。

        ★ 与 ok 是两件事：一次工具都没调的执行也可能「成功」返回文本，
          但那段文本没有触碰过任何外部状态。
        """

        return any(v > 0 for v in self.tool_calls.values())

    @property
    def parallel_width(self) -> int:
        return self.graph.parallel_width

    @property
    def is_real_collaboration(self) -> bool:
        """★ 自查：这次执行到底有没有在协作。

        判据是**结构**，不是意图：
          · 多于一个子任务
          · 至少存在一条依赖边（否则只是并行的独立任务）

        写成一个属性，是为了让"退化成广播"这件事可被观察 ——
        先前那个 round_robin 没有任何地方能看出它没在协作。
        """

        return len(self.plan.subtasks) > 1 and self.plan.edge_count > 0


def _pick_agent(sub: SubTask, agents: Dict[str, Any], matcher: Optional[Any]) -> Any:
    """为子任务点将。"""

    if sub.agent_name and sub.agent_name in agents:
        return agents[sub.agent_name]
    if matcher is not None:
        # ★ 子任务级点将用规则庙算，不再调模型。
        #
        #   matcher.match(desc, agents) 在不传 assessment 时会调
        #   assessor.assess() —— 那是一次 LLM 调用，而这里每个子任务都会走到。
        #   实测三个子任务 = 三次额外庙算，587 tokens，只为拿一个打分依据。
        fast = None
        assessor = getattr(matcher, "assessor", None)
        if assessor is not None and hasattr(assessor, "assess_fast"):
            try:
                fast = assessor.assess_fast(sub.description)
            except Exception:               # noqa: BLE001
                fast = None
        # ★ matcher.match 收的是 Dict[name, Agent]，不是 list。
        #
        #   第一版传了 list(agents.values())，参数类型不对，
        #   而外面那个 `except Exception: pass` **把错误整个吞掉了** ——
        #   于是每个子任务都静默退回"战力最高者"，
        #   实测三个不同的子任务全部点了同一员将。
        #
        # ★ 症状：点将功能看起来在工作（有结果、不报错），
        #   实际从未生效。这与本项目里其它几处静默失效同型：
        #   一个吞掉异常的 except，把"没跑成"伪装成"跑了但结果如此"。
        #
        #   现在只吞掉**预期内**的匹配失败，其余照常抛出。
        ranked = matcher.match(sub.description, agents, assessment=fast)
        if ranked:
            name = getattr(ranked[0], "agent_name", None)
            if name in agents:
                return agents[name]
    # 兜底：无 matcher 时取战力最高者
    return max(
        agents.values(),
        key=lambda a: a.profile.stats.total_power() if getattr(a, "profile", None) else 0,
    )



def _configure(agent: Any, sub: SubTask, *, ledger: Any = None) -> Any:
    """按子任务类型装配一份**独立**的执行副本。

    ════════════════════════════════════════════════════════
     ★ 为什么必须复制，而不是就地修改
    ════════════════════════════════════════════════════════

    同一个将领常被点给同一层里的多个子任务（无 matcher 时更是
    全部退回战力最高者）。而同层是**并行**执行的 ——
    若就地改配置，两个线程会同时写同一个对象，后写的覆盖先写的。

    实测：三个子任务（IR / Code / Write）跑完，观察到的配置
    只有 Code 和 Write 两种，**IR 的配置被 Code 整个冲掉了**。
    症状是那个子任务拿着错误的工具和验收策略执行，
    而全程不报错 —— 又一个"看起来在工作"的静默失效。

    ★ 修法是复制而不是加锁：加锁会把同层并行退化成串行，
      而并行正是这一层的价值所在（实测 0.41s vs 串行 0.6s）。
    """

    import copy

    from bingfu.graph import route_for

    # 浅复制 + 独立的工具映射；llm 与 profile 共享（它们是只读的）
    agent = copy.copy(agent)
    agent.tools = []
    agent._tool_functions = {}

    plan = route_for(sub.category or "*")
    agent.category = sub.category
    agent.max_revisions = plan.max_revisions
    agent.max_output_tokens = plan.max_output_tokens

    # 工具：按类别配发，而不是一律给全
    #
    # ★ 给写作任务一个 Python 解释器不会有帮助，只会增加跑偏的机会，
    #   而且每一轮都要把工具定义塞进上下文。实测工具让检索类的
    #   token 涨 19.6 倍 —— 这个量级下"多给一个反正不亏"不成立。
    belt = None
    if plan.tools_enabled and sub.category:
        from bingfu.tools import belt_for
        # ★ 把将领身上的知识库交给这条新工具带。
        #   copy.copy 保留了 agent.knowledge，但工具带是新建的，
        #   不显式传就等于每次派活都换一个空知识库。
        belt = belt_for(sub.category, knowledge=getattr(agent, "knowledge", None))
        for name, fn in belt.as_functions().items():
            agent.add_tool(name, fn)
    # ★ 随身兵器要在类别工具之后重新装上。
    #
    #   上面那两行把 `_tool_functions` 清空了 —— 那是为了让不同类别的
    #   子任务拿到各自该有的工具（这个设计是对的）。但文件读写不是
    #   「某一类的附加能力」，而是每一类都要用的随身装备：写作任务
    #   同样要读素材、要把稿子落盘。
    #
    #   不重装的话，调用方给将领配的工具一进编排就没了，
    #   而将领照常回话 —— 只是从未碰过任何文件。
    if hasattr(agent, "rearm_base_tools"):
        agent.rearm_base_tools()

    agent._toolbelt = belt

    # 验收：默认由路由表决定，而不是等调用方来填
    #
    # ★ 这是"接进框架路径"的关键。先前 verify 是 Agent 上的一个
    #   可选钩子，只有实验去填它 —— 普通用户拿到的框架里那一层
    #   等于不存在。
    if plan.verify_enabled and sub.category:
        from bingfu.verify import verify_output

        # ★ 只算**本子任务**写下的产物，所以要在开跑前先照一张相。
        #
        #   账本是跨子任务共用的（熔断要靠它）。直接读全量清单的话，
        #   上一个子任务写过文件，就会让**这一个**子任务的空洞产出
        #   也免于门禁 —— 那是把一处误判换成了另一处漏判。
        _before = frozenset(getattr(ledger, "written", ()) or ())

        def _verify(output: str, _cat=sub.category, _belt=belt,
                    _led=ledger, _before=_before):
            # 来源集来自工具**实际抓回**的原文；工具没被调用时为空，
            # 溯源检查会自动跳过，而不是把"没有证据"误判成"数字全是编的"
            sources = _belt.sources() if _belt else []
            # ★ 产物落盘时，回复只是回执 —— 把落盘清单交给验收，
            #   否则「已完成：写了 A 和 B」这句话会因为太短被打回重做，
            #   而重做会把整件事再干一遍。
            now = frozenset(getattr(_led, "written", ()) or ()) if _led else frozenset()
            artifacts = tuple(sorted(now - _before))
            return verify_output(output, category=_cat, artifacts=artifacts,
                                 sources=sources or [sub.description])

        agent.verify_fn = _verify
    else:
        agent.verify_fn = None

    return agent

def run_plan(
    plan: Plan,
    agents: Dict[str, Any],
    *,
    matcher: Optional[Any] = None,
    max_workers: int = 4,
    on_step: Optional[Any] = None,
    checkpointer: Optional[Any] = None,
    thread_id: str = "default",
) -> OrchestrationResult:
    """按依赖关系执行拆解图，同层并行。

    ★ 上游产物必须真的喂给下游 —— 这是"依赖"二字的全部含义。
      只按顺序跑而不传递产物，与并行跑没有区别，
      那样的"依赖"只是一个装饰。
    """

    import threading

    # ★ 一次执行一本账，所有子任务共用。
    #
    #   实测 agg-merge 里两个子任务各把 a/b/c.txt 读了一遍 ——
    #   第二个子任务没有任何途径得知第一个已经读过。账本就是那个途径。
    from bingfu.ledger import CallLedger
    ledger = CallLedger()

    graph = GraphOrchestrator(max_workers=max_workers)
    assignments: Dict[str, str] = {}
    # 同层并行，累加要加锁 —— 否则丢计数，而丢了看不出来
    tool_calls: Dict[str, int] = {}
    token_total = [0]
    tool_lock = threading.Lock()

    for sub in plan.subtasks:
        agent = _pick_agent(sub, agents, matcher)
        assignments[sub.id] = agent.name

        def _node(inputs: Dict[str, Any], _sub=sub, _agent=agent) -> str:
            # ★ 按子任务类型配置这一次执行。
            #
            #   路由表（bingfu/graph.py 的 ROUTES）原本只有实验在用，
            #   框架自身跑任务时一律走同一套配置。而实测分类别差异极大：
            #   检索类 12/12、代码类 3/12 —— 一套流水线套用于五类
            #   异质任务本来就是错的。
            #
            #   这里是把那张表接进框架路径的地方。
            _agent.ledger = ledger      # 副本共用同一本账
            # ★ 先挂账本再配置：_configure 要在开跑前给账本照相，
            #   照相必须发生在**这个**子任务动手之前。
            _agent = _configure(_agent, _sub, ledger=ledger)

            # ★ 进度回调要在**副本**上设，且要带上子任务 id。
            #
            #   同层并行时有多个副本同时在跑，没有 id 的话，
            #   界面上就是几条交错的「调用了 read_file」——
            #   看不出哪条属于哪个子任务，等于没有进度。
            if on_step is not None:
                def _step(ev, _sid=_sub.id, _cat=_sub.category):
                    on_step({**ev, "subtask": _sid, "subtask_category": _cat})
                if hasattr(_agent, "set_progress_callback"):
                    _agent.set_progress_callback(_step)

            context = ""
            if _sub.depends_on:
                parts = []
                for dep in _sub.depends_on:
                    val = inputs.get(dep)
                    if val:
                        parts.append(f"【{dep} 的产出】\n{val}")
                if parts:
                    context = (
                        "以下是你的前置任务已经完成的内容，请在此基础上继续：\n\n"
                        + "\n\n".join(parts)
                        + "\n\n────────\n"
                    )
            out = _agent.execute(context + _sub.description)

            # ★ 在副本还活着的时候把轨迹收走。
            #   函数返回后这个副本就没人引用了，轨迹随之消失。
            counts = getattr(_agent, "tool_call_counts", None) or {}
            used = int(getattr(_agent, "_last_run_tokens", 0) or 0)
            if counts or used:
                with tool_lock:
                    for k, v in counts.items():
                        tool_calls[k] = tool_calls.get(k, 0) + int(v)
                    token_total[0] += used
            return out

        graph.add(sub.id, _node, depends_on=sub.depends_on)

    # ★ 断点透传到编排层，否则它只是 graph.py 里一个没人用得到的能力。
    result = graph.run(checkpointer=checkpointer, thread_id=thread_id)

    # 汇总：按拓扑顺序拼接成功节点的产出
    chunks: List[str] = []
    for layer in result.layers:
        for name in layer:
            node = result.nodes.get(name)
            if node and node.status is NodeStatus.OK and node.output:
                chunks.append(str(node.output))
    output = "\n\n".join(chunks)

    return OrchestrationResult(
        tokens=token_total[0],
        breaker_hits=ledger.stats()["breaker_hits"],
        overwrites=ledger.stats()["overwrites"],
        overwrite_note=ledger.overwrite_report(),
        output=output, plan=plan, graph=result, assignments=assignments,
        tool_calls=dict(tool_calls),
    )


#: 低于这个复杂度就不拆解，单将直取。
#:
#: ★ 「运筹帷幄」的本意是**先算，再决定要不要动兵** ——
#:   而这个框架此前无论任务大小都要走完整流程：庙算 → 拆解 →
#:   分层执行。对小任务，这套开销是净损耗。
#:
#:   实测「写三个小文件」：拆成三个子任务后，每个子任务都要
#:   独立跑一遍 ReAct（各自的系统提示、各自的工具定义、各自的收尾），
#:   13 次调用；而同一件事一个将领顺手做完只要两三次。
#:   拆解的收益必须能覆盖拆解的成本，覆盖不了就不该拆。
#:
#: ★ 阈值取 5，对应复杂度分档的「中」以下（1–3 易 / 4–6 中）。
#:   这是一个**可以被数据推翻**的取值，不是定论 ——
#:   bench/ 下的跨框架实测就是用来推翻它的。
SOLO_BELOW_COMPLEXITY = 5


def _stable_solo_id(task: str) -> str:
    from bingfu.plan_check import stable_subtask_id
    return stable_subtask_id(task)


def orchestrate(
    task: str,
    agents: Dict[str, Any],
    *,
    llm: Optional[Any] = None,
    assessment: Optional[Any] = None,
    solo_below: int = SOLO_BELOW_COMPLEXITY,   # 保留形参：旧调用方仍在传
    force_decompose: bool = False,
    matcher: Optional[Any] = None,
    max_workers: int = 4,
    on_step: Optional[Any] = None,
    checkpointer: Optional[Any] = None,
    thread_id: str = "default",
) -> OrchestrationResult:
    """拆解 → 点将 → 按依赖执行 → 汇总。"""

    # ★ 算完再决定要不要动兵，而且**这一步不花钱**。
    #
    #   闸门读的是任务结构（需要几种将领、产出几件东西），
    #   不是一个 LLM 给的复杂度分 —— 后者有噪声、几乎从不改变结论，
    #   而且问错了问题。详见 assessment.needs_orchestration。
    #
    #   仍然走 run_plan，是为了让工具装配、进度回调、断点、
    #   工具计数这些机制保持同一条路径：省的是开销，不是可观测性。
    from bingfu.assessment import needs_orchestration

    if force_decompose or needs_orchestration(task, assessment):
        plan = decompose(task, agents, llm)
    else:
        plan = Plan([SubTask(id=_stable_solo_id(task), description=task,
                             category=infer_category(task))], source="solo")
    return run_plan(plan, agents, matcher=matcher, max_workers=max_workers,
                    on_step=on_step, checkpointer=checkpointer,
                    thread_id=thread_id)
