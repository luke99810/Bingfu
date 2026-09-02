"""
Complete Tactic Library — 26 formalized tactic patterns.

Each tactic has been carefully designed with:
- Style vectors calibrated against task categories
- Preconditions grounded in the TacticalContext
- Action sequences derived from Sun Tzu's text
- Task affinity validated against empirical patterns

See tactic_definitions.py for the TacticDefinition class.
"""

from typing import Dict, List
import numpy as np

from .tactic_definitions import (
    TacticCategory, TacticDefinition, TacticalContext, CombatStyle
)


# ── Precondition Helpers ──────────────────────────────────────

def _has_intel(ctx: TacticalContext, _agent_power) -> bool:
    return ctx.has_intelligence

def _needs_intel(ctx: TacticalContext, _agent_power) -> bool:
    return not ctx.has_intelligence

def _any_regime(ctx: TacticalContext, _agent_power) -> bool:
    return True

def _defensive_regime(ctx: TacticalContext, _agent_power) -> bool:
    return ctx.regime == "defensive"

def _defensive_or_balanced(ctx: TacticalContext, _agent_power) -> bool:
    return ctx.regime in ("defensive", "balanced")

def _balanced_regime(ctx: TacticalContext, _agent_power) -> bool:
    return ctx.regime == "balanced"

def _balanced_or_offensive(ctx: TacticalContext, _agent_power) -> bool:
    return ctx.regime in ("balanced", "offensive")

def _offensive_regime(ctx: TacticalContext, _agent_power) -> bool:
    return ctx.regime == "offensive"

def _high_morale(ctx: TacticalContext, _agent_power) -> bool:
    return ctx.morale >= 60

def _low_supplies(ctx: TacticalContext, _agent_power) -> bool:
    return ctx.supplies < 50

def _high_complexity(ctx: TacticalContext, _agent_power) -> bool:
    return ctx.task_complexity >= 7

def _medium_plus_complexity(ctx: TacticalContext, _agent_power) -> bool:
    return ctx.task_complexity >= 4

def _low_complexity(ctx: TacticalContext, _agent_power) -> bool:
    return ctx.task_complexity <= 3

def _urgent(ctx: TacticalContext, _agent_power) -> bool:
    return ctx.urgency >= 0.7

def _creative(ctx: TacticalContext, _agent_power) -> bool:
    return ctx.creativity_required >= 0.6

def _collaborative(ctx: TacticalContext, _agent_power) -> bool:
    return ctx.collaboration_required >= 0.5

def _high_agent_strategy(_, agent_power) -> bool:
    return agent_power[2] >= 70  # strategy dimension

def _high_agent_attack(_, agent_power) -> bool:
    return agent_power[0] >= 70  # attack dimension

def _high_agent_speed(_, agent_power) -> bool:
    return agent_power[3] >= 70  # speed dimension

def _high_agent_intelligence(_, agent_power) -> bool:
    return agent_power[4] >= 70  # intelligence dimension

def _high_agent_defense(_, agent_power) -> bool:
    return agent_power[1] >= 70  # defense dimension

def _and(*preds):
    return lambda ctx, ap: all(p(ctx, ap) for p in preds)

def _or(*preds):
    return lambda ctx, ap: any(p(ctx, ap) for p in preds)


# ── Style Vectors ─────────────────────────────────────────────
# Each vector ∈ [0,1]⁵ represents how the tactic distributes
# emphasis across: (attack, defense, strategy, speed, intelligence)

SV = {
    # Ch1: 始计篇 — Reconnaissance & Planning
    "知己知彼": (0.10, 0.15, 0.40, 0.30, 0.80),
    "庙算":     (0.15, 0.25, 0.80, 0.10, 0.75),

    # Ch2: 作战篇 — Speed & Resource
    "兵贵速胜": (0.65, 0.10, 0.25, 0.90, 0.30),
    "取用于国": (0.25, 0.55, 0.60, 0.20, 0.45),

    # Ch3: 谋攻篇 — Victory & Alliance
    "不战而胜": (0.05, 0.40, 0.90, 0.10, 0.85),
    "联横合纵": (0.15, 0.50, 0.75, 0.25, 0.70),

    # Ch4: 军形篇 — Defense & Terrain
    "先为不可胜": (0.10, 0.95, 0.40, 0.15, 0.45),
    "地生度":     (0.20, 0.85, 0.55, 0.20, 0.50),

    # Ch5: 兵势篇 — Orthodox/Unorthodox
    "奇正相生": (0.45, 0.30, 0.70, 0.50, 0.65),
    "出其不意": (0.60, 0.10, 0.55, 0.85, 0.50),

    # Ch6: 虚实篇 — Control & Flexibility
    "致人而不致于人": (0.40, 0.50, 0.75, 0.40, 0.60),
    "因敌而变":       (0.30, 0.35, 0.70, 0.60, 0.75),

    # Ch7: 军争篇 — Indirect & Morale
    "迁直之计": (0.25, 0.35, 0.85, 0.30, 0.70),
    "夺气夺心": (0.50, 0.30, 0.55, 0.45, 0.60),

    # Ch8: 九变篇 — Adaptability
    "变通在地": (0.30, 0.40, 0.75, 0.55, 0.70),
    "九变":     (0.20, 0.45, 0.85, 0.25, 0.80),

    # Ch9: 行军篇 — Intelligence & Position
    "察敌之情": (0.05, 0.20, 0.50, 0.60, 0.75),
    "处军之利": (0.15, 0.70, 0.55, 0.25, 0.50),

    # Ch10: 地形篇 — Terrain & Soldier
    "六地之用": (0.20, 0.75, 0.55, 0.30, 0.50),
    "兵卒之道": (0.35, 0.50, 0.40, 0.45, 0.55),

    # Ch11: 九地篇 — Nine Grounds & Deployment
    "九地之变": (0.30, 0.45, 0.80, 0.40, 0.65),
    "并敌一向": (0.85, 0.20, 0.30, 0.90, 0.35),

    # Ch12: 火攻篇 — Fire & Coordinated Attack
    "火攻有五": (0.80, 0.10, 0.35, 0.65, 0.40),
    "五火之变": (0.70, 0.30, 0.55, 0.55, 0.60),

    # Ch13: 用间篇 — Espionage
    "用间有五": (0.05, 0.15, 0.65, 0.25, 0.85),
    "五间俱起": (0.10, 0.20, 0.70, 0.35, 0.80),
}


# ── Build Full Library ────────────────────────────────────────

def build_tactic_library() -> Dict[str, TacticDefinition]:
    """Construct the complete 26-tactic library."""
    T = {}

    # ━━━━ Chapter 1: 始计篇 — Estimates ━━━━
    T["知己知彼"] = TacticDefinition(
        name="知己知彼",
        category=TacticCategory.RECONNAISSANCE,
        chapter=1,
        description="Know yourself and know your enemy — gather intelligence before committing forces",
        style_vector=SV["知己知彼"],
        combat_style=CombatStyle.SCOUT,
        precondition=_needs_intel,
        action_sequence=[
            "Information reconnaissance: collect task requirements and constraints",
            "Capability assessment: evaluate available agents against task profile",
            "Gap analysis: identify knowledge or resource gaps",
            "Report synthesis: produce structured intelligence report",
            "Recommendation: propose optimal approach based on intelligence",
        ],
        expected_outcome="Comprehensive task understanding with identified risks and capability requirements",
        task_affinity=["IR", "Data"],
        sun_tzu_quote="知彼知己，百战不殆；不知彼而知己，一胜一负；不知彼，不知己，每战必殆。",
        strength_regime="any",
    )

    T["庙算"] = TacticDefinition(
        name="庙算",
        category=TacticCategory.STRATEGIC_PLANNING,
        chapter=1,
        description="Strategic planning before action — estimate probabilities and plan accordingly",
        style_vector=SV["庙算"],
        combat_style=CombatStyle.STRATEGIC,
        precondition=_and(_high_complexity, _high_agent_strategy),
        action_sequence=[
            "Situation analysis: decompose complex task into components",
            "Option generation: enumerate feasible approaches",
            "Risk-benefit evaluation: assess each option's probability of success",
            "Resource planning: allocate agents and tools optimally",
            "Contingency planning: prepare fallback strategies",
            "Execute with monitoring: implement while tracking deviations",
        ],
        expected_outcome="Well-structured execution plan with quantified success probabilities",
        task_affinity=["Reason", "Data"],
        sun_tzu_quote="夫未战而庙算胜者，得算多也；未战而庙算不胜者，得算少也。多算胜，少算不胜。",
        strength_regime="any",
    )

    # ━━━━ Chapter 2: 作战篇 — Waging War ━━━━
    T["兵贵速胜"] = TacticDefinition(
        name="兵贵速胜",
        category=TacticCategory.RAPID_WARFARE,
        chapter=2,
        description="Speed is essential — rapid execution before the enemy can react",
        style_vector=SV["兵贵速胜"],
        combat_style=CombatStyle.BRAVE,
        precondition=_and(_urgent, _high_agent_speed),
        action_sequence=[
            "Rapid mobilization: activate highest-speed agent immediately",
            "Minimum viable analysis: essential requirements only",
            "Parallel execution: run independent sub-tasks concurrently",
            "Quick validation: rapid testing, accept minor imperfections",
            "Iterate if needed: fast feedback loop for correction",
        ],
        expected_outcome="Fast delivery of acceptable-quality output under time pressure",
        task_affinity=["Code", "IR"],
        sun_tzu_quote="兵贵胜，不贵久。",
        strength_regime="offensive",
    )

    T["取用于国"] = TacticDefinition(
        name="取用于国",
        category=TacticCategory.RESOURCE_MANAGEMENT,
        chapter=2,
        description="Leverage local resources — use existing assets rather than creating from scratch",
        style_vector=SV["取用于国"],
        combat_style=CombatStyle.COMMAND,
        precondition=_and(_low_supplies, _medium_plus_complexity),
        action_sequence=[
            "Resource audit: catalog available tools, data, and prior outputs",
            "Reuse assessment: identify existing components that can be adapted",
            "Integration planning: design minimal new work to connect existing pieces",
            "Efficient allocation: assign tasks to minimize token consumption",
            "Quality check: verify integrated solution meets requirements",
        ],
        expected_outcome="Task completion with minimal resource (token) consumption",
        task_affinity=["Code", "Write"],
        sun_tzu_quote="取用于国，因粮于敌，故军食可足也。",
        strength_regime="any",
    )

    # ━━━━ Chapter 3: 谋攻篇 — Attack by Stratagem ━━━━
    T["不战而胜"] = TacticDefinition(
        name="不战而胜",
        category=TacticCategory.TOTAL_VICTORY,
        chapter=3,
        description="Win without fighting — achieve objectives through superior strategy rather than brute force",
        style_vector=SV["不战而胜"],
        combat_style=CombatStyle.STRATEGIC,
        precondition=_and(_high_agent_strategy, _high_agent_intelligence),
        action_sequence=[
            "Deep analysis: understand the fundamental problem, not just the surface ask",
            "Alternative framing: reframe the problem to reveal simpler solutions",
            "Strategic shortcut: identify the minimal intervention that achieves the goal",
            "Elegant execution: implement the refined solution with precision",
            "Verification: confirm the solution meets the underlying need",
        ],
        expected_outcome="Elegant, minimal solution that addresses root cause, not symptoms",
        task_affinity=["Reason", "Write"],
        sun_tzu_quote="是故百战百胜，非善之善者也；不战而屈人之兵，善之善者也。",
        strength_regime="any",
    )

    T["联横合纵"] = TacticDefinition(
        name="联横合纵",
        category=TacticCategory.STRATEGIC_ALLIANCE,
        chapter=3,
        description="Build strategic alliances — combine multiple agents' complementary strengths",
        style_vector=SV["联横合纵"],
        combat_style=CombatStyle.COMMAND,
        precondition=_and(_collaborative, _balanced_or_offensive),
        action_sequence=[
            "Capability mapping: identify each agent's unique strengths",
            "Task partitioning: decompose task to maximize complementarity",
            "Agent pairing: assign sub-tasks to agent pairs with aligned skills",
            "Coordination protocol: establish communication and handoff rules",
            "Integration: synthesize individual contributions into unified output",
        ],
        expected_outcome="Synergistic output that exceeds any single agent's capability",
        task_affinity=["Code", "Data", "Write"],
        sun_tzu_quote="上兵伐谋，其次伐交，其次伐兵，其下攻城。",
        strength_regime="balanced",
    )

    # ━━━━ Chapter 4: 军形篇 — Tactical Dispositions ━━━━
    T["先为不可胜"] = TacticDefinition(
        name="先为不可胜",
        category=TacticCategory.DEFENSIVE_SUPERIORITY,
        chapter=4,
        description="Secure invincibility first — defensive posture, ensure correctness before speed",
        style_vector=SV["先为不可胜"],
        combat_style=CombatStyle.COMMAND,
        precondition=_and(_defensive_or_balanced, _high_agent_defense),
        action_sequence=[
            "Risk assessment: identify all potential failure modes",
            "Safety measures: implement validation, error handling, and guardrails",
            "Defensive execution: proceed cautiously with continuous verification",
            "Iterative refinement: improve incrementally, never compromising stability",
            "Final audit: comprehensive quality check before delivery",
        ],
        expected_outcome="Reliable, well-tested output with minimal errors",
        task_affinity=["Code", "Data"],
        sun_tzu_quote="昔之善战者，先为不可胜，以待敌之可胜。不可胜在己，可胜在敌。",
        strength_regime="defensive",
    )

    T["地生度"] = TacticDefinition(
        name="地生度",
        category=TacticCategory.TERRAIN_ADVANTAGE,
        chapter=4,
        description="Exploit terrain advantage — analyze task structure to identify natural leverage points",
        style_vector=SV["地生度"],
        combat_style=CombatStyle.STRATEGIC,
        precondition=_and(_medium_plus_complexity, _high_agent_strategy),
        action_sequence=[
            "Structure analysis: map task dependencies and bottlenecks",
            "Leverage identification: find high-impact, low-effort interventions",
            "Path optimization: design execution order for maximum efficiency",
            "Bottleneck resolution: prioritize critical path items",
            "Progressive delivery: release value incrementally",
        ],
        expected_outcome="Structured task execution optimizing for high-impact contributions",
        task_affinity=["Data", "Reason"],
        sun_tzu_quote="兵法：一曰度，二曰量，三曰数，四曰称，五曰胜。地生度，度生量，量生数，数生称，称生胜。",
        strength_regime="any",
    )

    # ━━━━ Chapter 5: 兵势篇 — Energy ━━━━
    T["奇正相生"] = TacticDefinition(
        name="奇正相生",
        category=TacticCategory.REGULAR_VS_SPECIAL,
        chapter=5,
        description="Combine orthodox and unorthodox — use standard methods for the main task, creative methods for edges",
        style_vector=SV["奇正相生"],
        combat_style=CombatStyle.COMMAND,
        precondition=_balanced_regime,
        action_sequence=[
            "Orthodox baseline: establish conventional solution as fallback",
            "Unorthodox exploration: generate creative alternative approaches",
            "Comparison: evaluate orthodox vs unorthodox on key metrics",
            "Hybrid construction: combine best elements of both approaches",
            "Validation: ensure hybrid solution is both innovative and reliable",
        ],
        expected_outcome="Solution that is both reliable (orthodox) and creative (unorthodox)",
        task_affinity=["Write", "Reason", "Code"],
        sun_tzu_quote="凡战者，以正合，以奇胜。故善出奇者，无穷如天地，不竭如江河。",
        strength_regime="balanced",
    )

    T["出其不意"] = TacticDefinition(
        name="出其不意",
        category=TacticCategory.SURPRISE_ATTACK,
        chapter=5,
        description="Strike where unexpected — bypass conventional approaches with innovative shortcuts",
        style_vector=SV["出其不意"],
        combat_style=CombatStyle.BRAVE,
        precondition=_and(_creative, _or(_urgent, _balanced_regime)),
        action_sequence=[
            "Convention audit: identify standard approaches being used",
            "Assumption challenge: question each constraint — is it real?",
            "Novel path: propose solution from unconventional angle",
            "Rapid prototype: quickly test the novel approach",
            "Surprise delivery: present innovative solution before expectations solidify",
        ],
        expected_outcome="Innovative solution that breaks conventional patterns",
        task_affinity=["Write", "Code"],
        sun_tzu_quote="攻其无备，出其不意。此兵家之胜，不可先传也。",
        strength_regime="any",
    )

    # ━━━━ Chapter 6: 虚实篇 — Weakness and Strength ━━━━
    T["致人而不致于人"] = TacticDefinition(
        name="致人而不致于人",
        category=TacticCategory.CONTROL_ENEMY,
        chapter=6,
        description="Control the engagement — proactively shape the task rather than reactively responding",
        style_vector=SV["致人而不致于人"],
        combat_style=CombatStyle.COMMAND,
        precondition=_and(_medium_plus_complexity, _any_regime),
        action_sequence=[
            "Proactive framing: define success criteria before starting",
            "Constraint management: negotiate or relax unnecessary constraints",
            "Paced execution: control work tempo to avoid reactive scrambling",
            "Strategic disclosure: release outputs when advantageous, not when pressured",
            "Definitive closure: clearly signal completion on your terms",
        ],
        expected_outcome="Well-paced, controlled execution with proactive quality management",
        task_affinity=["Reason", "Data", "Write"],
        sun_tzu_quote="故善战者，致人而不致于人。",
        strength_regime="any",
    )

    T["因敌而变"] = TacticDefinition(
        name="因敌而变",
        category=TacticCategory.FLEXIBLE_RESPONSE,
        chapter=6,
        description="Adapt like water — continuously adjust approach based on emerging information",
        style_vector=SV["因敌而变"],
        combat_style=CombatStyle.STRATEGIC,
        precondition=_and(_high_complexity, _any_regime),
        action_sequence=[
            "Initial assessment: form starting hypothesis",
            "Iterative sensing: continuously check if approach is working",
            "Adaptation trigger: define thresholds for strategy change",
            "Fluid response: switch tactics mid-execution if needed",
            "Convergence check: ensure adaptation leads toward goal, not drift",
        ],
        expected_outcome="Adaptive execution that improves strategy based on real-time feedback",
        task_affinity=["Code", "Reason", "Data"],
        sun_tzu_quote="夫兵形象水，水之形避高而趋下，兵之形避实而击虚。水因地而制流，兵因敌而制胜。",
        strength_regime="any",
    )

    # ━━━━ Chapter 7: 军争篇 — Maneuvering ━━━━
    T["迁直之计"] = TacticDefinition(
        name="迁直之计",
        category=TacticCategory.INDIRECT_APPROACH,
        chapter=7,
        description="Turn the indirect into direct — solve a harder sub-problem first to make the main problem easy",
        style_vector=SV["迁直之计"],
        combat_style=CombatStyle.STRATEGIC,
        precondition=_and(_high_complexity, _high_agent_strategy),
        action_sequence=[
            "Problem mapping: identify the core difficulty blocking progress",
            "Subsidiary attack: solve a related but different problem first",
            "Knowledge transfer: apply insights from subsidiary to main problem",
            "Simplified execution: now-tractable main problem is solved efficiently",
            "Synthesis: combine both solutions into comprehensive output",
        ],
        expected_outcome="Breakthrough on hard problems through indirect problem-solving",
        task_affinity=["Reason", "Code"],
        sun_tzu_quote="军争之难者，以迂为直，以患为利。",
        strength_regime="any",
    )

    T["夺气夺心"] = TacticDefinition(
        name="夺气夺心",
        category=TacticCategory.MORALE_WINNING,
        chapter=7,
        description="Target morale — maintain high agent confidence and momentum throughout execution",
        style_vector=SV["夺气夺心"],
        combat_style=CombatStyle.BRAVE,
        precondition=_high_morale,
        action_sequence=[
            "Early win: secure a quick, visible success to build confidence",
            "Momentum maintenance: chain successes, avoid stalls",
            "Confidence signaling: communicate progress clearly and frequently",
            "Energy management: prevent burnout through paced execution",
            "Celebratory closure: frame completion as a significant achievement",
        ],
        expected_outcome="High-energy execution with sustained momentum and agent confidence",
        task_affinity=["Write", "Code"],
        sun_tzu_quote="故三军可夺气，将军可夺心。是故朝气锐，昼气惰，暮气归。",
        strength_regime="any",
    )

    # ━━━━ Chapter 8: 九变篇 — Variation in Tactics ━━━━
    T["变通在地"] = TacticDefinition(
        name="变通在地",
        category=TacticCategory.ADAPTABILITY,
        chapter=8,
        description="Adapt to circumstances — modify approach based on task-specific characteristics",
        style_vector=SV["变通在地"],
        combat_style=CombatStyle.STRATEGIC,
        precondition=_and(_any_regime, _creative),
        action_sequence=[
            "Context sensing: deeply understand the specific task environment",
            "Template adaptation: modify standard approach to fit context",
            "Constraint-aware design: build solution that respects all constraints",
            "Stakeholder alignment: ensure solution meets all parties' needs",
            "Flexible delivery: format output for maximum consumption",
        ],
        expected_outcome="Tailored solution that fits the specific context perfectly",
        task_affinity=["Write", "IR"],
        sun_tzu_quote="故将通于九变之利者，知用兵矣；将不通于九变之利者，虽知地形，不能得地之利矣。",
        strength_regime="any",
    )

    T["九变"] = TacticDefinition(
        name="九变",
        category=TacticCategory.STRATEGIC_FLEXIBILITY,
        chapter=8,
        description="Consider both advantage and risk — evaluate each decision from multiple perspectives",
        style_vector=SV["九变"],
        combat_style=CombatStyle.STRATEGIC,
        precondition=_and(_high_complexity, _high_agent_strategy, _high_agent_intelligence),
        action_sequence=[
            "Multi-angle analysis: examine problem from all relevant perspectives",
            "Pros/cons enumeration: list advantages and risks of each approach",
            "Hybrid construction: combine complementary approaches, avoid conflicting ones",
            "Risk mitigation: build safeguards for identified risks",
            "Balanced execution: proceed with awareness of both opportunities and dangers",
        ],
        expected_outcome="Comprehensive solution that accounts for multi-faceted considerations",
        task_affinity=["Reason"],
        sun_tzu_quote="是故智者之虑，必杂于利害。杂于利而务可信也，杂于害而患可解也。",
        strength_regime="any",
    )

    # ━━━━ Chapter 9: 行军篇 — The Army on the March ━━━━
    T["察敌之情"] = TacticDefinition(
        name="察敌之情",
        category=TacticCategory.INTELLIGENCE_GATHERING,
        chapter=9,
        description="Read enemy signals — continuously gather task-relevant information during execution",
        style_vector=SV["察敌之情"],
        combat_style=CombatStyle.SCOUT,
        precondition=_any_regime,
        action_sequence=[
            "Signal detection: monitor for changes in task requirements",
            "Pattern recognition: identify recurring themes in requirements",
            "Context enrichment: supplement task with domain knowledge",
            "Early warning: detect potential issues before they become problems",
            "Informed adaptation: adjust approach based on new intelligence",
        ],
        expected_outcome="Information-rich execution that catches issues early",
        task_affinity=["IR", "Data"],
        sun_tzu_quote="敌近而静者，恃其险也；远而挑战者，欲人之进也。",
        strength_regime="any",
    )

    T["处军之利"] = TacticDefinition(
        name="处军之利",
        category=TacticCategory.TERRAIN_SELECTION,
        chapter=9,
        description="Position advantageously — choose the best execution environment and tools",
        style_vector=SV["处军之利"],
        combat_style=CombatStyle.COMMAND,
        precondition=_and(_medium_plus_complexity, _any_regime),
        action_sequence=[
            "Environment assessment: evaluate available tooling and platforms",
            "Optimal selection: choose the best environment for this specific task",
            "Setup optimization: configure tools for maximum effectiveness",
            "Resource positioning: ensure all needed resources are accessible",
            "Execution from advantage: leverage environmental strengths",
        ],
        expected_outcome="Optimal tool and environment selection reducing execution friction",
        task_affinity=["Code", "Data"],
        sun_tzu_quote="凡军好高而恶下，贵阳而贱阴，养生而处实，军无百疾，是谓必胜。",
        strength_regime="any",
    )

    # ━━━━ Chapter 10: 地形篇 — Terrain ━━━━
    T["六地之用"] = TacticDefinition(
        name="六地之用",
        category=TacticCategory.TERRAIN_TACTICS,
        chapter=10,
        description="Six terrain types — recognize task archetypes and apply matching strategies",
        style_vector=SV["六地之用"],
        combat_style=CombatStyle.STRATEGIC,
        precondition=_and(_medium_plus_complexity, _high_agent_strategy),
        action_sequence=[
            "Task classification: identify which of six archetypes the task matches",
            "Pattern retrieval: recall successful strategies for this archetype",
            "Strategy adaptation: customize archetype strategy for specific task",
            "Expected-outcome calibration: set realistic expectations based on archetype difficulty",
            "Archetype-aware execution: apply proven patterns with task-specific adjustments",
        ],
        expected_outcome="Pattern-matched execution leveraging prior task archetype knowledge",
        task_affinity=["Code", "Reason", "Data"],
        sun_tzu_quote="地形有通者，有挂者，有支者，有隘者，有险者，有远者。",
        strength_regime="any",
    )

    T["兵卒之道"] = TacticDefinition(
        name="兵卒之道",
        category=TacticCategory.SOLDIER_MANAGEMENT,
        chapter=10,
        description="Lead troops consistently — maintain consistent quality standards across executions",
        style_vector=SV["兵卒之道"],
        combat_style=CombatStyle.COMMAND,
        precondition=_and(_collaborative, _any_regime),
        action_sequence=[
            "Standard setting: define clear quality criteria upfront",
            "Process consistency: follow established protocols reliably",
            "Quality monitoring: continuous check against standards",
            "Feedback loop: address deviations promptly and constructively",
            "Consistent delivery: ensure uniform quality across all outputs",
        ],
        expected_outcome="Reliable, consistent output quality across repeated executions",
        task_affinity=["Code", "Data"],
        sun_tzu_quote="令素行以教其民，则民服；令不素行以教其民，则民不服。",
        strength_regime="any",
    )

    # ━━━━ Chapter 11: 九地篇 — Nine Situations ━━━━
    T["九地之变"] = TacticDefinition(
        name="九地之变",
        category=TacticCategory.NINE_TERRAINS,
        chapter=11,
        description="Nine ground types — adjust collaboration intensity based on task urgency and interdependence",
        style_vector=SV["九地之变"],
        combat_style=CombatStyle.STRATEGIC,
        precondition=_and(_high_complexity, _collaborative),
        action_sequence=[
            "Situation typing: classify the task's collaboration requirements",
            "Intensity calibration: set appropriate coordination level",
            "Agent positioning: assign roles based on situation demands",
            "Collaboration mode: select synchronous or asynchronous coordination",
            "Dynamic adjustment: modify collaboration intensity as task evolves",
        ],
        expected_outcome="Appropriately-calibrated collaboration avoiding both under- and over-coordination",
        task_affinity=["Code", "Write"],
        sun_tzu_quote="用兵之法，有散地，有轻地，有争地，有交地，有衢地，有重地，有圮地，有围地，有死地。",
        strength_regime="any",
    )

    T["并敌一向"] = TacticDefinition(
        name="并敌一向",
        category=TacticCategory.RAPID_DEPLOYMENT,
        chapter=11,
        description="Concentrate force — focus all resources on a single critical objective",
        style_vector=SV["并敌一向"],
        combat_style=CombatStyle.ASSAULT,
        precondition=_and(_offensive_regime, _high_agent_attack),
        action_sequence=[
            "Critical path identification: find the single most important deliverable",
            "Force concentration: allocate maximum resources to critical task",
            "Focused execution: eliminate all distractions and secondary tasks",
            "Breakthrough: achieve overwhelming quality on the critical item",
            "Consolidation: leverage breakthrough to accelerate remaining work",
        ],
        expected_outcome="Exceptional quality on the most critical deliverable",
        task_affinity=["Code", "Write"],
        sun_tzu_quote="故为兵之事，在于顺详敌之意，并敌一向，千里杀将，此谓巧能成事者也。",
        strength_regime="offensive",
    )

    # ━━━━ Chapter 12: 火攻篇 — Fire Attack ━━━━
    T["火攻有五"] = TacticDefinition(
        name="火攻有五",
        category=TacticCategory.FIRE_ATTACK,
        chapter=12,
        description="Five fire attack methods — deploy five distinct execution patterns for maximum impact",
        style_vector=SV["火攻有五"],
        combat_style=CombatStyle.ASSAULT,
        precondition=_and(_offensive_regime, _high_agent_attack),
        action_sequence=[
            "Method selection: choose from five execution patterns based on task",
            "Rapid ignition: start with highest-impact sub-task immediately",
            "Spread acceleration: let initial success cascade to dependent tasks",
            "Burn-through: complete all tasks with relentless momentum",
            "Aftermath assessment: verify completeness and quality post-execution",
        ],
        expected_outcome="Rapid, comprehensive task completion with cascading momentum",
        task_affinity=["Code", "IR"],
        sun_tzu_quote="凡火攻有五：一曰火人，二曰火积，三曰火辎，四曰火库，五曰火队。",
        strength_regime="offensive",
    )

    T["五火之变"] = TacticDefinition(
        name="五火之变",
        category=TacticCategory.COORDINATED_ATTACK,
        chapter=12,
        description="Multi-modal coordinated attack — orchestrate diverse execution methods simultaneously",
        style_vector=SV["五火之变"],
        combat_style=CombatStyle.COMMAND,
        precondition=_and(_balanced_or_offensive, _medium_plus_complexity),
        action_sequence=[
            "Modality decomposition: break task into differently-shaped sub-tasks",
            "Method matching: assign distinct execution methods to each sub-task",
            "Synchronized launch: initiate all sub-tasks with coordinated timing",
            "Cross-modality synergy: ensure sub-task outputs complement each other",
            "Unified delivery: present integrated multi-modal output",
        ],
        expected_outcome="Rich, multi-faceted output leveraging diverse execution methods",
        task_affinity=["Write", "Data", "Code"],
        sun_tzu_quote="以火佐攻者明，以水佐攻者强。水可以绝，不可以夺。",
        strength_regime="balanced",
    )

    # ━━━━ Chapter 13: 用间篇 — Use of Spies ━━━━
    T["用间有五"] = TacticDefinition(
        name="用间有五",
        category=TacticCategory.ESPIONAGE,
        chapter=13,
        description="Five spy types — deploy diverse information-gathering agents for comprehensive intelligence",
        style_vector=SV["用间有五"],
        combat_style=CombatStyle.SCOUT,
        precondition=_and(_needs_intel, _high_agent_intelligence),
        action_sequence=[
            "Source diversification: identify five distinct information sources",
            "Parallel collection: gather from all sources simultaneously",
            "Cross-validation: compare and verify information across sources",
            "Synthesis: integrate verified intelligence into coherent picture",
            "Actionable intelligence: produce decision-ready intelligence report",
        ],
        expected_outcome="Comprehensive, verified intelligence enabling informed decisions",
        task_affinity=["IR", "Reason"],
        sun_tzu_quote="故用间有五：有因间，有内间，有反间，有死间，有生间。五间俱起，莫知其道，是谓神纪。",
        strength_regime="any",
    )

    T["五间俱起"] = TacticDefinition(
        name="五间俱起",
        category=TacticCategory.INTELLIGENCE_NETWORK,
        chapter=13,
        description="Networked intelligence — create a web of information sources for 360-degree awareness",
        style_vector=SV["五间俱起"],
        combat_style=CombatStyle.SCOUT,
        precondition=_and(_has_intel, _high_complexity),
        action_sequence=[
            "Network activation: engage all intelligence sources simultaneously",
            "Cross-referencing: build connections between disparate information",
            "Pattern emergence: identify patterns invisible to any single source",
            "Strategic insight: derive novel insights from information network",
            "Decision support: provide comprehensive basis for strategic choices",
        ],
        expected_outcome="Emergent strategic insights from networked intelligence sources",
        task_affinity=["IR", "Data", "Reason"],
        sun_tzu_quote="五间俱起，莫知其道，是谓神纪，人君之宝也。",
        strength_regime="any",
    )

    return T


# ── Singleton ─────────────────────────────────────────────────

_TACTIC_LIBRARY: Dict[str, TacticDefinition] = {}

def get_tactic_library() -> Dict[str, TacticDefinition]:
    """Get (lazily built) full tactic library."""
    global _TACTIC_LIBRARY
    if not _TACTIC_LIBRARY:
        _TACTIC_LIBRARY = build_tactic_library()
    return _TACTIC_LIBRARY


def get_tactic(name: str) -> TacticDefinition:
    """Get a single tactic by name."""
    return get_tactic_library()[name]


def get_tactics_by_regime(regime: str) -> List[TacticDefinition]:
    """Get tactics applicable to a strength regime."""
    return [t for t in get_tactic_library().values()
            if t.strength_regime == regime or t.strength_regime == "any"]


def get_tactics_by_style(style: CombatStyle) -> List[TacticDefinition]:
    """Get tactics associated with a combat style."""
    return [t for t in get_tactic_library().values()
            if t.combat_style == style]
