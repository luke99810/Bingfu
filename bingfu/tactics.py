"""
BingFu Tactic Engine (战术引擎) — Unified optimization-based tactic selection.

Refactored v0.7.0: Replaces if-else branching with a proper optimization
framework over the 26-tactic library. The SelectTactic algorithm now:

  1. Builds TacticalContext from TaskAssessment + agent pool analysis
  2. Filters tactics by preconditions (φ_pre)
  3. Computes multi-objective Q(t, a) for each compatible (tactic, agent) pair
  4. Returns argmax (t*, a*, score, explanation)

Integrates with: TaskAssessor, TaskMatcher, Commander for end-to-end
strategy-driven agent coordination.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from .tactic_definitions import TacticCategory, TacticDefinition, TacticalContext
from .tactic_library import (
    get_tactic_library, get_tactic, get_tactics_by_regime, get_tactics_by_style
)
from .profile import GeneralProfile, CombatStyle, CombatStats
from .assessment import TaskAssessment, TaskAssessor


# ── Selection Result ──────────────────────────────────────────

@dataclass
class TacticSelection:
    """Result of a single tactic-agent pair evaluation."""
    tactic: TacticDefinition
    agent_name: str
    alignment_score: float      # cos_sim(tactic.style_vector, task_features)
    power_score: float          # agent's capability match
    combined_score: float       # weighted Q(t, a)
    is_applicable: bool
    explanation: str

    def __repr__(self):
        return (f"TacticSelection(tactic={self.tactic.name}, agent={self.agent_name}, "
                f"score={self.combined_score:.3f}, applicable={self.is_applicable})")


@dataclass
class OptimizationResult:
    """Full result of tactic-agent optimization."""
    selected_tactic: TacticDefinition
    selected_agent_name: str
    combined_score: float
    alignment_score: float
    power_score: float
    all_evaluations: List[TacticSelection]
    tactical_context: TacticalContext
    explanation: str


# ── Tactic Engine ─────────────────────────────────────────────

class TacticEngine:
    """
    Unified tactic selection engine.

    Implements the SelectTactic algorithm as a constrained optimization:

        (t*, a*) = argmax_{t∈T, a∈A} Q(τ, t, a)
        s.t. φ_pre(t, a_context) = 1

    where Q = w₁·Align(t.s, c(τ)) + w₂·PowerScore(a, τ) + w₃·History(a, τ)

    Attributes:
        library: The complete 26-tactic repertoire
        weights: (w₁, w₂, w₃) for alignment, power, and history components
        assessor: TaskAssessor for building tactical context
    """

    def __init__(
        self,
        weights: Tuple[float, float, float] = (0.35, 0.45, 0.20),
        center_alignment: bool = False,
        episodic: Optional[Any] = None,
        assessor: Optional[TaskAssessor] = None,
    ):
        self.library = get_tactic_library()
        self.weights = weights

        # ★ 中心化对齐 —— 默认**关闭**，这是个刻意的保守选择。
        #
        #   问题是真的：26 个战术的 style_vector 全为非负，
        #   余弦相似度因此天然聚集在高位 —— 两两余弦中位数 0.892，
        #   对任一任务全部 26 个都打 0.6 以上。
        #   实测 20 条基准任务里 19 条选中同一个战术。
        #   所谓"从 26 个战术中择优"，实际没有在择。
        #
        #   中心化（减去战术库重心）把跨度从 0.28 拉到约 1.5，
        #   五条代表任务选出的战术从 2 种增加到 3 种 —— 改善是真的。
        #
        # ★ 那为什么不默认打开？
        #
        #   因为唯二显著的正效应（Data p=0.039、IR p=0.005）
        #   正是在"永远选五火之变"的状态下测出来的。
        #   换一个战术，IR 的 12/12 可能就没了。
        #
        #   "更有区分度"是过程指标，"成功率更高"才是结果指标，
        #   两者并不自动等价。在拿到对比数据之前把它设为默认，
        #   就是用一个未经验证的改动去替换一个已被验证的行为。
        self.center_alignment = center_alignment
        self._library_centroid = None

        # ★ 闭合一个断了很久的回路。
        #
        #   打分公式是 Q = 0.35·对齐 + 0.45·战力 + **0.20·历史**，
        #   而 history_scores 这个参数**从未被任何调用方传入** ——
        #   历史项恒为常数 0.5。占 20% 权重的一整项是死重。
        #
        #   接口在、参数在、注释在，就是没人喂数据；
        #   而它不报错，因为 0.5 是个完全合法的值。
        #
        # ★ 传入 episodic 后，历史分由战报自动派生，
        #   且**样本不足的格子不参与**（见 MIN_SAMPLES 的说明）：
        #   "没有足够证据"与"表现是 0.5"必须区分，
        #   否则一个只跑过一次且失败的将领会被当成"确凿地不行"。
        self.episodic = episodic
        self.assessor = assessor or TaskAssessor()
        self._selection_log: List[OptimizationResult] = []

    # ── Public API ────────────────────────────────────────────

    def select_tactic(
        self,
        task: str,
        agents: Dict[str, 'Agent'],
        task_assessment: Optional[TaskAssessment] = None,
        history_scores: Optional[Dict[str, Dict[str, float]]] = None,
        top_k: int = 3,
    ) -> OptimizationResult:
        """
        Core algorithm: select optimal (tactic, agent) pair.

        Algorithm 1 from the paper: SelectTactic(task, agents)

        ════════════════════════════════════════════════════════
         ★ 「联合优化」目前并不是联合的
        ════════════════════════════════════════════════════════

        合计分的形式是：

            Q(t, a) = w₁·alignment(t) + w₂·power(a) + w₃·history(a)

        alignment 只依赖战术，power 与 history 只依赖将领 ——
        **这是可加可分的**。因此在 26×7 = 182 个组合上取 argmax，
        等价于两个独立的 argmax：战术由 alignment 单独决定，
        将领由 power+history 单独决定。

        也就是说，遍历 182 个组合的开销付出了，
        但没有买到任何"战术与将领相互适配"的效果 ——
        除了 is_applicable 前置条件带来的少量耦合
        （实测 C1 上 40 个组合全部通过，即耦合为零）。

        ★ 要真正联合，需要一个**交互项** —— 例如
          tactic.combat_style 与 general.style 的匹配度。
          这两个字段在领域模型里都存在，但打分时都没有用到。

        这条说明保留在这里，是因为方法名与文档字符串都在
        声称"联合优化"，而实现并没有做到。
        不写出来的话，下一个读者会以为它做了。

        Args:
            task: Natural language task description
            agents: Available agents {name: Agent}
            task_assessment: Pre-computed assessment (optional)
            history_scores: Per-agent historical performance {agent_name: {task_type: score}}
            top_k: Return top-k in all_evaluations

        Returns:
            OptimizationResult with best (tactic, agent), score, and explanation
        """
        # 历史分：调用方没传就从战报库派生
        if history_scores is None and self.episodic is not None:
            from bingfu.memory import history_scores as _derive
            history_scores = _derive(self.episodic)

        # Step 1: Assess task → extract features
        if task_assessment is None:
            assessment = self.assessor.assess(task)
        else:
            assessment = task_assessment

        # Step 2: Build TacticalContext
        ctx = self._build_context(task, assessment, agents)

        # Step 3: Extract task feature vector for alignment computation
        task_features = ctx.to_task_features()

        # Step 4: Filter tactics and score all (tactic, agent) pairs
        evaluations = []
        agent_powers = self._extract_agent_powers(agents)

        for tactic in self.library.values():
            for agent_name, agent in agents.items():
                agent_power = agent_powers.get(agent_name,
                                               np.array([50, 50, 50, 50, 50], dtype=np.float64))

                # Check precondition
                applicable = tactic.is_applicable(ctx, agent_power)

                if not applicable:
                    # Still record but with zero score for transparency
                    evaluations.append(TacticSelection(
                        tactic=tactic,
                        agent_name=agent_name,
                        alignment_score=0.0,
                        power_score=0.0,
                        combined_score=0.0,
                        is_applicable=False,
                        explanation=f"Precondition not met: {tactic.name} requires "
                                    f"strength_regime={tactic.strength_regime}, "
                                    f"current regime={ctx.regime}",
                    ))
                    continue

                # Compute alignment: cos_sim(tactic.style_vector, task_features)
                alignment = self._alignment(tactic, task_features)

                # Compute power score: agent's weighted capability match
                power = self._compute_power_score(agent_power, task_features)

                # Compute history score
                history = 0.5  # default neutral
                if history_scores and agent_name in history_scores:
                    task_type = self._infer_task_type(task)
                    history = history_scores[agent_name].get(task_type, 0.5)

                # Combined Q(t, a)
                w1, w2, w3 = self.weights
                combined = (w1 * alignment) + (w2 * power) + (w3 * history)

                explanation = self._generate_explanation(
                    tactic, agent_name, alignment, power, history, combined, ctx
                )

                evaluations.append(TacticSelection(
                    tactic=tactic,
                    agent_name=agent_name,
                    alignment_score=alignment,
                    power_score=power,
                    combined_score=combined,
                    is_applicable=True,
                    explanation=explanation,
                ))

        # Step 5: Sort by combined score descending
        evaluations.sort(key=lambda e: e.combined_score, reverse=True)

        # Step 6: Select best
        best = evaluations[0] if evaluations else None

        if best is None or best.combined_score == 0.0:
            # Fallback: return first applicable tactic found
            for ev in evaluations:
                if ev.is_applicable:
                    best = ev
                    break

        if best is None:
            # ★ 把**真实原因**交回去，而不是一句放之四海皆准的失败。
            #
            #   此前无论什么情况都报「No applicable tactic found for task」——
            #   而最常见的触发原因其实是 `agents` 是空的。看到那句话的人
            #   （或模型）会去查战术库和任务措辞，方向完全错了。
            #
            #   两种情况的修法**完全不同**：
            #     没有将领   → 先 add_agent，与战术库无关
            #     有将领但全不适用 → 是前置条件太严或战场态势极端
            if not agents:
                raise ValueError(
                    "无法选择战术：没有可用的将领（agents 为空）。"
                    "请先注册至少一位将领 —— 这与战术库无关。"
                )
            raise ValueError(
                f"无法选择战术：{len(agents)} 位将领与库中 {len(self.library)} 个战术"
                f"两两组合后没有任何一对满足前置条件。"
                f"\n  战场态势：{ctx.regime}（我方 {ctx.self_strength:.0f} / 敌方 {ctx.enemy_strength:.0f}）"
                f"\n  任务：{task[:100]}"
            )

        result = OptimizationResult(
            selected_tactic=best.tactic,
            selected_agent_name=best.agent_name,
            combined_score=best.combined_score,
            alignment_score=best.alignment_score,
            power_score=best.power_score,
            all_evaluations=evaluations[:top_k],
            tactical_context=ctx,
            explanation=best.explanation,
        )

        self._selection_log.append(result)
        return result

    def quick_select(
        self,
        task: str,
        agents: Dict[str, 'Agent'],
        task_assessment: Optional[TaskAssessment] = None,
    ) -> Tuple[TacticDefinition, str, float]:
        """
        Quick tactic selection returning (tactic, agent_name, score).
        Convenience wrapper around select_tactic for Commander integration.
        """
        result = self.select_tactic(task, agents, task_assessment)
        return (result.selected_tactic, result.selected_agent_name, result.combined_score)

    def get_log(self) -> List[OptimizationResult]:
        """Retrieve tactic selection history."""
        return self._selection_log

    def get_sun_tzu_quote(self, tactic_name: str) -> str:
        """Get the Sun Tzu quote associated with a tactic."""
        tactic = get_tactic(tactic_name)
        return tactic.sun_tzu_quote

    # ── Internal Methods ──────────────────────────────────────

    def _build_context(
        self,
        task: str,
        assessment: TaskAssessment,
        agents: Dict[str, 'Agent'],
    ) -> TacticalContext:
        """Build TacticalContext from TaskAssessment + agent pool analysis."""
        agent_powers = self._extract_agent_powers(agents)
        if agent_powers:
            avg_power = np.mean(list(agent_powers.values()), axis=0)
            self_strength = float(np.mean(avg_power))
        else:
            self_strength = 50.0

        enemy_strength = float(assessment.enemy_power)

        # Derive metadata from task
        has_intel = len(assessment.required_capabilities) >= 2
        urgency = self._estimate_urgency(task, assessment)
        creativity = self._estimate_creativity(task, assessment)
        collaboration = self._estimate_collaboration(assessment)

        return TacticalContext(
            self_strength=self_strength,
            enemy_strength=enemy_strength,
            terrain=self._map_terrain(task),
            weather=self._map_weather(assessment),
            morale=self._estimate_morale(agents),
            supplies=float(100 - assessment.complexity_score * 5),
            time_factor="夜间" if urgency > 0.7 else "白天",
            has_intelligence=has_intel,
            task_complexity=assessment.complexity_score,
            required_capabilities=assessment.required_capabilities,
            urgency=urgency,
            creativity_required=creativity,
            collaboration_required=collaboration,
        )

    def _extract_agent_powers(self, agents: Dict[str, 'Agent']) -> Dict[str, np.ndarray]:
        """Extract 5-dim power vectors from agent profiles."""
        powers = {}
        for name, agent in agents.items():
            profile = getattr(agent, 'profile', None)
            if profile and hasattr(profile, 'stats'):
                s = profile.stats
                powers[name] = np.array([
                    s.attack, s.defense, s.strategy, s.speed, s.intelligence
                ], dtype=np.float64)
            else:
                powers[name] = np.array([50, 50, 50, 50, 50], dtype=np.float64)
        return powers

    def _alignment(self, tactic, task_features: np.ndarray) -> float:
        """战术与任务的对齐分。

        默认走原始余弦；``center_alignment=True`` 时改为对战术库
        重心做中心化后再算余弦。

        中心化衡量的是"相对于其它战术，这一个是否更贴合本任务"，
        而不是"绝对重合度" —— 后者在全非负向量上没有区分力。
        """

        if not self.center_alignment:
            return tactic.alignment_score(task_features)

        if self._library_centroid is None:
            self._library_centroid = np.mean(
                [t.style_vector for t in self.library.values()], axis=0
            )
        mu = self._library_centroid
        v = np.asarray(tactic.style_vector, dtype=np.float64) - mu
        f = np.asarray(task_features, dtype=np.float64) - mu
        nv, nf = np.linalg.norm(v), np.linalg.norm(f)
        if nv == 0 or nf == 0:
            return 0.0
        return float(np.dot(v, f) / (nv * nf))

    def _compute_power_score(
        self, agent_power: np.ndarray, task_features: np.ndarray
    ) -> float:
        """
        Compute weighted power matching score.

        PowerScore(a, τ) = Σᵢ λᵢ(τ) · pᵢ⁽ᵃ⁾ / 10

        where λᵢ(τ) = softmax(task_features) with temperature α=1.0

        ════════════════════════════════════════════════════════
         ★ 归一化曾经差 10 倍
        ════════════════════════════════════════════════════════

        原来写的是 ``agent_power / 10.0``，而 ``agent_power`` 的取值是
        1–100（五维属性的定义域），除以 10 得到的是 **0.1–10**，
        不是注释声称的 0–1。

        后果：power 项在合计分里的实际贡献约 3.88，
        而带任务信息的 alignment 项（权重 0.35、取值跨度约 0.28）
        贡献仅约 0.10 —— **信号被一个与战术无关的常数压掉了 97%**。

        实测：C1 任务上前十名战术的合计分是 4.306 到 4.208，
        总跨度 0.099，全部由 alignment 贡献，其余 3.98 是常数。

        ★ 但要说清楚：修这个 bug **不改变选出哪个战术**。
          因为 power 只依赖将领、alignment 只依赖战术，
          合计分是可加可分的（见 select_tactic 的说明），
          加一个常数不影响排序。

          修它的意义在于分数重新落回 [0,1]，可解释、可比较，
          而不是让人误以为 4.306 是个"高分"。
        """

        # Task-adaptive weights via softmax
        alpha = 1.0
        exp_features = np.exp(alpha * task_features * 5)  # scale for better distribution
        lambdas = exp_features / exp_features.sum()

        # 五维属性定义域是 1–100，除以 100 才落到 [0,1]
        norm_power = agent_power / 100.0

        return float(np.dot(lambdas, norm_power))

    def _infer_task_type(self, task: str) -> str:
        """Infer task category from description text."""
        task_lower = task.lower()
        if any(kw in task_lower for kw in ['search', 'find', 'retrieve', 'research', '调研', '搜索', '查']):
            return "IR"
        elif any(kw in task_lower for kw in ['code', 'program', 'develop', 'implement', '代码', '编程', '开发']):
            return "Code"
        elif any(kw in task_lower for kw in ['data', 'analyze', 'statistics', 'visualize', '数据', '分析']):
            return "Data"
        elif any(kw in task_lower for kw in ['write', 'content', 'article', 'blog', '写', '文章', '创作']):
            return "Write"
        elif any(kw in task_lower for kw in ['reason', 'logic', 'strategy', 'think', '推理', '逻辑', '策略']):
            return "Reason"
        return "General"

    def _estimate_urgency(self, task: str, assessment: TaskAssessment) -> float:
        urgency_keywords = ['紧急', 'immediately', '马上', 'urgent', 'asap', '快速', '立即', '立刻']
        count = sum(1 for kw in urgency_keywords if kw in task.lower())
        # Higher complexity with urgency keywords → higher urgency
        base = min(1.0, count * 0.3 + 0.2)
        # Complexity also contributes to perceived urgency
        base += assessment.complexity_score / 30.0
        return min(1.0, base)

    def _estimate_creativity(self, task: str, assessment: TaskAssessment) -> float:
        creative_keywords = ['creative', 'innovative', 'novel', 'new', 'design', '创意', '设计',
                             '创作', 'write', '新颖', '创新', '艺术']
        count = sum(1 for kw in creative_keywords if kw in task.lower())
        base = min(1.0, count * 0.25 + 0.15)
        # Need capabilities like "文案写作" or "设计创意" → creative task
        creative_caps = {'文案写作', '设计创意', 'Write', 'Design'}
        if creative_caps & set(assessment.required_capabilities):
            base += 0.3
        return min(1.0, base)

    def _estimate_collaboration(self, assessment: TaskAssessment) -> float:
        collaborative_caps = {'沟通协调', '团队', '协调', 'Collaboration', 'Teamwork'}
        if collaborative_caps & set(assessment.required_capabilities):
            return 0.8
        if len(assessment.required_capabilities) >= 4:
            return 0.7  # multi-capability tasks likely need collaboration
        return 0.3

    def _estimate_morale(self, agents: Dict[str, 'Agent']) -> float:
        """Estimate agent morale from profile quality and agent state."""
        if not agents:
            return 50.0
        morale = 0.0
        for agent in agents.values():
            profile = getattr(agent, 'profile', None)
            if profile and hasattr(profile, 'stats'):
                morale += profile.stats.total_power() / 5.0
            else:
                morale += 50.0
        return morale / len(agents)

    def _map_terrain(self, task: str) -> str:
        """Map task description to metaphorical terrain."""
        task_lower = task.lower()
        if any(kw in task_lower for kw in ['安全', 'security', '测试', 'test', '稳定']):
            return "山地"  # Mountain — defensive, careful
        elif any(kw in task_lower for kw in ['快速', 'fast', 'urgent', '紧急']):
            return "平原"  # Plain — open, fast movement
        elif any(kw in task_lower for kw in ['数据', 'data', 'stream', '流']):
            return "河流"  # River — flowing, continuous
        elif any(kw in task_lower for kw in ['复杂', 'complex', 'system', '系统']):
            return "森林"  # Forest — complex, requires navigation
        return "平原"

    def _map_weather(self, assessment: TaskAssessment) -> str:
        """Map assessment to metaphorical weather."""
        if assessment.complexity_score >= 8:
            return "暴风"   # Storm — extreme conditions
        elif assessment.complexity_score >= 6:
            return "雨天"   # Rain — challenging
        elif assessment.complexity_score >= 4:
            return "阴天"   # Overcast — moderate
        else:
            return "晴"     # Clear — favorable

    def _generate_explanation(
        self,
        tactic: TacticDefinition,
        agent_name: str,
        alignment: float,
        power: float,
        history: float,
        combined: float,
        ctx: TacticalContext,
    ) -> str:
        """Generate human-readable explanation for the tactic selection."""
        parts = []

        if alignment > 0.7:
            parts.append(f"战术「{tactic.name}」与任务特征高度对齐 (cos_sim={alignment:.2f})")
        elif alignment > 0.4:
            parts.append(f"战术「{tactic.name}」与任务特征中等对齐 (cos_sim={alignment:.2f})")
        else:
            parts.append(f"战术「{tactic.name}」与任务特征对齐度较低 (cos_sim={alignment:.2f})")

        if power > 0.7:
            parts.append(f"将领「{agent_name}」战力充沛 (power={power:.2f})")
        elif power > 0.4:
            parts.append(f"将领「{agent_name}」战力充足 (power={power:.2f})")

        if history > 0.7:
            parts.append(f"历史表现优秀 (history={history:.2f})")

        parts.append(f"当前态势: {ctx.regime} (ratio={ctx.strength_ratio:.2f})")
        parts.append(f"综合评分 Q={combined:.3f}")

        return "；".join(parts)


# ── Sun Tzu Agent ─────────────────────────────────────────────

class SunTzuAgent:
    """
    Sun Tzu strategic advisor — provides tactic recommendations and wisdom.
    Wraps TacticEngine with conversational interface.
    """

    def __init__(self, name: str = "孙子", assessor: Optional[TaskAssessor] = None):
        self.name = name
        self.role = "军师"
        self.engine = TacticEngine(assessor=assessor)

    def analyze_and_recommend(
        self,
        task: str,
        agents: Dict[str, 'Agent'],
        task_assessment: Optional[TaskAssessment] = None,
    ) -> OptimizationResult:
        """Analyze task and recommend optimal tactic + agent."""
        return self.engine.select_tactic(task, agents, task_assessment)

    def get_wisdom(self) -> str:
        """Get a random Sun Tzu wisdom quote."""
        quotes = [
            "兵者，国之大事，死生之地，存亡之道，不可不察也。",
            "知彼知己，百战不殆；不知彼而知己，一胜一负；不知彼，不知己，每战必殆。",
            "上兵伐谋，其次伐交，其次伐兵，其下攻城。",
            "善战者，立于不败之地，而不失敌之败也。",
            "兵贵胜，不贵久。",
            "其疾如风，其徐如林，侵掠如火，不动如山。",
            "战势不过奇正，奇正相生，不可胜穷也。",
            "善战者，致人而不致于人。",
            "攻其无备，出其不意。",
            "善用兵者，修道而保法，故能为胜败之政。",
        ]
        import random
        return random.choice(quotes)

    def explain_tactic(self, tactic_name: str) -> str:
        """Get detailed explanation of a tactic."""
        tactic = get_tactic(tactic_name)
        lines = [
            f"【{tactic.name}】— {tactic.category.value}",
            f"章节：第{tactic.chapter}篇",
            f"描述：{tactic.description}",
            f"行动序列：",
        ]
        for i, step in enumerate(tactic.action_sequence, 1):
            lines.append(f"  {i}. {step}")
        lines.append(f"预期结果：{tactic.expected_outcome}")
        lines.append(f"孙子曰：{tactic.sun_tzu_quote}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"SunTzuAgent(name='{self.name}', role='{self.role}')"


# ── Demo ──────────────────────────────────────────────────────

def demo_optimization():
    """Demonstrate the unified optimization-based tactic selection."""
    print("=" * 60)
    print("BingFu Tactic Engine — Optimization Demo")
    print("=" * 60)

    from .presets import PRESET_GENERALS, get_preset
    from .agent import Agent

    engine = TacticEngine()

    # Create agents
    agents = {}
    for name in ["韩信", "白起", "诸葛亮", "项羽"]:
        profile = get_preset(name)
        agent = Agent(name=name, role="将军", profile=profile)
        agents[name] = agent

    # Test tasks
    tasks = [
        ("紧急修复生产环境的P0 bug：支付系统在高并发下出现数据不一致",
         "Code"),
        ("撰写一篇关于量子计算在药物发现中应用的深度技术博客",
         "Write"),
        ("分析2024年中国新能源汽车市场数据，预测2025年趋势并生成可视化报告",
         "Data"),
        ("设计一个微服务架构的系统，支持千万级用户的高并发访问",
         "Code"),
    ]

    for task, expected_type in tasks:
        print(f"\n{'─'*60}")
        print(f"任务 [{expected_type}]: {task[:60]}...")
        print(f"{'─'*60}")

        result = engine.select_tactic(task, agents)

        print(f"✅ 最优选择: 战术「{result.selected_tactic.name}」+ "
              f"将领「{result.selected_agent_name}」")
        print(f"   综合得分: {result.combined_score:.3f}")
        print(f"   对齐得分: {result.alignment_score:.3f}")
        print(f"   战力得分: {result.power_score:.3f}")
        print(f"   态势: {result.tactical_context.regime} "
              f"(ratio={result.tactical_context.strength_ratio:.2f})")
        print(f"   解释: {result.explanation}")

        print(f"\n   Top-3 备选方案:")
        for i, ev in enumerate(result.all_evaluations[:3], 1):
            print(f"   {i}. {ev.tactic.name} + {ev.agent_name} "
                  f"(Q={ev.combined_score:.3f}, applicable={ev.is_applicable})")


if __name__ == "__main__":
    demo_optimization()
