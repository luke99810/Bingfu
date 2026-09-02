"""
Tactic Definitions — Complete formalization of 26 tactic patterns
from Sun Tzu's 13 chapters as computable strategy primitives.

Each tactic is a first-class object with:
  - StyleVector: 5-dim vector encoding strategic emphasis
  - Precondition: applicability predicate
  - ActionSequence: ordered execution steps
  - TaskAffinity: which task categories benefit most

This module provides the TACTIC_LIBRARY — the full repertoire
of strategy patterns used by the TacticEngine optimizer.
"""

from typing import List, Dict, Callable, Optional, Tuple
from enum import Enum
import numpy as np

from .profile import CombatStyle


class TacticCategory(str, Enum):
    """13 strategic categories corresponding to Sun Tzu's 13 chapters."""
    # Ch1: 始计篇 — Estimates
    STRATEGIC_PLANNING = "庙算"          # Estimate before action
    RECONNAISSANCE = "知己知彼"           # Know yourself and enemy

    # Ch2: 作战篇 — Waging War
    RAPID_WARFARE = "兵贵速胜"            # Speed is essential
    RESOURCE_MANAGEMENT = "取用于国"       # Leverage local resources

    # Ch3: 谋攻篇 — Attack by Stratagem
    TOTAL_VICTORY = "不战而胜"            # Win without fighting
    STRATEGIC_ALLIANCE = "联横合纵"       # Build alliances

    # Ch4: 军形篇 — Tactical Dispositions
    DEFENSIVE_SUPERIORITY = "先为不可胜"   # Secure invincibility first
    TERRAIN_ADVANTAGE = "地生度"          # Exploit terrain

    # Ch5: 兵势篇 — Energy
    REGULAR_VS_SPECIAL = "奇正相生"       # Orthodox + unorthodox
    SURPRISE_ATTACK = "出其不意"          # Strike unexpectedly

    # Ch6: 虚实篇 — Weakness and Strength
    CONTROL_ENEMY = "致人而不致于人"       # Control engagement
    FLEXIBLE_RESPONSE = "因敌而变"        # Adapt like water

    # Ch7: 军争篇 — Maneuvering
    INDIRECT_APPROACH = "迁直之计"         # Turn indirect into direct
    MORALE_WINNING = "夺气夺心"           # Target morale

    # Ch8: 九变篇 — Variation in Tactics
    ADAPTABILITY = "变通在地"            # Adapt to circumstances
    STRATEGIC_FLEXIBILITY = "九变"        # Consider both sides

    # Ch9: 行军篇 — The Army on the March
    INTELLIGENCE_GATHERING = "察敌之情"   # Read enemy signals
    TERRAIN_SELECTION = "处军之利"        # Position advantageously

    # Ch10: 地形篇 — Terrain
    TERRAIN_TACTICS = "六地之用"          # Six terrain types
    SOLDIER_MANAGEMENT = "兵卒之道"       # Lead troops

    # Ch11: 九地篇 — Nine Situations
    NINE_TERRAINS = "九地之变"            # Nine ground types
    RAPID_DEPLOYMENT = "并敌一向"         # Concentrate force

    # Ch12: 火攻篇 — Fire Attack
    FIRE_ATTACK = "火攻有五"             # Five fire methods
    COORDINATED_ATTACK = "五火之变"       # Multi-modal attack

    # Ch13: 用间篇 — Use of Spies
    ESPIONAGE = "用间有五"               # Five spy types
    INTELLIGENCE_NETWORK = "五间俱起"     # Networked intelligence


class TacticDefinition:
    """
    Complete formal definition of a tactic pattern.

    A tactic is a 5-tuple:
        t = (name, φ_pre, s_vec, A_seq, ψ_post)

    where:
      - φ_pre is a precondition predicate: (context, agent) → bool
      - s_vec is a 5-dim style vector ∈ [0, 1]⁵
      - A_seq is an ordered list of action steps
      - ψ_post is the expected outcome description
    """

    def __init__(
        self,
        name: str,
        category: TacticCategory,
        chapter: int,
        description: str,
        style_vector: Tuple[float, float, float, float, float],
        combat_style: CombatStyle,
        precondition: Callable[['TacticalContext', object], bool],
        action_sequence: List[str],
        expected_outcome: str,
        task_affinity: List[str],
        sun_tzu_quote: str,
        strength_regime: str = "balanced",  # "defensive", "balanced", "offensive"
    ):
        self.name = name
        self.category = category
        self.chapter = chapter
        self.description = description
        self.style_vector = np.array(style_vector, dtype=np.float64)
        self.combat_style = combat_style
        self.precondition = precondition
        self.action_sequence = action_sequence
        self.expected_outcome = expected_outcome
        self.task_affinity = task_affinity
        self.sun_tzu_quote = sun_tzu_quote
        self.strength_regime = strength_regime

    def is_applicable(self, context: 'TacticalContext', agent_power: np.ndarray) -> bool:
        """Check if this tactic is applicable given the context and agent."""
        try:
            return self.precondition(context, agent_power)
        except Exception:
            return False

    def alignment_score(self, task_features: np.ndarray) -> float:
        """
        Compute cosine similarity between tactic style vector and task features.

        Args:
            task_features: 5-dim feature vector from TaskAssessment

        Returns:
            Cosine similarity ∈ [-1, 1]
        """
        dot = np.dot(self.style_vector, task_features)
        norm_t = np.linalg.norm(self.style_vector)
        norm_f = np.linalg.norm(task_features)
        if norm_t == 0 or norm_f == 0:
            return 0.0
        return dot / (norm_t * norm_f)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category.value,
            "chapter": self.chapter,
            "description": self.description,
            "style_vector": self.style_vector.tolist(),
            "combat_style": self.combat_style.value,
            "action_sequence": self.action_sequence,
            "expected_outcome": self.expected_outcome,
            "task_affinity": self.task_affinity,
            "strength_regime": self.strength_regime,
        }


class TacticalContext:
    """
    Battlefield state passed to tactic preconditions.
    Derived from TaskAssessment + agent pool analysis.
    """

    def __init__(
        self,
        self_strength: float = 50.0,
        enemy_strength: float = 50.0,
        terrain: str = "平原",
        weather: str = "晴",
        morale: float = 50.0,
        supplies: float = 100.0,
        time_factor: str = "白天",
        has_intelligence: bool = False,
        task_complexity: int = 5,
        required_capabilities: Optional[List[str]] = None,
        urgency: float = 0.5,
        creativity_required: float = 0.5,
        collaboration_required: float = 0.5,
    ):
        self.self_strength = self_strength
        self.enemy_strength = max(enemy_strength, 1)
        self.terrain = terrain
        self.weather = weather
        self.morale = morale
        self.supplies = supplies
        self.time_factor = time_factor
        self.has_intelligence = has_intelligence
        self.task_complexity = task_complexity
        self.required_capabilities = required_capabilities or []
        self.urgency = urgency
        self.creativity_required = creativity_required
        self.collaboration_required = collaboration_required

    @property
    def strength_ratio(self) -> float:
        return self.self_strength / self.enemy_strength

    @property
    def regime(self) -> str:
        if self.strength_ratio < 0.5:
            return "defensive"
        elif self.strength_ratio <= 2.0:
            return "balanced"
        else:
            return "offensive"

    def to_task_features(self) -> np.ndarray:
        """Convert context to 5-dim task feature vector for tactic alignment."""
        return np.array([
            self.task_complexity / 10.0,
            self.urgency,
            self.creativity_required,
            self.collaboration_required,
            self.enemy_strength / 200.0,
        ], dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "self_strength": self.self_strength,
            "enemy_strength": self.enemy_strength,
            "strength_ratio": self.strength_ratio,
            "regime": self.regime,
            "terrain": self.terrain,
            "weather": self.weather,
            "morale": self.morale,
            "supplies": self.supplies,
            "time_factor": self.time_factor,
            "has_intelligence": self.has_intelligence,
            "task_complexity": self.task_complexity,
            "urgency": self.urgency,
            "creativity_required": self.creativity_required,
            "collaboration_required": self.collaboration_required,
        }
