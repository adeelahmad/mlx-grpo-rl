"""Hierarchical reward aggregator — composes multiple rewards.

Combines multiple reward functions with configurable weights,
gating, and anti-gaming mechanisms.

    hierarchical = HierarchicalReward(
        rewards=[
            ("accuracy", AccuracyReward(), 0.5),
            ("format", FormatReward(), 0.2),
            ("reasoning", ReasoningReward(), 0.2),
            ("efficiency", EfficiencyReward(), 0.1),
        ],
        gate_threshold=0.1,  # Zero all if accuracy < 0.1
        gate_reward="accuracy",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..engine.result import RolloutResult
from .base import BaseReward


@dataclass(slots=True)
class RewardBreakdown:
    """Detailed breakdown of hierarchical reward computation."""

    total: float
    components: dict[str, float]
    weights: dict[str, float]
    gated: bool = False
    raw_components: dict[str, float] = field(default_factory=dict)


class HierarchicalReward(BaseReward, name="hierarchical"):
    """Multi-tier reward aggregator with gating and smoothing."""

    def __init__(
        self,
        rewards: list[tuple[str, BaseReward, float]] | None = None,
        gate_reward: str | None = None,
        gate_threshold: float = 0.1,
        smoothing: float = 0.0,
        anti_gaming: bool = True,
    ):
        """
        Args:
            rewards: List of (name, reward_fn, weight) tuples
            gate_reward: Name of gating reward (zeros all if below threshold)
            gate_threshold: Minimum gate reward to enable other rewards
            smoothing: EMA smoothing factor (0 = no smoothing)
            anti_gaming: Enable anti-gaming mechanisms
        """
        self._rewards = rewards or []
        self.gate_reward = gate_reward
        self.gate_threshold = gate_threshold
        self.smoothing = smoothing
        self.anti_gaming = anti_gaming
        self._ema: dict[str, float] = {}

    def add(self, name: str, reward: BaseReward, weight: float = 1.0) -> HierarchicalReward:
        """Add a reward component. Returns self for chaining."""
        self._rewards.append((name, reward, weight))
        return self

    def compute(self, result: RolloutResult, answer: str | None = None) -> float:
        return self.compute_detailed(result, answer).total

    def compute_detailed(
        self,
        result: RolloutResult,
        answer: str | None = None,
    ) -> RewardBreakdown:
        """Compute reward with full breakdown."""
        if not self._rewards:
            return RewardBreakdown(0.0, {}, {})

        # Compute all component rewards
        raw: dict[str, float] = {}
        for name, reward_fn, _ in self._rewards:
            try:
                raw[name] = reward_fn.compute(result, answer)
            except Exception:
                raw[name] = 0.0

        # Gating: if gate reward is below threshold, zero everything
        if (
            self.gate_reward
            and self.gate_reward in raw
            and raw[self.gate_reward] < self.gate_threshold
        ):
            return RewardBreakdown(
                total=0.0,
                components={name: 0.0 for name, _, _ in self._rewards},
                weights={name: w for name, _, w in self._rewards},
                gated=True,
                raw_components=raw,
            )

        # Anti-gaming: penalize perfect scores on format without accuracy
        if self.anti_gaming:
            raw = self._apply_anti_gaming(raw)

        # Weighted combination
        total_weight = sum(w for _, _, w in self._rewards)
        if total_weight == 0:
            return RewardBreakdown(0.0, raw, {})

        components: dict[str, float] = {}
        weighted_sum = 0.0
        for name, _, weight in self._rewards:
            value = raw.get(name, 0.0)
            # Optional EMA smoothing
            if self.smoothing > 0 and name in self._ema:
                value = self.smoothing * self._ema[name] + (1 - self.smoothing) * value
            self._ema[name] = value

            components[name] = value
            weighted_sum += weight * value

        total = weighted_sum / total_weight

        return RewardBreakdown(
            total=max(0.0, min(1.0, total)),
            components=components,
            weights={name: w for name, _, w in self._rewards},
            gated=False,
            raw_components=raw,
        )

    def _apply_anti_gaming(self, raw: dict[str, float]) -> dict[str, float]:
        """Anti-gaming: prevent high format/reasoning scores with zero accuracy."""
        accuracy = raw.get("accuracy")
        if accuracy is not None and accuracy < 0.1:
            # Cap non-accuracy rewards when accuracy is near zero
            capped = dict(raw)
            for key in capped:
                if key != "accuracy":
                    capped[key] = min(capped[key], 0.3)
            return capped
        return raw
