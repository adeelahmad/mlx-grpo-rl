"""Adaptive scaffold boundary discovery using Thompson Sampling and curve fitting.

Finds min/optimal/max scaffold ratios per (type, difficulty) pair by observing
rewards across a sweep of scaffold values. Starts with linear exploration,
then uses Thompson Sampling + quadratic fitting for exploitation.

Usage:
    optimizer = ScaffoldBoundaryOptimizer(n_exploration_samples=50)
    ratio = optimizer.select_scaffold_ratio(sample_type="math", difficulty=0.6)
    # run rollout with ratio...
    optimizer.observe_reward(sample_type="math", difficulty=0.6, ratio=ratio, reward=0.85)
    boundaries = optimizer.get_boundaries(sample_type="math", difficulty=0.6)
    # boundaries -> {"min": 0.1, "optimal": 0.5, "max": 0.8, "samples": 15}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

_log = logging.getLogger(__name__)


@dataclass
class BoundaryResult:
    """Extracted scaffold boundaries from fitted curve."""

    min_effective: float
    optimal: float
    max_effective: float
    curve_coeffs: tuple[float, float, float]  # (a, b, c) for ax^2 + bx + c
    samples_used: int
    mean_reward: float


class ScaffoldBoundaryOptimizer:
    """Finds optimal scaffold boundaries per (type, difficulty) pair."""

    def __init__(
        self,
        n_exploration_samples: int = 50,
        difficulty_buckets: list[float] | None = None,
        reward_threshold: float = 0.5,
        min_samples_for_fit: int = 5,
    ) -> None:
        """Initialize optimizer.

        Args:
            n_exploration_samples: Linear sweep samples before Thompson Sampling
            difficulty_buckets: Quantization points for difficulty [0.25, 0.5, 0.75]
            reward_threshold: Reward >= this is "effective" for min/max boundaries
            min_samples_for_fit: Don't fit curve until N observations recorded
        """
        self.n_exploration_samples = n_exploration_samples
        self.difficulty_buckets = difficulty_buckets or [0.25, 0.5, 0.75]
        self.reward_threshold = reward_threshold
        self.min_samples_for_fit = min_samples_for_fit

        # observations: {(type, bucket): [(ratio, reward), ...]}
        self.observations: dict[tuple[str, int], list[tuple[float, float]]] = {}

        # next scaffold ratio per (type, bucket)
        self.next_ratios: dict[tuple[str, int], float] = {}

    def _quantize_difficulty(self, difficulty: float) -> int:
        """Map [0, 1] difficulty to bucket index [0, N-1]."""
        if difficulty < self.difficulty_buckets[0]:
            return 0
        for i, bucket in enumerate(self.difficulty_buckets[:-1]):
            if bucket <= difficulty < self.difficulty_buckets[i + 1]:
                return i + 1
        return len(self.difficulty_buckets) - 1

    def select_scaffold_ratio(
        self,
        sample_type: str,
        difficulty: float,
    ) -> float:
        """Select next scaffold ratio to try.

        During exploration (first n_exploration_samples), uses linear sweep [0, 1].
        After exploration, uses Thompson Sampling around the observed peak.

        Args:
            sample_type: Task type (e.g., "math", "mcq")
            difficulty: Prompt difficulty [0, 1]

        Returns:
            Scaffold ratio [0, 1] to use for this sample
        """
        bucket = self._quantize_difficulty(difficulty)
        key = (sample_type, bucket)

        # First call: initialize exploration
        if key not in self.observations:
            self.observations[key] = []
            self.next_ratios[key] = 0.0
            return 0.0

        obs = self.observations[key]
        n_obs = len(obs)

        # Exploration phase: linear sweep
        if n_obs < self.n_exploration_samples:
            step = 1.0 / self.n_exploration_samples
            ratio = step * n_obs
            self.next_ratios[key] = ratio
            return ratio

        # Exploitation phase: Thompson Sampling
        if n_obs >= self.min_samples_for_fit:
            ratio = self._thompson_sample(obs)
        else:
            # Not enough samples yet, continue exploration
            ratio = np.random.uniform(0.0, 1.0)

        self.next_ratios[key] = ratio
        return ratio

    def _thompson_sample(self, observations: list[tuple[float, float]]) -> float:
        """Thompson Sampling: sample from posterior around observed peak.

        Fits quadratic curve, finds peak, samples from normal around peak.
        Bounds to [0, 1].
        """
        ratios = np.array([r for r, _ in observations])
        rewards = np.array([rw for _, rw in observations])

        # Fit quadratic
        try:
            coeffs = np.polyfit(ratios, rewards, 2)
            a, b, c = coeffs
        except Exception:
            return np.random.uniform(0.0, 1.0)

        # Peak of ax^2 + bx + c is at x = -b / (2a)
        peak = np.mean(ratios) if abs(a) < 1e-6 else -b / (2 * a)

        # Thompson: sample from normal(peak, std_of_observed_rewards)
        std = np.std(rewards) + 1e-3
        sample = np.random.normal(peak, std / 2.0)
        return float(np.clip(sample, 0.0, 1.0))

    def observe_reward(
        self,
        sample_type: str,
        difficulty: float,
        ratio: float,
        reward: float,
    ) -> None:
        """Record observation of (ratio, reward) for a sample.

        Args:
            sample_type: Task type
            difficulty: Prompt difficulty [0, 1]
            ratio: Scaffold ratio that was used
            reward: Resulting reward
        """
        bucket = self._quantize_difficulty(difficulty)
        key = (sample_type, bucket)

        if key not in self.observations:
            self.observations[key] = []

        self.observations[key].append((float(ratio), float(reward)))

    def get_boundaries(
        self,
        sample_type: str,
        difficulty: float,
    ) -> BoundaryResult | None:
        """Extract min/optimal/max scaffold boundaries from observations.

        Fits a quadratic curve to the observations, finds the peak (optimal),
        and bounds where reward >= threshold (min/max).

        Returns None if insufficient observations.
        """
        bucket = self._quantize_difficulty(difficulty)
        key = (sample_type, bucket)

        obs = self.observations.get(key, [])
        if len(obs) < self.min_samples_for_fit:
            return None

        ratios = np.array([r for r, _ in obs])
        rewards = np.array([rw for _, rw in obs])

        # Fit quadratic
        try:
            coeffs = np.polyfit(ratios, rewards, 2)
            a, b, c = coeffs
        except Exception:
            return None

        # Peak
        optimal = np.mean(ratios) if abs(a) < 1e-6 else -b / (2 * a)
        optimal = float(np.clip(optimal, 0.0, 1.0))

        # Find min/max where reward >= threshold via quadratic evaluation
        def quad_val(x: float) -> float:
            return float(a * x**2 + b * x + c)

        # Scan [0, 1] for boundary crossings
        min_eff, max_eff = self._find_boundaries(quad_val, optimal, self.reward_threshold)

        return BoundaryResult(
            min_effective=min_eff,
            optimal=optimal,
            max_effective=max_eff,
            curve_coeffs=(float(a), float(b), float(c)),
            samples_used=len(obs),
            mean_reward=float(np.mean(rewards)),
        )

    def _find_boundaries(
        self,
        quad_fn: callable,
        peak: float,
        threshold: float,
    ) -> tuple[float, float]:
        """Find left/right bounds where quad_fn(x) >= threshold."""
        # Scan left from peak
        min_bound = peak
        for x in np.linspace(peak, 0.0, 100):
            if quad_fn(x) >= threshold:
                min_bound = x
            else:
                break

        # Scan right from peak
        max_bound = peak
        for x in np.linspace(peak, 1.0, 100):
            if quad_fn(x) >= threshold:
                max_bound = x
            else:
                break

        return float(min_bound), float(max_bound)

    def get_all_boundaries(
        self,
        sample_type: str,
    ) -> dict[int, BoundaryResult | None]:
        """Get boundaries for all observed difficulty buckets of a type."""
        result = {}
        for bucket in range(len(self.difficulty_buckets)):
            key = (sample_type, bucket)
            obs = self.observations.get(key, [])
            if len(obs) >= self.min_samples_for_fit:
                difficulty = (
                    self.difficulty_buckets[bucket]
                    if bucket < len(self.difficulty_buckets)
                    else 1.0
                )
                result[bucket] = self.get_boundaries(sample_type, difficulty)
        return result
