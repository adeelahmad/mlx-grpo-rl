"""Tests for scaffold boundary optimizer and integration."""

from __future__ import annotations

import numpy as np

from mlx_grpo.training.prompt_difficulty_estimator import PromptDifficultyEstimator
from mlx_grpo.training.scaffold_boundary_optimizer import (
    BoundaryResult,
    ScaffoldBoundaryOptimizer,
)


def test_boundary_optimizer_exploration():
    """Test linear exploration phase."""
    optimizer = ScaffoldBoundaryOptimizer(n_exploration_samples=5)

    # First 5 calls should return linear sweep
    expected = [0.0, 0.2, 0.4, 0.6, 0.8]
    for _i, exp_ratio in enumerate(expected):
        ratio = optimizer.select_scaffold_ratio("math", 0.5)
        assert abs(ratio - exp_ratio) < 1e-6, f"Expected {exp_ratio}, got {ratio}"

        # Record observation
        optimizer.observe_reward("math", 0.5, ratio, 0.5)


def test_boundary_optimizer_thompson_sampling():
    """Test Thompson Sampling after exploration."""
    optimizer = ScaffoldBoundaryOptimizer(n_exploration_samples=3)

    # Do 3 exploration steps
    for _ in range(3):
        ratio = optimizer.select_scaffold_ratio("math", 0.5)
        # Simulate peak around 0.5
        reward = 0.5 - (ratio - 0.5) ** 2
        optimizer.observe_reward("math", 0.5, ratio, reward)

    # Now should do Thompson Sampling
    # Multiple calls should explore around the peak
    ratios = []
    for _ in range(10):
        ratio = optimizer.select_scaffold_ratio("math", 0.5)
        reward = 0.5 - (ratio - 0.5) ** 2 + np.random.normal(0, 0.02)
        reward = max(0, min(1, reward))
        optimizer.observe_reward("math", 0.5, ratio, reward)
        ratios.append(ratio)

    # Should have explored in Thompson phase
    assert len(set(r for r in ratios[:5])) > 1, "Thompson Sampling not exploring"


def test_boundary_optimizer_curve_fitting():
    """Test quadratic curve fitting."""
    optimizer = ScaffoldBoundaryOptimizer(n_exploration_samples=10)

    # Create synthetic observations with quadratic curve peaking at 0.5
    np.random.seed(42)
    for ratio in np.linspace(0, 1, 15):
        reward = 0.5 - (ratio - 0.5) ** 2 + np.random.normal(0, 0.05)
        reward = max(0, min(1, reward))
        optimizer.observe_reward("math", 0.6, ratio, reward)

    boundaries = optimizer.get_boundaries("math", 0.6)
    assert boundaries is not None, "Should have boundaries with 15 samples"
    assert abs(boundaries.optimal - 0.5) < 0.15, "Peak should be near 0.5"
    assert boundaries.min_effective < boundaries.optimal, "Min should be left of peak"
    assert boundaries.max_effective > boundaries.optimal, "Max should be right of peak"


def test_boundary_optimizer_difficulty_buckets():
    """Test difficulty bucketing."""
    optimizer = ScaffoldBoundaryOptimizer(
        difficulty_buckets=[0.33, 0.67],
    )

    # Same type, different difficulties should use different buckets
    optimizer.select_scaffold_ratio("math", 0.1)  # Bucket 0
    optimizer.select_scaffold_ratio("math", 0.5)  # Bucket 1
    optimizer.select_scaffold_ratio("math", 0.9)  # Bucket 1 (last bucket)

    # Observe in first bucket
    optimizer.observe_reward("math", 0.1, 0.5, 0.8)

    # Should only have 1 observation in first bucket
    key = ("math", 0)
    assert len(optimizer.observations.get(key, [])) == 1


def test_difficulty_estimator_basic():
    """Test difficulty estimation."""
    estimator = PromptDifficultyEstimator()

    # Short, simple MCQ should be easy
    easy = estimator.estimate(
        prompt="What is 1+1?",
        answer="2",
        sample_type="mcq",
    )

    # Long, complex math should be hard
    hard = estimator.estimate(
        prompt="Prove that for all n > 0, the sum of the first n cubes equals (n(n+1)/2)^2.",
        answer="By mathematical induction...",
        sample_type="math",
    )

    assert hard > easy, "Math should be harder than simple MCQ"
    assert 0.0 <= easy <= 1.0, "Difficulty should be in [0, 1]"
    assert 0.0 <= hard <= 1.0, "Difficulty should be in [0, 1]"


def test_difficulty_estimator_type_weights():
    """Test that different types have different base difficulties."""
    estimator = PromptDifficultyEstimator()

    same_prompt = "Same test prompt for all types"
    same_answer = "Same answer"

    mcq_diff = estimator.estimate(
        prompt=same_prompt,
        answer=same_answer,
        sample_type="mcq",
    )
    math_diff = estimator.estimate(
        prompt=same_prompt,
        answer=same_answer,
        sample_type="math",
    )

    # Math should be harder than MCQ due to base type difficulty
    assert math_diff > mcq_diff, "Math base difficulty should exceed MCQ"


def test_boundary_result_structure():
    """Test BoundaryResult dataclass."""
    result = BoundaryResult(
        min_effective=0.2,
        optimal=0.5,
        max_effective=0.8,
        curve_coeffs=(-1.0, 1.0, 0.0),
        samples_used=10,
        mean_reward=0.6,
    )

    assert result.min_effective < result.optimal
    assert result.optimal < result.max_effective
    assert result.samples_used == 10
    assert result.mean_reward == 0.6


def test_boundary_optimizer_multiple_types():
    """Test that different task types are tracked separately."""
    optimizer = ScaffoldBoundaryOptimizer(n_exploration_samples=3)

    # Sample for math type at difficulty 0.1 (bucket 0)
    for _ in range(5):
        ratio = optimizer.select_scaffold_ratio("math", 0.1)
        optimizer.observe_reward("math", 0.1, ratio, 0.7)

    # Sample for mcq type at difficulty 0.1 (bucket 0)
    for _ in range(5):
        ratio = optimizer.select_scaffold_ratio("mcq", 0.1)
        optimizer.observe_reward("mcq", 0.1, ratio, 0.9)

    # Should have separate observations for each type
    math_key = ("math", 0)
    mcq_key = ("mcq", 0)
    assert len(optimizer.observations[math_key]) == 5
    assert len(optimizer.observations[mcq_key]) == 5


def test_boundary_optimizer_no_early_return():
    """Test that boundaries are None with insufficient samples."""
    optimizer = ScaffoldBoundaryOptimizer(min_samples_for_fit=10)

    # Add only 2 samples
    optimizer.observe_reward("math", 0.5, 0.3, 0.7)
    optimizer.observe_reward("math", 0.5, 0.7, 0.6)

    # Should return None
    result = optimizer.get_boundaries("math", 0.5)
    assert result is None, "Should not fit with only 2 samples"

    # Add 8 more samples
    for _ in range(8):
        optimizer.observe_reward("math", 0.5, 0.5, 0.8)

    # Now should fit
    result = optimizer.get_boundaries("math", 0.5)
    assert result is not None, "Should fit with 10 samples"
