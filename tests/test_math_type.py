"""Tests for math type helpers — _count_answer_parts and budget scaling."""

from __future__ import annotations

import pytest

from mlx_grpo.types.builtins.math_helpers import _count_answer_parts


@pytest.mark.parametrize(
    "answer,prompt,expected",
    [
        ("42", "", 1),
        ("1,000,000", "", 1),  # formatted number — NOT multi-part
        ("a=36,b=18,c=2,d=6", "", 4),  # 4-part with = signs
        ("x=3,y=5", "", 2),
        ("", "G7.1 Find vol. G7.2 Find area.", 2),
        ("", "(a) Compute X (b) Compute Y (c) Find Z", 3),
        ("a=36,b=18", "(a) solve (b) solve (c) solve", 3),  # max(2,3)=3
        ("", "", 1),  # empty inputs → 1
        ("100", "Find the value.", 1),  # single-part prompt
        ("m=5,n=10", "", 2),  # 2-part with letters+= before comma
    ],
)
def test_count_answer_parts(answer: str, prompt: str, expected: int) -> None:
    assert _count_answer_parts(answer, prompt) == expected


def test_budget_scaling_single_part() -> None:
    """Single-part answer: no budget scaling."""
    budget = 2048
    num_parts = _count_answer_parts("42", "")
    assert num_parts == 1
    # No scaling for single-part
    scaled = int(budget * (1.0 + (num_parts - 1) * 0.5)) if num_parts > 1 else budget
    assert scaled == 2048


def test_budget_scaling_two_parts() -> None:
    """2-part answer: 1.5x scaling."""
    budget = 2048
    num_parts = _count_answer_parts("x=3,y=5", "")
    assert num_parts == 2
    scaled = int(budget * (1.0 + (num_parts - 1) * 0.5))
    assert scaled == 3072


def test_budget_scaling_four_parts() -> None:
    """4-part answer: 2.5x scaling, capped at ceiling."""
    budget = 2048
    ceiling = 4096
    num_parts = _count_answer_parts("a=36,b=18,c=2,d=6", "")
    assert num_parts == 4
    scaled = min(int(budget * (1.0 + (num_parts - 1) * 0.5)), ceiling)
    assert scaled == min(5120, ceiling)  # 5120 capped to 4096
    assert scaled == 4096


def test_recovery_budget_reservation() -> None:
    """Recovery bonus is reserved from thinking budget before cap."""
    thinking_budget = 2048
    bonus_budget = 128
    max_completion_tokens = 2048
    answer_max_tokens = 32

    # Reserve bonus
    after_reserve = max(256, thinking_budget - bonus_budget)
    assert after_reserve == 1920

    # Cap at global limit
    floor = max(512, int(max_completion_tokens * 0.75))
    ceiling = max(floor, max_completion_tokens - answer_max_tokens)
    effective = min(after_reserve, ceiling)

    # thinking(1920) + recovery(128) = 2048 == max_completion_tokens ✓
    assert effective + bonus_budget <= max_completion_tokens


def test_post_boxed_config_defaults() -> None:
    """post_boxed_tokens and post_boxed_reward_threshold in config."""
    from mlx_grpo.types.builtins.math import MathType

    cfg = MathType().config()
    assert cfg.get("post_boxed_tokens") == 0
    assert cfg.get("post_boxed_reward_threshold") == 0.8
