"""Tests for adaptive escalation budget and prefix helpers."""

from __future__ import annotations

import pytest

from mlx_grpo.training import _prefix_from_thinking
from mlx_grpo.types.builtins.math_helpers import _compute_thinking_budget

# ── _compute_thinking_budget ──────────────────────────────────────────


@pytest.mark.parametrize("is_scaffolded,expected_bonus", [(True, 128), (False, 0)])
def test_budget_bonus_only_for_scaffolded(is_scaffolded: bool, expected_bonus: int) -> None:
    cfg = {"thinking_budget": 2048, "recovery_bonus_tokens": 128, "answer_max_tokens": 128}
    _, bonus = _compute_thinking_budget(cfg, "x=3,y=5", "", 0, is_scaffolded)
    assert bonus == expected_bonus


def test_budget_reserves_room_for_answer() -> None:
    cfg = {"thinking_budget": 4096, "answer_max_tokens": 128}
    budget, _ = _compute_thinking_budget(cfg, "42", "", 2048, False)
    assert budget <= 2048 - 128


def test_budget_multipart_scales_up() -> None:
    cfg = {"thinking_budget": 2048, "thinking_budget_ceiling": 4096, "answer_max_tokens": 128}
    b1, _ = _compute_thinking_budget(cfg, "42", "", 0, False)
    b4, _ = _compute_thinking_budget(cfg, "a=1,b=2,c=3,d=4", "", 0, False)
    assert b4 > b1


def test_budget_reservation_prevents_starvation() -> None:
    # bonus is reserved; thinking budget alone must fit within cap minus answer_reserve
    cfg = {"thinking_budget": 2048, "recovery_bonus_tokens": 128, "answer_max_tokens": 32}
    budget, bonus = _compute_thinking_budget(cfg, "42", "", 2048, True)
    assert bonus == 128
    assert budget <= 2048 - 32  # thinking phase alone leaves room for answer


def test_budget_no_completion_cap_when_zero() -> None:
    cfg = {"thinking_budget": 1024}
    budget, _ = _compute_thinking_budget(cfg, "42", "", 0, False)
    assert budget == 1024


def test_budget_floor_prevents_tiny_window() -> None:
    # max_completion_tokens=600 → floor=max(512, 450)=512; ceiling=max(512,472)=512
    cfg = {"thinking_budget": 4096, "answer_max_tokens": 128}
    budget, _ = _compute_thinking_budget(cfg, "42", "", 600, False)
    assert budget >= 512


# ── _prefix_from_thinking ─────────────────────────────────────────────


class _MockBackend:
    """Word-tokenizing backend for deterministic tests."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


def test_prefix_strips_think_tag() -> None:
    b = _MockBackend()
    out = _prefix_from_thinking("<think>word1 word2 word3", 2, b)
    assert "word1" in out
    assert "word3" not in out


def test_prefix_zero_returns_empty() -> None:
    assert _prefix_from_thinking("some text", 0, _MockBackend()) == ""


def test_prefix_longer_than_text() -> None:
    b = _MockBackend()
    out = _prefix_from_thinking("a b c", 100, b)
    assert "a" in out and "b" in out and "c" in out


def test_prefix_no_think_tag() -> None:
    b = _MockBackend()
    out = _prefix_from_thinking("step1 step2 step3", 2, b)
    assert "step1" in out
    assert "step3" not in out


def test_prefix_empty_text_returns_empty() -> None:
    assert _prefix_from_thinking("", 10, _MockBackend()) == ""


def test_prefix_only_think_tag() -> None:
    assert _prefix_from_thinking("<think>", 10, _MockBackend()) == ""
