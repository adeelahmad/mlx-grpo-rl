"""Tests for vision reward functions (task 9.3)."""

from __future__ import annotations

import pytest

from mlx_grpo.rewards.base import BaseReward
from mlx_grpo.rewards.vision import (
    VisionBleuReward,
    VisionExactMatchReward,
    VisionRougeLReward,
    VisionTokenF1Reward,
)

# ── Helper ────────────────────────────────────────────────────────────


def _make_result(text: str):
    """Create a minimal mock result with the given completion text."""
    from unittest.mock import MagicMock

    result = MagicMock()
    result.completion_text = text
    return result


# ── Registration ──────────────────────────────────────────────────────


def test_vision_rewards_registered():
    names = BaseReward.list_rewards()
    assert "vision_exact" in names
    assert "vision_f1" in names
    assert "vision_bleu" in names
    assert "vision_rouge_l" in names


def test_vision_rewards_instantiable_from_registry():
    for name in ("vision_exact", "vision_f1", "vision_bleu", "vision_rouge_l"):
        reward = BaseReward.get(name)
        assert isinstance(reward, BaseReward)


# ── VisionExactMatchReward ────────────────────────────────────────────


@pytest.fixture
def exact():
    return VisionExactMatchReward()


def test_exact_match_perfect(exact):
    assert exact.compute(_make_result("cat"), "cat") == 1.0


def test_exact_match_case_insensitive(exact):
    assert exact.compute(_make_result("Cat"), "cat") == 1.0


def test_exact_match_whitespace_stripped(exact):
    assert exact.compute(_make_result("  cat  "), "cat") == 1.0


def test_exact_mismatch(exact):
    assert exact.compute(_make_result("dog"), "cat") == 0.0


def test_exact_no_reference(exact):
    assert exact.compute(_make_result("cat"), None) == 0.0


def test_exact_empty_reference(exact):
    assert exact.compute(_make_result("cat"), "") == 0.0


# ── VisionTokenF1Reward ───────────────────────────────────────────────


@pytest.fixture
def f1_reward():
    return VisionTokenF1Reward()


def test_f1_perfect(f1_reward):
    assert f1_reward.compute(_make_result("a black cat"), "a black cat") == pytest.approx(1.0)


def test_f1_partial(f1_reward):
    score = f1_reward.compute(_make_result("black cat"), "a black cat sitting")
    assert 0.0 < score < 1.0


def test_f1_no_overlap(f1_reward):
    assert f1_reward.compute(_make_result("dog bark"), "cat meow") == 0.0


def test_f1_empty_prediction(f1_reward):
    assert f1_reward.compute(_make_result(""), "cat") == 0.0


def test_f1_empty_reference(f1_reward):
    assert f1_reward.compute(_make_result("cat"), "") == 0.0


def test_f1_no_reference(f1_reward):
    assert f1_reward.compute(_make_result("cat"), None) == 0.0


def test_f1_single_word_match(f1_reward):
    assert f1_reward.compute(_make_result("cat"), "cat") == pytest.approx(1.0)


def test_f1_repeated_tokens(f1_reward):
    # "cat cat" vs "cat" — precision=0.5, recall=1.0, F1=0.667
    score = f1_reward.compute(_make_result("cat cat"), "cat")
    assert score == pytest.approx(2 / 3, rel=1e-3)


# ── VisionBleuReward ─────────────────────────────────────────────────


@pytest.fixture
def bleu():
    return VisionBleuReward()


def test_bleu_perfect(bleu):
    assert bleu.compute(_make_result("black cat"), "black cat") == pytest.approx(1.0)


def test_bleu_partial_overlap(bleu):
    # "black cat" vs "a black cat" — pred=["black","cat"], 2/2 clipped
    score = bleu.compute(_make_result("black cat"), "a black cat")
    assert score == pytest.approx(1.0)


def test_bleu_no_overlap(bleu):
    assert bleu.compute(_make_result("dog"), "cat") == 0.0


def test_bleu_empty_prediction(bleu):
    assert bleu.compute(_make_result(""), "cat") == 0.0


def test_bleu_no_reference(bleu):
    assert bleu.compute(_make_result("cat"), None) == 0.0


def test_bleu_clipping(bleu):
    # "cat cat cat" vs "cat" — clipped count=1, precision=1/3
    score = bleu.compute(_make_result("cat cat cat"), "cat")
    assert score == pytest.approx(1 / 3, rel=1e-3)


# ── VisionRougeLReward ────────────────────────────────────────────────


@pytest.fixture
def rouge():
    return VisionRougeLReward()


def test_rouge_perfect(rouge):
    assert rouge.compute(_make_result("a black cat"), "a black cat") == pytest.approx(1.0)


def test_rouge_partial(rouge):
    score = rouge.compute(_make_result("black cat"), "a black cat")
    assert 0.0 < score < 1.0


def test_rouge_no_overlap(rouge):
    assert rouge.compute(_make_result("dog bark"), "cat meow") == 0.0


def test_rouge_empty_prediction(rouge):
    assert rouge.compute(_make_result(""), "cat") == 0.0


def test_rouge_no_reference(rouge):
    assert rouge.compute(_make_result("cat"), None) == 0.0


def test_rouge_out_of_order(rouge):
    # LCS of ["cat", "black"] vs ["black", "cat"] is 1
    score = rouge.compute(_make_result("cat black"), "black cat")
    assert 0.0 < score < 1.0


def test_rouge_subsumes_reference(rouge):
    # Pred contains all ref tokens in order → ROUGE-L recall = 1
    score = rouge.compute(_make_result("the big black cat sat on the mat"), "black cat")
    assert score > 0.0


# ── Cross-reward sanity ───────────────────────────────────────────────


def test_all_rewards_zero_on_no_answer():
    result = _make_result("something")
    for name in ("vision_exact", "vision_f1", "vision_bleu", "vision_rouge_l"):
        assert BaseReward.get(name).compute(result, None) == 0.0


def test_all_rewards_perfect_on_exact_match():
    for name in ("vision_exact", "vision_f1", "vision_bleu", "vision_rouge_l"):
        r = BaseReward.get(name).compute(_make_result("black cat"), "black cat")
        assert r == pytest.approx(1.0), f"{name} expected 1.0, got {r}"
