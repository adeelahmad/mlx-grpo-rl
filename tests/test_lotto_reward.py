"""Tests for rewards/lotto.py — TattsLotto multi-component reward."""

from __future__ import annotations

import pytest

from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource
from mlx_grpo.rewards.lotto import (
    LottoReward,
    _combine,
    _degen_cap,
    _extract_pred,
    _gaming_penalty,
    _hard_zero,
    _parse_nums,
    _score_format,
    _score_near,
    _score_positional,
    _score_supp,
)


def _r(text: str) -> RolloutResult:
    return RolloutResult(
        segments=[Segment([ord(c) for c in text], text, SegmentSource.GENERATED, False)]
    )


def _valid_box(nums: list[int]) -> str:
    """Make a valid \\boxed{} answer with 6 numbers 1-45."""
    return f"\\boxed{{{' '.join(str(n) for n in nums)}}}"


def _valid_text(nums: list[int], think: bool = True) -> str:
    box = _valid_box(nums)
    if think:
        return f"<think>some reasoning here</think>\n{box}"
    return box


# ── _parse_nums ────────────────────────────────────────────────────────────────


class TestParseNums:
    def test_basic(self):
        assert _parse_nums("1 2 3 4 5 6") == [1, 2, 3, 4, 5, 6]

    def test_deduplicates(self):
        assert len(_parse_nums("1 1 2 3 4 5")) <= 6

    def test_filters_out_of_range(self):
        result = _parse_nums("0 46 1 2 3 4 5 6")
        assert all(1 <= n <= 45 for n in result)

    def test_takes_at_most_6(self):
        assert len(_parse_nums("1 2 3 4 5 6 7 8")) == 6

    def test_sorted(self):
        result = _parse_nums("6 5 4 3 2 1")
        assert result == sorted(result)

    def test_empty(self):
        assert _parse_nums("no numbers here") == []

    def test_single_number(self):
        assert _parse_nums("7") == [7]


# ── _extract_pred ──────────────────────────────────────────────────────────────


class TestExtractPred:
    def test_from_boxed(self):
        text = "\\boxed{1 2 3 4 5 6}"
        assert _extract_pred(text) == [1, 2, 3, 4, 5, 6]

    def test_fallback_no_boxed(self):
        text = "1 2 3 4 5 6"
        assert _extract_pred(text) == [1, 2, 3, 4, 5, 6]


# ── _hard_zero ─────────────────────────────────────────────────────────────────


class TestHardZero:
    def test_no_boxed(self):
        assert _hard_zero("1 2 3 4 5 6", [1, 2, 3, 4, 5, 6]) is True

    def test_wrong_count(self):
        assert _hard_zero("\\boxed{1 2 3 4 5}", [1, 2, 3, 4, 5]) is True

    def test_out_of_range(self):
        assert _hard_zero("\\boxed{0 2 3 4 5 6}", []) is True

    def test_duplicates(self):
        assert _hard_zero("\\boxed{1 1 2 3 4 5}", []) is True

    def test_valid(self):
        assert _hard_zero("\\boxed{1 2 3 4 5 6}", [1, 2, 3, 4, 5, 6]) is False


# ── _score_positional ──────────────────────────────────────────────────────────


class TestScorePositional:
    def test_no_matches(self):
        assert _score_positional([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]) == 0.0

    def test_all_match(self):
        assert _score_positional([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]) == 1.0

    def test_empty_pred(self):
        assert _score_positional([], [1, 2, 3]) == 0.0

    def test_empty_gt(self):
        assert _score_positional([1, 2, 3], []) == 0.0

    def test_partial_match(self):
        score = _score_positional([1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 5, 6])
        assert 0.0 < score < 1.0


# ── _score_near ────────────────────────────────────────────────────────────────


class TestScoreNear:
    def test_exact_match(self):
        score = _score_near([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
        assert score == 1.0

    def test_near_miss(self):
        # Each pred is within radius=2 of gt
        score = _score_near([2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6])
        assert score > 0.5  # partial credit for near misses

    def test_no_match(self):
        score = _score_near([1, 2, 3, 4, 5, 6], [40, 41, 42, 43, 44, 45])
        assert score == 0.0

    def test_empty(self):
        assert _score_near([], [1, 2, 3]) == 0.0
        assert _score_near([1, 2, 3], []) == 0.0


# ── _score_supp ────────────────────────────────────────────────────────────────


class TestScoreSupp:
    def test_no_supp(self):
        assert _score_supp([1, 2, 3, 4, 5, 6], []) == 0.0

    def test_match(self):
        assert _score_supp([1, 2, 3, 4, 5, 40], [40]) == pytest.approx(0.5)

    def test_two_matches(self):
        score = _score_supp([1, 2, 3, 4, 40, 41], [40, 41])
        assert score == pytest.approx(1.0)

    def test_out_of_range_supp_ignored(self):
        assert _score_supp([1, 2, 3, 4, 5, 6], [0, 46]) == 0.0

    def test_empty_pred(self):
        assert _score_supp([], [40]) == 0.0


# ── _score_format ──────────────────────────────────────────────────────────────


class TestScoreFormat:
    def test_no_structure(self):
        assert _score_format("1 2 3 4 5 6") == 0.0

    def test_has_think_open(self):
        score = _score_format("<think>reasoning")
        assert score > 0.0

    def test_has_think_close(self):
        score = _score_format("</think>")
        assert score > 0.0

    def test_has_boxed(self):
        score = _score_format("\\boxed{1 2 3 4 5 6}")
        assert score > 0.0

    def test_full_format(self):
        text = "<think>\nsome long reasoning here\n</think>\n\\boxed{1 2 3 4 5 6}"
        score = _score_format(text)
        assert score > 0.5

    def test_trailing_text_penalty(self):
        text = "\\boxed{1 2 3 4 5 6} followed by more text here that is too long"
        score_no_trailing = _score_format("\\boxed{1 2 3 4 5 6}")
        score_trailing = _score_format(text)
        assert score_trailing < score_no_trailing


# ── _gaming_penalty ────────────────────────────────────────────────────────────


class TestGamingPenalty:
    def test_no_penalty(self):
        assert _gaming_penalty("clean text", [1, 5, 12, 23, 34, 42]) == 0.0

    def test_double_boxed(self):
        text = "\\boxed{1 2 3} \\boxed{4 5 6}"
        pen = _gaming_penalty(text, [])
        assert pen >= 0.3

    def test_hedging_words(self):
        text = "maybe 1 2 3 4 5 6"
        pen = _gaming_penalty(text, [])
        assert pen >= 0.3

    def test_eos_token(self):
        text = "<|endoftext|>"
        pen = _gaming_penalty(text, [])
        assert pen >= 0.3

    def test_consecutive_numbers(self):
        # Numbers 1-6 form consecutive sequence
        pen = _gaming_penalty("text", [1, 2, 3, 4, 5, 6])
        assert pen >= 0.5

    def test_clustered_small(self):
        pen = _gaming_penalty("text", [1, 2, 3, 4, 5, 7])
        assert pen >= 0.25

    def test_all_small(self):
        pen = _gaming_penalty("text", [1, 2, 3, 4, 5, 8])
        assert pen >= 0.35

    def test_all_large(self):
        pen = _gaming_penalty("text", [35, 36, 37, 38, 39, 40])
        assert pen >= 0.35

    def test_lazy_pattern(self):
        pen = _gaming_penalty("text", [1, 2, 3, 4, 5, 6])
        assert pen > 0.0

    def test_capped_at_1(self):
        # Even worst case penalties don't exceed 1.0
        text = "maybe \\boxed{x} \\boxed{y} <|endoftext|>"
        assert _gaming_penalty(text, [1, 2, 3, 4, 5, 6]) <= 1.0


# ── _degen_cap ─────────────────────────────────────────────────────────────────


class TestDegenCap:
    def test_no_cap_normal(self):
        assert _degen_cap([3, 8, 15, 23, 31, 42]) is None

    def test_all_consecutive(self):
        assert _degen_cap([1, 2, 3, 4, 5, 6]) == pytest.approx(0.02)

    def test_four_consecutive(self):
        cap = _degen_cap([1, 2, 3, 4, 5, 10])
        assert cap is not None and cap <= 0.05

    def test_arithmetic_seq(self):
        cap = _degen_cap([5, 10, 15, 20, 25, 30])
        assert cap is not None

    def test_all_low(self):
        cap = _degen_cap([1, 3, 5, 7, 9, 13])
        assert cap is not None

    def test_all_high(self):
        cap = _degen_cap([30, 33, 36, 38, 40, 43])
        assert cap is not None

    def test_too_clustered(self):
        cap = _degen_cap([10, 11, 12, 13, 14, 18])
        assert cap is not None

    def test_short_list(self):
        assert _degen_cap([1, 2, 3]) is None

    def test_lazy_list(self):
        cap = _degen_cap([1, 2, 3, 4, 5, 6])
        assert cap == pytest.approx(0.02)


# ── _combine ──────────────────────────────────────────────────────────────────


class TestCombine:
    def test_no_supp(self):
        scores = {"value": 1.0, "positional": 1.0, "near": 1.0, "supp": 0.0, "fmt": 1.0}
        result = _combine(scores, has_supp=False)
        # without supp weight, total should still be 1.0
        assert result == pytest.approx(1.0)

    def test_with_supp(self):
        scores = {"value": 1.0, "positional": 1.0, "near": 1.0, "supp": 1.0, "fmt": 1.0}
        result = _combine(scores, has_supp=True)
        assert result == pytest.approx(1.0)

    def test_all_zero(self):
        scores = {"value": 0.0, "positional": 0.0, "near": 0.0, "supp": 0.0, "fmt": 0.0}
        assert _combine(scores, False) == pytest.approx(0.0)

    def test_partial(self):
        scores = {"value": 1.0, "positional": 0.0, "near": 0.0, "supp": 0.0, "fmt": 0.0}
        result = _combine(scores, False)
        assert 0.0 < result < 1.0


# ── LottoReward (integration) ─────────────────────────────────────────────────


class TestLottoReward:
    def setup_method(self):
        self.r = LottoReward()

    def test_no_text(self):
        assert self.r.compute(_r(""), "1 2 3 4 5 6") == 0.0

    def test_no_answer(self):
        assert self.r.compute(_r("\\boxed{1 2 3 4 5 6}"), None) == 0.0

    def test_hard_zero_no_boxed(self):
        assert self.r.compute(_r("1 2 3 4 5 6"), "1 2 3 4 5 6") == 0.0

    def test_hard_zero_wrong_count(self):
        assert self.r.compute(_r("\\boxed{1 2 3 4 5}"), "1 2 3 4 5 6") == 0.0

    def test_hard_zero_out_of_range(self):
        assert self.r.compute(_r("\\boxed{0 2 3 4 5 6}"), "1 2 3 4 5 6") == 0.0

    def test_hard_zero_duplicates(self):
        assert self.r.compute(_r("\\boxed{1 1 2 3 4 5}"), "1 2 3 4 5 6") == 0.0

    def test_perfect_match(self):
        nums = [3, 8, 15, 23, 34, 42]
        text = _valid_text(nums)
        answer = " ".join(str(n) for n in nums)
        score = self.r.compute(_r(text), answer)
        assert score > 0.7  # high score for perfect match with good format

    def test_no_match(self):
        pred_nums = [1, 2, 3, 4, 5, 6]
        gt_nums = [10, 20, 30, 40, 41, 45]
        text = _valid_text(pred_nums, think=False)
        answer = " ".join(str(n) for n in gt_nums)
        score = self.r.compute(_r(text), answer)
        # Capped due to consecutive [1,2,3,4,5,6] pattern
        assert score <= 0.1

    def test_answer_as_boxed(self):
        nums = [3, 8, 15, 23, 34, 42]
        text = _valid_text(nums, think=False)
        answer = f"\\boxed{{{' '.join(str(n) for n in nums)}}}"
        score = self.r.compute(_r(text), answer)
        assert score > 0.0

    def test_gaming_penalty_applied(self):
        nums = [3, 8, 15, 23, 34, 42]
        text = f"maybe \\boxed{{{' '.join(str(n) for n in nums)}}}"
        answer = " ".join(str(n) for n in nums)
        score_clean = self.r.compute(_r(_valid_text(nums, think=False)), answer)
        score_gaming = self.r.compute(_r(text), answer)
        assert score_gaming < score_clean

    def test_degen_cap_applied(self):
        nums = [1, 2, 3, 4, 5, 6]
        text = _valid_text(nums)
        answer = "1 2 3 4 5 6"
        score = self.r.compute(_r(text), answer)
        assert score <= 0.05  # capped at 0.02 (all consecutive)

    def test_supplementary_score(self):
        nums = [3, 8, 15, 23, 34, 42]
        text = _valid_text(nums)
        answer = " ".join(str(n) for n in nums)
        result = _r(text)
        result.metadata["_supp_ground_truth"] = [42]
        score_with_supp = self.r.compute(result, answer)
        assert score_with_supp > 0.0

    def test_positional_gt_override(self):
        nums = [3, 8, 15, 23, 34, 42]
        text = _valid_text(nums)
        result = _r(text)
        result.metadata["_positional_ground_truth"] = [3, 8, 15, 23, 34, 42]
        answer = " ".join(str(n) for n in nums)
        score = self.r.compute(result, answer)
        assert score > 0.5

    def test_returns_between_0_and_1(self):
        for nums in [[3, 8, 15, 23, 34, 42], [1, 5, 10, 20, 30, 40]]:
            text = _valid_text(nums)
            score = self.r.compute(_r(text), " ".join(str(n) for n in nums))
            assert 0.0 <= score <= 1.0
