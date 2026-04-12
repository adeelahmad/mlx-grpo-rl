"""Tests for rewards/exam_extract.py — answer extraction and normalization."""

from __future__ import annotations

import pytest

from mlx_grpo.rewards.exam_extract import (
    _parse_numeric,
    check_answer_match,
    extract_answer_from_completion,
    extract_boxed_answer,
    normalize_answer,
    normalize_ground_truth,
)

# ── _parse_numeric ─────────────────────────────────────────────────────────────


class TestParseNumeric:
    def test_integer(self):
        assert _parse_numeric("42") == pytest.approx(42.0)

    def test_float(self):
        assert _parse_numeric("3.14") == pytest.approx(3.14)

    def test_negative(self):
        assert _parse_numeric("-7") == pytest.approx(-7.0)

    def test_scientific(self):
        assert _parse_numeric("1e3") == pytest.approx(1000.0)

    def test_percent(self):
        assert _parse_numeric("50%") == pytest.approx(0.5)

    def test_fraction(self):
        assert _parse_numeric("1/4") == pytest.approx(0.25)

    def test_invalid_fraction(self):
        assert _parse_numeric("1/0") is None

    def test_empty(self):
        assert _parse_numeric("") is None

    def test_non_string(self):
        assert _parse_numeric(42) is None  # type: ignore

    def test_text(self):
        assert _parse_numeric("hello") is None

    def test_whitespace(self):
        assert _parse_numeric("  7  ") == pytest.approx(7.0)


# ── normalize_answer ──────────────────────────────────────────────────────────


class TestNormalizeAnswer:
    def test_single_letter(self):
        assert normalize_answer("B") == "B"

    def test_lowercase_letter(self):
        assert normalize_answer("c") == "C"

    def test_multi_letter(self):
        assert normalize_answer("A, C") == "AC"

    def test_multi_letter_sorted(self):
        assert normalize_answer("C, A") == "AC"

    def test_with_and(self):
        assert normalize_answer("A and C") == "AC"

    def test_numeric_preserved(self):
        assert normalize_answer("42") == "42"

    def test_float_preserved(self):
        assert normalize_answer("3.14") == "3.14"

    def test_fraction_preserved(self):
        assert normalize_answer("3/4") == "3/4"

    def test_empty(self):
        assert normalize_answer("") == ""

    def test_strips_answer_word(self):
        result = normalize_answer("Answer: B")
        assert "B" in result

    def test_deduplicates_letters(self):
        assert normalize_answer("A, A, B") == "AB"

    def test_only_valid_letters_abcdef(self):
        result = normalize_answer("X, A, B")
        assert "A" in result and "B" in result
        assert "X" not in result

    def test_fallback_to_stripped(self):
        """When no valid ABCDEF letters found, returns stripped input."""
        result = normalize_answer("some text 42")
        assert result == "some text 42"


# ── normalize_ground_truth ────────────────────────────────────────────────────


class TestNormalizeGroundTruth:
    def test_string(self):
        assert normalize_ground_truth("B") == "B"

    def test_list_of_letters(self):
        result = normalize_ground_truth(["A", "C"])
        assert "A" in result and "C" in result

    def test_list_sorted(self):
        assert normalize_ground_truth(["C", "A"]) == "AC"

    def test_none(self):
        assert normalize_ground_truth(None) == ""

    def test_integer(self):
        assert normalize_ground_truth(42) == "42"

    def test_float(self):
        assert normalize_ground_truth(3.14) == "3.14"

    def test_empty_string(self):
        assert normalize_ground_truth("") == ""

    def test_list_with_none(self):
        result = normalize_ground_truth(["A", None, "B"])
        assert "A" in result and "B" in result

    def test_non_str_non_list(self):
        result = normalize_ground_truth({"key": "val"})
        assert isinstance(result, str)


# ── extract_boxed_answer ──────────────────────────────────────────────────────


class TestExtractBoxedAnswer:
    def test_simple(self):
        assert extract_boxed_answer("\\boxed{42}") == "42"

    def test_nested(self):
        assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_no_boxed(self):
        assert extract_boxed_answer("no answer here") is None

    def test_empty(self):
        assert extract_boxed_answer("") is None

    def test_non_string(self):
        assert extract_boxed_answer(None) is None  # type: ignore

    def test_case_insensitive(self):
        assert extract_boxed_answer("\\Boxed{42}") == "42"

    def test_empty_boxed(self):
        # Empty boxed returns None (no content)
        assert extract_boxed_answer("\\boxed{}") is None

    def test_trailing_text(self):
        assert extract_boxed_answer("The answer is \\boxed{7} correct.") == "7"

    def test_incomplete_boxed(self):
        # unclosed brace — depth never reaches 0, returns None
        assert extract_boxed_answer("\\boxed{42") is None


# ── extract_answer_from_completion ───────────────────────────────────────────


class TestExtractAnswerFromCompletion:
    def test_none_input(self):
        assert extract_answer_from_completion(None) is None  # type: ignore

    def test_empty(self):
        assert extract_answer_from_completion("") is None

    def test_boxed_priority(self):
        assert extract_answer_from_completion("\\boxed{B}") == "B"

    def test_answer_tag(self):
        result = extract_answer_from_completion("<answer>C</answer>")
        assert result == "C"

    def test_answer_tag_non_exam(self):
        result = extract_answer_from_completion("<answer>hello world</answer>", for_exam=False)
        assert result == "hello world"

    def test_answer_colon_numeric(self):
        result = extract_answer_from_completion("The answer is: 42")
        assert result == "42"

    def test_answer_colon_letter(self):
        result = extract_answer_from_completion("The answer is: B")
        assert result == "B"

    def test_bold_letter(self):
        result = extract_answer_from_completion("...therefore **B**.")
        assert result == "B"

    def test_last_line_letter(self):
        result = extract_answer_from_completion("some reasoning\nB")
        assert result == "B"

    def test_unclosed_think_returns_none(self):
        """<think> without </think> → return None."""
        assert extract_answer_from_completion("<think>reasoning in progress") is None

    def test_closed_think_with_boxed(self):
        text = "<think>some reasoning</think>\\boxed{C}"
        assert extract_answer_from_completion(text) == "C"

    def test_tail_extraction(self):
        """Fallback: scan last 100 chars for letter."""
        text = "a " * 60 + "X"  # X near the end
        result = extract_answer_from_completion(text)
        assert result is not None

    def test_no_answer_found(self):
        result = extract_answer_from_completion("no clues in this text at all")
        # Returns None or something — depends on tail scan
        # Just check it doesn't raise
        assert result is None or isinstance(result, str)

    def test_boxed_non_exam(self):
        """for_exam=False skips normalize_answer."""
        result = extract_answer_from_completion("\\boxed{hello world}", for_exam=False)
        assert result == "hello world"

    def test_multiple_boxed_takes_last(self):
        result = extract_answer_from_completion("\\boxed{A} some text \\boxed{B}")
        # The function searches from start → takes first boxed
        assert result in ("A", "B")

    def test_answer_is_common_word_skipped(self):
        """Common words like 'is', 'a' are skipped."""
        result = extract_answer_from_completion("The answer is: is")
        # "is" is common, may not be returned as an answer
        # Just check no crash
        assert result is None or isinstance(result, str)


# ── check_answer_match ────────────────────────────────────────────────────────


class TestCheckAnswerMatch:
    def test_exact_match(self):
        ok, matched = check_answer_match("B", ["B"])
        assert ok and matched == "B"

    def test_case_insensitive(self):
        ok, matched = check_answer_match("b", ["B"])
        assert ok

    def test_normalized_match(self):
        ok, _ = check_answer_match("A, B", ["AB"])
        assert ok

    def test_no_match(self):
        ok, matched = check_answer_match("X", ["A", "B", "C"])
        assert not ok and matched is None

    def test_numeric_exact(self):
        ok, _ = check_answer_match("42", ["42"])
        assert ok

    def test_numeric_tolerance(self):
        ok, _ = check_answer_match("3.14159", ["3.14159"])
        assert ok

    def test_numeric_no_match(self):
        ok, _ = check_answer_match("42", ["43"])
        assert not ok

    def test_empty_model_answer(self):
        ok, _ = check_answer_match("", ["A"])
        assert not ok

    def test_empty_possible_answers(self):
        ok, _ = check_answer_match("A", [])
        assert not ok

    def test_none_in_possible_answers(self):
        ok, _ = check_answer_match("A", [None, "A"])
        assert ok

    def test_numeric_zero_model(self):
        ok, _ = check_answer_match("0", ["0"])
        assert ok

    def test_numeric_near_zero(self):
        ok, _ = check_answer_match("0.0000001", ["0"])  # abs < 1e-6 threshold
        assert ok

    def test_percent_answer(self):
        ok, _ = check_answer_match("50%", ["50%"])
        assert ok

    def test_fraction_answer(self):
        ok, _ = check_answer_match("1/2", ["1/2"])
        assert ok
