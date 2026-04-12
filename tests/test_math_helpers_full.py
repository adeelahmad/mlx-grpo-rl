"""Tests for types/builtins/math_helpers.py."""

from __future__ import annotations

from mlx_grpo.types.builtins.math_helpers import (
    _compute_thinking_budget,
    _count_answer_parts,
    _extract_boxed,
    _extract_last_boxed,
    _normalize_math_answer,
)

# ── _extract_boxed ──────────────────────────────────────────────────────────


class TestExtractBoxed:
    def test_simple(self):
        assert _extract_boxed("\\boxed{42}") == "42"

    def test_nested_braces(self):
        result = _extract_boxed("\\boxed{\\frac{1}{2}}")
        assert result == "\\frac{1}{2}"

    def test_no_boxed(self):
        assert _extract_boxed("no boxed here") is None

    def test_empty(self):
        assert _extract_boxed("") is None

    def test_trailing_text(self):
        assert _extract_boxed("answer is \\boxed{7} correct") == "7"

    def test_incomplete_no_close(self):
        # No closing brace — returns text up to end
        result = _extract_boxed("\\boxed{42")
        assert result == "42"

    def test_whitespace_stripped(self):
        result = _extract_boxed("\\boxed{  A  }")
        assert result == "A"

    def test_expression(self):
        result = _extract_boxed("\\boxed{x^2 + 1}")
        assert result == "x^2 + 1"

    def test_deeply_nested(self):
        result = _extract_boxed("\\boxed{a{b{c}}}")
        assert result is not None
        assert "a" in result


# ── _extract_last_boxed ─────────────────────────────────────────────────────


class TestExtractLastBoxed:
    def test_single_returns_it(self):
        assert _extract_last_boxed("\\boxed{42}") == "42"

    def test_picks_last_not_first(self):
        text = "Earlier I tried \\boxed{5} but the answer is \\boxed{7}"
        assert _extract_last_boxed(text) == "7"

    def test_no_boxed_returns_none(self):
        assert _extract_last_boxed("no boxed here") is None

    def test_empty_returns_none(self):
        assert _extract_last_boxed("") is None

    def test_nested_braces(self):
        assert _extract_last_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_injected_prefix_scenario(self):
        # Simulates injected prefix (wrong intermediate) + model answer (correct)
        text = "<think>So \\boxed{3} is wrong.</think>\n\\boxed{7}"
        assert _extract_last_boxed(text) == "7"

    def test_whitespace_stripped(self):
        assert _extract_last_boxed("\\boxed{  A  }") == "A"

    def test_incomplete_no_close(self):
        assert _extract_last_boxed("\\boxed{42") == "42"


# ── _normalize_math_answer ──────────────────────────────────────────────────


class TestNormalizeMathAnswer:
    def test_plain_number(self):
        assert _normalize_math_answer("42") == "42"

    def test_strips_dollar(self):
        result = _normalize_math_answer("$42$")
        assert "$" not in result
        assert "42" in result

    def test_strips_escaped_dollar(self):
        result = _normalize_math_answer("\\$42")
        assert "\\$" not in result

    def test_text_command(self):
        result = _normalize_math_answer("\\text{answer}")
        assert "answer" in result
        assert "\\text" not in result

    def test_mathrm(self):
        result = _normalize_math_answer("\\mathrm{kg}")
        assert "kg" in result

    def test_frac_simplified(self):
        # Numeric fracs are now evaluated to decimal; no raw \\frac remains
        result = _normalize_math_answer("\\frac{1}{2}")
        assert "\\frac" not in result
        assert result == "0.5"  # numeric frac → decimal

    def test_dfrac_simplified(self):
        result = _normalize_math_answer("\\dfrac{3}{4}")
        assert "\\dfrac" not in result
        assert result == "0.75"  # numeric dfrac → decimal

    def test_dfrac_equals_frac_after_normalization(self):
        # \\dfrac{3}{4} and \\frac{3}{4} must normalize identically
        assert _normalize_math_answer("\\dfrac{3}{4}") == _normalize_math_answer("\\frac{3}{4}")

    def test_frac_evaluated_to_decimal(self):
        # frac{99}{100} → "0.99" so decimal targets match LaTeX fraction predictions
        assert _normalize_math_answer("\\frac{99}{100}") == "0.99"

    def test_dfrac_evaluated_to_decimal(self):
        assert _normalize_math_answer("\\dfrac{3}{4}") == "0.75"

    def test_frac_decimal_matches_plain_decimal(self):
        # The key invariant: \\frac{X}{Y} and its decimal form must normalize equal
        assert _normalize_math_answer("\\frac{99}{100}") == _normalize_math_answer("0.99")
        assert _normalize_math_answer("\\frac{3}{4}") == _normalize_math_answer("0.75")
        assert _normalize_math_answer("\\dfrac{1}{2}") == _normalize_math_answer("0.5")

    def test_frac_non_numeric_unchanged(self):
        # Symbolic fractions must not be evaluated
        result = _normalize_math_answer("\\frac{x+1}{2}")
        assert "frac" in result  # kept as fraction string, not evaluated

    def test_frac_zero_denominator_unchanged(self):
        result = _normalize_math_answer("\\frac{1}{0}")
        assert "frac" in result  # division by zero → not evaluated

    def test_cdot_to_multiply(self):
        result = _normalize_math_answer("3\\cdot 4")
        assert "*" in result

    def test_times_to_multiply(self):
        result = _normalize_math_answer("3\\times 4")
        assert "*" in result

    def test_div_to_slash(self):
        result = _normalize_math_answer("6\\div 2")
        assert "/" in result

    def test_percent(self):
        result = _normalize_math_answer("50\\%")
        assert "%" in result

    def test_pi(self):
        result = _normalize_math_answer("2\\pi")
        assert "pi" in result

    def test_normalizes_whitespace(self):
        result = _normalize_math_answer("  3   +   4  ")
        assert "  " not in result

    def test_non_string(self):
        result = _normalize_math_answer(42)
        assert result == "42"


# ── _count_answer_parts ─────────────────────────────────────────────────────


class TestCountAnswerParts:
    def test_single(self):
        assert _count_answer_parts("42", "") == 1

    def test_multi_comma_letter(self):
        # a=36,b=18 → 2 parts
        count = _count_answer_parts("a=36,b=18", "")
        assert count == 2

    def test_formatted_number(self):
        # 1,000,000 — no letter adjacent, single part
        count = _count_answer_parts("1,000,000", "")
        assert count == 1

    def test_multi_part_prompt(self):
        # Prompt has (a), (b) → 2 parts
        count = _count_answer_parts("42", "Find (a) area and (b) perimeter")
        assert count >= 2

    def test_part_labels(self):
        count = _count_answer_parts("x", "Part 1: find x. Part 2: find y.")
        assert count >= 2

    def test_roman_numerals(self):
        count = _count_answer_parts("x", "Solve (i) and (ii) and (iii)")
        assert count >= 3

    def test_empty_prompt(self):
        count = _count_answer_parts("42", "")
        assert count >= 1


# ── _compute_thinking_budget ────────────────────────────────────────────────


class TestComputeThinkingBudget:
    def test_basic(self):
        cfg = {"thinking_budget": 2048}
        budget, bonus = _compute_thinking_budget(cfg, "42", "", 0, False)
        assert budget == 2048
        assert bonus == 0

    def test_recovery_bonus_when_scaffolded(self):
        cfg = {"thinking_budget": 2048, "recovery_bonus_tokens": 128}
        budget, bonus = _compute_thinking_budget(cfg, "42", "", 0, True)
        assert bonus == 128
        assert budget == 2048 - 128

    def test_no_bonus_when_not_scaffolded(self):
        cfg = {"thinking_budget": 2048, "recovery_bonus_tokens": 128}
        budget, bonus = _compute_thinking_budget(cfg, "42", "", 0, False)
        assert bonus == 0
        assert budget == 2048

    def test_max_completion_cap(self):
        cfg = {"thinking_budget": 4096, "answer_max_tokens": 128}
        budget, _ = _compute_thinking_budget(cfg, "42", "", 2048, False)
        assert budget <= 2048 - 128

    def test_multipart_scales_up(self):
        cfg = {"thinking_budget": 2048, "thinking_budget_ceiling": 4096, "answer_max_tokens": 128}
        b1, _ = _compute_thinking_budget(cfg, "42", "", 0, False)
        b4, _ = _compute_thinking_budget(cfg, "a=1,b=2,c=3,d=4", "", 0, False)
        assert b4 > b1

    def test_ceiling_respected(self):
        cfg = {"thinking_budget": 2048, "thinking_budget_ceiling": 3000}
        b, _ = _compute_thinking_budget(cfg, "a=1,b=2,c=3,d=4", "", 0, False)
        assert b <= 3000

    def test_floor_prevention(self):
        # bonus_budget reduction shouldn't go below 256
        cfg = {"thinking_budget": 300, "recovery_bonus_tokens": 500}
        budget, _ = _compute_thinking_budget(cfg, "42", "", 0, True)
        assert budget >= 256

    def test_no_cap_when_max_completion_zero(self):
        cfg = {"thinking_budget": 2048}
        budget, _ = _compute_thinking_budget(cfg, "42", "", 0, False)
        assert budget == 2048

    def test_floor_applied_correctly(self):
        # max_completion_tokens=1000, floor = max(512, 750) = 750
        cfg = {"thinking_budget": 4096, "answer_max_tokens": 128}
        budget, _ = _compute_thinking_budget(cfg, "42", "", 1000, False)
        # ceiling = max(750, 1000-128) = max(750, 872) = 872
        assert budget <= 872
