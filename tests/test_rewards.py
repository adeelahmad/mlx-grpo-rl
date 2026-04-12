"""Tests for reward system."""

from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource
from mlx_grpo.rewards.accuracy import AccuracyReward, FormatReward
from mlx_grpo.rewards.base import BaseReward, register_lambda_reward, reward_fn


def _r(text):
    return RolloutResult(
        segments=[
            Segment([ord(c) for c in text], text, SegmentSource.GENERATED, False),
        ]
    )


class TestAccuracy:
    def setup_method(self):
        self.r = AccuracyReward()

    def test_boxed(self):
        assert self.r.compute(_r("\\boxed{42}"), "42") == 1.0
        assert self.r.compute(_r("\\boxed{99}"), "42") == 0.0

    def test_mcq(self):
        assert self.r.compute(_r("answer is (B)"), "B") == 1.0

    def test_numeric(self):
        assert self.r.compute(_r("3.14"), "3.14") == 1.0

    def test_no_answer(self):
        assert self.r.compute(_r("x"), None) == 0.0

    def test_first_boxed(self):
        assert self.r.compute(_r("\\boxed{1} \\boxed{2}"), "1") == 1.0

    def test_boxed_nested_braces(self):
        """Nested braces like \\frac{}{} should not truncate the answer."""
        assert (
            self.r.compute(
                _r("\\boxed{\\frac{1}{2}}"),
                "\\frac{1}{2}",
            )
            == 1.0
        )
        assert (
            self.r.compute(
                _r("\\boxed{a_{n}=\\frac{n(n-1)}{2}}"),
                "a_{n}=\\frac{n(n-1)}{2}",
            )
            == 1.0
        )


class TestBoxedExtraction:
    """Comprehensive tests for \\boxed{} extraction with nested braces."""

    def setup_method(self):
        self.r = AccuracyReward()

    # ── Simple answers ──
    def test_integer(self):
        assert self.r.compute(_r("\\boxed{42}"), "42") == 1.0

    def test_negative(self):
        assert self.r.compute(_r("\\boxed{-7}"), "-7") == 1.0

    def test_decimal(self):
        assert self.r.compute(_r("\\boxed{3.14}"), "3.14") == 1.0

    def test_single_letter(self):
        assert self.r.compute(_r("\\boxed{x}"), "x") == 1.0

    # ── Fractions (the original bug) ──
    def test_simple_frac(self):
        assert self.r.compute(_r("\\boxed{\\frac{1}{2}}"), "\\frac{1}{2}") == 1.0

    def test_frac_with_expressions(self):
        assert (
            self.r.compute(
                _r("\\boxed{\\frac{n(n-1)}{2}}"),
                "\\frac{n(n-1)}{2}",
            )
            == 1.0
        )

    def test_frac_with_subscript(self):
        assert (
            self.r.compute(
                _r("\\boxed{a_{n}=\\frac{n(n-1)}{2}}"),
                "a_{n}=\\frac{n(n-1)}{2}",
            )
            == 1.0
        )

    def test_nested_frac(self):
        """\\frac{\\frac{1}{2}}{3} — three levels of braces."""
        assert (
            self.r.compute(
                _r("\\boxed{\\frac{\\frac{1}{2}}{3}}"),
                "\\frac{\\frac{1}{2}}{3}",
            )
            == 1.0
        )

    # ── Other LaTeX commands with braces ──
    def test_sqrt_with_index(self):
        assert (
            self.r.compute(
                _r("\\boxed{\\sqrt[3]{8}}"),
                "\\sqrt[3]{8}",
            )
            == 1.0
        )

    def test_binom(self):
        assert (
            self.r.compute(
                _r("\\boxed{\\binom{n}{k}}"),
                "\\binom{n}{k}",
            )
            == 1.0
        )

    def test_sum_notation(self):
        assert (
            self.r.compute(
                _r("\\boxed{\\sum_{i=1}^{n} i}"),
                "\\sum_{i=1}^{n} i",
            )
            == 1.0
        )

    def test_multiple_subscripts(self):
        assert (
            self.r.compute(
                _r("\\boxed{x_{1}^{2}+x_{2}^{2}}"),
                "x_{1}^{2}+x_{2}^{2}",
            )
            == 1.0
        )

    # ── Edge cases ──
    def test_empty_boxed(self):
        # Empty boxed matches empty answer (correct)
        assert self.r.compute(_r("\\boxed{}"), "") == 1.0
        # Empty boxed doesn't match non-empty answer
        assert self.r.compute(_r("\\boxed{}"), "42") == 0.0

    def test_boxed_with_trailing_text(self):
        assert (
            self.r.compute(
                _r("The answer is \\boxed{42} as shown above"),
                "42",
            )
            == 1.0
        )

    def test_multiple_boxed_takes_first(self):
        assert (
            self.r.compute(
                _r("First \\boxed{1} then \\boxed{\\frac{3}{4}}"),
                "1",
            )
            == 1.0
        )

    def test_unclosed_boxed(self):
        """Unclosed \\boxed{ falls back to rest of text."""
        assert self.r.compute(_r("\\boxed{42"), "42") == 1.0

    def test_unclosed_nested(self):
        """Unclosed nested brace returns what's available."""
        assert (
            self.r.compute(
                _r("\\boxed{\\frac{1}{2}"),
                "\\frac{1}{2}",
            )
            == 1.0
        )

    # ── Incorrect answers ──
    def test_wrong_frac(self):
        assert (
            self.r.compute(
                _r("\\boxed{\\frac{1}{3}}"),
                "\\frac{1}{2}",
            )
            == 0.0
        )

    def test_truncated_frac_wrong(self):
        """The original bug: stop at first } gives truncated answer."""
        # If extraction were broken, this would extract "\\frac{1"
        assert (
            self.r.compute(
                _r("\\boxed{\\frac{1}{2}}"),
                "\\frac{1}{3}",
            )
            == 0.0
        )


class TestMathReward:
    """Tests for MathReward with LaTeX normalization."""

    def setup_method(self):
        from mlx_grpo.rewards.math import MathReward

        self.r = MathReward()

    def test_exact_match(self):
        assert self.r.compute(_r("\\boxed{42}"), "42") == 1.0

    def test_frac_nested(self):
        assert (
            self.r.compute(
                _r("\\boxed{\\frac{n(n-1)}{2}}"),
                "\\frac{n(n-1)}{2}",
            )
            == 1.0
        )

    def test_latex_normalization(self):
        """\\frac → frac, \\cdot → *, etc."""
        assert (
            self.r.compute(
                _r("\\boxed{2\\pi}"),
                "2\\pi",
            )
            == 1.0
        )

    def test_numeric_tolerance(self):
        assert self.r.compute(_r("\\boxed{3.14}"), "3.14159") >= 0.5

    def test_wrong_answer(self):
        assert self.r.compute(_r("\\boxed{7}"), "42") == 0.0

    def test_no_boxed(self):
        assert self.r.compute(_r("the answer is 42"), "42") == 0.0

    def test_no_answer(self):
        assert self.r.compute(_r("\\boxed{42}"), None) == 0.0

    def test_deeply_nested(self):
        assert (
            self.r.compute(
                _r("\\boxed{\\frac{\\sqrt{x^{2}+1}}{\\binom{n}{k}}}"),
                "\\frac{\\sqrt{x^{2}+1}}{\\binom{n}{k}}",
            )
            == 1.0
        )

    def test_dfrac_matches_frac(self):
        """\\dfrac should normalize same as \\frac."""
        assert self.r.compute(_r("\\boxed{\\dfrac{3}{4}}"), "\\frac{3}{4}") == 1.0

    def test_frac_matches_decimal(self):
        """\\frac{1}{2} should match '0.5' via fraction evaluation."""
        assert self.r.compute(_r("\\boxed{\\frac{1}{2}}"), "0.5") == 1.0

    def test_answer_var_fallback(self):
        """Falls back to answer variable when no boxed in generated text."""
        result = RolloutResult(
            segments=[
                Segment([1], "no boxed here", SegmentSource.GENERATED, False),
            ]
        )
        from mlx_grpo.engine.result import GenerateResult

        # Simple answer without trailing braces works
        result.variables["answer"] = GenerateResult(
            text="42",
            tokens=[1, 2],
            stop_reason="eos",
        )
        assert self.r.compute(result, "42") == 1.0


class TestFormat:
    def test_all_tags(self):
        assert FormatReward(required_tags=["<t>", "</t>"]).compute(_r("<t>x</t>")) == 1.0

    def test_missing_tag(self):
        assert FormatReward(required_tags=["<t>", "</t>"]).compute(_r("<t>x")) < 1.0

    def test_forbidden(self):
        assert FormatReward(forbidden=["ERR"]).compute(_r("has ERR")) < 1.0


class TestRegistry:
    def test_registered(self):
        assert "accuracy" in BaseReward.list_rewards()

    def test_decorator(self):
        @reward_fn("test_dec_r")
        def my_r(result, answer=None):
            return 0.42

        assert BaseReward.get("test_dec_r").compute(_r(""), None) == 0.42

    def test_lambda(self):
        register_lambda_reward("test_lam_r", lambda r, a: 0.99)
        assert BaseReward.get("test_lam_r").compute(_r(""), None) == 0.99
