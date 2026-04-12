"""Edge case and corner scenario tests for ConditionalRules."""

import numpy as np
import pytest

from mlx_grpo.config.loader import ConditionalRuleConfig
from mlx_grpo.engine.conditional_rules import ConditionalRulesEngine, RuleContext
from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource


class TestHappyPathEdgeCases:
    """Happy path edge cases that should succeed."""

    def test_empty_rules_list(self):
        """Empty rules list should pass through unchanged."""
        engine = ConditionalRulesEngine([])
        segment = Segment(
            tokens=[1, 2, 3],
            text="any text",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert segment.loss_weight == 1.0

    def test_rule_with_no_matches(self):
        """Rule that doesn't match should not modify segment."""
        rule = ConditionalRuleConfig(
            pattern="nonexistent", check="contains", gradient="0.0", reward="-0.5"
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="different text",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert segment.loss_weight == 1.0

    def test_multiplier_identity(self):
        """Multiplier of 1.0 should not change segment."""
        rule = ConditionalRuleConfig(pattern="test", check="contains", gradient="1.0", reward="1.0")
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert segment.loss_weight == 1.0

    def test_gradient_multiplier_as_lambda(self):
        """Gradient multiplier can be a lambda returning different values."""
        rule = ConditionalRuleConfig(
            pattern="test",
            check="contains",
            gradient="lambda ctx: 0.0 if not ctx.answer_correct else 0.5 if ctx.attempt_count > 1 else 1.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )

        # Case 1: wrong answer
        result1 = RolloutResult(segments=[segment])
        engine.apply_to_result(result1, reward=0.0, answer_correct=False)
        assert segment.loss_weight == 0.0

        # Case 2: right answer, multiple attempts
        segment.loss_weight = 1.0
        result2 = RolloutResult(segments=[segment])
        engine.apply_to_result(result2, reward=0.8, answer_correct=True, attempt_count=2)
        assert segment.loss_weight == 0.5

        # Case 3: right answer, first attempt
        segment.loss_weight = 1.0
        result3 = RolloutResult(segments=[segment])
        engine.apply_to_result(result3, reward=0.8, answer_correct=True, attempt_count=1)
        assert segment.loss_weight == 1.0

    def test_reward_multiplier_as_lambda(self):
        """Reward multiplier can be a lambda with complex logic."""
        rule = ConditionalRuleConfig(
            pattern="test",
            check="contains",
            gradient="0.0",
            reward="lambda ctx: -1.0 * abs(ctx.reward - 0.5) if ctx.is_naked else -0.2",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        result = RolloutResult(segments=[segment])

        # Naked with high reward: large penalty
        mult1 = engine._resolve_value(
            rule.reward,
            RuleContext(
                segment=segment,
                reward=0.9,
                answer_correct=True,
                is_naked=True,
            ),
        )
        assert mult1 == -0.4

        # Non-naked: small penalty
        mult2 = engine._resolve_value(
            rule.reward,
            RuleContext(
                segment=segment,
                reward=0.9,
                answer_correct=True,
                is_naked=False,
            ),
        )
        assert mult2 == -0.2

    def test_segment_tag_based_targeting(self):
        """Rules can target segments by tag (thinking vs answer)."""
        rule = ConditionalRuleConfig(
            when="lambda ctx: ctx.segment.tag == 'thinking'",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])

        thinking_seg = Segment(
            tokens=[1, 2, 3],
            text="thinking",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
            tag="thinking",
        )
        answer_seg = Segment(
            tokens=[4, 5, 6],
            text="answer",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
            tag="answer",
        )

        result = RolloutResult(segments=[thinking_seg, answer_seg])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)

        assert thinking_seg.loss_weight == 0.0
        assert answer_seg.loss_weight == 1.0

    def test_batch_processing_multiple_results(self):
        """Multiple results processed sequentially should all apply rules."""
        rule = ConditionalRuleConfig(pattern="bad", check="contains", gradient="0.0", reward="-0.5")
        engine = ConditionalRulesEngine([rule])

        results = [
            RolloutResult(
                segments=[
                    Segment(
                        tokens=[i],
                        text="bad output",
                        source=SegmentSource.GENERATED,
                        masked=False,
                        loss_weight=1.0,
                    )
                ]
            )
            for i in range(3)
        ]

        for result in results:
            engine.apply_to_result(result, reward=0.5, answer_correct=False)

        for result in results:
            assert result.segments[0].loss_weight == 0.0

    def test_very_large_multiplier(self):
        """Very large multiplier (amplification) should work."""
        rule = ConditionalRuleConfig(
            pattern="good", check="contains", gradient="10.0", reward="5.0"
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="good output",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert segment.loss_weight == 10.0

    def test_very_small_multiplier(self):
        """Very small multiplier should work (near zero but not zero)."""
        rule = ConditionalRuleConfig(
            pattern="weak", check="contains", gradient="0.001", reward="0.01"
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="weak signal",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert abs(segment.loss_weight - 0.001) < 1e-6

    def test_negative_multiplier(self):
        """Negative multiplier should negate/reverse (allowed or not?)."""
        # Negative gradient: reverses loss sign
        # Negative reward: reverses reward sign
        rule = ConditionalRuleConfig(
            pattern="test", check="contains", gradient="-0.5", reward="-1.0"
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        # Should allow negative (it reverses direction)
        assert segment.loss_weight == -0.5

    def test_regex_with_special_characters(self):
        """Regex with special characters should work."""
        rule = ConditionalRuleConfig(
            pattern=r"\$\d+\.\d{2}",  # Match $X.XX
            check="regex",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])

        matching_seg = Segment(
            tokens=[1, 2, 3],
            text="The cost is $10.50",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        non_matching_seg = Segment(
            tokens=[4, 5, 6],
            text="The cost is 10.50",
            source=SegmentSource.GENERATED,
            masked=False,
        )

        result = RolloutResult(segments=[matching_seg, non_matching_seg])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)

        assert engine._pattern_matches(matching_seg, rule) is True
        assert engine._pattern_matches(non_matching_seg, rule) is False

    def test_unicode_in_pattern(self):
        """Unicode characters in pattern should work."""
        rule = ConditionalRuleConfig(
            pattern="你好",  # Chinese "hello"
            check="contains",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="The response is 你好 world",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        assert engine._pattern_matches(segment, rule) is True

    def test_rule_ordering_matters(self):
        """Rule order should matter (each rule multiplies result)."""
        rules = [
            ConditionalRuleConfig(pattern="test", check="contains", gradient="0.5", reward="-0.5"),
            ConditionalRuleConfig(pattern="test", check="contains", gradient="0.5", reward="-0.5"),
        ]
        engine = ConditionalRulesEngine(rules)
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        # Both rules apply: 1.0 * 0.5 * 0.5 = 0.25
        assert segment.loss_weight == 0.25


class TestSadPathEdgeCases:
    """Sad path: error conditions that should fail gracefully."""

    def test_regex_timeout_on_complex_pattern(self):
        """Complex regex that could timeout should handle gracefully."""
        # Pathological regex that causes catastrophic backtracking
        rule = ConditionalRuleConfig(
            pattern="(a+)+b",  # Vulnerable to backtracking
            check="regex",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="a" * 20,  # Short string without 'b'
            source=SegmentSource.GENERATED,
            masked=False,
        )
        # Should not crash — regex matching should handle gracefully
        try:
            result = engine._pattern_matches(segment, rule)
            # If it returns False (no match), that's correct
            assert result is False
        except (TimeoutError, RuntimeError):
            # Timeout on pathological regex is also acceptable
            pass

    def test_missing_required_config_field(self):
        """Missing required field in ConditionalRuleConfig should fail."""
        # Missing both pattern and when
        with pytest.raises(TypeError):
            ConditionalRuleConfig(gradient="0.0", reward="-0.5")

        # Missing gradient
        with pytest.raises(TypeError):
            ConditionalRuleConfig(pattern="test", check="regex", reward="-0.5")

        # Missing reward
        with pytest.raises(TypeError):
            ConditionalRuleConfig(pattern="test", check="regex", gradient="0.0")

    def test_invalid_check_type(self):
        """Invalid check type should raise error."""
        with pytest.raises(ValueError):
            ConditionalRuleConfig(
                pattern="test",
                check="invalid_check",  # Not "regex" or "contains"
                gradient="0.0",
                reward="-0.5",
            )

    def test_invalid_target_segments(self):
        """Invalid target_segments should raise error."""
        with pytest.raises(ValueError):
            ConditionalRuleConfig(
                pattern="test",
                check="regex",
                gradient="0.0",
                reward="-0.5",
                target_segments="invalid_target",
            )

    def test_both_pattern_and_when_provided(self):
        """Both pattern and when provided — should be allowed or conflict?"""
        # Currently allowed, both should be checked
        rule = ConditionalRuleConfig(
            pattern="test",
            check="contains",
            when="lambda ctx: ctx.answer_correct",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        ctx = RuleContext(segment=segment, reward=0.5, answer_correct=False)
        # Both must match? Or either?
        # Currently undefined — should clarify

    def test_lambda_returns_non_boolean_for_when(self):
        """Lambda for 'when' returning non-boolean gets converted to bool."""
        rule = ConditionalRuleConfig(
            when="lambda ctx: 'not a boolean'",  # Returns string, converts to True
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        ctx = RuleContext(segment=segment, reward=0.5, answer_correct=True)
        # Non-boolean string converts to True (non-empty string)
        assert engine._when_matches(ctx, rule) is True

    def test_lambda_returns_non_numeric_for_gradient(self):
        """Lambda for 'gradient' returning non-numeric should fail."""
        rule = ConditionalRuleConfig(
            pattern="test",
            check="contains",
            gradient="lambda ctx: 'not a number'",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        ctx = RuleContext(segment=segment, reward=0.5, answer_correct=True)
        with pytest.raises((TypeError, ValueError)):
            engine._resolve_value(rule.gradient, ctx)

    def test_multiplier_nan(self):
        """Multiplier value of NaN should be handled."""
        rule = ConditionalRuleConfig(
            pattern="test",
            check="contains",
            gradient="lambda ctx: float('nan')",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        # Should either reject NaN or handle it safely
        try:
            engine.apply_to_result(result, reward=0.5, answer_correct=True)
            assert np.isnan(segment.loss_weight) or segment.loss_weight == 1.0
        except (ValueError, FloatingPointError):
            pass

    def test_multiplier_infinity(self):
        """Multiplier value of Infinity should be handled."""
        rule = ConditionalRuleConfig(
            pattern="test",
            check="contains",
            gradient="lambda ctx: float('inf')",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        # Should either reject inf or handle it
        try:
            engine.apply_to_result(result, reward=0.5, answer_correct=True)
            assert np.isinf(segment.loss_weight) or segment.loss_weight == 1.0
        except (ValueError, FloatingPointError):
            pass

    def test_segment_without_tokens(self):
        """Segment with no tokens should be handled."""
        rule = ConditionalRuleConfig(
            pattern="test", check="contains", gradient="0.0", reward="-0.5"
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[],  # Empty tokens
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        result = RolloutResult(segments=[segment])
        # Should handle gracefully
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert True  # Just shouldn't crash

    def test_segment_without_text(self):
        """Segment with no text should be handled."""
        rule = ConditionalRuleConfig(
            pattern="test", check="contains", gradient="0.0", reward="-0.5"
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="",  # Empty text
            source=SegmentSource.GENERATED,
            masked=False,
        )
        result = RolloutResult(segments=[segment])
        # Should not match pattern
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert segment.loss_weight == 1.0

    def test_result_without_segments(self):
        """Result with no segments should be handled."""
        rule = ConditionalRuleConfig(
            pattern="test", check="contains", gradient="0.0", reward="-0.5"
        )
        engine = ConditionalRulesEngine([rule])
        result = RolloutResult(segments=[])  # No segments
        # Should handle gracefully
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert True  # Just shouldn't crash

    def test_context_variable_not_available(self):
        """Accessing unavailable context variable should fail."""
        rule = ConditionalRuleConfig(
            when="lambda ctx: ctx.nonexistent_variable > 0.5",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        ctx = RuleContext(segment=segment, reward=0.5, answer_correct=True)
        with pytest.raises(AttributeError):
            engine._when_matches(ctx, rule)

    def test_static_method_wrong_signature(self):
        """Static method with wrong signature should fail."""

        class BadMethods:
            @staticmethod
            def no_args():  # Missing ctx parameter
                return True

        rule = ConditionalRuleConfig(
            when="BadMethods.no_args",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule], method_registry={"BadMethods": BadMethods})
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        ctx = RuleContext(segment=segment, reward=0.5, answer_correct=True)
        with pytest.raises(TypeError):
            engine._when_matches(ctx, rule)

    def test_static_method_returns_wrong_type(self):
        """Static method returning wrong type should fail."""

        class BadMethods:
            @staticmethod
            def returns_string(ctx):
                return "not a boolean"

        rule = ConditionalRuleConfig(
            when="BadMethods.returns_string",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule], method_registry={"BadMethods": BadMethods})
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        ctx = RuleContext(segment=segment, reward=0.5, answer_correct=True)
        # Method returns non-boolean string, which converts to True
        assert engine._when_matches(ctx, rule) is True
