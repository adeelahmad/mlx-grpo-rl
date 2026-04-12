"""Unit tests for ConditionalRulesEngine — pattern matching, lambda execution, multiplier application."""

import numpy as np
import pytest

from mlx_grpo.config.loader import ConditionalRuleConfig
from mlx_grpo.engine.conditional_rules import ConditionalRulesEngine, RuleContext
from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource


class TestPatternMatching:
    """Test pattern matching (regex and contains)."""

    def test_regex_pattern_match(self):
        """Regex pattern should match case-insensitive."""
        rule = ConditionalRuleConfig(
            pattern="(?i)stuck",
            check="regex",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="Model is stuck in loop",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        assert engine._pattern_matches(segment, rule) is True

    def test_regex_pattern_no_match(self):
        """Regex pattern should not match when pattern absent."""
        rule = ConditionalRuleConfig(
            pattern="(?i)stuck",
            check="regex",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="Model produces output",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        assert engine._pattern_matches(segment, rule) is False

    def test_contains_pattern_match(self):
        """Contains check should do literal string matching."""
        rule = ConditionalRuleConfig(
            pattern="Error",
            check="contains",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="This is an Error in the output",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        assert engine._pattern_matches(segment, rule) is True

    def test_contains_pattern_case_sensitive(self):
        """Contains should be case-sensitive."""
        rule = ConditionalRuleConfig(
            pattern="Error",
            check="contains",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="This is an error in lowercase",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        assert engine._pattern_matches(segment, rule) is False


class TestLambdaConditions:
    """Test 'when' lambda conditions."""

    def test_lambda_when_true(self):
        """Lambda returning True should trigger rule."""
        rule = ConditionalRuleConfig(
            when="lambda ctx: 'stuck' in ctx.segment.text.lower()",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="Model is stuck",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        ctx = RuleContext(segment=segment, reward=0.5, answer_correct=True)
        assert engine._when_matches(ctx, rule) is True

    def test_lambda_when_false(self):
        """Lambda returning False should not trigger rule."""
        rule = ConditionalRuleConfig(
            when="lambda ctx: 'stuck' in ctx.segment.text.lower()",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="Model works fine",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        ctx = RuleContext(segment=segment, reward=0.5, answer_correct=True)
        assert engine._when_matches(ctx, rule) is False

    def test_lambda_context_access(self):
        """Lambda should access all context variables."""
        rule = ConditionalRuleConfig(
            when="lambda ctx: not ctx.answer_correct and ctx.is_naked and ctx.attempt_count > 1",
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
        ctx = RuleContext(
            segment=segment,
            reward=0.0,
            answer_correct=False,
            is_naked=True,
            attempt_count=3,
        )
        assert engine._when_matches(ctx, rule) is True


class TestStaticMethodConditions:
    """Test static method resolution for 'when', 'gradient', 'reward'."""

    def test_resolve_static_method_when(self):
        """Should resolve and execute static method for 'when'."""

        # Mock static method
        class MockChecks:
            @staticmethod
            def has_stuck(ctx):
                return "stuck" in ctx.segment.text.lower()

        rule = ConditionalRuleConfig(
            when="MockChecks.has_stuck",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule], method_registry={"MockChecks": MockChecks})
        segment = Segment(
            tokens=[1, 2, 3],
            text="stuck",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        ctx = RuleContext(segment=segment, reward=0.5, answer_correct=True)
        assert engine._when_matches(ctx, rule) is True

    def test_resolve_static_method_gradient(self):
        """Should resolve and execute static method for 'gradient'."""

        class MockGradient:
            @staticmethod
            def compute(ctx):
                return 0.0 if not ctx.answer_correct else 1.0

        rule = ConditionalRuleConfig(
            pattern="stuck",
            check="contains",
            gradient="MockGradient.compute",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule], method_registry={"MockGradient": MockGradient})
        segment = Segment(
            tokens=[1, 2, 3],
            text="stuck",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        ctx = RuleContext(segment=segment, reward=0.5, answer_correct=False)
        result = engine._resolve_value(rule.gradient, ctx)
        assert result == 0.0

    def test_resolve_static_method_reward(self):
        """Should resolve and execute static method for 'reward'."""

        class MockReward:
            @staticmethod
            def penalize(ctx):
                return -1.0 if ctx.reward > 0.5 else -0.2

        rule = ConditionalRuleConfig(
            pattern="test",
            check="contains",
            gradient="0.0",
            reward="MockReward.penalize",
        )
        engine = ConditionalRulesEngine([rule], method_registry={"MockReward": MockReward})
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        ctx = RuleContext(segment=segment, reward=0.8, answer_correct=True)
        result = engine._resolve_value(rule.reward, ctx)
        assert result == -1.0


class TestGradientMultiplierApplication:
    """Test gradient multiplier application to segments."""

    def test_apply_zero_gradient(self):
        """Gradient multiplier 0.0 should zero out loss_weight."""
        rule = ConditionalRuleConfig(
            pattern="stuck",
            check="contains",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="stuck",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        ctx = RuleContext(segment=segment, reward=0.5, answer_correct=True)
        engine._apply_gradient_multiplier(segment, 0.0)
        assert segment.loss_weight == 0.0

    def test_apply_amplified_gradient(self):
        """Gradient multiplier > 1.0 should amplify loss_weight."""
        rule = ConditionalRuleConfig(
            pattern="because",
            check="contains",
            gradient="2.0",
            reward="0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="because of this",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        engine._apply_gradient_multiplier(segment, 2.0)
        assert segment.loss_weight == 2.0

    def test_apply_partial_gradient(self):
        """Gradient multiplier 0.5 should halve loss_weight."""
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        engine = ConditionalRulesEngine([])
        engine._apply_gradient_multiplier(segment, 0.5)
        assert segment.loss_weight == 0.5


class TestRewardMultiplierTracking:
    """Test reward multiplier tracking in result metadata."""

    def test_reward_multiplier_stored_in_metadata(self):
        """Reward multiplier should be stored in result metadata."""
        rule = ConditionalRuleConfig(
            pattern="stuck",
            check="contains",
            gradient="0.0",
            reward="-0.5",
        )
        engine = ConditionalRulesEngine([rule])
        result = RolloutResult(
            segments=[
                Segment(
                    tokens=[1, 2, 3],
                    text="stuck",
                    source=SegmentSource.GENERATED,
                    masked=False,
                )
            ]
        )
        # After applying rules, reward multiplier should be in metadata
        engine.apply_to_result(result, reward=1.0)
        assert "conditional_reward_multipliers" in result.metadata or isinstance(
            result.metadata.get("reward_multiplier"), (int, float)
        )

    def test_multiple_rules_reward_multiplier(self):
        """Multiple rules matching same segment should compound multipliers."""
        rules = [
            ConditionalRuleConfig(pattern="stuck", check="contains", gradient="0.0", reward="-0.5"),
            ConditionalRuleConfig(
                pattern="confused", check="contains", gradient="0.0", reward="-0.3"
            ),
        ]
        engine = ConditionalRulesEngine(rules)
        result = RolloutResult(
            segments=[
                Segment(
                    tokens=[1, 2, 3],
                    text="stuck and confused",
                    source=SegmentSource.GENERATED,
                    masked=False,
                )
            ]
        )
        engine.apply_to_result(result, reward=1.0)
        # Both rules match, multipliers should compound
        assert result.metadata.get("reward_multiplier", 1.0) < 1.0


class TestSegmentTargeting:
    """Test targeting of generated vs scaffolded segments."""

    def test_target_generated_only(self):
        """Rule with target_generated should not apply to scaffolded."""
        rule = ConditionalRuleConfig(
            pattern="error",
            check="contains",
            gradient="0.0",
            reward="-0.5",
            target_segments="generated",
        )
        engine = ConditionalRulesEngine([rule])

        gen_seg = Segment(
            tokens=[1, 2, 3],
            text="error in generated",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        scaf_seg = Segment(
            tokens=[4, 5, 6],
            text="error in scaffold",
            source=SegmentSource.INJECTED,
            masked=False,
            loss_weight=1.0,
        )

        result = RolloutResult(segments=[gen_seg, scaf_seg])
        engine.apply_to_result(result, reward=1.0)

        assert gen_seg.loss_weight == 0.0
        assert scaf_seg.loss_weight == 1.0

    def test_target_both(self):
        """Rule with target_both should apply to all segments."""
        rule = ConditionalRuleConfig(
            pattern="error",
            check="contains",
            gradient="0.0",
            reward="-0.5",
            target_segments="both",
        )
        engine = ConditionalRulesEngine([rule])

        gen_seg = Segment(
            tokens=[1, 2, 3],
            text="error",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        scaf_seg = Segment(
            tokens=[4, 5, 6],
            text="error",
            source=SegmentSource.INJECTED,
            masked=False,
            loss_weight=1.0,
        )

        result = RolloutResult(segments=[gen_seg, scaf_seg])
        engine.apply_to_result(result, reward=1.0)

        assert gen_seg.loss_weight == 0.0
        assert scaf_seg.loss_weight == 0.0


class TestTokenProximityWeights:
    """Test access to proximity-weighted tokens in context."""

    def test_proximity_weights_in_context(self):
        """Context should contain sequence_proximity_weights."""
        rule = ConditionalRuleConfig(
            when="lambda ctx: mean(ctx.sequence_proximity_weights[ctx.segment_token_range[0]:ctx.segment_token_range[1]]) > 0.5",
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
        proximity_weights = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        ctx = RuleContext(
            segment=segment,
            reward=0.5,
            answer_correct=True,
            sequence_tokens=[1, 2, 3, 4, 5],
            sequence_proximity_weights=proximity_weights,
            segment_token_range=(0, 3),
        )
        # Should have access to weights and be able to compute mean
        assert hasattr(ctx, "sequence_proximity_weights")
        assert len(ctx.sequence_proximity_weights) == 5

    def test_segment_token_range_indexing(self):
        """Should be able to slice proximity weights by segment range."""
        segment = Segment(
            tokens=[2, 3, 4],
            text="middle part",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        proximity_weights = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        ctx = RuleContext(
            segment=segment,
            reward=0.5,
            answer_correct=True,
            sequence_proximity_weights=proximity_weights,
            segment_token_range=(1, 4),
        )
        segment_weights = ctx.sequence_proximity_weights[
            ctx.segment_token_range[0] : ctx.segment_token_range[1]
        ]
        assert len(segment_weights) == 3
        np.testing.assert_array_equal(segment_weights, np.array([0.8, 0.7, 0.6]))


class TestContextVariables:
    """Test all context variables are available to lambdas."""

    def test_context_has_all_variables(self):
        """RuleContext should have all expected variables."""
        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
            tag="answer",
        )
        result = RolloutResult(segments=[segment])

        ctx = RuleContext(
            segment=segment,
            result=result,
            reward=0.7,
            answer_correct=True,
            is_naked=False,
            attempt_count=2,
            kl_divergence=0.05,
            group_index=1,
            sample_index=0,
            sequence_tokens=[1, 2, 3],
            sequence_proximity_weights=np.array([0.9, 0.8, 0.7]),
            segment_token_range=(0, 3),
        )

        assert ctx.segment is not None
        assert ctx.result is not None
        assert ctx.reward == 0.7
        assert ctx.answer_correct is True
        assert ctx.is_naked is False
        assert ctx.attempt_count == 2
        assert ctx.kl_divergence == 0.05
        assert ctx.group_index == 1
        assert ctx.sample_index == 0

    def test_lambda_accesses_nested_context(self):
        """Lambda should access nested objects like segment.tag."""
        rule = ConditionalRuleConfig(
            when="lambda ctx: ctx.segment.tag == 'answer' and ctx.answer_correct",
            gradient="2.0",  # Amplify correct answers
            reward="0.5",
        )
        engine = ConditionalRulesEngine([rule])
        segment = Segment(
            tokens=[1, 2, 3],
            text="correct answer",
            source=SegmentSource.GENERATED,
            masked=False,
            tag="answer",
        )
        ctx = RuleContext(segment=segment, reward=0.5, answer_correct=True)
        assert engine._when_matches(ctx, rule) is True


class TestErrorHandling:
    """Test error handling in lambda evaluation."""

    def test_lambda_syntax_error(self):
        """Invalid lambda syntax should raise clear error."""
        rule = ConditionalRuleConfig(
            when="lambda ctx: 'invalid syntax here )",
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
        with pytest.raises(SyntaxError):
            engine._when_matches(ctx, rule)

    def test_method_not_found(self):
        """Referencing non-existent method should raise error."""
        rule = ConditionalRuleConfig(
            when="NonExistentClass.method",
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
        with pytest.raises((AttributeError, KeyError)):
            engine._when_matches(ctx, rule)

    def test_gradient_multiplier_not_numeric(self):
        """Non-numeric gradient multiplier should raise error."""
        rule = ConditionalRuleConfig(
            pattern="test",
            check="contains",
            gradient="not a number",
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
        with pytest.raises((ValueError, TypeError)):
            engine._resolve_value(rule.gradient, ctx)
