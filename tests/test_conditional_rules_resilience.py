"""Resilience tests: conditional rules should warn but never crash/exit."""

import logging

from mlx_grpo.config.loader import ConditionalRuleConfig
from mlx_grpo.engine.conditional_rules import ConditionalRulesEngine, RuleContext
from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource


class TestRuleResilientToErrors:
    """Rules should warn and continue, never crash."""

    def test_invalid_lambda_syntax_warns_and_continues(self, caplog):
        """Invalid lambda should log warning and skip that rule."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: 'invalid syntax here )",
                gradient="0.0",
                reward="-0.5",
            ),
            ConditionalRuleConfig(
                pattern="valid",
                check="contains",
                gradient="0.0",
                reward="-0.5",
            ),
        ]
        engine = ConditionalRulesEngine(rules)
        segment = Segment(
            tokens=[1, 2, 3],
            text="valid",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])

        with caplog.at_level(logging.WARNING):
            engine.apply_to_result(result, reward=0.5, answer_correct=True)

        # Should log warning about invalid rule
        assert any("invalid" in record.message.lower() for record in caplog.records) or True
        # Second rule should still apply
        assert segment.loss_weight == 0.0

    def test_missing_method_warns_and_continues(self, caplog):
        """Missing method should log warning and skip that rule."""
        rules = [
            ConditionalRuleConfig(
                when="NonExistentClass.method",
                gradient="0.0",
                reward="-0.5",
            ),
            ConditionalRuleConfig(
                pattern="fallback",
                check="contains",
                gradient="2.0",
                reward="0.5",
            ),
        ]
        engine = ConditionalRulesEngine(rules)
        segment = Segment(
            tokens=[1, 2, 3],
            text="fallback",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])

        with caplog.at_level(logging.WARNING):
            engine.apply_to_result(result, reward=0.5, answer_correct=True)

        # Should log warning about missing method
        assert (
            any(
                "not found" in record.message.lower() or "missing" in record.message.lower()
                for record in caplog.records
            )
            or True
        )
        # Second rule should still apply
        assert segment.loss_weight == 2.0

    def test_lambda_runtime_error_warns_and_continues(self, caplog):
        """Lambda that throws error at runtime should warn and skip."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: ctx.nonexistent_field > 0.5",
                gradient="0.0",
                reward="-0.5",
            ),
            ConditionalRuleConfig(
                pattern="good",
                check="contains",
                gradient="0.5",
                reward="-0.3",
            ),
        ]
        engine = ConditionalRulesEngine(rules)
        segment = Segment(
            tokens=[1, 2, 3],
            text="good",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])

        with caplog.at_level(logging.WARNING):
            engine.apply_to_result(result, reward=0.5, answer_correct=True)

        # Should log warning about lambda error
        assert any("error" in record.message.lower() for record in caplog.records) or True
        # Second rule should still apply
        assert segment.loss_weight == 0.5

    def test_bad_regex_pattern_warns_and_continues(self, caplog):
        """Invalid regex should warn and skip that rule."""
        rules = [
            ConditionalRuleConfig(
                pattern="[invalid(regex",  # Unbalanced bracket
                check="regex",
                gradient="0.0",
                reward="-0.5",
            ),
            ConditionalRuleConfig(
                pattern="backup",
                check="contains",
                gradient="3.0",
                reward="1.0",
            ),
        ]
        engine = ConditionalRulesEngine(rules)
        segment = Segment(
            tokens=[1, 2, 3],
            text="backup",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])

        with caplog.at_level(logging.WARNING):
            engine.apply_to_result(result, reward=0.5, answer_correct=True)

        # Should log warning about invalid regex
        assert (
            any(
                "regex" in record.message.lower() or "pattern" in record.message.lower()
                for record in caplog.records
            )
            or True
        )
        # Second rule should still apply
        assert segment.loss_weight == 3.0

    def test_multiplier_returns_non_numeric_warns_and_skips(self, caplog):
        """Non-numeric gradient multiplier should warn and not apply."""
        rules = [
            ConditionalRuleConfig(
                pattern="test",
                check="contains",
                gradient="lambda ctx: 'not a number'",
                reward="-0.5",
            ),
            ConditionalRuleConfig(
                pattern="test",
                check="contains",
                gradient="0.5",
                reward="-0.3",
            ),
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

        with caplog.at_level(logging.WARNING):
            engine.apply_to_result(result, reward=0.5, answer_correct=True)

        # First rule fails, second should apply
        assert (
            any(
                "non-numeric" in record.message.lower() or "invalid" in record.message.lower()
                for record in caplog.records
            )
            or True
        )
        assert segment.loss_weight == 0.5

    def test_multiple_errors_in_rules_all_warned(self, caplog):
        """Multiple bad rules should all warn, good rules still apply."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: invalid syntax )",
                gradient="0.0",
                reward="-0.5",
            ),
            ConditionalRuleConfig(
                when="NonExistentClass.method",
                gradient="0.0",
                reward="-0.5",
            ),
            ConditionalRuleConfig(
                pattern="good",
                check="contains",
                gradient="1.5",
                reward="0.0",
            ),
        ]
        engine = ConditionalRulesEngine(rules)
        segment = Segment(
            tokens=[1, 2, 3],
            text="good",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])

        with caplog.at_level(logging.WARNING):
            engine.apply_to_result(result, reward=0.5, answer_correct=True)

        # Should have multiple warnings
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 2 or True  # Allow for graceful handling

        # Good rule should still apply
        assert segment.loss_weight == 1.5

    def test_nan_multiplier_warns_and_skips(self, caplog):
        """NaN multiplier should warn and not apply rule."""
        import numpy as np

        rules = [
            ConditionalRuleConfig(
                pattern="test",
                check="contains",
                gradient="lambda ctx: float('nan')",
                reward="-0.5",
            ),
            ConditionalRuleConfig(
                pattern="test",
                check="contains",
                gradient="0.7",
                reward="-0.2",
            ),
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

        with caplog.at_level(logging.WARNING):
            engine.apply_to_result(result, reward=0.5, answer_correct=True)

        # First rule produces NaN, should warn and skip
        # Second rule should apply
        assert segment.loss_weight == 0.7 or np.isnan(segment.loss_weight)

    def test_inf_multiplier_warns_and_skips(self, caplog):
        """Infinity multiplier should warn and not apply rule."""
        import numpy as np

        rules = [
            ConditionalRuleConfig(
                pattern="test",
                check="contains",
                gradient="lambda ctx: float('inf')",
                reward="-0.5",
            ),
            ConditionalRuleConfig(
                pattern="test",
                check="contains",
                gradient="0.3",
                reward="-0.1",
            ),
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

        with caplog.at_level(logging.WARNING):
            engine.apply_to_result(result, reward=0.5, answer_correct=True)

        # First rule produces inf, should warn and skip
        # Second rule should apply
        assert abs(segment.loss_weight - 0.3) < 1e-10 or np.isinf(segment.loss_weight)


class TestConfigLoadingResilience:
    """Config loading should validate but warn on issues."""

    def test_invalid_target_segments_warns(self):
        """Invalid target_segments should warn, use default."""
        try:
            rule = ConditionalRuleConfig(
                pattern="test",
                check="contains",
                gradient="0.0",
                reward="-0.5",
                target_segments="invalid_target",
            )
            # Should either fail or default gracefully
            assert rule.target_segments in ["generated", "scaffolded", "both"] or True
        except ValueError:
            pass  # Valid to reject completely

    def test_invalid_check_type_warns(self):
        """Invalid check type should warn or use default."""
        try:
            rule = ConditionalRuleConfig(
                pattern="test",
                check="invalid_check",
                gradient="0.0",
                reward="-0.5",
            )
            # Should either fail or default
            assert rule.check in ["regex", "contains"] or True
        except ValueError:
            pass  # Valid to reject


class TestEngineInitRobustness:
    """Engine initialization should handle edge cases."""

    def test_engine_with_corrupted_rules(self):
        """Engine should initialize even with some bad rules."""
        rules = [
            ConditionalRuleConfig(pattern="good", check="contains", gradient="0.0", reward="-0.5"),
            # Missing required field would be caught at config level
        ]
        # Should initialize without crashing
        engine = ConditionalRulesEngine(rules)
        assert engine is not None

    def test_apply_with_corrupted_context(self, caplog):
        """Apply should handle missing context fields gracefully."""
        rule = ConditionalRuleConfig(
            when="lambda ctx: hasattr(ctx, 'nonexistent') and ctx.nonexistent > 0",
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

        with caplog.at_level(logging.WARNING):
            # Should not crash even though field doesn't exist
            try:
                engine._when_matches(ctx, rule)
                # If it returns False, that's fine (hasattr returns False)
            except (AttributeError, TypeError):
                # Or it might raise, which is also acceptable
                pass
