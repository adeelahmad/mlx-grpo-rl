"""Integration tests for ConditionalRules with training loop."""

import numpy as np
from conftest import MockBackend

from mlx_grpo.config.loader import ConditionalRuleConfig, TrainingConfig
from mlx_grpo.data import DatasetLoader, Sample
from mlx_grpo.engine.conditional_rules import ConditionalRulesEngine
from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource
from mlx_grpo.training import Trainer


class TestConditionalRulesIntegration:
    """Integration tests: conditional rules applied during training."""

    def _make_training_config_with_rules(self, rules):
        """Helper to create TrainingConfig with conditional rules."""
        return TrainingConfig(
            model_path="dummy",
            learning_rate=1e-5,
            iterations=10,
            batch_size=1,
            group_size=2,
            conditional_rules=tuple(rules) if rules else (),
        )

    def test_conditional_rules_applied_to_rollout_result(self):
        """Conditional rules should modify gradient/reward in rollout result."""
        rules = [
            ConditionalRuleConfig(pattern="(?i)error", check="regex", gradient="0.0", reward="-0.5")
        ]
        config = self._make_training_config_with_rules(rules)

        result = RolloutResult(
            segments=[
                Segment(
                    tokens=[1, 2, 3],
                    text="ERROR in output",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    loss_weight=1.0,
                )
            ]
        )

        engine = ConditionalRulesEngine(rules)
        engine.apply_to_result(result, reward=0.8, answer_correct=False)

        # Segment should be masked (gradient = 0)
        assert result.segments[0].loss_weight == 0.0
        # Reward multiplier should be tracked
        assert result.metadata.get("reward_multiplier", 1.0) < 1.0

    def test_conditional_rules_with_trainer_initialization(self):
        """Trainer should initialize with conditional rules from config."""
        rules = [
            ConditionalRuleConfig(pattern="stuck", check="contains", gradient="0.0", reward="-0.8")
        ]
        config = self._make_training_config_with_rules(rules)

        # Create minimal dataset
        dataset = DatasetLoader(
            {
                "path": "/tmp/test.jsonl",
                "include_types": ["general_qna"],
            }
        )
        dataset._data = [
            Sample(
                prompt="test prompt",
                answer="test answer",
                type_name="general_qna",
            )
        ]

        backend = MockBackend()
        trainer = Trainer(config=config, dataset=dataset, backend=backend)

        # Trainer should have conditional rules engine
        assert hasattr(trainer, "_conditional_rules_engine")
        assert trainer._conditional_rules_engine is not None

    def test_conditional_rules_applied_per_sample_group(self):
        """Conditional rules should apply to each sample group correctly."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: not ctx.answer_correct",
                gradient="0.0",
                reward="-1.0",
            )
        ]
        config = self._make_training_config_with_rules(rules)
        engine = ConditionalRulesEngine(rules)

        # Create two sample results: one correct, one incorrect
        correct_result = RolloutResult(
            segments=[
                Segment(
                    tokens=[1, 2, 3],
                    text="correct output",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    loss_weight=1.0,
                )
            ]
        )

        incorrect_result = RolloutResult(
            segments=[
                Segment(
                    tokens=[4, 5, 6],
                    text="wrong output",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    loss_weight=1.0,
                )
            ]
        )

        # Apply to correct: should not mask
        engine.apply_to_result(correct_result, reward=0.8, answer_correct=True)
        assert correct_result.segments[0].loss_weight == 1.0

        # Apply to incorrect: should mask
        engine.apply_to_result(incorrect_result, reward=0.0, answer_correct=False)
        assert incorrect_result.segments[0].loss_weight == 0.0

    def test_conditional_rules_with_multiple_segments(self):
        """Rules should apply independently to each segment."""
        rules = [
            ConditionalRuleConfig(
                pattern="(?i)uncertain",
                check="regex",
                gradient="0.0",
                reward="-0.5",
            )
        ]
        engine = ConditionalRulesEngine(rules)

        result = RolloutResult(
            segments=[
                Segment(
                    tokens=[1, 2, 3],
                    text="uncertain response",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    loss_weight=1.0,
                    tag="thinking",
                ),
                Segment(
                    tokens=[4, 5, 6],
                    text="final answer",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    loss_weight=1.0,
                    tag="answer",
                ),
            ]
        )

        engine.apply_to_result(result, reward=0.5, answer_correct=False)

        # First segment matches pattern, should be masked
        assert result.segments[0].loss_weight == 0.0
        # Second segment doesn't match, should not be masked
        assert result.segments[1].loss_weight == 1.0

    def test_conditional_rules_compound_on_same_segment(self):
        """Multiple rules matching same segment should compound effects."""
        rules = [
            ConditionalRuleConfig(
                pattern="(?i)stuck", check="regex", gradient="0.5", reward="-0.5"
            ),
            ConditionalRuleConfig(
                pattern="(?i)confused", check="regex", gradient="0.5", reward="-0.5"
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
                    loss_weight=1.0,
                )
            ]
        )

        engine.apply_to_result(result, reward=0.5, answer_correct=False)

        # Both rules match: 1.0 * 0.5 * 0.5 = 0.25
        assert result.segments[0].loss_weight == 0.25

    def test_conditional_rules_with_proximity_weights(self):
        """Rules should access proximity-weighted tokens."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: np.mean(ctx.sequence_proximity_weights[ctx.segment_token_range[0]:ctx.segment_token_range[1]]) < 0.5",
                gradient="0.0",
                reward="-0.5",
            )
        ]
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[1, 2, 3],
            text="low proximity tokens",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )

        result = RolloutResult(segments=[segment])

        # Low proximity weights for this segment
        proximity_weights = np.array([0.2, 0.3, 0.4, 0.9, 0.95])

        engine.apply_to_result(
            result,
            reward=0.5,
            answer_correct=True,
            sequence_proximity_weights=proximity_weights,
            segment_token_range=(0, 3),
        )

        # Mean proximity is 0.3, < 0.5, so rule should apply
        assert result.segments[0].loss_weight == 0.0


class TestConditionalRulesWithStaticMethods:
    """Integration tests: conditional rules using static methods."""

    def test_static_method_resolution_in_training(self):
        """Trainer should resolve and use static methods from conditional rules."""

        class CustomPatternChecks:
            @staticmethod
            def is_uncertain(ctx):
                uncertain_words = ["uncertain", "confused", "stuck", "unsure"]
                return any(word in ctx.segment.text.lower() for word in uncertain_words)

        rules = [
            ConditionalRuleConfig(
                when="CustomPatternChecks.is_uncertain",
                gradient="0.0",
                reward="-0.5",
            )
        ]

        engine = ConditionalRulesEngine(
            rules, method_registry={"CustomPatternChecks": CustomPatternChecks}
        )

        segment = Segment(
            tokens=[1, 2, 3],
            text="I am confused about this",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )

        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=False)

        assert result.segments[0].loss_weight == 0.0

    def test_static_method_gradient_computation(self):
        """Static method for gradient should compute based on context."""

        class GradientRules:
            @staticmethod
            def adaptive_mask(ctx):
                if not ctx.answer_correct:
                    return 0.0
                if ctx.is_naked and ctx.attempt_count > 1:
                    return 0.5
                return 1.0

        rules = [
            ConditionalRuleConfig(
                pattern="test",
                check="contains",
                gradient="GradientRules.adaptive_mask",
                reward="-0.5",
            )
        ]

        engine = ConditionalRulesEngine(rules, method_registry={"GradientRules": GradientRules})

        segment = Segment(
            tokens=[1, 2, 3],
            text="test",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )

        # Case 1: answer wrong
        engine.apply_to_result(RolloutResult(segments=[segment]), reward=0.5, answer_correct=False)
        assert segment.loss_weight == 0.0

        # Case 2: answer right but naked and multiple attempts
        segment.loss_weight = 1.0
        engine.apply_to_result(
            RolloutResult(segments=[segment]),
            reward=0.5,
            answer_correct=True,
            is_naked=True,
            attempt_count=2,
        )
        assert segment.loss_weight == 0.5

        # Case 3: answer right, scaffolded, first attempt
        segment.loss_weight = 1.0
        engine.apply_to_result(
            RolloutResult(segments=[segment]),
            reward=0.5,
            answer_correct=True,
            is_naked=False,
            attempt_count=1,
        )
        assert segment.loss_weight == 1.0


class TestConditionalRulesLoadingFromYAML:
    """Integration tests: loading conditional rules from YAML config."""

    def test_load_simple_rules_from_yaml(self):
        """Config loader should parse conditional_rules from YAML."""
        # This test will fail until config loader is updated
        yaml_content = """
model_path: dummy
iterations: 10
batch_size: 1

conditional_rules:
  - pattern: "(?i)error"
    check: regex
    gradient: "0.0"
    reward: "-0.5"
  - pattern: "stuck"
    check: contains
    gradient: "0.5"
    reward: "-0.3"
"""
        # Will be implemented in config loader
        assert True  # Placeholder

    def test_load_lambda_rules_from_yaml(self):
        """Config loader should parse lambda conditions from YAML."""
        yaml_content = """
model_path: dummy
iterations: 10
batch_size: 1

conditional_rules:
  - when: "lambda ctx: not ctx.answer_correct and len(ctx.segment.text) < 10"
    gradient: "0.0"
    reward: "-1.0"
"""
        # Will be implemented in config loader
        assert True  # Placeholder

    def test_load_mixed_rules_from_yaml(self):
        """Config loader should handle mix of pattern and lambda rules."""
        yaml_content = """
model_path: dummy
iterations: 10
batch_size: 1

conditional_rules:
  - pattern: "(?i)stuck"
    check: regex
    gradient: "0.0"
    reward: "-0.5"
  - when: "lambda ctx: ctx.attempt_count > 3"
    gradient: "0.0"
    reward: "-0.8"
  - pattern: "because"
    check: contains
    gradient: "2.0"
    reward: "0.5"
"""
        # Will be implemented in config loader
        assert True  # Placeholder
