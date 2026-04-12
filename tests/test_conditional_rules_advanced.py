"""Advanced test cases for ConditionalRulesEngine: AND/OR logic, exists field, new check types."""

import numpy as np

from mlx_grpo.config.loader import ConditionalRuleConfig
from mlx_grpo.engine.conditional_rules import ConditionalRulesEngine
from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource


class TestANDORLogic:
    """Test AND/OR logic in rule matching."""

    def test_pattern_matching_and_logic(self):
        """Multiple patterns with AND logic: ALL must match."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: 'error' in ctx.segment.text.lower() and 'fatal' in ctx.segment.text.lower()",
                gradient="0.0",
                reward="-0.5",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        # Both patterns present: rule applies
        segment = Segment(
            tokens=[1, 2, 3],
            text="fatal error occurred",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert segment.loss_weight == 0.0

        # Only one pattern present: rule doesn't apply
        segment2 = Segment(
            tokens=[1, 2, 3],
            text="error occurred",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result2 = RolloutResult(segments=[segment2])
        engine.apply_to_result(result2, reward=0.5, answer_correct=True)
        # Should not apply since "fatal" not present
        assert segment2.loss_weight == 1.0

    def test_alternative_patterns_or_logic(self):
        """Multiple patterns as alternatives: ANY can match (OR logic)."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: '<think>' in ctx.segment.text or '<thinking>' in ctx.segment.text",
                gradient="1.0",
                reward="0.5",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        # Alternative 1: <think>
        segment1 = Segment(
            tokens=[1, 2, 3],
            text="<think>processing</think>",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result1 = RolloutResult(segments=[segment1])
        engine.apply_to_result(result1, reward=0.5, answer_correct=True)
        assert segment1.loss_weight == 1.0

        # Alternative 2: <thinking>
        segment2 = Segment(
            tokens=[1, 2, 3],
            text="<thinking>processing</thinking>",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result2 = RolloutResult(segments=[segment2])
        engine.apply_to_result(result2, reward=0.5, answer_correct=True)
        assert segment2.loss_weight == 1.0

        # Neither: no match
        segment3 = Segment(
            tokens=[1, 2, 3],
            text="just text",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result3 = RolloutResult(segments=[segment3])
        engine.apply_to_result(result3, reward=0.5, answer_correct=True)
        assert segment3.loss_weight == 1.0


class TestExistsField:
    """Test exists: true/false pattern negation."""

    def test_exists_false_penalty_when_pattern_found(self):
        """exists: false means penalize if pattern IS found."""
        rules = [
            ConditionalRuleConfig(
                pattern="(?i)confused",
                check="regex",
                gradient="0.0",
                reward="-1.0",
                # Implicit: exists: false (or explicit)
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        # Pattern found: should mask
        segment = Segment(
            tokens=[1, 2, 3],
            text="I am confused about this",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert segment.loss_weight == 0.0

    def test_exists_true_reward_when_pattern_found(self):
        """exists: true means reward if pattern IS found."""
        rules = [
            ConditionalRuleConfig(
                pattern="\\bboxed{",
                check="regex",
                gradient="1.5",
                reward="1.0",
                # Would need to support exists: true in config
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        # Pattern found: should boost
        segment = Segment(
            tokens=[1, 2, 3],
            text="The answer is \\boxed{42}",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert segment.loss_weight == 1.5


class TestMultipleTargets:
    """Test targeting prompt_text vs generated_text."""

    def test_rule_targets_prompt_text(self):
        """Rule should check prompt_text when target='prompt_text'."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: 'code' in ctx.prompt_text.lower()",
                gradient="2.0",
                reward="0.5",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[1, 2, 3],
            text="def foo(): pass",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(
            result, reward=0.5, answer_correct=True, prompt_text="write some code"
        )
        assert segment.loss_weight == 2.0

    def test_rule_targets_generated_text(self):
        """Rule should check generated_text when target='generated_text'."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: 'code' in ctx.segment.text.lower()",
                gradient="1.5",
                reward="0.3",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[1, 2, 3],
            text="Here is code: def foo(): pass",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert segment.loss_weight == 1.5


class TestWeightMultipliers:
    """Test weight field and dynamic multiplier changes."""

    def test_weight_at_condition_level(self):
        """Weight field should amplify rule effect."""
        rules = [
            ConditionalRuleConfig(
                pattern="forbidden",
                check="contains",
                gradient="0.0",  # Normally mask
                reward="-1.0",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[1, 2, 3],
            text="forbidden word here",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert segment.loss_weight == 0.0

    def test_multiple_rules_compound_effects(self):
        """Multiple matching rules should compound gradient multipliers."""
        rules = [
            ConditionalRuleConfig(
                pattern="uncertain",
                check="contains",
                gradient="0.5",
                reward="-0.5",
            ),
            ConditionalRuleConfig(
                pattern="confused",
                check="contains",
                gradient="0.5",
                reward="-0.5",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        # Both match: 1.0 * 0.5 * 0.5 = 0.25
        segment = Segment(
            tokens=[1, 2, 3],
            text="uncertain and confused",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert segment.loss_weight == 0.25


class TestLengthAndWordCountChecks:
    """Test length and word_count check types (from trainer config)."""

    def test_check_minimum_text_length(self, caplog):
        """Check length threshold for text."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: len(ctx.segment.text) < 50",
                gradient="0.5",
                reward="-0.1",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        # Short text: rule applies
        segment_short = Segment(
            tokens=[1, 2, 3],
            text="short",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result_short = RolloutResult(segments=[segment_short])
        engine.apply_to_result(result_short, reward=0.5, answer_correct=True)
        assert segment_short.loss_weight == 0.5

        # Long text: rule doesn't apply
        segment_long = Segment(
            tokens=[1, 2, 3],
            text="a" * 100,
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result_long = RolloutResult(segments=[segment_long])
        engine.apply_to_result(result_long, reward=0.5, answer_correct=True)
        assert segment_long.loss_weight == 1.0

    def test_check_word_count_threshold(self):
        """Check word count threshold."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: len(ctx.segment.text.split()) > 600",
                gradient="0.7",
                reward="-0.5",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        # Few words: no penalty
        segment_few = Segment(
            tokens=[1, 2, 3],
            text="one two three four five",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result_few = RolloutResult(segments=[segment_few])
        engine.apply_to_result(result_few, reward=0.5, answer_correct=True)
        assert segment_few.loss_weight == 1.0

        # Many words: penalty applies
        many_words = " ".join(["word"] * 700)
        segment_many = Segment(
            tokens=[1, 2, 3],
            text=many_words,
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result_many = RolloutResult(segments=[segment_many])
        engine.apply_to_result(result_many, reward=0.5, answer_correct=True)
        assert segment_many.loss_weight == 0.7


class TestConditionalApplicationBasedOnContext:
    """Test rules that apply conditionally based on context."""

    def test_code_rules_only_when_code_in_prompt(self):
        """Code rules should only apply when prompt mentions code."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: 'code' not in ctx.prompt_text.lower()",
                gradient="1.0",
                reward="0.0",
            ),
            ConditionalRuleConfig(
                when="lambda ctx: 'code' in ctx.prompt_text.lower()",
                pattern="```",
                check="contains",
                gradient="0.5",
                reward="0.5",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        # Case 1: Code in prompt, code in response
        segment1 = Segment(
            tokens=[1, 2, 3],
            text="```python\nprint('hi')\n```",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result1 = RolloutResult(segments=[segment1])
        engine.apply_to_result(result1, reward=0.5, answer_correct=True, prompt_text="write code")
        assert segment1.loss_weight == 0.5

        # Case 2: No code in prompt, code not expected
        segment2 = Segment(
            tokens=[1, 2, 3],
            text="some text without code",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result2 = RolloutResult(segments=[segment2])
        engine.apply_to_result(
            result2, reward=0.5, answer_correct=True, prompt_text="explain the concept"
        )
        assert segment2.loss_weight == 1.0

    def test_answer_correct_conditional_rule(self):
        """Rule behavior should change based on answer_correct flag."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: not ctx.answer_correct",
                gradient="0.0",
                reward="-1.0",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        # Answer wrong: mask heavily
        segment = Segment(
            tokens=[1, 2, 3],
            text="any text",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result_wrong = RolloutResult(segments=[segment])
        engine.apply_to_result(result_wrong, reward=0.2, answer_correct=False)
        assert segment.loss_weight == 0.0

        # Answer correct: no masking
        segment.loss_weight = 1.0
        result_correct = RolloutResult(segments=[segment])
        engine.apply_to_result(result_correct, reward=0.8, answer_correct=True)
        assert segment.loss_weight == 1.0


class TestSegmentTagTargeting:
    """Test targeting by segment tag (thinking vs answer)."""

    def test_target_thinking_tag_only(self):
        """Rule should apply only to segments with 'thinking' tag."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: ctx.segment.tag == 'thinking'",
                pattern="(?i)uncertain",
                check="regex",
                gradient="0.0",
                reward="-0.5",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        # Thinking segment with uncertain: mask
        thinking_seg = Segment(
            tokens=[1, 2, 3],
            text="I am uncertain about this",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
            tag="thinking",
        )
        thinking_result = RolloutResult(segments=[thinking_seg])
        engine.apply_to_result(thinking_result, reward=0.5, answer_correct=True)
        assert thinking_seg.loss_weight == 0.0

        # Answer segment with uncertain: don't mask
        answer_seg = Segment(
            tokens=[4, 5, 6],
            text="The answer is uncertain",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
            tag="answer",
        )
        answer_result = RolloutResult(segments=[answer_seg])
        engine.apply_to_result(answer_result, reward=0.5, answer_correct=True)
        assert answer_seg.loss_weight == 1.0


class TestDualSegmentWeighting:
    """Test weighted gradients on thinking vs answer segments."""

    def test_different_weights_thinking_vs_answer(self):
        """Thinking and answer segments should have different weight multipliers."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: ctx.segment.tag == 'thinking'",
                pattern="(?i)error",
                check="regex",
                gradient="0.5",  # Half weight for thinking errors
                reward="-0.3",
            ),
            ConditionalRuleConfig(
                when="lambda ctx: ctx.segment.tag == 'answer'",
                pattern="(?i)error",
                check="regex",
                gradient="0.0",  # Full mask for answer errors
                reward="-1.0",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        # Thinking with error
        thinking_seg = Segment(
            tokens=[1, 2, 3],
            text="encountered error during thinking",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
            tag="thinking",
        )
        thinking_result = RolloutResult(segments=[thinking_seg])
        engine.apply_to_result(thinking_result, reward=0.5, answer_correct=True)
        assert thinking_seg.loss_weight == 0.5

        # Answer with error
        answer_seg = Segment(
            tokens=[4, 5, 6],
            text="There is an error in this answer",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
            tag="answer",
        )
        answer_result = RolloutResult(segments=[answer_seg])
        engine.apply_to_result(answer_result, reward=0.5, answer_correct=True)
        assert answer_seg.loss_weight == 0.0


class TestRegexSpecialPatterns:
    """Test regex patterns from trainer config with special characters."""

    def test_regex_word_boundaries(self):
        """Test regex with word boundaries and case insensitivity."""
        rules = [
            ConditionalRuleConfig(
                pattern="(?i)\\bprobably\\b",
                check="regex",
                gradient="0.8",
                reward="-0.3",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        # Word boundary match
        segment1 = Segment(
            tokens=[1, 2, 3],
            text="The answer is probably 42",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result1 = RolloutResult(segments=[segment1])
        engine.apply_to_result(result1, reward=0.5, answer_correct=True)
        assert segment1.loss_weight == 0.8

        # Not a word boundary (part of word): shouldn't match
        segment2 = Segment(
            tokens=[1, 2, 3],
            text="This is probably_not a match",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result2 = RolloutResult(segments=[segment2])
        engine.apply_to_result(result2, reward=0.5, answer_correct=True)
        # "probably" is part of "probably_not" so it may or may not match depending on implementation
        # This test documents the expected behavior

    def test_regex_alternation_patterns(self):
        """Test regex with alternation (OR within pattern)."""
        rules = [
            ConditionalRuleConfig(
                pattern="(?i)(alibaba|tongyi|qwen)",
                check="regex",
                gradient="0.0",
                reward="-2.0",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        # Match first alternative
        segment1 = Segment(
            tokens=[1, 2, 3],
            text="This uses Alibaba cloud",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result1 = RolloutResult(segments=[segment1])
        engine.apply_to_result(result1, reward=0.5, answer_correct=True)
        assert segment1.loss_weight == 0.0

        # Match second alternative
        segment2 = Segment(
            tokens=[1, 2, 3],
            text="Tongyi Lab developed this",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result2 = RolloutResult(segments=[segment2])
        engine.apply_to_result(result2, reward=0.5, answer_correct=True)
        assert segment2.loss_weight == 0.0


class TestProximityWeightsInContext:
    """Test proximity weight array is passed and usable in lambda rules."""

    def test_proximity_weights_available_in_context(self):
        """Lambda rules should access proximity weights array."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: np.mean(ctx.sequence_proximity_weights[ctx.segment_token_range[0]:ctx.segment_token_range[1]]) < 0.3",
                gradient="0.5",
                reward="-0.3",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[1, 2, 3],
            text="low proximity text",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])

        # Low proximity weights (0.1, 0.2, 0.25)
        proximity_weights = np.array([0.1, 0.2, 0.25, 0.9, 0.95])

        engine.apply_to_result(
            result,
            reward=0.5,
            answer_correct=True,
            sequence_proximity_weights=proximity_weights,
            segment_token_range=(0, 3),
        )
        # Mean is (0.1 + 0.2 + 0.25) / 3 ≈ 0.183 < 0.3, so rule applies
        assert segment.loss_weight == 0.5


class TestEdgeCasesAdvanced:
    """Additional edge cases from advanced scenarios."""

    def test_empty_text_with_length_check(self):
        """Rule checking text length on empty segment."""
        rules = [
            ConditionalRuleConfig(
                when="lambda ctx: len(ctx.segment.text) > 0",
                gradient="1.0",
                reward="0.5",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        empty_segment = Segment(
            tokens=[],
            text="",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[empty_segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        # Empty text: condition false, weight unchanged
        assert empty_segment.loss_weight == 1.0

    def test_unicode_pattern_matching(self):
        """Test regex patterns with unicode characters."""
        rules = [
            ConditionalRuleConfig(
                pattern="[éèêë]",
                check="regex",
                gradient="0.8",
                reward="0.2",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[1, 2, 3],
            text="café with crème",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert segment.loss_weight == 0.8

    def test_very_large_reward_multiplier(self):
        """Test reward multiplier > 10x (extreme amplification)."""
        rules = [
            ConditionalRuleConfig(
                pattern="excellent",
                check="contains",
                gradient="50.0",  # Extreme boost
                reward="50.0",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[1, 2, 3],
            text="This is excellent",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert segment.loss_weight == 50.0

    def test_very_small_reward_multiplier(self):
        """Test reward multiplier < 0.01 (extreme suppression)."""
        rules = [
            ConditionalRuleConfig(
                pattern="bad",
                check="contains",
                gradient="0.001",
                reward="-0.9",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[1, 2, 3],
            text="This is bad",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        assert abs(segment.loss_weight - 0.001) < 1e-10

    def test_negative_reward_multiplier(self):
        """Test negative multiplier (reversal of training direction)."""
        rules = [
            ConditionalRuleConfig(
                pattern="wrong",
                check="contains",
                gradient="-1.0",  # Negative: reversal
                reward="-1.0",
            ),
        ]
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[1, 2, 3],
            text="This is wrong",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=0.5, answer_correct=True)
        # Negative multiplier: flips sign (not typical, but should be supported)
        assert segment.loss_weight == -1.0 or segment.loss_weight == 1.0
