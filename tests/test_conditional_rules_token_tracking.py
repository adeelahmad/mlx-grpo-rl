"""Tests for token-level change tracking in ConditionalRulesEngine."""

from mlx_grpo.config.loader import ConditionalRuleConfig
from mlx_grpo.engine.conditional_rules import ConditionalRulesEngine
from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource


class TestTokenChangeTracking:
    """Verify token-level changes are recorded with complete lineage."""

    def test_token_changes_recorded_on_gradient_application(self):
        """Token changes should be recorded in result.metadata['token_changes']."""
        rules = (
            ConditionalRuleConfig(pattern="bad", check="contains", gradient="0.0", reward="-0.5"),
        )
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[100, 101, 102],
            text="bad output",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=1.0, answer_correct=False)

        changes = result.metadata.get("token_changes", [])
        assert len(changes) == 3
        assert all(c["token_id"] in [100, 101, 102] for c in changes)

    def test_token_change_structure_complete_lineage(self):
        """Token change record should contain complete lineage fields."""
        rules = (
            ConditionalRuleConfig(pattern="test", check="contains", gradient="0.5", reward="0.2"),
        )
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[50],
            text="test data",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=1.0, answer_correct=False)

        changes = result.metadata.get("token_changes", [])
        assert len(changes) == 1

        change = changes[0]
        assert "token_id" in change
        assert "token_index" in change
        assert "segment_index" in change
        assert "before_weight" in change
        assert "after_weight" in change
        assert "multiplier" in change
        assert "rule_idx" in change
        assert "change_type" in change
        assert "match_reason" in change

    def test_token_change_values_correct(self):
        """Token change values should accurately reflect the modification."""
        rules = (
            ConditionalRuleConfig(pattern="boost", check="contains", gradient="2.0", reward="0.0"),
        )
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[200, 201],
            text="boost signal",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=1.0, answer_correct=False)

        changes = result.metadata.get("token_changes", [])
        assert len(changes) == 2

        for i, change in enumerate(changes):
            assert change["token_id"] == 200 + i
            assert change["token_index"] == i
            assert change["segment_index"] == 0
            assert change["before_weight"] == 1.0
            assert change["after_weight"] == 2.0
            assert change["multiplier"] == 2.0
            assert change["rule_idx"] == 0
            assert change["change_type"] == "gradient_token_level"
            assert "boost" in change["match_reason"]

    def test_token_changes_multiple_segments(self):
        """Token changes from multiple segments should be tracked separately."""
        rules = (
            ConditionalRuleConfig(pattern="test", check="contains", gradient="0.5", reward="0.0"),
        )
        engine = ConditionalRulesEngine(rules)

        result = RolloutResult(
            segments=[
                Segment(
                    tokens=[10, 11],
                    text="test A",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    loss_weight=1.0,
                ),
                Segment(
                    tokens=[20, 21, 22],
                    text="test B",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    loss_weight=1.0,
                ),
            ]
        )
        engine.apply_to_result(result, reward=1.0, answer_correct=False)

        changes = result.metadata.get("token_changes", [])
        assert len(changes) == 5  # 2 + 3 tokens

        # Verify segment indices
        changes_seg0 = [c for c in changes if c["segment_index"] == 0]
        changes_seg1 = [c for c in changes if c["segment_index"] == 1]
        assert len(changes_seg0) == 2
        assert len(changes_seg1) == 3

    def test_token_changes_multiple_rules(self):
        """Token changes should track which rule applied to each token."""
        rules = (
            ConditionalRuleConfig(pattern="bad", check="contains", gradient="0.0", reward="0.0"),
            ConditionalRuleConfig(pattern="good", check="contains", gradient="1.5", reward="0.0"),
        )
        engine = ConditionalRulesEngine(rules)

        result = RolloutResult(
            segments=[
                Segment(
                    tokens=[30, 31],
                    text="bad result",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    loss_weight=1.0,
                ),
                Segment(
                    tokens=[40, 41],
                    text="good result",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    loss_weight=1.0,
                ),
            ]
        )
        engine.apply_to_result(result, reward=1.0, answer_correct=False)

        changes = result.metadata.get("token_changes", [])
        assert len(changes) == 4

        rule_0_changes = [c for c in changes if c["rule_idx"] == 0]
        rule_1_changes = [c for c in changes if c["rule_idx"] == 1]
        assert len(rule_0_changes) == 2
        assert len(rule_1_changes) == 2

    def test_token_changes_with_lambda_condition(self):
        """Token changes should record rule matches from lambda conditions."""
        rules = (
            ConditionalRuleConfig(
                when="lambda ctx: ctx.answer_correct", gradient="1.3", reward="0.0"
            ),
        )
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[60, 61, 62],
            text="correct",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=1.0, answer_correct=True)

        changes = result.metadata.get("token_changes", [])
        assert len(changes) == 3
        assert all("ctx.answer_correct" in c["match_reason"] for c in changes)

    def test_token_changes_no_record_for_identity_multiplier(self):
        """Identity multiplier (1.0) records changes with effective_mult=1.0 for transparency."""
        rules = (
            ConditionalRuleConfig(pattern="pass", check="contains", gradient="1.0", reward="0.0"),
        )
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[70, 71],
            text="pass through",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=1.0, answer_correct=False)

        changes = result.metadata.get("token_changes", [])
        assert len(changes) == 2  # Pattern matched, record for transparency
        assert all(c["effective_multiplier"] == 1.0 for c in changes)

    def test_token_changes_empty_when_no_rules_match(self):
        """Token changes should be empty when rules don't match."""
        rules = (
            ConditionalRuleConfig(
                pattern="impossible", check="contains", gradient="0.0", reward="0.0"
            ),
        )
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[80, 81],
            text="some text",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=1.0, answer_correct=False)

        changes = result.metadata.get("token_changes", [])
        assert len(changes) == 0

    def test_token_changes_preserves_segment_token_order(self):
        """Token changes should preserve the order of tokens in segment."""
        rules = (
            ConditionalRuleConfig(pattern="order", check="contains", gradient="0.8", reward="0.0"),
        )
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[100, 200, 300, 400],
            text="order check",
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=1.0, answer_correct=False)

        changes = result.metadata.get("token_changes", [])
        assert len(changes) == 4
        assert changes[0]["token_id"] == 100
        assert changes[1]["token_id"] == 200
        assert changes[2]["token_id"] == 300
        assert changes[3]["token_id"] == 400

    def test_token_changes_tracked_per_rule_application(self):
        """Token changes should track each rule application independently."""
        rules = (
            ConditionalRuleConfig(pattern="mark", check="contains", gradient="0.6", reward="0.0"),
            ConditionalRuleConfig(
                when="lambda ctx: len(ctx.segment.text) > 5", gradient="0.9", reward="0.0"
            ),
        )
        engine = ConditionalRulesEngine(rules)

        segment = Segment(
            tokens=[90, 91, 92],
            text="marked",  # Matches both rules
            source=SegmentSource.GENERATED,
            masked=False,
            loss_weight=1.0,
        )
        result = RolloutResult(segments=[segment])
        engine.apply_to_result(result, reward=1.0, answer_correct=False)

        changes = result.metadata.get("token_changes", [])
        assert len(changes) == 6  # 3 tokens × 2 rules

        # First rule application
        rule_0_changes = [c for c in changes if c["rule_idx"] == 0]
        assert len(rule_0_changes) == 3
        assert all(c["before_weight"] == 1.0 and c["after_weight"] == 0.6 for c in rule_0_changes)

        # Second rule application
        rule_1_changes = [c for c in changes if c["rule_idx"] == 1]
        assert len(rule_1_changes) == 3
        assert all(c["before_weight"] == 0.6 and c["after_weight"] == 0.54 for c in rule_1_changes)
