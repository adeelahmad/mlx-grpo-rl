"""Tests for ExamReward opt-in flags: require_accuracy, multiplicative_accuracy,
accuracy_weight_override. All default-flag tests verify parity with old behaviour."""

from unittest.mock import MagicMock

import pytest

from mlx_grpo.rewards.exam import ExamReward

GOOD_FORMAT = (
    "<think>Let me reason carefully. The question asks which letter. It must be C.</think>\n"
    "\\boxed{C}"
)
CORRECT_ANSWER = "C"
GT_CORRECT = "C"
GT_WRONG = "D"  # model said C but gt is D


def _make_result(completion, generated=None, variables=None):
    r = MagicMock()
    r.completion_text = completion
    r.generated_text = generated if generated is not None else completion
    r.variables = variables or {}
    return r


class TestExamRequireAccuracy:
    def test_false_default_wrong_answer_gets_nonzero(self):
        r = ExamReward(require_accuracy=False)
        result = _make_result(GOOD_FORMAT)
        reward = r.compute(result, answer=GT_WRONG)
        assert reward > 0.0

    def test_true_wrong_answer_gets_wrong_max_zero(self):
        r = ExamReward(require_accuracy=True, wrong_max=0.0)
        result = _make_result(GOOD_FORMAT)
        reward = r.compute(result, answer=GT_WRONG)
        assert reward == pytest.approx(0.0)

    def test_true_correct_answer_unaffected(self):
        r_default = ExamReward(require_accuracy=False)
        r_gate = ExamReward(require_accuracy=True)
        result = _make_result(GOOD_FORMAT)
        rew_default = r_default.compute(result, answer=GT_CORRECT)
        rew_gate = r_gate.compute(result, answer=GT_CORRECT)
        assert rew_gate == pytest.approx(rew_default, abs=1e-6)

    def test_wrong_max_custom_nonzero(self):
        r = ExamReward(require_accuracy=True, wrong_max=0.05)
        result = _make_result(GOOD_FORMAT)
        reward = r.compute(result, answer=GT_WRONG)
        assert reward == pytest.approx(0.05)

    def test_gate_fires_before_penalties(self):
        r = ExamReward(require_accuracy=True, wrong_max=0.0)
        result = _make_result(GOOD_FORMAT)
        assert r.compute(result, answer=GT_WRONG) == 0.0


class TestExamMultiplicative:
    def test_false_default_parity(self):
        r_old = ExamReward()
        r_new = ExamReward(multiplicative_accuracy=False)
        result = _make_result(GOOD_FORMAT)
        assert r_old.compute(result, GT_WRONG) == pytest.approx(
            r_new.compute(result, GT_WRONG), abs=1e-9
        )

    def test_true_wrong_gets_lower_than_additive(self):
        # multiplicative: acc=0 → raw=0 → clamped=0 → sigmoid(0) ≈ 0.119
        # additive (default): acc=0 → raw>0 from format/reasoning ≈ 0.183
        # multiplicative reward is strictly lower for wrong answers
        r_mult = ExamReward(multiplicative_accuracy=True)
        r_add = ExamReward(multiplicative_accuracy=False)
        result = _make_result(GOOD_FORMAT)
        rew_mult = r_mult.compute(result, answer=GT_WRONG)
        rew_add = r_add.compute(result, answer=GT_WRONG)
        assert rew_mult < rew_add

    def test_true_correct_gets_positive_reward(self):
        r = ExamReward(multiplicative_accuracy=True)
        result = _make_result(GOOD_FORMAT)
        reward = r.compute(result, answer=GT_CORRECT)
        assert reward > 0.0


class TestExamAccuracyWeightOverride:
    def test_override_none_uses_default(self):
        r = ExamReward(accuracy_weight_override=None)
        assert r.weights.accuracy == pytest.approx(0.75)

    def test_override_changes_weight(self):
        r = ExamReward(accuracy_weight_override=0.90)
        assert r.weights.accuracy == pytest.approx(0.90)

    def test_higher_accuracy_weight_reduces_wrong_reward(self):
        r_default = ExamReward()
        r_high = ExamReward(accuracy_weight_override=0.90)
        result = _make_result(GOOD_FORMAT)
        rew_default = r_default.compute(result, GT_WRONG)
        rew_high = r_high.compute(result, GT_WRONG)
        assert rew_high <= rew_default + 1e-6


class TestExamPlateau:
    def test_wrong_answer_reward_below_0_5_default(self):
        r = ExamReward()
        result = _make_result(GOOD_FORMAT)
        reward = r.compute(result, answer=GT_WRONG)
        assert reward < 0.5

    def test_require_accuracy_breaks_plateau_all_wrong_zero(self):
        r = ExamReward(require_accuracy=True)
        for wrong_choice in ["A", "B", "D", "E"]:
            # Replace the boxed answer specifically (not first occurrence of letter)
            completion = f"<think>Let me reason carefully.</think>\n\\boxed{{{wrong_choice}}}"
            result = _make_result(completion)
            assert r.compute(result, answer=GT_CORRECT) == 0.0


class TestExamDefaultParity:
    """When all flags are at defaults, output must be byte-for-byte identical to old behaviour."""

    def test_default_weights_unchanged(self):
        r = ExamReward()
        assert r.weights.accuracy == pytest.approx(0.75)
        assert r.weights.boxed == pytest.approx(0.02)
        assert r.weights.format == pytest.approx(0.03)
        assert r.weights.reasoning == pytest.approx(0.20)

    def test_default_flags_false(self):
        r = ExamReward()
        assert r._require_accuracy is False
        assert r._multiplicative_accuracy is False

    def test_all_defaults_match_explicit_false(self):
        r_implicit = ExamReward()
        r_explicit = ExamReward(
            require_accuracy=False,
            wrong_max=0.0,
            multiplicative_accuracy=False,
            accuracy_weight_override=None,
        )
        result = _make_result(GOOD_FORMAT)
        for answer in (GT_CORRECT, GT_WRONG):
            assert r_implicit.compute(result, answer) == pytest.approx(
                r_explicit.compute(result, answer), abs=1e-9
            )

    def test_no_answer_returns_low_reward(self):
        r = ExamReward()
        result = _make_result(GOOD_FORMAT)
        reward = r.compute(result, answer=None)
        # With acc=0.0, format/reasoning still contribute small amount
        assert 0.0 <= reward < 0.5
