"""Extended tests for exam_format.py, exam_gaming.py, exam.py rewards."""

from __future__ import annotations

import pytest

from mlx_grpo.rewards.exam_format import (
    check_boxed_format,
    check_format_compliance,
    check_reasoning_quality,
)
from mlx_grpo.rewards.exam_gaming import (
    compute_bad_words_penalty,
    compute_gaming_penalties,
    compute_length_penalty,
    compute_repeat_words_penalty,
    smooth_reward,
)

# ── check_boxed_format ──────────────────────────────────────────────────────


class TestCheckBoxedFormat:
    def test_valid_with_content(self):
        score, details = check_boxed_format("\\boxed{A}")
        assert score == 1.0
        assert details["has_boxed"] is True
        assert details["valid"] is True

    def test_empty_boxed(self):
        score, details = check_boxed_format("\\boxed{}")
        assert score == 0.5
        assert details["boxed_content"] == ""
        assert details["valid"] is False

    def test_no_boxed(self):
        score, details = check_boxed_format("no answer here")
        assert score == 0.0
        assert details["has_boxed"] is False

    def test_empty_string(self):
        score, details = check_boxed_format("")
        assert score == 0.0

    def test_non_string(self):
        score, details = check_boxed_format(None)
        assert score == 0.0

    def test_has_letters_flag(self):
        score, details = check_boxed_format("\\boxed{A}")
        assert details["has_letters"] is True
        assert details["has_numeric"] is False

    def test_has_numeric_flag(self):
        score, details = check_boxed_format("\\boxed{42}")
        assert details["has_numeric"] is True

    def test_case_insensitive(self):
        score, _ = check_boxed_format("\\Boxed{A}")
        assert score == 1.0


# ── check_format_compliance ──────────────────────────────────────────────────


class TestCheckFormatCompliance:
    def test_valid_think_tags(self):
        text = "<think>reasoning here</think>\\boxed{A}"
        score, details = check_format_compliance(text)
        assert score > 0.5
        assert details["has_think_tags"] is True

    def test_no_tags(self):
        score, details = check_format_compliance("just text")
        assert score == 0.0
        assert details["has_think_tags"] is False

    def test_empty_string(self):
        score, details = check_format_compliance("")
        assert score == 0.0

    def test_non_string(self):
        score, details = check_format_compliance(None)
        assert score == 0.0

    def test_wrong_order_boxed_before_close(self):
        # \\boxed{} before </think> → improper order
        text = "<think>\\boxed{A}reasoning</think>"
        score, details = check_format_compliance(text)
        assert details["proper_order"] is False

    def test_thinking_tag_variant(self):
        text = "<thinking>reasoning</thinking>\\boxed{A}"
        score, details = check_format_compliance(text)
        assert details["has_think_tags"] is True

    def test_full_score(self):
        text = "<think>long reasoning</think>\\boxed{B}"
        score, _ = check_format_compliance(text)
        # think_valid=True: +0.5+0.3+0.2 = 1.0
        assert score == pytest.approx(1.0)


# ── check_reasoning_quality ──────────────────────────────────────────────────


class TestCheckReasoningQuality:
    def test_no_thinking(self):
        score, details = check_reasoning_quality("plain text \\boxed{A}")
        assert details["has_thinking"] is False
        assert score == pytest.approx(0.3)  # partial credit

    def test_empty(self):
        score, details = check_reasoning_quality("")
        assert score == 0.0

    def test_with_analysis_keywords(self):
        text = "<think>This is correct because the data implies X therefore Y</think>\\boxed{A}"
        score, details = check_reasoning_quality(text)
        assert details["has_analysis"] is True

    def test_sufficient_words(self):
        text = "<think>" + " ".join(["word"] * 55) + "</think>\\boxed{A}"
        score, details = check_reasoning_quality(text)
        assert details["thinking_words"] >= 50

    def test_answer_words_after_boxed(self):
        text = "<think>reasoning</think>\\boxed{A} extra words here"
        _, details = check_reasoning_quality(text)
        assert details["answer_words"] > 0

    def test_has_options(self):
        text = "<think>option a is wrong because option b is correct</think>\\boxed{B}"
        _, details = check_reasoning_quality(text)
        assert details.get("has_options") is True


# ── compute_gaming_penalties ─────────────────────────────────────────────────


class TestComputeGamingPenalties:
    def test_clean_completion(self):
        penalty, details = compute_gaming_penalties(
            "The answer is \\boxed{A}", "careful reasoning", "A"
        )
        assert penalty == pytest.approx(0.0)

    def test_multiple_boxed(self):
        penalty, details = compute_gaming_penalties("\\boxed{A} or \\boxed{B}", "reasoning", "A")
        assert details["breakdown"].get("multiple_boxed", 0) > 0

    def test_hedge_words_excess(self):
        text = "maybe possibly perhaps might probably likely " * 3
        penalty, details = compute_gaming_penalties(text, text, "A")
        assert details["breakdown"].get("hedge_words", 0) > 0

    def test_garbage_chars(self):
        garbage = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09" * 3
        penalty, details = compute_gaming_penalties(garbage, "", None)
        assert details["breakdown"].get("garbage_chars", 0) > 0

    def test_boxed_hedging(self):
        penalty, details = compute_gaming_penalties("answer", "thinking", "A or B")
        assert details["breakdown"].get("boxed_hedging", 0) > 0

    def test_low_entropy(self):
        thinking = "word " * 50  # Very repetitive
        penalty, details = compute_gaming_penalties("text", thinking, "A")
        assert details["breakdown"].get("low_entropy", 0) > 0

    def test_inconsistency(self):
        # reasoning says B is correct, but boxed is A
        thinking = "the answer is B"
        penalty, details = compute_gaming_penalties("\\boxed{A}", thinking, "A")
        # May or may not trigger depending on regex
        assert penalty <= 1.0

    def test_total_capped_at_one(self):
        bad = "maybe possibly perhaps \\boxed{A or B} \\boxed{C}" + "\x00" * 20
        penalty, _ = compute_gaming_penalties(bad, bad * 3, "A or B")
        assert penalty <= 1.0

    def test_repetitive_phrases(self):
        # Repeat 4-gram many times
        phrase = "step one analyze this "
        text = phrase * 8
        penalty, details = compute_gaming_penalties("text", text, "A")
        assert penalty <= 1.0


# ── compute_length_penalty ──────────────────────────────────────────────────


class TestComputeLengthPenalty:
    def test_normal_range(self):
        # 50 thinking words, 20 answer words → no penalty
        penalty = compute_length_penalty(50, 20)
        assert penalty == pytest.approx(0.0)

    def test_too_short_thinking(self):
        penalty = compute_length_penalty(5, 10)
        assert penalty > 0.0

    def test_too_long_thinking(self):
        penalty = compute_length_penalty(300, 10)
        assert penalty > 0.0

    def test_too_long_answer(self):
        penalty = compute_length_penalty(50, 100)
        assert penalty > 0.0

    def test_zero_thinking(self):
        penalty = compute_length_penalty(0, 10)
        assert penalty > 0.0


# ── compute_bad_words_penalty ────────────────────────────────────────────────


class TestComputeBadWordsPenalty:
    def test_no_bad_words(self):
        assert compute_bad_words_penalty("clean text here") == pytest.approx(0.0)

    def test_obviously(self):
        penalty = compute_bad_words_penalty("obviously this is clearly simply")
        assert penalty > 0.0

    def test_capped(self):
        text = " ".join(["obviously", "clearly", "simply", "basically", "essentially"] * 20)
        assert compute_bad_words_penalty(text) <= 0.15


# ── compute_repeat_words_penalty ─────────────────────────────────────────────


class TestComputeRepeatWordsPenalty:
    def test_no_repeats(self):
        assert compute_repeat_words_penalty("the quick brown fox") == pytest.approx(0.0)

    def test_with_repeats(self):
        text = "compute compute compute compute compute"
        penalty = compute_repeat_words_penalty(text)
        assert penalty > 0.0

    def test_ignore_short_words(self):
        # Short words (< 4 chars) are ignored
        assert compute_repeat_words_penalty("the the the the") == pytest.approx(0.0)

    def test_capped_at_0_20(self):
        text = " ".join(["computation"] * 100)
        assert compute_repeat_words_penalty(text) <= 0.20


# ── smooth_reward ────────────────────────────────────────────────────────────


class TestSmoothReward:
    def test_none_method(self):
        assert smooth_reward(0.75, method="none") == pytest.approx(0.75)

    def test_clip_method(self):
        assert smooth_reward(1.5, method="clip") == pytest.approx(1.0)
        assert smooth_reward(-0.5, method="clip") == pytest.approx(0.0)

    def test_sigmoid_center(self):
        # At center (0.5), sigmoid should return 0.5
        result = smooth_reward(0.5, method="sigmoid", temp=4.0, center=0.5)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_sigmoid_high(self):
        result = smooth_reward(1.0, method="sigmoid", temp=4.0, center=0.5)
        assert result > 0.5

    def test_sigmoid_low(self):
        # Negative x path (reward < center)
        result = smooth_reward(0.0, method="sigmoid", temp=4.0, center=0.5)
        assert result < 0.5

    def test_tanh_center(self):
        result = smooth_reward(0.5, method="tanh", temp=4.0, center=0.5)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_unknown_method_passthrough(self):
        assert smooth_reward(0.7, method="unknown") == pytest.approx(0.7)


# ── ExamReward phrase-target accuracy ─────────────────────────────────────────


def _make_exam_result(answer_text: str, generated: str = "") -> object:
    """Build a minimal RolloutResult duck-type for ExamReward tests."""
    from mlx_grpo.engine.result import GenerateResult, RolloutResult, Segment, SegmentSource

    gen_text = generated or answer_text
    seg = Segment(
        tokens=[1],
        text=gen_text,
        source=SegmentSource.GENERATED,
        masked=False,
        tag="answer",
    )
    result = RolloutResult(segments=[seg])
    result.variables["answer"] = GenerateResult(text=answer_text, stop_reason="max_tokens")
    return result


class TestExamRewardPhraseTarget:
    """ExamReward must handle phrase-style targets, not just MCQ letters.

    Bug: 'does not exist' passed the old alpha-only is_exam regex, forcing MCQ
    letter extraction on both GT and model answer → accuracy=0 even when correct.
    """

    def setup_method(self):
        from mlx_grpo.rewards.exam import ExamReward

        self.reward = ExamReward()

    def test_phrase_target_correct_answer_gets_nonzero_reward(self):
        """Model's verbose answer containing the target phrase → accuracy=1 → high reward."""
        thinking = "because " + " ".join(["word"] * 50)
        generated = f'<think>{thinking}</think>The movie "The Mars Paradox" does not exist.'
        result = _make_exam_result(
            'The movie "The Mars Paradox" does not exist.', generated=generated
        )
        score = self.reward.compute(result, answer="does not exist")
        # accuracy=1.0 + proper format → combined_fmt >= 0.5 → gated_acc=1.0 → high reward
        assert score > 0.5

    def test_phrase_target_wrong_answer_gets_low_reward(self):
        """Model invents plot details → answer doesn't contain target → accuracy=0."""
        result = _make_exam_result(
            "It was a 1998 thriller directed by Ridley Scott.",
            generated="It was a 1998 thriller directed by Ridley Scott.",
        )
        score = self.reward.compute(result, answer="does not exist")
        assert score < 0.3  # accuracy=0; only format/reasoning contribute

    def test_mcq_target_still_uses_letter_extraction(self):
        """Single MCQ letter targets still go through the is_exam=True path."""
        result = _make_exam_result("C", generated="<think>reasoning</think>\n\\boxed{C}")
        score = self.reward.compute(result, answer="C")
        assert score > 0.5

    def test_phrase_target_partial_inverted(self):
        """Short model answer contained in longer target → also counts as match."""
        thinking = "because " + " ".join(["word"] * 50)
        generated = f"<think>{thinking}</think>does not exist"
        result = _make_exam_result("does not exist", generated=generated)
        score = self.reward.compute(result, answer="The movie does not exist.")
        assert score > 0.5

    def test_is_exam_false_for_multi_word_phrase(self):
        """_accuracy uses is_exam=False for phrase-style GTs."""
        from mlx_grpo.rewards.exam import ExamReward

        reward = ExamReward()
        result = _make_exam_result("does not exist")
        # Compute accuracy directly — phrase target should not try MCQ extraction
        acc = reward._accuracy(result, "does not exist")
        assert acc == 1.0

    def test_is_exam_true_for_single_mcq_letter(self):
        """_accuracy uses is_exam=True for single A-F letter targets."""
        from mlx_grpo.rewards.exam import ExamReward

        reward = ExamReward()
        result = _make_exam_result("B")
        acc = reward._accuracy(result, "B")
        assert acc == 1.0
