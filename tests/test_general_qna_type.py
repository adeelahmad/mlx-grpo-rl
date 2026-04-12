"""Tests for types/builtins/general_qna.py — GeneralQnAType."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource
from mlx_grpo.types.builtins.general_qna import GeneralQnAType, _tokenize

# ── _tokenize ────────────────────────────────────────────────────────────────


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_removes_stopwords(self):
        tokens = _tokenize("the cat is on the mat")
        assert "the" not in tokens
        assert "is" not in tokens

    def test_short_words_removed(self):
        # Words len <= 1 removed
        tokens = _tokenize("a b c cat")
        # 'a', 'b', 'c' all len 1, removed. 'cat' len 3 > 1, kept
        assert "cat" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_only_stopwords(self):
        assert _tokenize("the and or but") == []


# ── GeneralQnAType ────────────────────────────────────────────────────────────


def _make_rollout_result(text: str = "", answer_text: str = "") -> RolloutResult:
    segments = []
    if text:
        segments.append(Segment([ord(c) for c in text], text, SegmentSource.GENERATED, False))
    if answer_text:
        from mlx_grpo.engine.result import GenerateResult

        variables = {"answer": GenerateResult(text=answer_text, tokens=[], stop_reason="stop")}
        result = RolloutResult(segments=segments, variables=variables)
    else:
        result = RolloutResult(segments=segments)
    return result


class TestGeneralQnATypeConfig:
    def test_config_has_required_keys(self):
        t = GeneralQnAType()
        cfg = t.config()
        assert "temperature" in cfg
        assert "thinking_budget" in cfg
        assert "answer_max_tokens" in cfg

    def test_chat_params(self):
        t = GeneralQnAType()
        cp = t.chat_params()
        assert cp.system_prompt is None  # system prompt disabled


class TestGeneralQnATypeReward:
    def test_no_answer(self):
        t = GeneralQnAType()
        r = _make_rollout_result("some text")
        # No ground truth → partial credit
        score = t.reward(r, answer=None)
        assert score == pytest.approx(0.3)

    def test_empty_completion(self):
        t = GeneralQnAType()
        r = _make_rollout_result("")
        score = t.reward(r, "Paris")
        assert score == pytest.approx(0.0)

    def test_perfect_overlap(self):
        t = GeneralQnAType()
        text = "Paris is the capital of France"
        r = _make_rollout_result(text)
        score = t.reward(r, text)
        assert score > 0.5

    def test_zero_overlap(self):
        t = GeneralQnAType()
        r = _make_rollout_result("banana orange apple")
        score = t.reward(r, "Paris France Europe")
        # Some words may not overlap — score should be low
        assert score < 0.5

    def test_partial_overlap(self):
        t = GeneralQnAType()
        r = _make_rollout_result("Paris is a city")
        score = t.reward(r, "Paris France")
        assert 0.0 < score <= 1.0

    def test_uses_answer_variable(self):
        t = GeneralQnAType()
        r = _make_rollout_result("scaffold ignored", answer_text="Paris capital France")
        score = t.reward(r, "Paris France Europe capital")
        assert score > 0.0

    def test_empty_ground_truth(self):
        t = GeneralQnAType()
        r = _make_rollout_result("some answer")
        # Only stopwords in ground truth → empty gt_words → partial credit
        score = t.reward(r, "the")
        assert score == pytest.approx(0.3)


class TestGeneralQnARollout:
    """Test rollout() via mock context."""

    def _make_ctx(self, stop_reason="stop_sequence", max_tokens=0):
        """Create mock RolloutContext."""
        ctx = MagicMock()
        ctx.type_config = {}
        ctx.max_completion_tokens = max_tokens

        # Mock batch scaffold
        ctx.batch.scaffold.min_ratio = 0.01
        ctx.batch.scaffold_thinking_ratio = 0.0  # no scaffolding
        ctx.batch.progress = 0.0
        ctx.batch.group_index = 0

        # Mock generate result
        think_result = MagicMock()
        think_result.stop_reason = stop_reason
        think_result.text = "careful reasoning"

        answer_result = MagicMock()
        answer_result.text = "42"

        ctx.generate.side_effect = [think_result, answer_result]
        ctx.callbacks.coin_flip.return_value = True
        return ctx

    def test_rollout_stop_sequence(self):
        t = GeneralQnAType()
        ctx = self._make_ctx(stop_reason="stop_sequence")
        t.rollout(ctx)
        # Should generate thinking and answer
        assert ctx.generate.call_count == 2

    def test_rollout_max_tokens_with_coin_flip(self):
        """When max_tokens hit but coin_flip=True, inject close tag and continue."""
        t = GeneralQnAType()
        ctx = self._make_ctx(stop_reason="max_tokens")
        ctx.callbacks.coin_flip.return_value = True
        t.rollout(ctx)
        ctx.inject.assert_called()  # inject </think>

    def test_rollout_max_tokens_without_coin_flip(self):
        """When max_tokens hit and coin_flip=False, return early."""
        t = GeneralQnAType()
        ctx = self._make_ctx(stop_reason="max_tokens")
        ctx.callbacks.coin_flip.return_value = False
        t.rollout(ctx)
        # Should generate thinking but no answer (returns early)
        assert ctx.generate.call_count == 1

    def test_rollout_with_max_completion_tokens(self):
        """max_completion_tokens=1000 should cap thinking_budget."""
        t = GeneralQnAType()
        ctx = self._make_ctx(max_tokens=1000)
        ctx.type_config = {"thinking_budget": 2048, "answer_max_tokens": 256}
        t.rollout(ctx)
        # First generate call should have max_tokens <= 1000-256
        # Just ensure it was called
        assert ctx.generate.call_count >= 1

    def test_rollout_with_scaffold(self):
        """When scaffold_thinking_ratio > min_ratio, inject scaffold."""
        t = GeneralQnAType()
        ctx = self._make_ctx()
        ctx.batch.scaffold_thinking_ratio = 0.5  # above min_ratio=0.01
        ctx.answer = "<think>scaffold text</think>"
        ctx.encode.return_value = [1, 2, 3, 4, 5]
        ctx.decode.return_value = "scaffold"
        ctx.callbacks.strip_boxed_tail.return_value = "scaffold text"
        ctx.callbacks.slice_answer_tokens.return_value = [1, 2]
        ctx.batch.scaffold.mode = "prefix"
        ctx.batch.scaffold.effective_tokens.return_value = 3

        t.rollout(ctx)
        # Should have injected scaffold
        assert ctx.inject.call_count >= 1
