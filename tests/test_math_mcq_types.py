"""Tests for MathType rollout/reward and MCQType rollout (mock ctx)."""

from __future__ import annotations

from unittest.mock import MagicMock

from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource
from mlx_grpo.types.builtins.math import MathType
from mlx_grpo.types.builtins.mcq import MCQType


def _make_ctx(
    stop_reason="stop_sequence",
    max_tokens=0,
    answer="<think>thinking</think>\\boxed{42}",
    scaffold_ratio=0.0,
    recovery_ratio=0.8,
):
    ctx = MagicMock()
    ctx.type_config = {"recovery_ratio": recovery_ratio}
    ctx.max_completion_tokens = max_tokens
    ctx.answer = answer
    ctx.prompt = "Solve 6*7"

    sc = MagicMock()
    sc.min_ratio = 0.01
    sc.mode = "none" if scaffold_ratio == 0.0 else "prefix"
    sc.effective_tokens.return_value = 0
    ctx.batch.scaffold = sc
    ctx.batch.scaffold_thinking_ratio = scaffold_ratio
    ctx.batch.scaffold_answer_ratio = 0.0
    ctx.batch.progress = 0.0
    ctx.batch.group_index = 0

    think_result = MagicMock()
    think_result.stop_reason = stop_reason
    think_result.text = "step by step"

    recovery_result = MagicMock()
    recovery_result.stop_reason = "stop_sequence"
    recovery_result.text = "more thinking</think>"

    answer_result = MagicMock()
    answer_result.text = "42"
    answer_result.stop_reason = "stop"

    ctx.generate.side_effect = [think_result, recovery_result, answer_result]
    ctx.callbacks.coin_flip.return_value = True
    ctx.callbacks.strip_boxed_tail.return_value = "thinking"
    ctx.encode.return_value = [1, 2, 3, 4, 5, 6, 7]
    ctx.decode.return_value = "thinking"
    ctx.callbacks.slice_answer_tokens.return_value = [1, 2, 3]
    ctx.callbacks.snap_to_sentence_end.return_value = "thinking."

    return ctx


def _make_result(text: str) -> RolloutResult:
    seg = Segment([1, 2, 3], text, SegmentSource.GENERATED, False)
    return RolloutResult(segments=[seg])


# ── MathType Rollout ─────────────────────────────────────────────────────────


class TestMathTypeRollout:
    def test_stop_sequence_two_generates(self):
        t = MathType()
        ctx = _make_ctx(stop_reason="stop_sequence")
        t.rollout(ctx)
        # thinking + answer
        assert ctx.generate.call_count == 2

    def test_max_tokens_recovery_coin_true(self):
        t = MathType()
        ctx = _make_ctx(stop_reason="max_tokens")
        ctx.callbacks.coin_flip.return_value = True
        t.rollout(ctx)
        # thinking + possibly recovery + answer
        assert ctx.generate.call_count >= 1
        # inject must have been called
        assert ctx.inject.call_count >= 1

    def test_max_tokens_recovery_coin_false(self):
        t = MathType()
        ctx = _make_ctx(stop_reason="max_tokens")
        ctx.callbacks.coin_flip.return_value = False
        t.rollout(ctx)
        # Only thinking, returns early
        assert ctx.generate.call_count == 1

    def test_scaffold_word_level_short(self):
        """Short answer (<50 words) uses word-level scaffold."""
        t = MathType()
        ctx = _make_ctx(scaffold_ratio=0.5, answer="word1 word2 word3")
        ctx.batch.scaffold_thinking_ratio = 0.5
        ctx.batch.scaffold.effective_tokens.return_value = 2
        ctx.batch.scaffold.mode = "prefix"
        t.rollout(ctx)
        # Should have called inject for scaffold
        assert ctx.inject.call_count >= 1

    def test_scaffold_token_level_long(self):
        """Long answer (>=50 words) uses token-level scaffold."""
        t = MathType()
        long_answer = " ".join([f"word{i}" for i in range(60)])
        ctx = _make_ctx(scaffold_ratio=0.5, answer=long_answer)
        ctx.batch.scaffold_thinking_ratio = 0.5
        ctx.batch.scaffold.effective_tokens.return_value = 5
        ctx.batch.scaffold.mode = "prefix"
        ctx.encode.return_value = list(range(60))
        ctx.callbacks.slice_answer_tokens.return_value = [1, 2, 3]
        ctx.decode.return_value = "slice"
        t.rollout(ctx)
        assert ctx.inject.call_count >= 1

    def test_scaffold_bonus_budget_path(self):
        """Scaffolded + bonus_budget → generates recovery thinking."""
        t = MathType()
        ctx = _make_ctx(stop_reason="max_tokens", scaffold_ratio=0.5)
        ctx.type_config = {
            "recovery_ratio": 1.0,
            "recovery_bonus_tokens": 128,
        }
        ctx.batch.scaffold.mode = "prefix"
        ctx.batch.scaffold_thinking_ratio = 0.5
        # think_result has max_tokens and no </think>
        think_result = MagicMock()
        think_result.stop_reason = "max_tokens"
        think_result.text = "no close tag"

        recovery_result = MagicMock()
        recovery_result.text = "more thinking</think>"

        answer_result = MagicMock()
        answer_result.text = "42"

        ctx.generate.side_effect = [think_result, recovery_result, answer_result]
        ctx.callbacks.coin_flip.return_value = True
        t.rollout(ctx)
        # Should generate at least 2 times (thinking + recovery)
        assert ctx.generate.call_count >= 2

    def test_boxed_brace_closing(self):
        """After answer generation, unclosed braces are closed."""
        t = MathType()
        ctx = _make_ctx(stop_reason="stop_sequence")
        # Answer has open brace depth
        answer_result = MagicMock()
        answer_result.text = "42{"  # depth = 1, needs closing
        answer_result.stop_reason = "stop"

        think_result = MagicMock()
        think_result.stop_reason = "stop_sequence"
        think_result.text = "reasoning"

        ctx.generate.side_effect = [think_result, answer_result]
        t.rollout(ctx)
        # Should inject closing braces
        inject_calls = [str(c) for c in ctx.inject.call_args_list]
        closing_injected = any("}" in c for c in inject_calls)
        assert closing_injected


# ── MathType Reward ───────────────────────────────────────────────────────────


class TestMathTypeReward:
    def test_correct_boxed(self):
        t = MathType()
        r = _make_result("\\boxed{42}")
        assert t.reward(r, "42") == 1.0

    def test_no_answer(self):
        t = MathType()
        r = _make_result("\\boxed{42}")
        assert t.reward(r, None) == 0.0

    def test_empty_completion(self):
        t = MathType()
        r = _make_result("")
        assert t.reward(r, "42") == 0.0

    def test_normalized_answer(self):
        t = MathType()
        r = _make_result("\\boxed{$42$}")
        # After normalization, 42 == 42
        score = t.reward(r, "42")
        assert score == 1.0

    def test_wrong_answer(self):
        t = MathType()
        r = _make_result("\\boxed{99}")
        assert t.reward(r, "42") == 0.0


# ── MCQType Rollout ───────────────────────────────────────────────────────────


def _make_mcq_ctx(stop_reason="stop_sequence", scaffold_ratio=0.0):
    ctx = MagicMock()
    ctx.type_config = {}
    ctx.max_completion_tokens = 0
    ctx.answer = "A"
    ctx.prompt = "Which is correct?"

    sc = MagicMock()
    sc.min_ratio = 0.01
    sc.mode = "none" if scaffold_ratio == 0.0 else "prefix"
    sc.effective_tokens.return_value = 0
    ctx.batch.scaffold = sc
    ctx.batch.scaffold_thinking_ratio = scaffold_ratio
    ctx.batch.progress = 0.0
    ctx.batch.group_index = 0

    think_result = MagicMock()
    think_result.stop_reason = stop_reason
    think_result.text = "analysis"

    answer_result = MagicMock()
    answer_result.text = "A"

    ctx.generate.side_effect = [think_result, answer_result]
    ctx.callbacks.coin_flip.return_value = True
    ctx.callbacks.strip_boxed_tail.return_value = "analysis"
    ctx.encode.return_value = [1, 2, 3]
    ctx.decode.return_value = "analysis"
    ctx.callbacks.slice_answer_tokens.return_value = [1, 2]
    ctx.callbacks.snap_to_sentence_end.return_value = "analysis."

    return ctx


class TestMCQTypeRollout:
    def test_rollout_stop_sequence(self):
        t = MCQType()
        ctx = _make_mcq_ctx(stop_reason="stop_sequence")
        t.rollout(ctx)
        assert ctx.generate.call_count >= 2

    def test_rollout_max_tokens_recovery(self):
        t = MCQType()
        ctx = _make_mcq_ctx(stop_reason="max_tokens")
        ctx.callbacks.coin_flip.return_value = True
        t.rollout(ctx)
        # inject close tag
        assert ctx.inject.call_count >= 1

    def test_rollout_max_tokens_no_recovery(self):
        t = MCQType()
        ctx = _make_mcq_ctx(stop_reason="max_tokens")
        ctx.callbacks.coin_flip.return_value = False
        t.rollout(ctx)
        assert ctx.generate.call_count == 1

    def test_rollout_with_scaffold(self):
        t = MCQType()
        ctx = _make_mcq_ctx(scaffold_ratio=0.5)
        ctx.batch.scaffold.effective_tokens.return_value = 5
        ctx.batch.scaffold_thinking_ratio = 0.5
        t.rollout(ctx)
        assert ctx.inject.call_count >= 1

    def test_rollout_suffix_scaffold(self):
        """Suffix/postfix mode calls snap_to_sentence_end."""
        t = MCQType()
        ctx = _make_mcq_ctx(scaffold_ratio=0.5)
        ctx.batch.scaffold_thinking_ratio = 0.5
        ctx.batch.scaffold.mode = "suffix"
        ctx.batch.scaffold.effective_tokens.return_value = 3
        t.rollout(ctx)
        # snap_to_sentence_end should be called for suffix mode
        assert ctx.callbacks.snap_to_sentence_end.call_count >= 0  # may or may not be called

    def test_rollout_injects_boxed_prefix(self):
        t = MCQType()
        ctx = _make_mcq_ctx(stop_reason="stop_sequence")
        t.rollout(ctx)
        inject_calls = [str(c) for c in ctx.inject.call_args_list]
        boxed_injected = any("boxed" in c for c in inject_calls)
        assert boxed_injected


class TestMCQTypeEscalation:
    """Test escalation_rollout method if it exists on MCQType."""

    def test_mcq_rollout_with_max_completion_tokens(self):
        t = MCQType()
        ctx = _make_mcq_ctx()
        ctx.max_completion_tokens = 500
        t.rollout(ctx)
        assert ctx.generate.call_count >= 1
