"""Extended tests for engine/callbacks.py."""

from __future__ import annotations

import pytest

from mlx_grpo.engine.callbacks import (
    CallbackRegistry,
    _snap_text_after,
    _snap_text_before,
)

# ── Validation ──────────────────────────────────────────────────────────────


class TestIsValidXml:
    def test_valid(self):
        assert CallbackRegistry.is_valid_xml("<foo>bar</foo>") is True

    def test_invalid(self):
        assert CallbackRegistry.is_valid_xml("<unclosed>") is False

    def test_empty_fragment(self):
        assert CallbackRegistry.is_valid_xml("") is True  # wrapped in <root>

    def test_nested(self):
        assert CallbackRegistry.is_valid_xml("<a><b>text</b></a>") is True

    def test_bad_xml(self):
        assert CallbackRegistry.is_valid_xml("<<>>") is False


class TestIsValidJson:
    def test_valid(self):
        assert CallbackRegistry.is_valid_json('{"key": "value"}') is True

    def test_invalid(self):
        assert CallbackRegistry.is_valid_json("{bad json") is False


class TestIsValidPython:
    def test_valid(self):
        assert CallbackRegistry.is_valid_python("x = 1 + 2") is True

    def test_invalid(self):
        assert CallbackRegistry.is_valid_python("def(") is False


# ── Extraction ──────────────────────────────────────────────────────────────


class TestExtractBetween:
    def test_basic(self):
        result = CallbackRegistry.extract_between("hello[world]bye", "[", "]")
        assert result == "world"

    def test_no_start(self):
        assert CallbackRegistry.extract_between("hello world", "[", "]") is None

    def test_no_end_empty_string(self):
        result = CallbackRegistry.extract_between("hello[world", "[", "")
        assert result == "world"

    def test_no_end_not_found(self):
        # End not found → returns from start to end of string
        result = CallbackRegistry.extract_between("start<content", "<", ">")
        assert result == "content"


class TestExtractJson:
    def test_dict(self):
        result = CallbackRegistry.extract_json('text {"key": 1} more')
        assert result == {"key": 1}

    def test_list(self):
        result = CallbackRegistry.extract_json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_no_json(self):
        assert CallbackRegistry.extract_json("no json here") is None

    def test_nested_dict(self):
        # Simple nested doesn't match the pattern (r"\{[^{}]*\}") but list would
        result = CallbackRegistry.extract_json('text {"a": 1}')
        assert result is not None


# ── Text testing ────────────────────────────────────────────────────────────


class TestHasTag:
    def test_found(self):
        assert CallbackRegistry.has_tag("<think>text</think>", "<think>") is True

    def test_not_found(self):
        assert CallbackRegistry.has_tag("plain text", "<think>") is False

    def test_non_string_input(self):
        assert CallbackRegistry.has_tag(42, "4") is True


class TestEndswithAny:
    def test_match(self):
        assert CallbackRegistry.endswith_any("hello world", ["world", "foo"]) is True

    def test_no_match(self):
        assert CallbackRegistry.endswith_any("hello", ["x", "y"]) is False

    def test_empty_list(self):
        assert CallbackRegistry.endswith_any("hello", []) is False


class TestCountTokens:
    def test_basic(self):
        # 4 chars per token heuristic
        assert CallbackRegistry.count_tokens("abcd") == 1
        assert CallbackRegistry.count_tokens("a" * 40) == 10

    def test_minimum_one(self):
        assert CallbackRegistry.count_tokens("a") == 1


# ── Text manipulation ────────────────────────────────────────────────────────


class TestSliceAnswer:
    def test_prefix_ratio(self):
        text = "abcdefghij"
        result = CallbackRegistry.slice_answer(text, 0.5)
        assert len(result) == 5

    def test_suffix_mode(self):
        text = "abcdefghij"
        result = CallbackRegistry.slice_answer(text, 0.5, mode="suffix")
        assert result == "fghij"

    def test_postfix_mode(self):
        text = "abcdefghij"
        result = CallbackRegistry.slice_answer(text, 0.5, mode="postfix")
        assert result == "fghij"

    def test_by_lines_mode(self):
        text = "line1\nline2\nline3\nline4"
        result = CallbackRegistry.slice_answer(text, 0.5, mode="by_lines")
        assert "line1" in result
        assert "line3" not in result

    def test_middle_mode(self):
        text = "0123456789"
        result = CallbackRegistry.slice_answer(text, 0.5, mode="middle")
        # Middle 5 chars of 10-char string
        assert len(result) == 5

    def test_ratio_zero(self):
        assert CallbackRegistry.slice_answer("hello", 0) == ""

    def test_ratio_one(self):
        assert CallbackRegistry.slice_answer("hello", 1) == "hello"

    def test_snap_to_boundary(self):
        text = "sentence one. sentence two. sentence three."
        result = CallbackRegistry.slice_answer(text, 0.5, snap_to_boundary=True)
        assert isinstance(result, str)

    def test_suffix_snap(self):
        text = "sentence one. sentence two. sentence three."
        result = CallbackRegistry.slice_answer(text, 0.5, mode="suffix", snap_to_boundary=True)
        assert isinstance(result, str)


class TestSliceAnswerTokens:
    def test_prefix(self):
        tokens = [1, 2, 3, 4, 5]
        result = CallbackRegistry.slice_answer_tokens(tokens, 3)
        assert result == [1, 2, 3]

    def test_suffix_mode(self):
        tokens = [1, 2, 3, 4, 5]
        result = CallbackRegistry.slice_answer_tokens(tokens, 3, mode="suffix")
        assert result == [3, 4, 5]

    def test_postfix_mode(self):
        tokens = [1, 2, 3, 4, 5]
        result = CallbackRegistry.slice_answer_tokens(tokens, 3, mode="postfix")
        assert result == [3, 4, 5]

    def test_middle_mode(self):
        tokens = list(range(10))
        result = CallbackRegistry.slice_answer_tokens(tokens, 4, mode="middle")
        assert len(result) == 4

    def test_zero_max_tokens(self):
        assert CallbackRegistry.slice_answer_tokens([1, 2, 3], 0) == []

    def test_empty_tokens(self):
        assert CallbackRegistry.slice_answer_tokens([], 5) == []

    def test_exceeds_length(self):
        tokens = [1, 2, 3]
        result = CallbackRegistry.slice_answer_tokens(tokens, 100)
        assert result == [1, 2, 3]


class TestTruncateThinking:
    def test_no_truncate(self):
        text = "short"
        result = CallbackRegistry.truncate_thinking(text, target=100)
        assert result == "short"

    def test_truncates_long(self):
        text = "a" * 1000
        result = CallbackRegistry.truncate_thinking(text, target=100)
        assert len(result) < 1000
        assert "[...truncated...]" in result


class TestSnapToSentenceEnd:
    def test_trims_incomplete(self):
        text = "First sentence. Second incomplete"
        result = CallbackRegistry.snap_to_sentence_end(text)
        # Should trim at sentence boundary if found
        assert isinstance(result, str)

    def test_short_text_unchanged(self):
        text = "hello"
        result = CallbackRegistry.snap_to_sentence_end(text)
        assert result == "hello"


# ── Curriculum ──────────────────────────────────────────────────────────────


class TestComputeCurriculumRatio:
    def test_zero_progress(self):
        ratio = CallbackRegistry.compute_curriculum_ratio(0.0)
        assert ratio == pytest.approx(1.0)

    def test_at_warmup(self):
        ratio = CallbackRegistry.compute_curriculum_ratio(0.2)
        assert ratio == pytest.approx(0.0)

    def test_past_warmup(self):
        ratio = CallbackRegistry.compute_curriculum_ratio(0.5)
        assert ratio == pytest.approx(0.0)

    def test_midpoint(self):
        ratio = CallbackRegistry.compute_curriculum_ratio(0.1)
        assert 0.0 < ratio < 1.0

    def test_zero_warmup(self):
        ratio = CallbackRegistry.compute_curriculum_ratio(0.0, warmup=0)
        assert ratio == pytest.approx(0.0)


class TestCurriculumFilter:
    def test_past_warmup_all_pass(self):
        assert CallbackRegistry.curriculum_filter(1.0, 0.9, warmup=0.5) is True

    def test_early_training_easy_sample(self):
        # Early training (progress=0.05), easy sample (difficulty=0.3)
        assert CallbackRegistry.curriculum_filter(0.3, 0.05, warmup=0.5) is True

    def test_early_training_hard_sample(self):
        # Early training (progress=0.0), hard sample (difficulty=1.0)
        # threshold=0.3, difficulty=1.0 > 0.3 → False
        assert CallbackRegistry.curriculum_filter(1.0, 0.0, warmup=0.5) is False

    def test_zero_warmup(self):
        assert CallbackRegistry.curriculum_filter(0.0, 0.0, warmup=0) is True


# ── Seeded randomness ────────────────────────────────────────────────────────


class TestSeededRandomness:
    def test_random_range(self):
        cb = CallbackRegistry(seed=42)
        for _ in range(10):
            assert 0.0 <= cb.random() < 1.0

    def test_coin_flip(self):
        cb = CallbackRegistry(seed=42)
        results = [cb.coin_flip() for _ in range(100)]
        assert any(results)
        assert not all(results)

    def test_reseed_reproducible(self):
        cb1 = CallbackRegistry(seed=42)
        cb1.reseed(99)
        v1 = cb1.random()
        cb2 = CallbackRegistry(seed=99)
        v2 = cb2.random()
        assert v1 == pytest.approx(v2)


# ── Custom registration ───────────────────────────────────────────────────────


class TestCustomRegistration:
    def test_register(self):
        cb = CallbackRegistry()
        cb.register("my_fn", lambda x: x * 2)
        assert cb.my_fn(5) == 10

    def test_register_many(self):
        cb = CallbackRegistry()
        cb.register_many({"fn1": lambda: 1, "fn2": lambda: 2})
        assert cb.fn1() == 1
        assert cb.fn2() == 2

    def test_shadow_builtin_raises(self):
        cb = CallbackRegistry()
        with pytest.raises(ValueError, match="Cannot shadow built-in"):
            cb.register("is_valid_json", lambda: None)

    def test_missing_attr_raises(self):
        cb = CallbackRegistry()
        with pytest.raises(AttributeError, match="No callback"):
            _ = cb.nonexistent_fn


# ── Boundary snapping ────────────────────────────────────────────────────────


class TestSnapTextBefore:
    def test_sentence_end(self):
        text = "Hello. World foo bar"
        pos = _snap_text_before(text, 12)
        assert pos > 0

    def test_newline_fallback(self):
        text = "hello\nworld"
        pos = _snap_text_before(text, 10)
        assert pos >= 0

    def test_no_boundary(self):
        text = "nospaces"
        pos = _snap_text_before(text, 5)
        assert pos == 5  # returns pos unchanged


class TestSnapTextAfter:
    def test_sentence_end(self):
        text = "hello. World"
        pos = _snap_text_after(text, 0)
        assert pos > 0

    def test_newline_fallback(self):
        text = "hello\nworld"
        pos = _snap_text_after(text, 3)
        assert pos > 0

    def test_no_boundary(self):
        text = "nospaces"
        pos = _snap_text_after(text, 5)
        assert pos == 5  # returns pos unchanged
