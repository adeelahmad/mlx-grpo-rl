"""Tests for callback registry."""

import pytest

from mlx_grpo.engine.callbacks import CallbackRegistry


class TestBuiltins:
    def setup_method(self):
        self.cb = CallbackRegistry(seed=42)

    def test_json(self):
        assert self.cb.is_valid_json('{"a":1}')
        assert not self.cb.is_valid_json("{bad")

    def test_python(self):
        assert self.cb.is_valid_python("x = 1")
        assert not self.cb.is_valid_python("def (")

    def test_extract_boxed(self):
        assert self.cb.extract_boxed("\\boxed{42}") == "42"
        assert self.cb.extract_boxed("none") is None

    def test_extract_boxed_nested(self):
        """Nested braces like \\frac{}{} inside \\boxed{}."""
        assert self.cb.extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
        assert (
            self.cb.extract_boxed("\\boxed{a_{n}=\\frac{n(n-1)}{2}}") == "a_{n}=\\frac{n(n-1)}{2}"
        )
        # Unclosed brace fallback
        assert self.cb.extract_boxed("\\boxed{x") == "x"

    def test_extract_boxed_edge_cases(self):
        """Edge cases: empty, deeply nested, trailing text."""
        # Empty content
        assert self.cb.extract_boxed("\\boxed{}") == ""
        # Triple-nested: \frac{\frac{1}{2}}{3}
        assert (
            self.cb.extract_boxed("\\boxed{\\frac{\\frac{1}{2}}{3}}") == "\\frac{\\frac{1}{2}}{3}"
        )
        # Trailing text after boxed
        assert self.cb.extract_boxed("So \\boxed{42} is the answer") == "42"
        # Subscripts and superscripts
        assert self.cb.extract_boxed("\\boxed{x_{1}^{2}+y_{2}^{3}}") == "x_{1}^{2}+y_{2}^{3}"
        # No \\boxed at all
        assert self.cb.extract_boxed("just text") is None
        # Binom
        assert self.cb.extract_boxed("\\boxed{\\binom{n}{k}}") == "\\binom{n}{k}"

    def test_extract_between(self):
        assert self.cb.extract_between("<a>hi</a>", "<a>", "</a>") == "hi"

    def test_slice_prefix(self):
        assert self.cb.slice_answer("abcdefghij", 0.5) == "abcde"
        assert self.cb.slice_answer("text", 0.0) == ""
        assert self.cb.slice_answer("text", 1.0) == "text"

    def test_slice_by_lines(self):
        assert self.cb.slice_answer("a\nb\nc\nd", 0.5, "by_lines") == "a\nb"

    def test_truncate_thinking(self):
        long = "A" * 1000
        r = self.cb.truncate_thinking(long, target=100)
        assert len(r) < 200
        assert "[...truncated...]" in r

    def test_seeded_random(self):
        a, b = CallbackRegistry(42), CallbackRegistry(42)
        assert a.random() == b.random()

    def test_coin_flip(self):
        results = [self.cb.coin_flip(0.5) for _ in range(100)]
        assert any(results) and not all(results)


class TestCustom:
    def test_register(self):
        cb = CallbackRegistry()
        cb.register("dbl", lambda x: x * 2)
        assert cb.dbl(5) == 10

    def test_no_shadow(self):
        with pytest.raises(ValueError):
            CallbackRegistry().register("is_valid_json", lambda: None)

    def test_unknown(self):
        with pytest.raises(AttributeError):
            CallbackRegistry().nope("x")


class TestSnapToBoundary:
    """Tests for sentence boundary snapping in slice_answer."""

    def test_prefix_snaps_to_sentence_end(self):
        text = "First sentence. Second sentence. Third sentence."
        # ratio=0.5 → 24 chars → naive cut lands mid-"Second"
        result = CallbackRegistry.slice_answer(text, 0.5, "prefix", snap_to_boundary=True)
        assert result == "First sentence. "

    def test_prefix_no_boundary_falls_back(self):
        text = "onewordnoboundaries"
        result = CallbackRegistry.slice_answer(text, 0.5, "prefix", snap_to_boundary=True)
        # No sentence boundary found → falls back to naive cut
        assert result == text[:9]

    def test_prefix_snap_disabled(self):
        text = "First sentence. Second sentence."
        result = CallbackRegistry.slice_answer(text, 0.5, "prefix", snap_to_boundary=False)
        # Naive cut at 16 chars: "First sentence. " → exact
        assert result == text[:16]

    def test_suffix_snaps_to_boundary(self):
        text = "First sentence. Second sentence. Third sentence."
        result = CallbackRegistry.slice_answer(text, 0.5, "suffix", snap_to_boundary=True)
        # Should start at a sentence boundary
        assert result.startswith("Second") or result.startswith("Third")

    def test_snap_with_question_mark(self):
        text = "What is 2+2? The answer is 4. Done."
        result = CallbackRegistry.slice_answer(text, 0.5, "prefix", snap_to_boundary=True)
        assert result.rstrip().endswith("?")

    def test_snap_with_newline(self):
        text = "Line one\nLine two\nLine three\nLine four"
        result = CallbackRegistry.slice_answer(text, 0.5, "prefix", snap_to_boundary=True)
        assert result.endswith("\n")

    def test_full_ratio_ignores_snap(self):
        text = "Some text."
        assert CallbackRegistry.slice_answer(text, 1.0, "prefix", snap_to_boundary=True) == text

    def test_zero_ratio_ignores_snap(self):
        assert CallbackRegistry.slice_answer("text", 0.0, "prefix", snap_to_boundary=True) == ""


class TestSlicePositionModes:
    """slice_answer and slice_answer_tokens — suffix/postfix/middle modes."""

    def test_suffix_takes_end(self):
        # "abcdefghij" len=10, ratio=0.4 → last 4 chars
        assert CallbackRegistry.slice_answer("abcdefghij", 0.4, "suffix") == "ghij"

    def test_postfix_alias_matches_suffix(self):
        assert CallbackRegistry.slice_answer("abcdefghij", 0.4, "postfix") == "ghij"

    def test_middle_center_slice(self):
        # "0123456789" len=10, ratio=0.4 → budget=4, mid=5, start=3, end=7
        assert CallbackRegistry.slice_answer("0123456789", 0.4, "middle") == "3456"

    def test_slice_tokens_suffix(self):
        tokens = list(range(100))
        assert CallbackRegistry.slice_answer_tokens(tokens, 10, "suffix") == list(range(90, 100))

    def test_slice_tokens_postfix_alias(self):
        tokens = list(range(100))
        assert CallbackRegistry.slice_answer_tokens(tokens, 10, "postfix") == list(range(90, 100))

    def test_slice_tokens_middle_center(self):
        # 100 tokens, budget=20 → mid=50, half=10, start=40, end=60
        tokens = list(range(100))
        assert CallbackRegistry.slice_answer_tokens(tokens, 20, "middle") == list(range(40, 60))

    def test_slice_tokens_budget_exceeds_length(self):
        assert CallbackRegistry.slice_answer_tokens([1, 2, 3], 100) == [1, 2, 3]

    def test_slice_tokens_empty(self):
        assert CallbackRegistry.slice_answer_tokens([], 10) == []


class TestSnapToSentenceEnd:
    def test_trims_incomplete_trailing_sentence(self):
        # Boundary at ~60% of text → gets trimmed
        prefix = "x" * 30
        text = prefix + " The answer is B. memory requirements ranging from"
        result = CallbackRegistry.snap_to_sentence_end(text)
        assert result.endswith("B.")

    def test_preserves_text_with_no_boundary(self):
        text = "no sentence boundary here at all"
        assert CallbackRegistry.snap_to_sentence_end(text) == text

    def test_boundary_in_first_quarter_not_trimmed(self):
        # Sentence end early in text → leave unchanged to avoid over-truncation
        text = "A. " + "x" * 200 + " incomplete fragment"
        result = CallbackRegistry.snap_to_sentence_end(text)
        assert len(result) > 100

    def test_question_mark_boundary(self):
        text = "Is this correct?\nMaybe not entirely clear"
        result = CallbackRegistry.snap_to_sentence_end(text)
        assert result == "Is this correct?"
