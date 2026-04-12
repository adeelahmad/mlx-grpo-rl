"""Extended tests for engine/context.py and engine/result.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlx_grpo.engine.context import BatchContext, RolloutContext
from mlx_grpo.engine.result import GenerateResult, RolloutResult, Segment, SegmentSource
from mlx_grpo.scaffold import ScaffoldConfig

# ── BatchContext ─────────────────────────────────────────────────────────────


class TestBatchContextProperties:
    def test_scaffold_ratio_defaults(self):
        bc = BatchContext()
        assert bc.scaffold_ratio >= 0.0

    def test_scaffold_thinking_ratio(self):
        bc = BatchContext(scaffold=ScaffoldConfig(mode="prefix"))
        ratio = bc.scaffold_thinking_ratio
        assert ratio >= 0.0

    def test_scaffold_answer_ratio(self):
        bc = BatchContext(scaffold=ScaffoldConfig(mode="prefix"))
        ratio = bc.scaffold_answer_ratio
        assert ratio >= 0.0

    def test_is_last_group_true(self):
        bc = BatchContext(group_index=3, group_size=4)
        assert bc.is_last_group is True

    def test_is_last_group_false(self):
        bc = BatchContext(group_index=2, group_size=4)
        assert bc.is_last_group is False

    def test_progress(self):
        bc = BatchContext(iteration=5, total_iters=10)
        assert bc.progress == pytest.approx(0.5)

    def test_progress_zero_iters(self):
        bc = BatchContext(iteration=0, total_iters=0)
        assert bc.progress == pytest.approx(0.0)


# ── RolloutContext ────────────────────────────────────────────────────────────


def _make_backend():
    """Create mock backend."""
    backend = MagicMock()
    backend.generate.return_value = GenerateResult(
        text="test output", tokens=[116, 101, 115, 116], stop_reason="stop"
    )
    return backend


def _make_ctx(backend=None, prompt_tokens=None, max_completion_tokens=0):
    if backend is None:
        backend = _make_backend()
    if prompt_tokens is None:
        prompt_tokens = [1, 2, 3]
    return RolloutContext(
        backend=backend,
        prompt="test prompt",
        prompt_tokens=prompt_tokens,
        max_completion_tokens=max_completion_tokens,
    )


class TestRolloutContextGenerate:
    def test_basic_generate(self):
        ctx = _make_ctx()
        result = ctx.generate(max_tokens=50)
        assert isinstance(result, GenerateResult)
        assert result.text == "test output"

    def test_generate_short_circuited(self):
        ctx = _make_ctx()
        ctx.short_circuited = True
        result = ctx.generate(max_tokens=50)
        assert result.stop_reason == "short_circuited"
        assert result.text == ""

    def test_generate_with_tag_config_override(self):
        """Segment-level config in type_config overrides generate params."""
        ctx = _make_ctx()
        ctx.type_config = {"segments": {"thinking": {"temperature": 0.3}}}
        result = ctx.generate(max_tokens=100, temperature=0.7, tag="thinking")
        # Should not raise; temperature was overridden to 0.3 internally
        assert result is not None

    def test_max_completion_tokens_cap(self):
        """When max_completion_tokens set, subsequent generates are capped."""
        backend = _make_backend()
        ctx = _make_ctx(backend=backend, max_completion_tokens=10)
        # Generate once with large request
        ctx.generate(max_tokens=100, tag="thinking")
        # The backend was called with capped tokens
        call_args = backend.generate.call_args
        params = call_args[0][1]  # Second positional arg = GenerateParams
        assert params.max_tokens <= 10

    def test_generate_with_stop_sequences(self):
        ctx = _make_ctx()
        result = ctx.generate(max_tokens=50, stop=["</think>"])
        assert result is not None


class TestRolloutContextInject:
    def test_inject_creates_segment(self):
        ctx = _make_ctx()
        ctx.inject("prefix text", masked=True, tag="scaffold")
        # Should have injected segment
        assert len(ctx._segments) == 1
        seg = ctx._segments[0]
        assert seg.source == SegmentSource.INJECTED

    def test_inject_not_masked(self):
        ctx = _make_ctx()
        ctx.inject("answer text", masked=False)
        seg = ctx._segments[0]
        assert not seg.masked

    def test_inject_with_tokens(self):
        backend = _make_backend()
        backend.encode.return_value = [1, 2, 3]
        ctx = _make_ctx(backend=backend)
        ctx.inject("some text", masked=True)
        assert len(ctx._prefix_tokens) > 3  # prompt + injected


class TestRolloutContextCapture:
    def test_capture_result(self):
        ctx = _make_ctx()
        result = ctx.generate(max_tokens=50, tag="thinking")
        ctx.capture("thinking", result)
        assert "thinking" in ctx._variables  # internal dict

    def test_capture_accessible_in_result(self):
        ctx = _make_ctx()
        result = ctx.generate(max_tokens=50)
        ctx.capture("my_var", result)
        built = ctx.build_result()
        assert "my_var" in built.variables


class TestRolloutContextMetadata:
    def test_set_meta(self):
        ctx = _make_ctx()
        ctx.set_meta("key", "value")
        assert ctx._metadata["key"] == "value"

    def test_set_meta_overwrite(self):
        ctx = _make_ctx()
        ctx.set_meta("key", "v1")
        ctx.set_meta("key", "v2")
        assert ctx._metadata["key"] == "v2"


class TestRolloutContextResult:
    def test_build_result(self):
        ctx = _make_ctx()
        ctx.generate(max_tokens=50, tag="thinking")
        result = ctx.build_result()
        assert isinstance(result, RolloutResult)

    def test_empty_result(self):
        ctx = _make_ctx()
        result = ctx.build_result()
        assert isinstance(result, RolloutResult)

    def test_metadata_in_result(self):
        ctx = _make_ctx()
        ctx.set_meta("test_key", 42)
        result = ctx.build_result()
        assert result.metadata.get("test_key") == 42


class TestRolloutContextEncDec:
    def test_encode(self):
        backend = _make_backend()
        backend.encode.return_value = [104, 101]
        ctx = _make_ctx(backend=backend)
        tokens = ctx.encode("he")
        assert tokens == [104, 101]

    def test_decode(self):
        backend = _make_backend()
        backend.decode.return_value = "hello"
        ctx = _make_ctx(backend=backend)
        text = ctx.decode([104, 101, 108])
        assert text == "hello"


# ── RolloutResult ─────────────────────────────────────────────────────────────


class TestRolloutResult:
    def _make_result(self, text: str = "output") -> RolloutResult:
        seg = Segment(
            tokens=[ord(c) for c in text],
            text=text,
            source=SegmentSource.GENERATED,
            masked=False,
        )
        return RolloutResult(segments=[seg])

    def test_completion_text(self):
        r = self._make_result("hello world")
        assert r.completion_text == "hello world"

    def test_completion_tokens(self):
        r = self._make_result("abc")
        assert r.completion_tokens == [97, 98, 99]

    def test_generated_text(self):
        # Only GENERATED segments
        gen_seg = Segment([1, 2], "gen", SegmentSource.GENERATED, False)
        inj_seg = Segment([3, 4], "inj", SegmentSource.INJECTED, True)
        r = RolloutResult(segments=[gen_seg, inj_seg])
        assert r.generated_text == "gen"

    def test_variables_property(self):
        r = RolloutResult(
            segments=[Segment([1], "x", SegmentSource.GENERATED, False)],
            variables={"my_var": GenerateResult("value", [], "stop")},
        )
        assert "my_var" in r.variables

    def test_metadata_empty_default(self):
        r = self._make_result()
        assert isinstance(r.metadata, dict)

    def test_completion_token_count(self):
        r = self._make_result("abc")
        assert len(r.completion_tokens) == 3
