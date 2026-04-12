"""Tests for RolloutContext and Executor."""

import pytest

from mlx_grpo.engine.context import BatchContext, RolloutContext
from mlx_grpo.engine.executor import RolloutExecutor
from mlx_grpo.engine.result import SegmentSource


class TestBatchContext:
    def test_progress(self):
        assert BatchContext(iteration=50, total_iters=100).progress == 0.5

    def test_group_flags(self):
        assert BatchContext(group_index=0, group_size=4).is_first_group
        assert BatchContext(group_index=3, group_size=4).is_last_group

    def test_frozen(self):
        with pytest.raises(AttributeError):
            BatchContext().iteration = 99


class TestRolloutContext:
    def test_generate(self, mock_backend):
        ctx = RolloutContext(mock_backend, "p", [1, 2])
        r = ctx.generate(max_tokens=10)
        assert r.text
        built = ctx.build_result()
        # segments[0] is now the PROMPT segment; generated content starts at [1]
        assert built.segments[0].source == SegmentSource.PROMPT
        assert built.segments[1].source == SegmentSource.GENERATED

    def test_inject_masked(self, mock_backend):
        ctx = RolloutContext(mock_backend, "p", [1])
        ctx.inject("forced", masked=True)
        assert ctx.build_result().segments[1].masked

    def test_inject_unmasked(self, mock_backend):
        ctx = RolloutContext(mock_backend, "p", [1])
        ctx.inject("train", masked=False)
        assert not ctx.build_result().segments[1].masked

    def test_prefix_grows(self, mock_backend):
        ctx = RolloutContext(mock_backend, "p", [1])
        n = len(ctx._prefix_tokens)
        ctx.generate(max_tokens=5)
        assert len(ctx._prefix_tokens) > n

    def test_capture_var(self, mock_backend):
        ctx = RolloutContext(mock_backend, "p", [1])
        r = ctx.generate(max_tokens=5)
        ctx.capture("out", r)
        assert ctx.var("out") is r
        assert ctx.var("nope") is None

    def test_tags(self, mock_backend):
        ctx = RolloutContext(mock_backend, "p", [1])
        ctx.generate(max_tokens=3, tag="thinking")
        ctx.inject("x", tag="close")
        ctx.generate(max_tokens=3, tag="answer")
        assert [s.tag for s in ctx.build_result().segments] == [
            "prompt",
            "thinking",
            "close",
            "answer",
        ]

    def test_segment_overrides(self, mock_backend):
        """Segment-level type_config overrides replace hardcoded generate() params."""
        type_config = {
            "segments": {
                "answer": {"temperature": 0.1, "top_k": 5, "max_tokens": 32},
            },
        }
        ctx = RolloutContext(mock_backend, "p", [1], type_config=type_config)
        # Builtin would pass temperature=0.3, but segment override wins
        ctx.generate(max_tokens=512, temperature=0.3, tag="answer")
        # Verify override reached the backend by checking the params used
        # MockBackend doesn't record params, so validate indirectly: max_tokens
        # was capped to 32 (from segment override), so at most 32 tokens
        built = ctx.build_result()
        assert len(built.segments[1].tokens) <= 32

    def test_segment_overrides_no_tag(self, mock_backend):
        """No override when generate() is called without a tag."""
        type_config = {"segments": {"answer": {"max_tokens": 1}}}
        ctx = RolloutContext(mock_backend, "p", [1], type_config=type_config)
        ctx.generate(max_tokens=512, tag="")
        built = ctx.build_result()
        # Without tag, segment override should NOT apply — tokens can exceed 1
        assert len(built.segments[1].tokens) > 1

    def test_segment_overrides_unmatched_tag(self, mock_backend):
        """Override only applies to matching tag."""
        type_config = {"segments": {"answer": {"max_tokens": 1}}}
        ctx = RolloutContext(mock_backend, "p", [1], type_config=type_config)
        ctx.generate(max_tokens=512, tag="thinking")
        built = ctx.build_result()
        assert len(built.segments[1].tokens) > 1

    def test_segment_overrides_xtc(self, mock_backend):
        """XTC and min_tokens_to_keep flow through segment overrides."""
        type_config = {
            "segments": {
                "thinking": {
                    "xtc_probability": 0.3,
                    "xtc_threshold": 0.2,
                    "min_tokens_to_keep": 2,
                    "min_p": 0.05,
                },
            },
        }
        ctx = RolloutContext(mock_backend, "p", [1], type_config=type_config)
        ctx.generate(max_tokens=10, tag="thinking")
        # Verify the generate succeeded (MockBackend ignores sampler params)
        built = ctx.build_result()
        assert len(built.segments) == 2  # prompt + thinking
        assert built.segments[1].tag == "thinking"

    def test_seg_accessor(self, mock_backend):
        """seg() merges flat config with segment overrides."""
        type_config = {
            "temperature": 0.7,
            "thinking_budget": 512,
            "recovery_ratio": 0.8,
            "segments": {
                "thinking": {"recovery_ratio": 0.3, "stop_sequences": ["</think>"]},
                "answer": {"temperature": 0.1},
            },
        }
        ctx = RolloutContext(mock_backend, "p", [1], type_config=type_config)
        # Segment override wins over flat key
        cfg = ctx.seg("thinking")
        assert cfg["recovery_ratio"] == 0.3
        assert cfg["stop_sequences"] == ["</think>"]
        assert cfg["thinking_budget"] == 512  # inherited from flat
        assert "segments" not in cfg  # stripped
        # Unmatched segment inherits flat config
        cfg_ans = ctx.seg("answer")
        assert cfg_ans["temperature"] == 0.1
        assert cfg_ans["recovery_ratio"] == 0.8  # flat, not overridden
        # Missing tag returns flat config unchanged
        cfg_none = ctx.seg("code")
        assert cfg_none["temperature"] == 0.7


class TestExecutor:
    def test_simple(self, mock_backend):
        def fn(ctx):
            ctx.generate(max_tokens=10, tag="out")

        r = RolloutExecutor(mock_backend).execute(fn, "Hello", type_name="t")
        assert r.type_name == "t"
        assert r.generated_token_count > 0

    def test_with_callbacks(self, mock_backend):
        from mlx_grpo.engine.callbacks import CallbackRegistry

        cb = CallbackRegistry()
        cb.register("yes", lambda _: True)

        def fn(ctx):
            if ctx.callbacks.yes(""):
                ctx.inject("y", masked=True)

        r = RolloutExecutor(mock_backend).execute(fn, "p", callbacks=cb)
        assert r.injected_token_count > 0
