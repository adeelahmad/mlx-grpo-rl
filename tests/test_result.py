"""Tests for engine result types."""

from mlx_grpo.engine.result import (
    GenerateResult,
    RolloutResult,
    Segment,
    SegmentSource,
)


class TestGenerateResult:
    def test_string_protocol(self):
        r = GenerateResult(text="hello world", tokens=[1, 2, 3])
        assert str(r) == "hello world"
        assert "hello" in r
        assert r.endswith("world")
        assert r.startswith("hello")
        assert len(r) == 11
        assert r.token_count == 3

    def test_defaults(self):
        r = GenerateResult()
        assert r.text == ""
        assert r.valid is True


class TestRolloutResult:
    def _make(self):
        return RolloutResult(
            segments=[
                Segment([1, 2], "prompt", SegmentSource.PROMPT, masked=True),
                Segment([3, 4, 5], "abc", SegmentSource.GENERATED, masked=False, tag="thinking"),
                Segment([6], ".", SegmentSource.INJECTED, masked=True),
                Segment([7, 8], "42", SegmentSource.GENERATED, masked=False, tag="answer"),
            ]
        )

    def test_completion_text(self):
        assert self._make().completion_text == "abc.42"

    def test_completion_tokens(self):
        assert self._make().completion_tokens == [3, 4, 5, 6, 7, 8]

    def test_loss_mask(self):
        # loss_mask returns full-length float array: 0.0=masked, 1.0+=active
        # Segments: PROMPT[2] GENERATED[3] INJECTED[1] GENERATED[2] = 8 tokens
        # PROMPT is masked by default, INJECTED is masked, so:
        # [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        mask = self._make().loss_mask
        assert len(mask) == 8  # Full length with all tokens
        assert mask[0:2] == [0.0, 0.0]  # PROMPT masked
        assert mask[2:5] == [1.0, 1.0, 1.0]  # GENERATED active
        assert mask[5:6] == [0.0]  # INJECTED masked
        assert mask[6:8] == [1.0, 1.0]  # GENERATED active

    def test_token_counts(self):
        r = self._make()
        assert r.generated_token_count == 5
        assert r.injected_token_count == 1
        assert r.prompt_token_count == 2

    def test_tagged_segments(self):
        assert self._make().tagged_segments("thinking")[0].text == "abc"

    def test_empty(self):
        r = RolloutResult()
        assert r.completion_text == ""
        assert r.loss_mask == []
