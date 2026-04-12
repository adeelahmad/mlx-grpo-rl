"""Tests for vision-aware loss masking across all backends.

Verifies that:
1. Vision tokens are properly masked in the loss computation
2. Text tokens receive gradients normally
3. No regressions in non-vision tasks
"""

from __future__ import annotations

from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource
from mlx_grpo.loss import compute_advantages, grpo_loss


class TestSegmentModality:
    """Test Segment modality field and loss mask computation."""

    def test_segment_modality_field_exists(self):
        """Segment should have modality field."""
        seg = Segment(
            tokens=[1, 2, 3],
            text="hello",
            source=SegmentSource.GENERATED,
            masked=False,
            modality="text",
        )
        assert seg.modality == "text"

    def test_segment_modality_vision_default_none(self):
        """Segment modality defaults to None."""
        seg = Segment(
            tokens=[1, 2, 3],
            text="hello",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        assert seg.modality is None

    def test_vision_tokens_always_masked(self):
        """Vision modality tokens should always have mask weight 0.0."""
        result = RolloutResult(
            segments=[
                Segment(
                    tokens=[101, 102],
                    text="[VIS]",
                    source=SegmentSource.GENERATED,
                    masked=False,  # Even though masked=False
                    modality="vision",  # modality=vision overrides
                ),
                Segment(
                    tokens=[103, 104],
                    text="hello",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    modality="text",
                ),
            ]
        )
        mask = result.loss_mask
        assert mask[0] == 0.0  # Vision token 1
        assert mask[1] == 0.0  # Vision token 2
        assert mask[2] == 1.0  # Text token 1
        assert mask[3] == 1.0  # Text token 2

    def test_text_modality_respects_masked_flag(self):
        """Text modality tokens respect the masked flag."""
        result = RolloutResult(
            segments=[
                Segment(
                    tokens=[101, 102],
                    text="hello",
                    source=SegmentSource.GENERATED,
                    masked=True,  # masked=True
                    modality="text",
                ),
                Segment(
                    tokens=[103, 104],
                    text="world",
                    source=SegmentSource.GENERATED,
                    masked=False,  # masked=False
                    modality="text",
                ),
            ]
        )
        mask = result.loss_mask
        assert mask[0] == 0.0  # Masked text
        assert mask[1] == 0.0
        assert mask[2] == 1.0  # Unmasked text
        assert mask[3] == 1.0

    def test_none_modality_behaves_like_before(self):
        """None modality should behave like original (no vision awareness)."""
        result = RolloutResult(
            segments=[
                Segment(
                    tokens=[101, 102],
                    text="hello",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    modality=None,  # None modality (default)
                ),
                Segment(
                    tokens=[103, 104],
                    text="world",
                    source=SegmentSource.GENERATED,
                    masked=True,
                    modality=None,
                ),
            ]
        )
        mask = result.loss_mask
        assert mask[0] == 1.0
        assert mask[1] == 1.0
        assert mask[2] == 0.0
        assert mask[3] == 0.0

    def test_prompt_tokens_always_excluded(self):
        """Prompt tokens should always be excluded (weight 0.0) by default."""
        result = RolloutResult(
            segments=[
                Segment(
                    tokens=[1, 2],
                    text="prompt",
                    source=SegmentSource.PROMPT,
                    masked=False,  # Masked=False but PROMPT source
                    modality="text",
                ),
                Segment(
                    tokens=[3, 4],
                    text="answer",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    modality="text",
                ),
            ]
        )
        mask = result.loss_mask
        assert len(mask) == 4  # Full length: prompt + generated
        assert mask[0] == 0.0  # Prompt tokens excluded
        assert mask[1] == 0.0
        assert mask[2] == 1.0  # Generated tokens included
        assert mask[3] == 1.0

    def test_loss_mask_with_mixed_modality(self):
        """Loss mask should handle mixed vision/text/none modalities."""
        result = RolloutResult(
            segments=[
                Segment(
                    tokens=[1],
                    text="[VIS]",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    modality="vision",
                ),
                Segment(
                    tokens=[2],
                    text="a",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    modality="text",
                ),
                Segment(
                    tokens=[3],
                    text="b",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    modality=None,
                ),
                Segment(
                    tokens=[4],
                    text="c",
                    source=SegmentSource.GENERATED,
                    masked=True,
                    modality="text",
                ),
            ]
        )
        mask = result.loss_mask
        assert mask == [0.0, 1.0, 1.0, 0.0]


class TestGRPOLossWithVisionMasking:
    """Test GRPO loss computation with vision tokens."""

    def test_grpo_loss_ignores_vision_tokens(self):
        """GRPO loss should be identical with/without vision tokens in mask."""
        # Setup: group of 2 rollouts
        log_probs = [
            [0.5, 0.6, 0.7, 0.8],  # Rollout 1: [vis, vis, text, text]
            [0.4, 0.5, 0.6, 0.7],  # Rollout 2: [text, text, text, text]
        ]
        ref_log_probs = [
            [0.4, 0.5, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6],
        ]
        advantages = [1.0, -0.5]

        # Mask with vision tokens (first 2 tokens of rollout 1 masked)
        masks_with_vision = [
            [0.0, 0.0, 1.0, 1.0],  # Vision tokens masked
            [1.0, 1.0, 1.0, 1.0],
        ]

        # Equivalent mask without vision tokens (just skip them)
        masks_without_vision = [
            [1.0, 1.0],  # Only text tokens
            [1.0, 1.0, 1.0, 1.0],
        ]

        # Compute loss with vision masking
        result_with_vision = grpo_loss(
            log_probs,
            ref_log_probs,
            advantages,
            masks_with_vision,
            beta=0.04,
        )

        # Compute loss without vision tokens (using only text portions)
        result_without_vision = grpo_loss(
            [[0.7, 0.8], [0.4, 0.5, 0.6, 0.7]],
            [[0.5, 0.6], [0.3, 0.4, 0.5, 0.6]],
            advantages,
            masks_without_vision,
            beta=0.04,
        )

        # Loss should be similar (might differ slightly due to advantage computation)
        assert abs(result_with_vision.loss - result_without_vision.loss) < 0.01, (
            "Vision masking should not affect loss computation"
        )

    def test_vision_tokens_dont_count_as_effective_tokens(self):
        """Effective token count should exclude masked vision tokens."""
        log_probs = [[0.5, 0.6, 0.7, 0.8]]
        ref_log_probs = [[0.4, 0.5, 0.5, 0.6]]
        advantages = [1.0]
        masks = [[0.0, 0.0, 1.0, 1.0]]  # First 2 masked (vision)

        result = grpo_loss(
            log_probs,
            ref_log_probs,
            advantages,
            masks,
            beta=0.04,
        )

        # Effective tokens should be 2, not 4
        assert result.effective_tokens == 2

    def test_advantages_computed_correctly(self):
        """Advantages should compute group-relative normalization."""
        rewards = [0.8, 0.6, 0.4]
        advantages = compute_advantages(rewards)

        # Mean = 0.6, std = sqrt(var) where var = [(0.2)^2 + (0)^2 + (-0.2)^2] / 3
        # var = 0.08/3 ≈ 0.0267, std ≈ 0.163
        mean = sum(rewards) / len(rewards)
        assert abs(mean - 0.6) < 1e-6
        assert abs(sum(advantages)) < 1e-6  # Advantages should sum to ~0


class TestSegmentModalityIntegration:
    """Integration tests for vision masking in rollouts."""

    def test_rollout_result_with_vision_segment(self):
        """RolloutResult with vision segments should mask correctly."""
        result = RolloutResult(
            segments=[
                Segment(
                    tokens=[101, 102, 103],
                    text="<image>",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    modality="vision",
                ),
                Segment(
                    tokens=[104, 105],
                    text="A dog",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    modality="text",
                ),
            ],
            prompt_tokens=[1, 2],
        )

        mask = result.loss_mask
        assert len(mask) == 5
        assert mask == [0.0, 0.0, 0.0, 1.0, 1.0]

    def test_completion_tokens_include_vision(self):
        """completion_tokens should include vision tokens (only excluded from loss)."""
        result = RolloutResult(
            segments=[
                Segment(
                    tokens=[101, 102],
                    text="<vis>",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    modality="vision",
                ),
                Segment(
                    tokens=[103, 104],
                    text="text",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    modality="text",
                ),
            ]
        )

        tokens = result.completion_tokens
        assert tokens == [101, 102, 103, 104]


class TestBackwardCompatibility:
    """Ensure changes don't break existing code."""

    def test_segment_without_modality_works(self):
        """Segments without modality field (old code) should still work."""
        seg = Segment(
            tokens=[1, 2],
            text="hello",
            source=SegmentSource.GENERATED,
            masked=False,
        )
        assert seg.modality is None

    def test_loss_mask_backward_compatible(self):
        """Loss mask should work identically for non-vision tasks."""
        result = RolloutResult(
            segments=[
                Segment(
                    tokens=[1, 2],
                    text="thinking",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    tag="thinking",
                ),
                Segment(
                    tokens=[3, 4],
                    text="answer",
                    source=SegmentSource.GENERATED,
                    masked=False,
                    tag="answer",
                ),
            ]
        )
        mask = result.loss_mask
        assert mask == [1.0, 1.0, 1.0, 1.0]


# ── End-of-File ──────────────────────────────────────────────────────
# Dependencies: test infrastructure only
# Usage: pytest tests/test_vision_loss_masking.py -v
# Status: New test file for vision masking feature
# ─────────────────────────────────────────────────────────────────────
