"""Tests for gradient-level utilities (_internals)."""

import pytest

from mlx_grpo._internals import (
    apply_gradient_mask,
    clip_gradient_norm,
    compute_gradient_alignment,
    compute_gradient_norm,
    create_layer_gradient_mask,
)


class TestGradientNorm:
    def test_flat_grads(self):
        grads = {"w1": [3.0, 4.0]}
        norm = compute_gradient_norm(grads)
        assert norm == pytest.approx(5.0)

    def test_nested_grads(self):
        grads = {"layer1": {"w": [3.0, 4.0]}}
        norm = compute_gradient_norm(grads)
        assert norm == pytest.approx(5.0)

    def test_empty(self):
        assert compute_gradient_norm({}) == 0.0


class TestGradientClipping:
    def test_no_clip_needed(self):
        grads = {"w": [0.1, 0.2]}
        clipped, norm = clip_gradient_norm(grads, max_norm=10.0)
        assert clipped == grads

    def test_clips_when_exceeds(self):
        grads = {"w": [3.0, 4.0]}  # norm = 5.0
        clipped, norm = clip_gradient_norm(grads, max_norm=2.5)
        assert norm == pytest.approx(5.0)
        clipped_norm = compute_gradient_norm(clipped)
        assert clipped_norm == pytest.approx(2.5, rel=0.01)


class TestGradientAlignment:
    def test_parallel_grads(self):
        grpo = {"w": [1.0, 0.0]}
        sft = {"w": [2.0, 0.0]}
        aligned, info = compute_gradient_alignment(grpo, sft, mode="project")
        assert info.cosine_similarity == pytest.approx(1.0, abs=0.01)

    def test_opposing_grads(self):
        grpo = {"w": [1.0, 0.0]}
        sft = {"w": [-1.0, 0.0]}
        _, info = compute_gradient_alignment(grpo, sft, mode="project")
        assert info.cosine_similarity == pytest.approx(-1.0, abs=0.01)

    def test_interpolate_mode(self):
        grpo = {"w": [1.0, 0.0]}
        sft = {"w": [0.0, 1.0]}
        aligned, _ = compute_gradient_alignment(
            grpo,
            sft,
            mode="interpolate",
            alpha=0.5,
        )
        assert aligned["w"][0] == pytest.approx(0.5)
        assert aligned["w"][1] == pytest.approx(0.5)

    def test_reject_mode_opposing(self):
        grpo = {"w": [1.0, 0.0]}
        sft = {"w": [-1.0, 0.0]}
        aligned, info = compute_gradient_alignment(grpo, sft, mode="reject")
        # Opposing component should be removed
        assert info.cosine_similarity < 0


class TestLayerMask:
    def test_freeze_patterns(self):
        layers = ["embed.weight", "layer.0.attn", "layer.1.attn", "lm_head"]
        mask = create_layer_gradient_mask(layers, freeze_patterns=["embed", "lm_head"])
        assert mask["embed.weight"] is False
        assert mask["lm_head"] is False
        assert mask["layer.0.attn"] is True

    def test_train_overrides_freeze(self):
        layers = ["layer.0.lora", "layer.0.weight", "embed"]
        mask = create_layer_gradient_mask(
            layers,
            freeze_patterns=["layer"],
            train_patterns=["lora"],
        )
        assert mask["layer.0.lora"] is True
        assert mask["layer.0.weight"] is False

    def test_apply_mask(self):
        grads = {"trainable": [1.0, 2.0], "frozen": [3.0, 4.0]}
        mask = {"trainable": True, "frozen": False}
        result = apply_gradient_mask(grads, mask)
        assert result["trainable"] == [1.0, 2.0]
        assert result["frozen"] == [0.0, 0.0]
