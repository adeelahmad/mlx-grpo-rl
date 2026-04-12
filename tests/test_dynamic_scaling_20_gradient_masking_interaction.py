"""Tests for dynamic scaling with existing masking and conditional rules."""

from __future__ import annotations

import numpy as np
import pytest

from mlx_grpo.loss.gradient_scaler import DynamicGradientScaler


class TestGradientScalingWithExistingMasking:
    """Test scaling works alongside existing masking and conditional rules."""

    def test_scaling_with_always_masked_tokens(self):
        """Test scaling when some tokens are always masked."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.5,
            max_multiplier=3.0,
        )

        # Gradient where padding tokens (last 2 positions) are masked
        grads = {
            "model.layers.20.q_proj": np.array(
                [
                    [1.0, 2.0, 0.0, 0.0],  # First 2 active, last 2 masked
                    [3.0, 4.0, 0.0, 0.0],
                ]
            ),
        }

        multipliers = [2.0] + [1.0] * 15
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Active tokens should be scaled
        assert result["model.layers.20.q_proj"][0, 0] == pytest.approx(2.0)
        assert result["model.layers.20.q_proj"][0, 1] == pytest.approx(4.0)

        # Masked tokens should remain zero
        assert result["model.layers.20.q_proj"][0, 2] == 0.0
        assert result["model.layers.20.q_proj"][0, 3] == 0.0

    def test_scaling_with_conditional_masking(self):
        """Test scaling with conditional masking (e.g., thinking vs answer)."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.2,
            max_multiplier=4.0,
        )

        # Simulated gradient with thinking (active) and answer (masked)
        grads = {
            "model.layers.20.q_proj": np.array(
                [
                    [1.0, 2.0, 3.0, 0.0, 0.0],  # First 3 thinking, last 2 answer
                    [4.0, 5.0, 6.0, 0.0, 0.0],
                ]
            ),
            "model.layers.21.q_proj": np.ones((2, 5)) * 0.5,  # All active (no masking)
        }

        multipliers = [2.5, 1.5] + [1.0] * 14
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Layer 20 thinking part scaled
        assert result["model.layers.20.q_proj"][0, 0] == pytest.approx(2.5)
        # Layer 20 answer part stays zero
        assert result["model.layers.20.q_proj"][0, 3] == 0.0

        # Layer 21 all scaled
        np.testing.assert_array_almost_equal(
            result["model.layers.21.q_proj"],
            np.ones((2, 5)) * 0.75,
        )

    def test_scaling_respects_existing_attention_masks(self):
        """Test that scaling works with attention masks (causal, etc)."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.3,
            max_multiplier=4.0,
        )

        # Gradient with causal masking pattern (upper triangle is masked)
        grads = {
            "model.layers.20.attn_grad": np.array(
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
        }

        multipliers = [2.0] + [1.0] * 15
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Active (lower triangle) should be scaled
        assert result["model.layers.20.attn_grad"][0, 0] == pytest.approx(2.0)
        assert result["model.layers.20.attn_grad"][1, 0] == pytest.approx(4.0)
        assert result["model.layers.20.attn_grad"][2, 2] == pytest.approx(12.0)

        # Masked (upper triangle) should remain zero
        assert result["model.layers.20.attn_grad"][0, 1] == 0.0
        assert result["model.layers.20.attn_grad"][0, 2] == 0.0
        assert result["model.layers.20.attn_grad"][1, 2] == 0.0

    def test_scaling_with_mixed_layer_masks(self):
        """Test with different masks applied to different layers."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.1,
            max_multiplier=5.0,
        )

        # Layer 20: Row masking (padding)
        # Layer 21: Column masking (feature selection)
        # Layer 22: No masking
        grads = {
            "model.layers.20.q_proj": np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [0.0, 0.0],  # Masked row
                ]
            ),
            "model.layers.21.q_proj": np.array(
                [
                    [1.0, 0.0, 2.0],
                    [3.0, 0.0, 4.0],
                ]
            ),
            "model.layers.22.q_proj": np.ones((2, 3)),
        }

        multipliers = [2.0, 1.5, 3.0] + [1.0] * 13
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Layer 20: active rows scaled
        assert result["model.layers.20.q_proj"][0, 0] == pytest.approx(2.0)
        # Layer 20: masked row stays zero
        assert result["model.layers.20.q_proj"][2, 0] == 0.0

        # Layer 21: active columns scaled
        assert result["model.layers.21.q_proj"][0, 0] == pytest.approx(1.5)
        # Layer 21: masked columns stay zero
        assert result["model.layers.21.q_proj"][0, 1] == 0.0

        # Layer 22: all scaled
        np.testing.assert_array_almost_equal(
            result["model.layers.22.q_proj"],
            np.ones((2, 3)) * 3.0,
        )

    def test_scaling_zero_weight_rules(self):
        """Test with zero-weight conditional rules (weight=0 → no gradient)."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.5,
            max_multiplier=3.0,
        )

        # Simulating a rule with weight=0 (zeros out gradients)
        grads = {
            "model.layers.20.q_proj": np.array(
                [
                    [1.0, 2.0, 0.0],  # Last column has zero weight
                    [3.0, 4.0, 0.0],
                ]
            ),
        }

        multipliers = [2.0] + [1.0] * 15
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Active values scaled
        assert result["model.layers.20.q_proj"][0, 0] == pytest.approx(2.0)
        # Zero-weighted column stays zero
        assert result["model.layers.20.q_proj"][0, 2] == 0.0

    def test_scaling_with_gradient_norm_clipping_interaction(self):
        """Test that scaling works before/after gradient norm clipping."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.1,
            max_multiplier=5.0,
        )

        # Gradient with large values (would be clipped)
        grads = {
            "model.layers.20.q_proj": np.array(
                [
                    [100.0, 200.0],
                    [300.0, 400.0],
                ]
            ),
        }

        multipliers = [0.1] + [1.0] * 15  # Downscale layer 20
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Scaling happens (downscaling large gradients aids clipping)
        assert result["model.layers.20.q_proj"][0, 0] == pytest.approx(10.0)
        assert result["model.layers.20.q_proj"][1, 1] == pytest.approx(40.0)

    def test_scaling_with_multiple_skip_conditions(self):
        """Test scaling when multiple skip conditions might apply."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.1,
            max_multiplier=5.0,
        )

        # Gradient with multiple skip conditions:
        # - Row 0: always skip (padding)
        # - Column 1: always skip (unused feature)
        # - Interior: apply scaling
        grads = {
            "model.layers.20.q_proj": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],  # Always skip row
                    [1.0, 0.0, 2.0, 0.0],  # Col 1 and 3 skip
                    [3.0, 0.0, 4.0, 0.0],
                ]
            ),
        }

        multipliers = [3.0] + [1.0] * 15
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Skip rows/cols remain zero
        assert result["model.layers.20.q_proj"][0, 0] == 0.0
        assert result["model.layers.20.q_proj"][1, 1] == 0.0

        # Interior values scaled
        assert result["model.layers.20.q_proj"][1, 0] == pytest.approx(3.0)
        assert result["model.layers.20.q_proj"][1, 2] == pytest.approx(6.0)

    def test_scaling_preserves_gradient_structure(self):
        """Test that scaling preserves gradient sparsity/structure."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=1.0,
            max_multiplier=5.0,
        )

        # Sparse gradient (typical for some attention/gating mechanisms)
        grad_sparse = np.zeros((10, 10))
        grad_sparse[1, 2] = 1.0
        grad_sparse[5, 5] = 2.0
        grad_sparse[8, 9] = 3.0

        grads = {"model.layers.20.q_proj": grad_sparse}
        multipliers = [2.0] + [1.0] * 15
        layer_names = [f"L{20 + i}" for i in range(16)]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Non-zero positions remain non-zero (in same locations)
        assert result["model.layers.20.q_proj"][1, 2] == pytest.approx(2.0)
        assert result["model.layers.20.q_proj"][5, 5] == pytest.approx(4.0)
        assert result["model.layers.20.q_proj"][8, 9] == pytest.approx(6.0)

        # Zero positions remain zero
        assert result["model.layers.20.q_proj"][0, 0] == 0.0
        assert result["model.layers.20.q_proj"][3, 7] == 0.0
