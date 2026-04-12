"""Tests for DynamicGradientScaler interaction with loss masking."""

from __future__ import annotations

import numpy as np
import pytest

from mlx_grpo.loss.gradient_scaler import DynamicGradientScaler


class TestApplyToGradientsMaskingInteraction:
    """Test that gradient scaling preserves masked zeros in gradients."""

    def test_zero_masked_tokens_stay_zero(self):
        """Test that zero values in gradients remain zero after scaling."""
        scaler = DynamicGradientScaler()
        # Gradient with explicit zeros (e.g., from masking)
        grads = {
            "model.layers.20.q_proj": np.array(
                [
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 3.0, 0.0, 4.0],
                ]
            ),
        }
        multipliers = [2.0]
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Zeros should remain zeros (0 * 2.0 = 0)
        assert result["model.layers.20.q_proj"][0, 0] == 0.0
        assert result["model.layers.20.q_proj"][1, 0] == 0.0
        assert result["model.layers.20.q_proj"][0, 2] == 0.0
        assert result["model.layers.20.q_proj"][1, 2] == 0.0

    def test_non_zero_masked_tokens_are_scaled(self):
        """Test that non-zero masked regions are still scaled."""
        scaler = DynamicGradientScaler()
        grads = {
            "model.layers.20.q_proj": np.array(
                [
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 3.0, 0.0, 4.0],
                ]
            ),
        }
        multipliers = [2.0]
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Non-zeros should be scaled
        assert result["model.layers.20.q_proj"][0, 1] == pytest.approx(2.0)
        assert result["model.layers.20.q_proj"][0, 3] == pytest.approx(4.0)
        assert result["model.layers.20.q_proj"][1, 1] == pytest.approx(6.0)
        assert result["model.layers.20.q_proj"][1, 3] == pytest.approx(8.0)

    def test_all_zeros_gradient_stays_zero(self):
        """Test that a fully masked (all-zero) gradient remains all-zero."""
        scaler = DynamicGradientScaler()
        grads = {
            "model.layers.20.q_proj": np.zeros((5, 5)),
        }
        multipliers = [5.0]  # High multiplier
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # All zeros should remain zeros
        np.testing.assert_array_equal(result["model.layers.20.q_proj"], np.zeros((5, 5)))

    def test_partial_masking_pattern(self):
        """Test gradient with checkerboard masking pattern."""
        scaler = DynamicGradientScaler()
        # Checkerboard: alternating zero/non-zero
        grad = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0, 2.0],
                [3.0, 0.0, 3.0, 0.0],
            ]
        )
        grads = {"model.layers.21.q_proj": grad.copy()}
        multipliers = [3.0]
        layer_names = ["L21"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        expected = grad * 3.0
        np.testing.assert_array_almost_equal(result["model.layers.21.q_proj"], expected)

    def test_row_masking_pattern(self):
        """Test gradient with rows masked out (e.g., padding tokens)."""
        scaler = DynamicGradientScaler()
        grad = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [0.0, 0.0, 0.0],  # Masked row
                [0.0, 0.0, 0.0],  # Masked row
            ]
        )
        grads = {"model.layers.22.q_proj": grad.copy()}
        multipliers = [2.5]
        layer_names = ["L22"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # First two rows scaled, masked rows stay zero
        np.testing.assert_array_almost_equal(
            result["model.layers.22.q_proj"][:2, :], grad[:2, :] * 2.5
        )
        masked_result = result["model.layers.22.q_proj"][2:, :]
        np.testing.assert_array_equal(masked_result, np.zeros((2, 3)))

    def test_negative_gradients_scaled_correctly(self):
        """Test that negative gradients (also valid in backprop) are scaled."""
        scaler = DynamicGradientScaler()
        grad = np.array(
            [
                [1.0, -1.0],
                [-2.0, 2.0],
            ]
        )
        grads = {"model.layers.20.q_proj": grad.copy()}
        multipliers = [2.0]
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        expected = grad * 2.0
        np.testing.assert_array_almost_equal(result["model.layers.20.q_proj"], expected)

    def test_multiple_layers_with_different_masks(self):
        """Test multiple layers with different masking patterns."""
        scaler = DynamicGradientScaler()

        grad_l20 = np.array([[1.0, 0.0], [0.0, 2.0]])  # Diagonal non-zero
        grad_l21 = np.array([[0.0, 0.0], [0.0, 0.0]])  # All zero (fully masked)
        grad_l22 = np.array([[3.0, 4.0], [5.0, 6.0]])  # Fully active

        grads = {
            "model.layers.20.q_proj": grad_l20.copy(),
            "model.layers.21.q_proj": grad_l21.copy(),
            "model.layers.22.q_proj": grad_l22.copy(),
        }

        multipliers = [2.0, 1.5, 3.0]
        layer_names = ["L20", "L21", "L22"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # L20: diagonal scaled
        np.testing.assert_array_almost_equal(result["model.layers.20.q_proj"], grad_l20 * 2.0)
        # L21: all zero
        np.testing.assert_array_equal(result["model.layers.21.q_proj"], np.zeros((2, 2)))
        # L22: all scaled
        np.testing.assert_array_almost_equal(result["model.layers.22.q_proj"], grad_l22 * 3.0)

    def test_very_small_nonzero_preserved(self):
        """Test that very small non-zero values are still scaled (not treated as masked)."""
        scaler = DynamicGradientScaler()
        grad = np.array(
            [
                [1e-8, 0.0],
                [1e-10, 0.0],
            ]
        )
        grads = {"model.layers.20.q_proj": grad.copy()}
        multipliers = [2.0]
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Small values should still be scaled
        assert result["model.layers.20.q_proj"][0, 0] == pytest.approx(2e-8)
        assert result["model.layers.20.q_proj"][1, 0] == pytest.approx(2e-10)

    def test_nan_in_gradient_not_scaled(self):
        """Test that NaN in gradients is handled gracefully."""
        scaler = DynamicGradientScaler()
        grad = np.array(
            [
                [1.0, np.nan],
                [2.0, 3.0],
            ]
        )
        grads = {"model.layers.20.q_proj": grad.copy()}
        multipliers = [2.0]
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Finite values should be scaled, NaN handling depends on numpy
        assert result["model.layers.20.q_proj"][1, 0] == pytest.approx(4.0)
        assert result["model.layers.20.q_proj"][1, 1] == pytest.approx(6.0)

    def test_scaling_multiplier_one_no_change(self):
        """Test that multiplier=1.0 results in no change (identity)."""
        scaler = DynamicGradientScaler()
        grad = np.array(
            [
                [1.5, 0.0, 2.5],
                [0.0, 3.0, 0.0],
            ]
        )
        grads = {"model.layers.20.q_proj": grad.copy()}
        multipliers = [1.0]
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # No change with multiplier 1.0
        np.testing.assert_array_equal(result["model.layers.20.q_proj"], grad)

    def test_scaling_multiplier_fraction(self):
        """Test that multiplier < 1.0 reduces gradients."""
        scaler = DynamicGradientScaler()
        grad = np.array(
            [
                [4.0, 6.0],
                [8.0, 10.0],
            ]
        )
        grads = {"model.layers.20.q_proj": grad.copy()}
        multipliers = [0.5]
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        expected = grad * 0.5
        np.testing.assert_array_almost_equal(result["model.layers.20.q_proj"], expected)
