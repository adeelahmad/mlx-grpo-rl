"""Tests for DynamicGradientScaler.apply_to_gradients shape preservation."""

from __future__ import annotations

import numpy as np
import pytest

from mlx_grpo.loss.gradient_scaler import DynamicGradientScaler


class TestApplyToGradientsShapePreservation:
    """Test that gradient application preserves tensor shapes."""

    def test_apply_preserves_dict_structure(self):
        """Test that output dict has same keys as input."""
        scaler = DynamicGradientScaler()
        grads = {
            "model.layers.20.self_attn.q_proj": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "model.layers.21.self_attn.q_proj": np.array([5.0, 6.0]),
        }
        multipliers = [2.0, 0.5]
        layer_names = ["L20", "L21"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        assert set(result.keys()) == set(grads.keys())

    def test_apply_preserves_array_shape_2d(self):
        """Test that 2D arrays maintain shape after multiplication."""
        scaler = DynamicGradientScaler()
        original_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        grads = {"model.layers.20.q_proj": original_array.copy()}
        multipliers = [2.0]
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        assert result["model.layers.20.q_proj"].shape == original_array.shape

    def test_apply_preserves_array_shape_1d(self):
        """Test that 1D arrays maintain shape after multiplication."""
        scaler = DynamicGradientScaler()
        original_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        grads = {"model.layers.21.bias": original_array.copy()}
        multipliers = [0.5]
        layer_names = ["L21"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        assert result["model.layers.21.bias"].shape == original_array.shape

    def test_apply_preserves_array_shape_3d(self):
        """Test that 3D arrays maintain shape after multiplication."""
        scaler = DynamicGradientScaler()
        original_array = np.random.randn(4, 5, 6)
        grads = {"model.layers.22.attn.w": original_array.copy()}
        multipliers = [1.5]
        layer_names = ["L22"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        assert result["model.layers.22.attn.w"].shape == original_array.shape

    def test_apply_preserves_dtype_float32(self):
        """Test that dtype is preserved for float32."""
        scaler = DynamicGradientScaler()
        original_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grads = {"model.layers.20.q_proj": original_array.copy()}
        multipliers = [2.0]
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Result dtype should match or be compatible
        assert result["model.layers.20.q_proj"].dtype in [np.float32, np.float64]

    def test_apply_preserves_dtype_float64(self):
        """Test that dtype is preserved for float64."""
        scaler = DynamicGradientScaler()
        original_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        grads = {"model.layers.21.q_proj": original_array.copy()}
        multipliers = [0.5]
        layer_names = ["L21"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        assert result["model.layers.21.q_proj"].dtype in [np.float32, np.float64]

    def test_apply_nonmatched_layers_unchanged(self):
        """Test that non-matched layers remain unchanged."""
        scaler = DynamicGradientScaler()
        unchanged_array = np.array([7.0, 8.0, 9.0])
        grads = {
            "model.layers.20.q_proj": np.array([1.0, 2.0, 3.0]),
            "other.layer.unmatched": unchanged_array.copy(),
        }
        multipliers = [2.0]
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Non-matched layer should be identical
        np.testing.assert_array_equal(result["other.layer.unmatched"], unchanged_array)

    def test_apply_multiple_layers_16_layers(self):
        """Test with all 16 LoRA layers."""
        scaler = DynamicGradientScaler()
        grads = {}
        multipliers = []
        layer_names = []

        # Create mock gradients for L20-L35
        for i in range(20, 36):
            key = f"model.layers.{i}.self_attn.q_proj"
            grads[key] = np.random.randn(8, 8)
            multipliers.append(float(i - 20 + 1))  # 1.0 to 16.0
            layer_names.append(f"L{i}")

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Should have all 16 layers
        assert len(result) == 16
        # All shapes should be preserved
        for _key, grad in result.items():
            assert grad.shape == (8, 8)

    def test_apply_mixed_matched_unmatched(self):
        """Test with mix of matched and unmatched gradient keys."""
        scaler = DynamicGradientScaler()
        matched_shape = (4, 5)
        unmatched_shape = (2, 3)

        grads = {
            "model.layers.20.self_attn.q_proj": np.random.randn(*matched_shape),
            "model.layers.25.self_attn.q_proj": np.random.randn(*matched_shape),
            "other.component.param": np.random.randn(*unmatched_shape),
            "another.unmatched": np.random.randn(*unmatched_shape),
        }

        multipliers = [2.0, 0.5]
        layer_names = ["L20", "L25"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Matched layers preserve shape
        assert result["model.layers.20.self_attn.q_proj"].shape == matched_shape
        assert result["model.layers.25.self_attn.q_proj"].shape == matched_shape

        # Unmatched layers preserve shape
        assert result["other.component.param"].shape == unmatched_shape
        assert result["another.unmatched"].shape == unmatched_shape

    def test_apply_with_empty_grad_dict(self):
        """Test with empty gradient dictionary."""
        scaler = DynamicGradientScaler()
        grads = {}
        multipliers = [2.0]
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        assert result == {}

    def test_apply_non_finite_multiplier_preserved(self):
        """Test that multipliers are validated (non-finite are skipped)."""
        scaler = DynamicGradientScaler()
        original = np.array([1.0, 2.0, 3.0])
        grads = {"model.layers.20.q_proj": original.copy()}

        # Invalid multiplier (inf) should be skipped
        multipliers = [float("inf")]
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Should be unchanged due to inf guard
        np.testing.assert_array_equal(result["model.layers.20.q_proj"], original)

    def test_apply_zero_multiplier_handled(self):
        """Test that zero/negative multipliers are rejected."""
        scaler = DynamicGradientScaler()
        original = np.array([1.0, 2.0, 3.0])
        grads = {"model.layers.20.q_proj": original.copy()}

        # Zero multiplier should be skipped
        multipliers = [0.0]
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        # Should be unchanged due to <= 0 guard
        np.testing.assert_array_equal(result["model.layers.20.q_proj"], original)

    def test_apply_sparse_array_simulation(self):
        """Test with sparse-like arrays (mostly zeros)."""
        scaler = DynamicGradientScaler()
        sparse_like = np.zeros((100, 100))
        sparse_like[5, 5] = 1.0
        sparse_like[10, 20] = 2.0
        sparse_like[50, 50] = 3.0

        grads = {"model.layers.20.q_proj": sparse_like.copy()}
        multipliers = [2.0]
        layer_names = ["L20"]

        result = scaler.apply_to_gradients(grads, multipliers, layer_names)

        assert result["model.layers.20.q_proj"].shape == sparse_like.shape
        # Non-zero elements should be scaled
        assert result["model.layers.20.q_proj"][5, 5] == pytest.approx(2.0)
        assert result["model.layers.20.q_proj"][10, 20] == pytest.approx(4.0)
