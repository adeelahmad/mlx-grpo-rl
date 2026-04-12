"""Tests for DynamicGradientScaler initialization and config validation."""

from __future__ import annotations

import pytest

from mlx_grpo.loss.gradient_scaler import DynamicGradientScaler


class TestDynamicGradientScalerInitialization:
    """Test DynamicGradientScaler creation and parameter validation."""

    def test_default_initialization(self):
        """Test creating scaler with defaults."""
        scaler = DynamicGradientScaler()
        assert scaler.angle_drift_threshold == 0.005
        assert scaler.max_multiplier == 5.0
        assert scaler.min_multiplier == 0.1

    def test_custom_thresholds(self):
        """Test creating scaler with custom parameters."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            max_multiplier=10.0,
            min_multiplier=0.2,
        )
        assert scaler.angle_drift_threshold == 0.01
        assert scaler.max_multiplier == 10.0
        assert scaler.min_multiplier == 0.2

    def test_invalid_min_multiplier_zero(self):
        """Test that min_multiplier must be > 0."""
        with pytest.raises(ValueError, match="min_multiplier must be > 0"):
            DynamicGradientScaler(min_multiplier=0.0)

    def test_invalid_min_multiplier_negative(self):
        """Test that min_multiplier cannot be negative."""
        with pytest.raises(ValueError, match="min_multiplier must be > 0"):
            DynamicGradientScaler(min_multiplier=-0.5)

    def test_invalid_max_multiplier_zero(self):
        """Test that max_multiplier must be > 0."""
        with pytest.raises(ValueError, match="max_multiplier must be > 0"):
            DynamicGradientScaler(max_multiplier=0.0)

    def test_invalid_max_multiplier_negative(self):
        """Test that max_multiplier cannot be negative."""
        with pytest.raises(ValueError, match="max_multiplier must be > 0"):
            DynamicGradientScaler(max_multiplier=-1.0)

    def test_invalid_bounds_min_equals_max(self):
        """Test that min_multiplier must be < max_multiplier."""
        with pytest.raises(ValueError, match="min_multiplier .* must be < max_multiplier"):
            DynamicGradientScaler(min_multiplier=1.0, max_multiplier=1.0)

    def test_invalid_bounds_min_greater_than_max(self):
        """Test that min_multiplier must be < max_multiplier."""
        with pytest.raises(ValueError, match="min_multiplier .* must be < max_multiplier"):
            DynamicGradientScaler(min_multiplier=5.0, max_multiplier=1.0)

    def test_valid_near_bounds(self):
        """Test that min and max can be very close but distinct."""
        scaler = DynamicGradientScaler(
            min_multiplier=1.0,
            max_multiplier=1.01,
        )
        assert scaler.min_multiplier == 1.0
        assert scaler.max_multiplier == 1.01

    def test_wide_range_multipliers(self):
        """Test with wide range of multipliers."""
        scaler = DynamicGradientScaler(
            min_multiplier=0.01,
            max_multiplier=100.0,
        )
        assert scaler.min_multiplier == 0.01
        assert scaler.max_multiplier == 100.0

    def test_threshold_positive(self):
        """Test that threshold can be various positive values."""
        for threshold in [0.001, 0.01, 0.1, 1.0]:
            scaler = DynamicGradientScaler(angle_drift_threshold=threshold)
            assert scaler.angle_drift_threshold == threshold

    def test_threshold_zero_allowed(self):
        """Test that threshold can be 0 (all drifts treated as high)."""
        scaler = DynamicGradientScaler(angle_drift_threshold=0.0)
        assert scaler.angle_drift_threshold == 0.0
