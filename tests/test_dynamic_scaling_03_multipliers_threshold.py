"""Tests for DynamicGradientScaler threshold behavior."""

from __future__ import annotations

import pytest

from mlx_grpo.loss.gradient_scaler import DynamicGradientScaler


class TestComputeMultipliersThreshold:
    """Test threshold behavior — drifts below threshold get min_multiplier."""

    def test_drift_just_below_threshold(self):
        """Test drift just below threshold gets min_multiplier."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.5,
            max_multiplier=3.0,
        )
        result = scaler.compute_multipliers([0.009999])
        assert result[0] == pytest.approx(0.5)

    def test_drift_exactly_at_threshold(self):
        """Test drift exactly at threshold is treated as low (< threshold, not <=)."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.5,
            max_multiplier=3.0,
        )
        result = scaler.compute_multipliers([0.01])
        # At threshold: drift == threshold, so drift < threshold is False → min_multiplier
        assert result[0] == pytest.approx(0.5)

    def test_drift_just_above_threshold(self):
        """Test drift just above threshold is scaled."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.5,
            max_multiplier=3.0,
        )
        result = scaler.compute_multipliers([0.010001])
        # Slightly above threshold, only this layer qualifies as high
        # ratio = 0.010001 / 0.010001 = 1.0
        # mult = 0.5 + 1.0 * 2.5 = 3.0
        assert result[0] == pytest.approx(3.0)

    def test_mixed_below_and_at_threshold(self):
        """Test mix of below and at-threshold drifts."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=1.0,
            max_multiplier=5.0,
        )
        drifts = [0.005, 0.01, 0.02]
        result = scaler.compute_multipliers(drifts)

        # drifts[0]=0.005 < 0.01 → 1.0
        # drifts[1]=0.01 >= 0.01 (treated as high) → scaled
        # drifts[2]=0.02 >= 0.01 → scaled
        # max_drift = 0.02
        # drifts[1]: ratio=0.01/0.02=0.5 → 1.0 + 0.5*4.0 = 3.0
        # drifts[2]: ratio=0.02/0.02=1.0 → 1.0 + 1.0*4.0 = 5.0

        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(3.0)
        assert result[2] == pytest.approx(5.0)

    def test_zero_threshold_all_high(self):
        """Test with threshold=0: all drifts are high."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.0,
            min_multiplier=1.0,
            max_multiplier=5.0,
        )
        drifts = [0.0, 0.01, 0.02, 0.03]
        result = scaler.compute_multipliers(drifts)

        # All >= 0.0, so max_drift = 0.03
        # drifts[0]=0.0: ratio=0.0/0.03=0.0 → 1.0 + 0.0*4.0 = 1.0
        # drifts[1]=0.01: ratio=0.01/0.03=0.333 → 1.0 + 0.333*4.0 ≈ 2.333
        # drifts[2]=0.02: ratio=0.02/0.03=0.667 → 1.0 + 0.667*4.0 ≈ 3.667
        # drifts[3]=0.03: ratio=0.03/0.03=1.0 → 1.0 + 1.0*4.0 = 5.0

        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.0 + (0.01 / 0.03) * 4.0, rel=1e-4)
        assert result[2] == pytest.approx(1.0 + (0.02 / 0.03) * 4.0, rel=1e-4)
        assert result[3] == pytest.approx(5.0)

    def test_very_high_threshold_all_below(self):
        """Test with very high threshold: all drifts below."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=1000.0,
            min_multiplier=0.5,
            max_multiplier=5.0,
        )
        drifts = [0.001, 0.01, 0.1, 1.0]
        result = scaler.compute_multipliers(drifts)

        # All < 1000.0 → all below threshold
        assert all(m == pytest.approx(0.5) for m in result)

    def test_single_high_multiplies_appropriately(self):
        """Test that single high-drift layer gets maximum."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.1,
            max_multiplier=10.0,
        )
        drifts = [0.001] * 15 + [0.1]
        result = scaler.compute_multipliers(drifts)

        # Only last layer is high (0.1), max_drift=0.1
        # All others: 0.001 < 0.01 → 0.1
        # Last: ratio=0.1/0.1=1.0 → 0.1 + 1.0*9.9 = 10.0

        assert all(m == pytest.approx(0.1) for m in result[:-1])
        assert result[-1] == pytest.approx(10.0)

    def test_threshold_boundary_precision(self):
        """Test threshold boundary with floating-point precision."""
        threshold = 0.00123456
        scaler = DynamicGradientScaler(
            angle_drift_threshold=threshold,
            min_multiplier=1.0,
            max_multiplier=5.0,
        )

        # Just below
        result_below = scaler.compute_multipliers([threshold - 1e-10])
        assert result_below[0] == pytest.approx(1.0)

        # Just above
        result_above = scaler.compute_multipliers([threshold + 1e-10])
        # Only this layer is high, so max_drift ≈ threshold
        # ratio ≈ 1.0
        assert result_above[0] == pytest.approx(5.0)

    def test_multiple_below_one_above(self):
        """Test 15 below + 1 above threshold."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.5,
            max_multiplier=3.0,
        )
        drifts = [0.001] * 15 + [0.05]
        result = scaler.compute_multipliers(drifts)

        # 15 below → 0.5
        # 1 above (0.05): ratio=1.0 → 0.5 + 1.0*2.5 = 3.0
        assert all(m == pytest.approx(0.5) for m in result[:-1])
        assert result[-1] == pytest.approx(3.0)
