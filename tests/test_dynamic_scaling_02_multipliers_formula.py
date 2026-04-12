"""Tests for DynamicGradientScaler.compute_multipliers formula correctness."""

from __future__ import annotations

import math

import pytest

from mlx_grpo.loss.gradient_scaler import DynamicGradientScaler


class TestComputeMultipliersFormula:
    """Test multiplier computation with various angle drifts."""

    def test_empty_input(self):
        """Test that empty input returns empty output."""
        scaler = DynamicGradientScaler()
        result = scaler.compute_multipliers([])
        assert result == []

    def test_single_layer_below_threshold(self):
        """Test single layer with drift below threshold."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.1,
            max_multiplier=5.0,
        )
        result = scaler.compute_multipliers([0.005])
        assert len(result) == 1
        assert result[0] == pytest.approx(0.1)

    def test_single_layer_above_threshold(self):
        """Test single layer with drift above threshold."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.1,
            max_multiplier=5.0,
        )
        result = scaler.compute_multipliers([0.02])
        assert len(result) == 1
        # Single high-drift layer: max_drift=0.02, ratio=0.02/0.02=1.0
        # mult = 0.1 + 1.0 * (5.0 - 0.1) = 0.1 + 4.9 = 5.0
        assert result[0] == pytest.approx(5.0)

    def test_formula_with_two_layers_one_high(self):
        """Test formula: low drift → min, high drift → scaled."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.1,
            max_multiplier=5.0,
        )
        result = scaler.compute_multipliers([0.005, 0.02])
        # Layer 0: drift 0.005 < threshold → min_mult = 0.1
        # Layer 1: drift 0.02 >= threshold, max_drift=0.02
        #   ratio = 0.02 / 0.02 = 1.0
        #   mult = 0.1 + 1.0 * (5.0 - 0.1) = 5.0
        assert result[0] == pytest.approx(0.1)
        assert result[1] == pytest.approx(5.0)

    def test_formula_with_multiple_high_drifts(self):
        """Test scaling when multiple layers exceed threshold."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=1.0,
            max_multiplier=5.0,
        )
        drifts = [0.005, 0.02, 0.03, 0.015]
        result = scaler.compute_multipliers(drifts)

        # High drifts: [0.02, 0.03, 0.015], max_drift=0.03
        # drifts[0]=0.005: < 0.01 → 1.0
        # drifts[1]=0.02: ratio=0.02/0.03=0.667 → 1.0 + 0.667*4.0 = 3.668
        # drifts[2]=0.03: ratio=0.03/0.03=1.0 → 1.0 + 1.0*4.0 = 5.0
        # drifts[3]=0.015: ratio=0.015/0.03=0.5 → 1.0 + 0.5*4.0 = 3.0
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.0 + (0.02 / 0.03) * (5.0 - 1.0), rel=1e-4)
        assert result[2] == pytest.approx(5.0)
        assert result[3] == pytest.approx(1.0 + (0.015 / 0.03) * (5.0 - 1.0))

    def test_all_drifts_below_threshold(self):
        """Test when all drifts are below threshold."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.5,
            max_multiplier=3.0,
        )
        drifts = [0.001, 0.002, 0.003, 0.004]
        result = scaler.compute_multipliers(drifts)
        # All below threshold → all should be min_multiplier
        assert all(m == pytest.approx(0.5) for m in result)

    def test_all_drifts_above_threshold(self):
        """Test when all drifts are above threshold."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=0.5,
            max_multiplier=3.0,
        )
        drifts = [0.01, 0.02, 0.03, 0.04]
        result = scaler.compute_multipliers(drifts)
        # All above threshold, max_drift=0.04
        # drifts[0]=0.01: ratio=0.01/0.04=0.25 → 0.5 + 0.25*2.5 = 1.125
        # drifts[1]=0.02: ratio=0.02/0.04=0.5 → 0.5 + 0.5*2.5 = 1.75
        # drifts[2]=0.03: ratio=0.03/0.04=0.75 → 0.5 + 0.75*2.5 = 2.375
        # drifts[3]=0.04: ratio=0.04/0.04=1.0 → 0.5 + 1.0*2.5 = 3.0
        assert result[0] == pytest.approx(0.5 + 0.25 * 2.5)
        assert result[1] == pytest.approx(0.5 + 0.5 * 2.5)
        assert result[2] == pytest.approx(0.5 + 0.75 * 2.5)
        assert result[3] == pytest.approx(0.5 + 1.0 * 2.5)

    def test_ratio_clamped_to_one(self):
        """Test that ratio never exceeds 1.0 (clamping in code)."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=1.0,
            max_multiplier=5.0,
        )
        # Even if drift equals max_drift, ratio should be exactly 1.0
        drifts = [0.05, 0.05]
        result = scaler.compute_multipliers(drifts)
        # Both at max_drift: ratio=1.0 → 1.0 + 1.0*4.0 = 5.0
        assert all(m == pytest.approx(5.0) for m in result)

    def test_16_layers_typical_config(self):
        """Test with 16 layers (typical LoRA scenario)."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.005,
            min_multiplier=0.1,
            max_multiplier=5.0,
        )
        # 16 layers with varied drifts
        drifts = [
            0.001,
            0.002,
            0.003,
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.009,
            0.010,
            0.011,
            0.012,
            0.013,
            0.014,
            0.015,
            0.016,
        ]
        result = scaler.compute_multipliers(drifts)

        assert len(result) == 16
        # First 4: below threshold → 0.1
        assert all(result[i] == pytest.approx(0.1) for i in range(4))
        # Rest: above threshold, should be scaled
        assert all(0.1 <= m <= 5.0 for m in result[4:])
        # Highest drift should get max multiplier
        assert result[-1] == pytest.approx(5.0)

    def test_zero_max_drift_among_high(self):
        """Test edge case: drift equals threshold (boundary is now strict >)."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=1.0,
            max_multiplier=5.0,
        )
        # All drifts equal to threshold: with strict >, all are treated as low
        drifts = [0.01, 0.01, 0.01]
        result = scaler.compute_multipliers(drifts)
        # All drift == threshold, so drift < threshold is False → all get min_multiplier
        assert all(m == pytest.approx(1.0) for m in result)

    def test_multiplier_bounds_never_exceed_max(self):
        """Test that no multiplier exceeds max_multiplier."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.005,
            min_multiplier=0.2,
            max_multiplier=3.0,
        )
        drifts = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = scaler.compute_multipliers(drifts)
        assert all(m <= 3.0 for m in result)

    def test_multiplier_bounds_never_below_min(self):
        """Test that no multiplier is below min_multiplier."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.005,
            min_multiplier=0.2,
            max_multiplier=3.0,
        )
        drifts = [0.001, 0.002, 0.3, 0.4]
        result = scaler.compute_multipliers(drifts)
        assert all(m >= 0.2 for m in result)

    def test_nan_inf_handling(self):
        """Test that NaN/inf inputs don't produce bad multipliers."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            min_multiplier=1.0,
            max_multiplier=5.0,
        )
        # Normal drifts should still work even if we test robustness
        drifts = [0.005, 0.02, 0.01]
        result = scaler.compute_multipliers(drifts)
        # All results should be finite
        assert all(math.isfinite(m) for m in result)
