"""Tests for EMABuffer.update EMA formula on subsequent calls."""

from __future__ import annotations

import pytest

from mlx_grpo.training.ema_state import EMABuffer


class TestEMAUpdateFormula:
    """Test EMA formula: α·raw + (1-α)·history."""

    def test_formula_basic(self):
        """Test basic EMA formula application."""
        buffer = EMABuffer(ema_alpha=0.25)

        # First call: raw becomes history
        raw1 = [2.0] * 16
        buffer.update(raw1)

        # Second call: apply EMA
        raw2 = [3.0] * 16
        result = buffer.update(raw2)

        # Expected: 0.25 * 3.0 + 0.75 * 2.0 = 0.75 + 1.5 = 2.25
        expected = [2.25] * 16
        assert result == expected

    def test_formula_mixed_values(self):
        """Test EMA formula with varied values."""
        buffer = EMABuffer(ema_alpha=0.25)

        raw1 = [1.0, 2.0, 3.0, 4.0] * 4
        buffer.update(raw1)

        raw2 = [2.0, 3.0, 4.0, 5.0] * 4
        result = buffer.update(raw2)

        # Per-layer calculation
        expected = []
        for r1, r2 in zip(raw1, raw2, strict=False):
            ema = 0.25 * r2 + 0.75 * r1
            expected.append(ema)

        assert result == expected

    def test_formula_alpha_high(self):
        """Test EMA with high alpha (more weight on raw)."""
        buffer = EMABuffer(ema_alpha=0.9)

        raw1 = [1.0] * 16
        buffer.update(raw1)

        raw2 = [10.0] * 16
        result = buffer.update(raw2)

        # Expected: 0.9 * 10.0 + 0.1 * 1.0 = 9.0 + 0.1 = 9.1
        expected = [9.1] * 16
        assert result == expected

    def test_formula_alpha_low(self):
        """Test EMA with low alpha (more weight on history)."""
        buffer = EMABuffer(ema_alpha=0.1)

        raw1 = [10.0] * 16
        buffer.update(raw1)

        raw2 = [1.0] * 16
        result = buffer.update(raw2)

        # Expected: 0.1 * 1.0 + 0.9 * 10.0 = 0.1 + 9.0 = 9.1
        expected = [9.1] * 16
        assert result == expected

    def test_formula_alpha_one(self):
        """Test EMA with alpha=1.0 (no history weight)."""
        buffer = EMABuffer(ema_alpha=1.0)

        raw1 = [5.0] * 16
        buffer.update(raw1)

        raw2 = [10.0] * 16
        result = buffer.update(raw2)

        # Expected: 1.0 * 10.0 + 0.0 * 5.0 = 10.0
        expected = [10.0] * 16
        assert result == expected

    def test_formula_convergence_direction(self):
        """Test that EMA converges toward raw value."""
        buffer = EMABuffer(ema_alpha=0.5)

        # Start with 1.0
        raw1 = [1.0] * 16
        buffer.update(raw1)

        # Step up to 10.0
        raw2 = [10.0] * 16
        result2 = buffer.update(raw2)

        # First step: (0.5 * 10.0 + 0.5 * 1.0) = 5.5
        assert result2 == [5.5] * 16

        # Step 3: EMA continues toward 10.0
        raw3 = [10.0] * 16
        result3 = buffer.update(raw3)

        # Second step: (0.5 * 10.0 + 0.5 * 5.5) = 7.75
        assert result3 == [7.75] * 16

        # Step 4: Gets even closer
        raw4 = [10.0] * 16
        result4 = buffer.update(raw4)

        # Third step: (0.5 * 10.0 + 0.5 * 7.75) = 8.875
        assert result4 == [8.875] * 16

    def test_formula_per_layer_independent(self):
        """Test that EMA is applied independently per layer."""
        buffer = EMABuffer(ema_alpha=0.25)

        raw1 = [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
        ]
        buffer.update(raw1)

        raw2 = [
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
            18.0,
            20.0,
            22.0,
            24.0,
            26.0,
            28.0,
            30.0,
            32.0,
        ]
        result = buffer.update(raw2)

        # Verify each layer independently
        for i in range(16):
            expected_i = 0.25 * raw2[i] + 0.75 * raw1[i]
            assert result[i] == pytest.approx(expected_i)

    def test_formula_negative_drifts(self):
        """Test EMA with negative values (shouldn't happen but test robustness)."""
        buffer = EMABuffer(ema_alpha=0.5)

        raw1 = [-1.0] * 16
        buffer.update(raw1)

        raw2 = [-10.0] * 16
        result = buffer.update(raw2)

        # Expected: 0.5 * -10.0 + 0.5 * -1.0 = -5.5
        expected = [-5.5] * 16
        assert result == expected

    def test_formula_mixed_sign(self):
        """Test EMA with mixed positive/negative values."""
        buffer = EMABuffer(ema_alpha=0.5)

        raw1 = [1.0, -1.0, 2.0, -2.0] * 4
        buffer.update(raw1)

        raw2 = [5.0, -5.0, 10.0, -10.0] * 4
        result = buffer.update(raw2)

        # Verify per-layer
        for i in range(16):
            expected_i = 0.5 * raw2[i] + 0.5 * raw1[i]
            assert result[i] == pytest.approx(expected_i)

    def test_formula_multiple_steps_convergence(self):
        """Test multi-step convergence toward target."""
        buffer = EMABuffer(ema_alpha=0.1)

        target = [100.0] * 16
        current = buffer.update([0.0] * 16)  # Start at 0

        # Run 10 steps, converge toward 100
        for _ in range(10):
            current = buffer.update(target)

        # After 10 steps with alpha=0.1, should be close to target
        # Formula: 100 * (1 - 0.9^n) where 0.9^10 ≈ 0.349
        # So ≈ 100 * (1 - 0.349) = 65.1
        # But more precisely: each step adds 10% toward the gap
        assert all(m > 50 for m in current)  # Should be substantially toward 100

    def test_formula_alternating_values(self):
        """Test EMA with alternating raw values."""
        buffer = EMABuffer(ema_alpha=0.5)

        raw1 = [0.0] * 16
        result1 = buffer.update(raw1)
        assert result1 == raw1

        raw2 = [10.0] * 16
        result2 = buffer.update(raw2)
        # Expected: 0.5 * 10.0 + 0.5 * 0.0 = 5.0
        assert result2 == [5.0] * 16

        raw3 = [0.0] * 16
        result3 = buffer.update(raw3)
        # Expected: 0.5 * 0.0 + 0.5 * 5.0 = 2.5
        assert result3 == [2.5] * 16

        raw4 = [10.0] * 16
        result4 = buffer.update(raw4)
        # Expected: 0.5 * 10.0 + 0.5 * 2.5 = 6.25
        assert result4 == [6.25] * 16

    def test_formula_precision_small_values(self):
        """Test EMA formula precision with small values."""
        buffer = EMABuffer(ema_alpha=0.25)

        raw1 = [1e-6] * 16
        buffer.update(raw1)

        raw2 = [2e-6] * 16
        result = buffer.update(raw2)

        # Expected: 0.25 * 2e-6 + 0.75 * 1e-6 = 0.5e-6 + 0.75e-6 = 1.25e-6
        for r in result:
            assert r == pytest.approx(1.25e-6)

    def test_formula_updates_internal_state(self):
        """Test that formula result updates internal state for next call."""
        buffer = EMABuffer(ema_alpha=0.5)

        raw1 = [1.0] * 16
        buffer.update(raw1)

        raw2 = [3.0] * 16
        buffer.update(raw2)
        # Expected: [2.0] * 16 (0.5*3 + 0.5*1)

        # Now verify internal state is updated
        raw3 = [5.0] * 16
        result3 = buffer.update(raw3)
        # result3 should use result2 as history
        # Expected: 0.5 * 5.0 + 0.5 * 2.0 = 3.5
        assert result3 == [3.5] * 16
