"""Tests for EMABuffer convergence behavior."""

from __future__ import annotations

import pytest

from mlx_grpo.training.ema_state import EMABuffer


class TestEMAConvergence:
    """Test that EMA converges to target values over multiple updates."""

    def test_convergence_to_constant_target(self):
        """Test convergence when raw is constant."""
        buffer = EMABuffer(ema_alpha=0.1)

        # Start from far away
        buffer.update([0.0] * 16)

        # Keep feeding constant target
        target = [100.0] * 16
        results = []

        for _ in range(50):
            result = buffer.update(target)
            results.append(result[0])

        # Should be monotonically increasing and approaching 100
        for i in range(1, len(results)):
            assert results[i] > results[i - 1]

        # Should be close to target at the end
        assert results[-1] > 95.0

    def test_convergence_rate_high_alpha(self):
        """Test that high alpha converges faster."""
        buffer_fast = EMABuffer(ema_alpha=0.8)
        buffer_slow = EMABuffer(ema_alpha=0.1)

        target = [100.0] * 16

        # Initialize both
        buffer_fast.update([0.0] * 16)
        buffer_slow.update([0.0] * 16)

        # Run 5 steps (not too many)
        for _ in range(5):
            fast_result = buffer_fast.update(target)
            slow_result = buffer_slow.update(target)

        # Fast should be closer to target
        assert fast_result[0] > slow_result[0]

    def test_convergence_oscillating_around_mean(self):
        """Test behavior with oscillating inputs around a mean."""
        buffer = EMABuffer(ema_alpha=0.2)

        values = [5.0, 15.0, 5.0, 15.0] * 4  # Oscillate 5-15
        buffer.update([10.0] * 16)  # Start near mean

        results = []
        for val in values:
            raw = [val] * 16
            result = buffer.update(raw)
            results.append(result[0])

        # Should oscillate but damped, staying closer to 10
        for r in results:
            assert 7.0 < r < 13.0

    def test_convergence_step_count_increments(self):
        """Test that step_count increments with each update."""
        buffer = EMABuffer()

        for i in range(20):
            buffer.update([float(i)] * 16)
            assert buffer.step_count == i + 1

    def test_convergence_multiple_step_sizes(self):
        """Test convergence with various alphas."""
        target = [10.0] * 16
        start = [0.0] * 16

        alphas = [0.05, 0.1, 0.25, 0.5]  # Exclude 0.75+ as they're too close
        results_by_alpha = {}

        for alpha in alphas:
            buffer = EMABuffer(ema_alpha=alpha)
            buffer.update(start)

            for _ in range(50):  # Fewer steps to show ordering
                result = buffer.update(target)

            results_by_alpha[alpha] = result[0]

        # Higher alpha should be closer to target after same number of steps
        for i in range(len(alphas) - 1):
            assert results_by_alpha[alphas[i]] < results_by_alpha[alphas[i + 1]]

    def test_convergence_from_below_target(self):
        """Test convergence starting below target."""
        buffer = EMABuffer(ema_alpha=0.25)

        buffer.update([1.0] * 16)
        target = [5.0] * 16

        prev = 1.0
        for _ in range(20):
            result = buffer.update(target)
            current = result[0]

            # Should be increasing toward target
            assert current > prev
            assert current <= 5.0

            prev = current

    def test_convergence_from_above_target(self):
        """Test convergence starting above target."""
        buffer = EMABuffer(ema_alpha=0.25)

        buffer.update([10.0] * 16)
        target = [1.0] * 16

        prev = 10.0
        for _ in range(20):
            result = buffer.update(target)
            current = result[0]

            # Should be decreasing toward target
            assert current < prev
            assert current >= 1.0

            prev = current

    def test_convergence_per_layer_independent(self):
        """Test that each layer converges independently."""
        buffer = EMABuffer(ema_alpha=0.2)

        # Initialize with different values per layer
        start = list(range(1, 17))  # [1, 2, 3, ..., 16]
        buffer.update(start)

        # Converge to different targets per layer
        target = [100 - i * 5 for i in range(16)]  # [100, 95, 90, ..., 20]

        # Run convergence
        for _ in range(50):
            result = buffer.update(target)

        # Each layer should have converged closer to its target
        # Note: We're checking general trend, not exact values
        assert result[0] > 50  # L20 should move toward 100
        assert result[-1] < 50  # L35 should move toward 20

    def test_convergence_exact_value_then_hold(self):
        """Test holding exact value when input becomes constant."""
        buffer = EMABuffer(ema_alpha=0.25)

        # Converge to a value
        buffer.update([2.0] * 16)

        for _ in range(5):
            result = buffer.update([3.0] * 16)

        # Now hold at that converged value
        converged = result[0]

        for _ in range(10):
            result = buffer.update([converged] * 16)

        # Should stay constant
        for _ in range(5):
            result_new = buffer.update([converged] * 16)
            assert result_new[0] == pytest.approx(converged)

    def test_convergence_with_extreme_alpha_zero_like(self):
        """Test convergence with very small alpha (nearly ignoring raw)."""
        buffer = EMABuffer(ema_alpha=0.01)

        buffer.update([10.0] * 16)
        target = [100.0] * 16

        # With alpha=0.01, convergence is very slow
        # After 10 steps, should barely move
        for _ in range(10):
            buffer.update(target)

        # Even after many steps, it should be much closer to starting point
        # than the target (at this small alpha)
        result = buffer.update(target)
        assert result[0] < 50.0  # Still much less than 100

    def test_convergence_bounded_by_min_max(self):
        """Test that convergence stays within reasonable bounds."""
        buffer = EMABuffer(ema_alpha=0.5)

        # Start low
        buffer.update([0.1] * 16)

        # Target high
        target = [5.0] * 16

        for _ in range(100):
            result = buffer.update(target)

        # Should be between bounds (shouldn't overshoot)
        assert all(0.1 <= m <= 5.0 for m in result)

    def test_convergence_negative_to_positive(self):
        """Test convergence from negative to positive (edge case)."""
        buffer = EMABuffer(ema_alpha=0.25)

        buffer.update([-5.0] * 16)
        target = [5.0] * 16

        prev = -5.0
        for _ in range(20):
            result = buffer.update(target)
            current = result[0]

            # Should be monotonically increasing
            assert current > prev
            prev = current

        # Should end positive
        assert result[0] > 0

    def test_convergence_stops_after_reaching_target(self):
        """Test that once at target, stays at target."""
        buffer = EMABuffer(ema_alpha=0.25)

        # Start at target directly
        target = [3.0] * 16
        buffer.update(target)

        # Keep feeding the same target
        for _ in range(20):
            result = buffer.update(target)

        # Should stay exactly at target
        assert all(m == pytest.approx(3.0) for m in result)
