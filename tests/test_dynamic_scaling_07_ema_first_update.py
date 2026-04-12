"""Tests for EMABuffer.update first call (initialization)."""

from __future__ import annotations

import pytest

from mlx_grpo.training.ema_state import EMABuffer


class TestEMAUpdateFirstCall:
    """Test first call to update() — should return raw without EMA."""

    def test_first_call_returns_raw(self):
        """Test that first update returns raw multipliers unchanged."""
        buffer = EMABuffer(ema_alpha=0.25)
        raw = [2.0, 0.5, 1.5, 1.0] * 4  # 16 values
        result = buffer.update(raw)

        assert result == raw

    def test_first_call_no_smoothing(self):
        """Test that first call applies no smoothing."""
        buffer = EMABuffer(ema_alpha=0.01)  # Low alpha (would heavily weight history)
        raw = [5.0, 3.0, 2.0, 4.0] * 4
        result = buffer.update(raw)

        # First call ignores alpha, returns raw
        assert result == raw

    def test_first_call_step_count_increments(self):
        """Test that step_count increments on first call."""
        buffer = EMABuffer()
        assert buffer.step_count == 0

        raw = [1.0] * 16
        buffer.update(raw)

        assert buffer.step_count == 1

    def test_first_call_updates_internal_state(self):
        """Test that first call updates internal multipliers."""
        buffer = EMABuffer()
        raw = [2.0, 0.5, 1.5, 1.0] * 4
        buffer.update(raw)

        # Internal state should match raw
        assert buffer.current_multipliers == raw

    def test_first_call_with_varied_values(self):
        """Test first call with various multiplier values."""
        buffer = EMABuffer()
        raw = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 0.2, 1.5, 2.5, 3.5, 4.5, 0.3, 1.2, 2.2, 3.2]
        result = buffer.update(raw)

        assert result == raw

    def test_first_call_with_all_ones(self):
        """Test first call when raw is all 1.0 (no scaling)."""
        buffer = EMABuffer()
        raw = [1.0] * 16
        result = buffer.update(raw)

        assert all(m == 1.0 for m in result)

    def test_first_call_with_min_multiplier(self):
        """Test first call with min_multiplier values."""
        buffer = EMABuffer()
        raw = [0.1] * 16  # All at minimum
        result = buffer.update(raw)

        assert all(m == 0.1 for m in result)

    def test_first_call_with_max_multiplier(self):
        """Test first call with max_multiplier values."""
        buffer = EMABuffer()
        raw = [5.0] * 16  # All at maximum
        result = buffer.update(raw)

        assert all(m == 5.0 for m in result)

    def test_first_call_with_mixed_extremes(self):
        """Test first call with mixed min and max values."""
        buffer = EMABuffer()
        raw = [0.1] * 8 + [5.0] * 8  # Half min, half max
        result = buffer.update(raw)

        assert result == raw

    def test_first_call_invalid_length(self):
        """Test that first call rejects invalid input length."""
        buffer = EMABuffer()
        with pytest.raises(ValueError, match="Expected 16 multipliers"):
            buffer.update([1.0] * 15)

    def test_first_call_too_many_values(self):
        """Test that first call rejects too many values."""
        buffer = EMABuffer()
        with pytest.raises(ValueError, match="Expected 16 multipliers"):
            buffer.update([1.0] * 17)

    def test_first_call_empty_list(self):
        """Test that first call rejects empty list."""
        buffer = EMABuffer()
        with pytest.raises(ValueError, match="Expected 16 multipliers"):
            buffer.update([])

    def test_first_call_different_alphas_same_result(self):
        """Test that first call result is independent of alpha."""
        raw = [2.0, 0.5, 1.5, 1.0] * 4

        for alpha in [0.01, 0.25, 0.5, 0.9, 1.0]:
            buffer = EMABuffer(ema_alpha=alpha)
            result = buffer.update(raw)
            assert result == raw

    def test_first_call_returns_list(self):
        """Test that update returns a list, not other type."""
        buffer = EMABuffer()
        raw = [1.0] * 16
        result = buffer.update(raw)

        assert isinstance(result, list)
        assert len(result) == 16

    def test_first_call_does_not_modify_input(self):
        """Test that update doesn't modify the input list."""
        buffer = EMABuffer()
        raw = [2.0, 0.5, 1.5, 1.0] * 4
        raw_copy = list(raw)

        buffer.update(raw)

        assert raw == raw_copy

    def test_sequential_first_calls_build_state(self):
        """Test that first call sets up state for subsequent calls."""
        buffer = EMABuffer(ema_alpha=0.25)

        # First call
        raw1 = [2.0] * 16
        result1 = buffer.update(raw1)
        assert result1 == raw1

        # Verify state is ready for second call
        assert buffer.step_count == 1
        assert buffer.current_multipliers == raw1

        # Second call should apply EMA
        raw2 = [3.0] * 16
        result2 = buffer.update(raw2)

        # Should apply EMA formula
        expected = [0.25 * 3.0 + 0.75 * 2.0 for _ in range(16)]
        assert result2 == expected
