"""Tests for EMABuffer initialization and defaults."""

from __future__ import annotations

import pytest

from mlx_grpo.training.ema_state import EMABuffer, EMAState


class TestEMABufferInitialization:
    """Test EMABuffer creation and default state."""

    def test_default_initialization(self):
        """Test creating EMABuffer with defaults."""
        buffer = EMABuffer()
        assert buffer.ema_alpha == 0.25
        assert buffer.step_count == 0
        assert len(buffer.current_multipliers) == 16
        assert all(m == 1.0 for m in buffer.current_multipliers)

    def test_custom_alpha(self):
        """Test creating EMABuffer with custom alpha."""
        buffer = EMABuffer(ema_alpha=0.5)
        assert buffer.ema_alpha == 0.5

    def test_alpha_high_value(self):
        """Test with high alpha (more weight on recent values)."""
        buffer = EMABuffer(ema_alpha=0.9)
        assert buffer.ema_alpha == 0.9

    def test_alpha_low_value(self):
        """Test with low alpha (more weight on history)."""
        buffer = EMABuffer(ema_alpha=0.01)
        assert buffer.ema_alpha == 0.01

    def test_alpha_exactly_one(self):
        """Test with alpha=1.0 (all weight on raw)."""
        buffer = EMABuffer(ema_alpha=1.0)
        assert buffer.ema_alpha == 1.0

    def test_alpha_too_low_rejected(self):
        """Test that alpha <= 0 is rejected."""
        with pytest.raises(ValueError, match="ema_alpha must be in"):
            EMABuffer(ema_alpha=0.0)

    def test_alpha_negative_rejected(self):
        """Test that negative alpha is rejected."""
        with pytest.raises(ValueError, match="ema_alpha must be in"):
            EMABuffer(ema_alpha=-0.5)

    def test_alpha_too_high_rejected(self):
        """Test that alpha > 1.0 is rejected."""
        with pytest.raises(ValueError, match="ema_alpha must be in"):
            EMABuffer(ema_alpha=1.01)

    def test_initial_multipliers_all_ones(self):
        """Test that initial multipliers are all 1.0 (no scaling)."""
        buffer = EMABuffer()
        mults = buffer.current_multipliers
        assert len(mults) == 16
        assert all(m == 1.0 for m in mults)

    def test_initial_step_count_zero(self):
        """Test that step_count starts at 0."""
        buffer = EMABuffer()
        assert buffer.step_count == 0

    def test_multiple_instances_independent(self):
        """Test that multiple EMABuffer instances are independent."""
        buffer1 = EMABuffer(ema_alpha=0.25)
        buffer2 = EMABuffer(ema_alpha=0.5)

        assert buffer1.ema_alpha == 0.25
        assert buffer2.ema_alpha == 0.5
        assert buffer1.step_count == buffer2.step_count == 0

    def test_ema_state_dataclass(self):
        """Test EMAState dataclass creation."""
        state = EMAState()
        assert state.ema_alpha == 0.25
        assert state.step_count == 0
        assert len(state.ema_multipliers) == 16
        assert all(m == 1.0 for m in state.ema_multipliers)

    def test_ema_state_custom_values(self):
        """Test EMAState with custom values."""
        custom_mults = [1.5] * 16
        state = EMAState(
            ema_multipliers=custom_mults,
            ema_alpha=0.5,
            step_count=10,
        )
        assert state.ema_multipliers == custom_mults
        assert state.ema_alpha == 0.5
        assert state.step_count == 10

    def test_ema_state_frozen(self):
        """Test that EMAState is frozen (immutable)."""
        state = EMAState()
        with pytest.raises(AttributeError):
            state.ema_alpha = 0.5

    def test_alpha_boundary_value_just_above_zero(self):
        """Test alpha with value just above 0."""
        buffer = EMABuffer(ema_alpha=1e-10)
        assert buffer.ema_alpha == pytest.approx(1e-10)

    def test_current_multipliers_returns_copy(self):
        """Test that current_multipliers returns a copy, not reference."""
        buffer = EMABuffer()
        mults1 = buffer.current_multipliers
        mults2 = buffer.current_multipliers

        # Should be equal but independent
        assert mults1 == mults2
        # Modifying one shouldn't affect the other or the buffer
        mults1[0] = 999.0
        assert buffer.current_multipliers[0] == 1.0
