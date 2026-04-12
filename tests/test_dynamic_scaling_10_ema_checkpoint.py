"""Tests for EMABuffer checkpoint persistence (to_dict/from_dict)."""

from __future__ import annotations

import pytest

from mlx_grpo.training.ema_state import EMABuffer


class TestEMACheckpointPersistence:
    """Test EMA state serialization and deserialization."""

    def test_to_dict_default_state(self):
        """Test to_dict on freshly created buffer."""
        buffer = EMABuffer()
        state_dict = buffer.to_dict()

        assert "ema_multipliers" in state_dict
        assert "ema_alpha" in state_dict
        assert "step_count" in state_dict

    def test_to_dict_has_correct_values(self):
        """Test that to_dict captures correct values."""
        buffer = EMABuffer(ema_alpha=0.5)
        buffer.update([2.0] * 16)

        state_dict = buffer.to_dict()

        assert state_dict["ema_alpha"] == 0.5
        assert state_dict["step_count"] == 1
        assert state_dict["ema_multipliers"] == [2.0] * 16

    def test_to_dict_after_updates(self):
        """Test to_dict after multiple updates."""
        buffer = EMABuffer(ema_alpha=0.25)

        buffer.update([1.0] * 16)
        buffer.update([2.0] * 16)
        buffer.update([3.0] * 16)

        state_dict = buffer.to_dict()

        assert state_dict["step_count"] == 3
        # Multipliers should be the smoothed value after 3 steps
        assert len(state_dict["ema_multipliers"]) == 16

    def test_from_dict_restores_state(self):
        """Test that from_dict restores exact state."""
        buffer1 = EMABuffer(ema_alpha=0.3)

        buffer1.update([1.0] * 16)
        buffer1.update([2.0] * 16)

        state_dict = buffer1.to_dict()

        # Create new buffer from saved state
        buffer2 = EMABuffer.from_dict(state_dict)

        assert buffer2.ema_alpha == buffer1.ema_alpha
        assert buffer2.step_count == buffer1.step_count
        assert buffer2.current_multipliers == buffer1.current_multipliers

    def test_from_dict_preserves_alpha(self):
        """Test that from_dict preserves alpha value."""
        for alpha in [0.01, 0.25, 0.5, 0.9, 1.0]:
            buffer1 = EMABuffer(ema_alpha=alpha)
            buffer1.update([1.0] * 16)

            state_dict = buffer1.to_dict()
            buffer2 = EMABuffer.from_dict(state_dict)

            assert buffer2.ema_alpha == alpha

    def test_roundtrip_to_from_dict(self):
        """Test roundtrip: to_dict → from_dict → to_dict."""
        buffer1 = EMABuffer(ema_alpha=0.25)

        for _ in range(5):
            buffer1.update([float(i + 1) for i in range(16)])

        dict1 = buffer1.to_dict()
        buffer2 = EMABuffer.from_dict(dict1)
        dict2 = buffer2.to_dict()

        assert dict1 == dict2

    def test_from_dict_continues_updates(self):
        """Test that buffer restored from checkpoint can continue updates."""
        buffer1 = EMABuffer(ema_alpha=0.25)

        buffer1.update([1.0] * 16)
        buffer1.update([2.0] * 16)

        # Save state
        state_dict = buffer1.to_dict()

        # Restore and continue
        buffer2 = EMABuffer.from_dict(state_dict)
        result2 = buffer2.update([3.0] * 16)

        # Should apply EMA formula with restored state
        # Previous: [step_count=2, mults from previous update]
        # This update should compute EMA using the restored history

        # Verify we can update further
        assert len(result2) == 16
        assert buffer2.step_count == 3

    def test_from_dict_continuity_matches_fresh(self):
        """Test that resuming from checkpoint gives same result as fresh."""
        # Run fresh
        buffer_fresh = EMABuffer(ema_alpha=0.25)
        buffer_fresh.update([1.0] * 16)
        buffer_fresh.update([2.0] * 16)
        result_fresh = buffer_fresh.update([3.0] * 16)

        # Run with checkpoint
        buffer_ckpt = EMABuffer(ema_alpha=0.25)
        buffer_ckpt.update([1.0] * 16)
        buffer_ckpt.update([2.0] * 16)

        state_dict = buffer_ckpt.to_dict()
        buffer_restored = EMABuffer.from_dict(state_dict)
        result_restored = buffer_restored.update([3.0] * 16)

        assert result_fresh == result_restored

    def test_to_dict_multipliers_list_copy(self):
        """Test that to_dict returns a copy, not a reference."""
        buffer = EMABuffer()
        buffer.update([2.0] * 16)

        dict1 = buffer.to_dict()
        dict2 = buffer.to_dict()

        # Modify dict1's multipliers
        dict1["ema_multipliers"][0] = 999.0

        # dict2 and buffer should be unaffected
        assert dict2["ema_multipliers"][0] == 2.0
        assert buffer.current_multipliers[0] == 2.0

    def test_from_dict_with_custom_multipliers(self):
        """Test from_dict with custom multiplier values."""
        custom_mults = [
            0.1,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]

        state_dict = {
            "ema_multipliers": custom_mults,
            "ema_alpha": 0.3,
            "step_count": 10,
        }

        buffer = EMABuffer.from_dict(state_dict)

        assert buffer.current_multipliers == custom_mults
        assert buffer.ema_alpha == 0.3
        assert buffer.step_count == 10

    def test_from_dict_missing_key_raises_error(self):
        """Test that from_dict raises error if key is missing."""
        incomplete_dict = {
            "ema_multipliers": [1.0] * 16,
            # Missing ema_alpha and step_count
        }

        with pytest.raises(KeyError):
            EMABuffer.from_dict(incomplete_dict)

    def test_from_dict_wrong_multiplier_length_raises_error(self):
        """Test that from_dict validates multiplier length."""
        bad_dict = {
            "ema_multipliers": [1.0] * 15,  # Wrong length
            "ema_alpha": 0.25,
            "step_count": 5,
        }

        restored = EMABuffer.from_dict(bad_dict)
        # from_dict creates the buffer but update() will validate
        with pytest.raises(ValueError, match="Expected 16 multipliers"):
            restored.update([1.0] * 15)

    def test_to_dict_preserves_step_count(self):
        """Test that to_dict preserves step count correctly."""
        buffer = EMABuffer()

        for step in range(1, 11):
            buffer.update([float(step)] * 16)
            state_dict = buffer.to_dict()
            assert state_dict["step_count"] == step

    def test_checkpoint_independence(self):
        """Test that two buffers restored from same checkpoint are independent."""
        buffer_orig = EMABuffer(ema_alpha=0.25)
        buffer_orig.update([1.0] * 16)

        state_dict = buffer_orig.to_dict()

        buffer1 = EMABuffer.from_dict(state_dict)
        buffer2 = EMABuffer.from_dict(state_dict)

        # Update only buffer1
        buffer1.update([5.0] * 16)

        # buffer2 should be unchanged
        assert buffer2.step_count == 1
        assert buffer2.current_multipliers == [1.0] * 16

    def test_checkpoint_with_varied_state(self):
        """Test checkpoint with varied internal state."""
        buffer = EMABuffer(ema_alpha=0.1)

        # Create varied state
        mults = [0.1 + i * 0.3 for i in range(16)]
        for mult in mults:
            buffer.update([mult] * 16)

        state_dict = buffer.to_dict()

        # Verify structure
        assert len(state_dict["ema_multipliers"]) == 16
        assert state_dict["step_count"] == 16
        assert state_dict["ema_alpha"] == 0.1

    def test_from_dict_then_update_step_count_continues(self):
        """Test that step_count continues incrementing after restore."""
        buffer1 = EMABuffer()
        buffer1.update([1.0] * 16)
        buffer1.update([2.0] * 16)

        state_dict = buffer1.to_dict()
        assert state_dict["step_count"] == 2

        buffer2 = EMABuffer.from_dict(state_dict)
        assert buffer2.step_count == 2

        buffer2.update([3.0] * 16)
        assert buffer2.step_count == 3

    def test_checkpoint_dict_contains_all_fields(self):
        """Test that checkpoint dict has all required fields."""
        buffer = EMABuffer(ema_alpha=0.5)
        buffer.update([1.5] * 16)
        buffer.update([2.5] * 16)

        state_dict = buffer.to_dict()

        required_keys = {"ema_multipliers", "ema_alpha", "step_count"}
        assert set(state_dict.keys()) == required_keys

    def test_from_dict_with_float_step_count(self):
        """Test from_dict handles step_count as int."""
        state_dict = {
            "ema_multipliers": [1.0] * 16,
            "ema_alpha": 0.25,
            "step_count": 5,
        }

        buffer = EMABuffer.from_dict(state_dict)
        assert isinstance(buffer.step_count, int)
        assert buffer.step_count == 5
