"""Test checkpoint save/restore with EMA state persistence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mlx_grpo.loss.gradient_scaler import DynamicGradientScaler
from mlx_grpo.training.ema_state import EMABuffer


class TestCheckpointRestoreResume:
    """Test saving EMA state to checkpoint and resuming training."""

    def test_save_ema_to_checkpoint_dict(self):
        """Test saving EMA state to checkpoint dict."""
        ema_buffer = EMABuffer(ema_alpha=0.25)

        # Simulate training
        ema_buffer.update([1.0] * 16)
        ema_buffer.update([2.0] * 16)

        # Save to checkpoint dict
        checkpoint = {"ema_state": ema_buffer.to_dict()}

        assert "ema_state" in checkpoint
        assert checkpoint["ema_state"]["step_count"] == 2

    def test_restore_ema_from_checkpoint_dict(self):
        """Test restoring EMA state from checkpoint dict."""
        # Original buffer
        ema1 = EMABuffer(ema_alpha=0.25)
        ema1.update([1.0] * 16)
        ema1.update([2.0] * 16)

        checkpoint = {"ema_state": ema1.to_dict()}

        # Restore from checkpoint
        ema2 = EMABuffer.from_dict(checkpoint["ema_state"])

        assert ema2.step_count == ema1.step_count
        assert ema2.current_multipliers == ema1.current_multipliers

    def test_checkpoint_at_step_50_resume_at_51(self):
        """Test checkpoint at step 50, resume at step 51."""
        ema = EMABuffer(ema_alpha=0.2)

        # Simulate 50 steps of training
        for step in range(50):
            raw_mults = [float(step % 5 + 1)] * 16
            ema.update(raw_mults)

        # At step 50, save checkpoint
        state_at_50 = ema.to_dict()

        # Restore and continue at step 51
        ema_resumed = EMABuffer.from_dict(state_at_50)
        assert ema_resumed.step_count == 50

        # Step 51
        raw_mults_51 = [5.0] * 16
        ema_resumed.update(raw_mults_51)

        # Verify step_count incremented
        assert ema_resumed.step_count == 51

    def test_checkpoint_json_serializable(self):
        """Test that checkpoint can be JSON serialized."""
        ema = EMABuffer(ema_alpha=0.25)
        ema.update([1.5] * 16)
        ema.update([2.5] * 16)

        checkpoint = {"ema_state": ema.to_dict()}

        # Should be JSON serializable
        json_str = json.dumps(checkpoint)
        restored_checkpoint = json.loads(json_str)

        # Should be able to restore from JSON
        ema_restored = EMABuffer.from_dict(restored_checkpoint["ema_state"])

        assert ema_restored.step_count == ema.step_count
        assert ema_restored.current_multipliers == ema.current_multipliers

    def test_checkpoint_file_save_load(self):
        """Test saving checkpoint to file and loading it."""
        ema = EMABuffer(ema_alpha=0.3)

        # Training
        ema.update([1.0] * 16)
        ema.update([2.0] * 16)

        checkpoint = {"ema_state": ema.to_dict(), "iteration": 2}

        # Save to file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(checkpoint, f)
            checkpoint_file = f.name

        try:
            # Load from file
            with open(checkpoint_file) as f:
                loaded_checkpoint = json.load(f)

            # Restore EMA
            ema_loaded = EMABuffer.from_dict(loaded_checkpoint["ema_state"])

            assert ema_loaded.step_count == ema.step_count
            assert ema_loaded.ema_alpha == ema.ema_alpha
            assert ema_loaded.current_multipliers == ema.current_multipliers
        finally:
            Path(checkpoint_file).unlink()

    def test_scaler_config_in_checkpoint(self):
        """Test saving scaler config alongside EMA state."""
        scaler = DynamicGradientScaler(
            angle_drift_threshold=0.01,
            max_multiplier=5.0,
            min_multiplier=0.1,
        )
        ema = EMABuffer(ema_alpha=0.25)

        ema.update([1.0] * 16)

        # Save checkpoint with both
        checkpoint = {
            "scaler_config": {
                "angle_drift_threshold": scaler.angle_drift_threshold,
                "max_multiplier": scaler.max_multiplier,
                "min_multiplier": scaler.min_multiplier,
            },
            "ema_state": ema.to_dict(),
            "iteration": 1,
        }

        # Verify all parts are present
        assert "scaler_config" in checkpoint
        assert "ema_state" in checkpoint
        assert checkpoint["scaler_config"]["angle_drift_threshold"] == 0.01

    def test_multiple_checkpoint_resume_cycles(self):
        """Test multiple save-resume cycles."""
        ema = EMABuffer(ema_alpha=0.15)

        # First training phase (steps 0-20)
        for step in range(20):
            ema.update([1.0 + step * 0.01] * 16)

        checkpoint1 = {"ema_state": ema.to_dict()}

        # Second training phase (steps 20-40)
        for step in range(20):
            ema.update([2.0 + step * 0.01] * 16)

        # Resume from checkpoint1
        ema_from_ckpt1 = EMABuffer.from_dict(checkpoint1["ema_state"])
        assert ema_from_ckpt1.step_count == 20

        # Continue
        for step in range(20):
            ema_from_ckpt1.update([2.0 + step * 0.01] * 16)

        # Should match checkpoint2
        assert ema_from_ckpt1.step_count == ema.step_count
        assert ema_from_ckpt1.current_multipliers == ema.current_multipliers

    def test_checkpoint_preserves_ema_convergence(self):
        """Test that resuming from checkpoint preserves convergence."""
        # Original buffer that trains for 50 steps
        ema1 = EMABuffer(ema_alpha=0.05)

        target = [10.0] * 16

        for _ in range(50):
            ema1.update(target)

        # Checkpoint at step 50
        checkpoint = ema1.to_dict()

        # Resumed buffer from checkpoint
        ema2 = EMABuffer.from_dict(checkpoint)

        # Both continue for 10 more steps
        for _ in range(10):
            ema1.update(target)
            ema2.update(target)

        # Should have converged to same value
        for v1, v2 in zip(ema1.current_multipliers, ema2.current_multipliers, strict=False):
            assert v1 == pytest.approx(v2)

    def test_checkpoint_different_alpha(self):
        """Test that checkpoint preserves correct alpha."""
        for alpha in [0.01, 0.25, 0.5, 0.99]:
            ema = EMABuffer(ema_alpha=alpha)
            ema.update([2.0] * 16)

            checkpoint = ema.to_dict()

            ema_restored = EMABuffer.from_dict(checkpoint)
            assert ema_restored.ema_alpha == alpha

    def test_checkpoint_with_varied_step_counts(self):
        """Test checkpointing at various step counts."""
        for num_steps in [1, 5, 10, 50, 100]:
            ema = EMABuffer()

            for _ in range(num_steps):
                ema.update([1.0] * 16)

            checkpoint = ema.to_dict()
            assert checkpoint["step_count"] == num_steps

            ema_restored = EMABuffer.from_dict(checkpoint)
            assert ema_restored.step_count == num_steps

    def test_checkpoint_immutability(self):
        """Test that modifying checkpoint doesn't affect buffer."""
        ema = EMABuffer()
        ema.update([1.0] * 16)

        checkpoint = ema.to_dict()
        original_mult = checkpoint["ema_multipliers"][0]

        # Modify checkpoint
        checkpoint["ema_multipliers"][0] = 999.0

        # Original buffer unchanged
        assert ema.current_multipliers[0] == original_mult

    def test_checkpoint_restore_independence(self):
        """Test that multiple restores are independent."""
        ema = EMABuffer(ema_alpha=0.25)
        ema.update([1.0] * 16)

        checkpoint = ema.to_dict()

        # Restore twice
        ema1 = EMABuffer.from_dict(checkpoint)
        ema2 = EMABuffer.from_dict(checkpoint)

        # Update only ema1
        ema1.update([5.0] * 16)

        # ema2 should be unchanged
        assert ema2.current_multipliers == [1.0] * 16
        assert ema1.current_multipliers != ema2.current_multipliers
