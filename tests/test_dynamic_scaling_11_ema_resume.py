"""Tests for EMABuffer training resumption from checkpoints."""

from __future__ import annotations

from mlx_grpo.training.ema_state import EMABuffer


class TestEMAResumeTraining:
    """Test resuming training with saved EMA state."""

    def test_resume_at_same_step_identical(self):
        """Test that resuming at same step gives identical state."""
        # Original training
        buffer1 = EMABuffer(ema_alpha=0.25)

        buffer1.update([1.0] * 16)
        buffer1.update([2.0] * 16)

        state_at_step_2 = buffer1.to_dict()

        # Save state and simulate crash
        checkpoint_dict = state_at_step_2

        # Resume from checkpoint
        buffer2 = EMABuffer.from_dict(checkpoint_dict)

        assert buffer2.step_count == 2
        assert buffer2.current_multipliers == buffer1.current_multipliers

    def test_resume_continue_training_produces_same_result(self):
        """Test that continuing from checkpoint matches fresh run."""
        # Scenario: train 2 steps, checkpoint, then train 2 more steps

        # Fresh run: train 4 steps
        buffer_fresh = EMABuffer(ema_alpha=0.25)
        for i in range(1, 5):
            buffer_fresh.update([float(i)] * 16)
        fresh_result = buffer_fresh.current_multipliers

        # Checkpoint run: train 2, save, resume, train 2 more
        buffer_ckpt = EMABuffer(ema_alpha=0.25)
        for i in range(1, 3):
            buffer_ckpt.update([float(i)] * 16)

        checkpoint = buffer_ckpt.to_dict()

        buffer_resumed = EMABuffer.from_dict(checkpoint)
        for i in range(3, 5):
            buffer_resumed.update([float(i)] * 16)
        resumed_result = buffer_resumed.current_multipliers

        assert fresh_result == resumed_result

    def test_resume_step_count_restored(self):
        """Test that step_count is correctly restored."""
        buffer1 = EMABuffer()

        for _ in range(50):
            buffer1.update([1.0] * 16)

        assert buffer1.step_count == 50

        checkpoint = buffer1.to_dict()

        buffer2 = EMABuffer.from_dict(checkpoint)
        assert buffer2.step_count == 50

        buffer2.update([2.0] * 16)
        assert buffer2.step_count == 51

    def test_resume_ema_alpha_preserved(self):
        """Test that EMA alpha is preserved across resume."""
        alpha_values = [0.01, 0.1, 0.25, 0.5, 0.9]

        for alpha in alpha_values:
            buffer1 = EMABuffer(ema_alpha=alpha)

            for _ in range(10):
                buffer1.update([1.5] * 16)

            checkpoint = buffer1.to_dict()

            buffer2 = EMABuffer.from_dict(checkpoint)
            assert buffer2.ema_alpha == alpha

    def test_resume_convergence_continues(self):
        """Test that convergence continues correctly after resume."""
        buffer = EMABuffer(ema_alpha=0.1)

        # Start from 0
        buffer.update([0.0] * 16)

        # Converge toward 100 for 20 steps
        for _ in range(20):
            buffer.update([100.0] * 16)

        # Save state
        checkpoint = buffer.to_dict()
        value_at_20 = buffer.current_multipliers[0]

        # Resume and continue convergence for 20 more steps
        buffer_resumed = EMABuffer.from_dict(checkpoint)

        for _ in range(20):
            buffer_resumed.update([100.0] * 16)

        value_at_40 = buffer_resumed.current_multipliers[0]

        # Should have converged further
        assert value_at_40 > value_at_20
        assert value_at_40 > 90.0  # Close to target

    def test_resume_with_changed_input(self):
        """Test resuming with different input values."""
        buffer1 = EMABuffer(ema_alpha=0.25)

        # Train toward 10
        for _ in range(5):
            buffer1.update([10.0] * 16)

        checkpoint = buffer1.to_dict()
        value_at_5 = buffer1.current_multipliers[0]

        # Resume and train toward different target (100)
        buffer_resumed = EMABuffer.from_dict(checkpoint)

        for _ in range(10):
            buffer_resumed.update([100.0] * 16)

        value_at_15 = buffer_resumed.current_multipliers[0]

        # Should have moved toward the new target
        assert value_at_15 > value_at_5
        assert value_at_15 > 50.0  # Significantly toward 100

    def test_resume_multiple_times(self):
        """Test multiple save-resume cycles."""
        buffer = EMABuffer(ema_alpha=0.2)

        # Step 0-10
        for i in range(10):
            buffer.update([float(i + 1)] * 16)

        checkpoint1 = buffer.to_dict()
        assert checkpoint1["step_count"] == 10

        # Step 10-20
        for i in range(10):
            buffer.update([float(i + 1)] * 16)

        checkpoint2 = buffer.to_dict()
        assert checkpoint2["step_count"] == 20

        # Resume from checkpoint2
        buffer_restored = EMABuffer.from_dict(checkpoint2)

        # Step 20-30
        for i in range(10):
            buffer_restored.update([float(i + 1)] * 16)

        assert buffer_restored.step_count == 30

        # Resume again from first checkpoint should have different state
        buffer_restored_old = EMABuffer.from_dict(checkpoint1)
        assert buffer_restored_old.step_count == 10
        assert buffer_restored_old.current_multipliers != buffer_restored.current_multipliers

    def test_resume_idempotent_checkpoint(self):
        """Test that checkpoint is idempotent (creating new buffer doesn't change it)."""
        buffer1 = EMABuffer(ema_alpha=0.3)

        for _ in range(15):
            buffer1.update([2.0] * 16)

        checkpoint = buffer1.to_dict()

        # Create multiple buffers from same checkpoint
        buffer2 = EMABuffer.from_dict(checkpoint)
        buffer3 = EMABuffer.from_dict(checkpoint)

        # Both should be identical
        assert buffer2.current_multipliers == buffer3.current_multipliers
        assert buffer2.step_count == buffer3.step_count

        # Updating one shouldn't affect the other
        buffer2.update([5.0] * 16)

        assert buffer2.step_count == 16
        assert buffer3.step_count == 15

    def test_resume_sparse_checkpointing(self):
        """Test checkpointing at intervals (e.g., every 5 steps)."""
        buffer = EMABuffer(ema_alpha=0.25)

        checkpoints = {}

        for step in range(50):
            buffer.update([float((step % 10) + 1)] * 16)

            if (step + 1) % 5 == 0:
                checkpoints[step + 1] = buffer.to_dict()

        # Verify checkpoints are consistent
        assert len(checkpoints) == 10
        for step in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            assert step in checkpoints
            assert checkpoints[step]["step_count"] == step

    def test_resume_after_long_training(self):
        """Test resuming after simulating long training run."""
        buffer = EMABuffer(ema_alpha=0.05)  # Low alpha = stable

        # Simulate 100 steps of training (reduced from 1000 for speed)
        for step in range(100):
            value = 50.0 + 10.0 * (step % 20 - 10)  # Oscillating around 50
            buffer.update([value] * 16)

        # At step 100, take checkpoint
        checkpoint_100 = buffer.to_dict()

        # Resume and continue
        buffer_resumed = EMABuffer.from_dict(checkpoint_100)
        assert buffer_resumed.step_count == 100

        # Continue for 50 more steps
        for step in range(50):
            value = 50.0 + 10.0 * ((step + 100) % 20 - 10)
            buffer_resumed.update([value] * 16)

        assert buffer_resumed.step_count == 150
        # Value should be somewhere in the oscillation range (low alpha means slow convergence)
        assert 30.0 < buffer_resumed.current_multipliers[0] < 70.0

    def test_resume_exact_checkpoint_data(self):
        """Test that checkpoint data exactly matches buffer state."""
        buffer = EMABuffer(ema_alpha=0.15)

        raw_values = [
            [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.0, 2.0, 3.0, 1.5],
            [2.0, 3.0, 4.0, 2.5, 3.5, 4.5, 2.0, 3.0, 4.0, 2.5, 3.5, 4.5, 2.0, 3.0, 4.0, 2.5],
            [1.5, 2.5, 3.5, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 2.0],
        ]

        for raw in raw_values:
            buffer.update(raw)

        # Get exact state
        exact_mults = buffer.current_multipliers
        exact_alpha = buffer.ema_alpha
        exact_step = buffer.step_count

        # Checkpoint and restore
        checkpoint = buffer.to_dict()
        buffer_restored = EMABuffer.from_dict(checkpoint)

        # Verify exact match
        assert buffer_restored.current_multipliers == exact_mults
        assert buffer_restored.ema_alpha == exact_alpha
        assert buffer_restored.step_count == exact_step

    def test_resume_large_step_count(self):
        """Test that large step counts are handled correctly."""
        buffer = EMABuffer()

        # Simulate large step count without actually iterating
        # We can do this by manually updating to match the expected state
        buffer.update([1.0] * 16)

        # Now artificially set step count to large value via dict manipulation
        checkpoint = buffer.to_dict()
        checkpoint["step_count"] = 1000000

        buffer_restored = EMABuffer.from_dict(checkpoint)
        assert buffer_restored.step_count == 1000000

        # Should still work
        buffer_restored.update([2.0] * 16)
        assert buffer_restored.step_count == 1000001
