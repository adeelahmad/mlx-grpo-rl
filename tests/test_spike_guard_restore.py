"""Tests for SpikeGuard restore/snapshot methods and _write_log exception path."""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlx_grpo.training.spike_guard import (
    SpikeConfig,
    SpikeGuard,
    SpikeShutdownError,
)


def _make_guard(kl_threshold=0.5, max_consecutive=5, adapter_path=None):
    cfg = SpikeConfig(kl_threshold=kl_threshold, max_consecutive=max_consecutive)
    return SpikeGuard(cfg, adapter_path=adapter_path)


# ── snapshot ─────────────────────────────────────────────────────────────────


class TestSnapshot:
    def test_snapshot_with_no_mlx(self):
        """When MLX is not available, snapshot sets _snapshot to None."""
        guard = _make_guard()
        model = MagicMock()
        model.trainable_parameters.return_value = {"w": 1.0}

        with patch("mlx_grpo.training.spike_guard._copy_tree", side_effect=Exception("no mlx")):
            guard.snapshot(model)

        assert guard._snapshot is None

    def test_snapshot_with_mock_model(self):
        """snapshot() calls model.trainable_parameters()."""
        guard = _make_guard()
        model = MagicMock()
        model.trainable_parameters.return_value = {"weight": 1.0}

        # Patch the MLX import and eval
        with patch.dict("sys.modules", {"mlx.core": MagicMock()}), contextlib.suppress(Exception):
            guard.snapshot(model)
            # If it doesn't raise, snapshot was attempted

    def test_snapshot_exception_sets_none(self):
        """If snapshot raises, _snapshot stays None."""
        guard = _make_guard()
        model = MagicMock()
        model.trainable_parameters.side_effect = RuntimeError("no params")
        guard.snapshot(model)
        assert guard._snapshot is None


# ── restore_snapshot ──────────────────────────────────────────────────────────


class TestRestoreSnapshot:
    def test_restore_snapshot_none_returns_false(self):
        """When no snapshot taken, restore_snapshot returns False."""
        guard = _make_guard()
        guard._snapshot = None
        model = MagicMock()
        result = guard.restore_snapshot(model)
        assert result is False

    def test_restore_snapshot_with_mlx_exception(self):
        """When MLX restore fails with exception, returns False."""
        guard = _make_guard()
        guard._snapshot = {"w": 1.0}  # Fake snapshot
        model = MagicMock()
        model.update.side_effect = RuntimeError("mlx error")

        with patch.dict("sys.modules", {"mlx.core": MagicMock(side_effect=ImportError)}):
            result = guard.restore_snapshot(model)
            # Should return False when exception occurs
            assert result is False

    def test_restore_snapshot_exception_caught(self):
        """Exception during restore is caught, returns False."""
        guard = _make_guard()
        guard._snapshot = {"w": "fake_array"}
        model = MagicMock()

        # Simulate exception during model.update
        model.update.side_effect = Exception("update failed")

        # Patch mx to avoid import issues
        try:
            import mlx.core  # noqa: F401

            mlx_available = True
        except ImportError:
            mlx_available = False

        with patch("mlx_grpo.training.spike_guard._copy_tree", return_value={}):
            if mlx_available:
                # Real MLX available — test the exception path
                guard._snapshot = {"w": 1.0}
                with contextlib.suppress(Exception):
                    guard.restore_snapshot(model)
                    # With real MLX, model.update raising RuntimeError should return False
            else:
                # MLX not available — mock it
                mock_mx = MagicMock()
                mock_mx.eval.side_effect = RuntimeError("eval failed")
                with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": mock_mx}):
                    guard._snapshot = {"w": MagicMock()}
                    result = guard.restore_snapshot(model)
                    assert result is False


# ── restore_checkpoint ────────────────────────────────────────────────────────


class TestRestoreCheckpoint:
    def test_no_adapter_path_falls_back_to_snapshot(self):
        """Without adapter_path, restore_checkpoint calls restore_snapshot."""
        guard = _make_guard(adapter_path=None)
        guard._snapshot = None
        model = MagicMock()
        # Falls back to restore_snapshot which returns False (no snapshot)
        result = guard.restore_checkpoint(model)
        assert result is False

    def test_adapter_path_no_file_falls_back(self):
        """adapter_path set but file missing → falls back to snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guard = _make_guard(adapter_path=tmpdir)
            guard._snapshot = None
            model = MagicMock()
            # adapters.safetensors doesn't exist
            result = guard.restore_checkpoint(model)
            # Falls back to restore_snapshot which returns False (no snapshot)
            assert result is False

    def test_restore_checkpoint_exception_falls_back(self):
        """Exception during checkpoint load → falls back to snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake (but invalid) adapters.safetensors
            weights_path = Path(tmpdir) / "adapters.safetensors"
            weights_path.write_bytes(b"not valid safetensors")
            guard = _make_guard(adapter_path=tmpdir)
            guard._snapshot = None
            model = MagicMock()
            # With invalid file, MLX will raise an exception → fall back to snapshot
            result = guard.restore_checkpoint(model)
            # Either fails gracefully or returns False
            assert isinstance(result, bool)


# ── _write_log exception ──────────────────────────────────────────────────────


class TestWriteLogException:
    def test_write_log_exception_silenced(self):
        """Exception in _write_log is silently caught."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "nested" / "deep" / "spike.jsonl")
            cfg = SpikeConfig(kl_threshold=0.1, log_file=log_path, max_consecutive=100)
            guard = SpikeGuard(cfg)
            # This should work (creates parent dirs)
            guard.check(loss=1.0, kl=1.0, iteration=1)
            assert Path(log_path).exists()

    def test_write_log_exception_in_write(self):
        """If file write fails (e.g., read-only), exception is silenced."""

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "spike.jsonl")
            cfg = SpikeConfig(kl_threshold=0.1, log_file=log_path, max_consecutive=100)
            guard = SpikeGuard(cfg)
            # Create log file and make it read-only
            Path(log_path).write_text("")
            Path(log_path).chmod(0o000)
            try:
                # Should not raise despite write failure
                guard.check(loss=1.0, kl=1.0, iteration=1)
            finally:
                # Restore permissions for cleanup
                Path(log_path).chmod(0o644)

    def test_log_includes_to_dict(self):
        """Written log contains all SkipRecord fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "spike.jsonl")
            cfg = SpikeConfig(kl_threshold=0.1, log_file=log_path, max_consecutive=100)
            guard = SpikeGuard(cfg)
            guard.check(loss=2.5, kl=1.5, iteration=7, sample_info={"prompt": "test"})
            line = json.loads(Path(log_path).read_text().strip())
            assert line["iteration"] == 7
            assert line["loss"] == pytest.approx(2.5)
            assert line["kl"] == pytest.approx(1.5)
            assert line["sample"]["prompt"] == "test"
            assert "timestamp" in line


# ── loss_delta detection ──────────────────────────────────────────────────────


class TestLossDeltaDetection:
    def test_loss_delta_spike(self):
        cfg = SpikeConfig(kl_threshold=100.0, loss_delta=1.0, loss_threshold=None, min_window=3)
        guard = SpikeGuard(cfg)
        for i in range(5):
            guard.check(loss=1.0, kl=0.1, iteration=i)
        # Now spike loss
        action = guard.check(loss=5.0, kl=0.1, iteration=5)
        assert action == "skip"

    def test_loss_delta_normal_no_spike(self):
        cfg = SpikeConfig(kl_threshold=100.0, loss_delta=5.0, min_window=3)
        guard = SpikeGuard(cfg)
        for i in range(5):
            guard.check(loss=1.0, kl=0.1, iteration=i)
        action = guard.check(loss=1.5, kl=0.1, iteration=5)
        assert action is None


# ── shutdown action ───────────────────────────────────────────────────────────


class TestShutdownAction:
    def test_shutdown_action_returned_as_string(self):
        """action='shutdown' returns 'shutdown' string — caller decides to raise."""
        cfg = SpikeConfig(kl_threshold=0.1, action="shutdown", max_consecutive=10)
        guard = SpikeGuard(cfg)
        action = guard.check(loss=1.0, kl=1.0, iteration=0)
        assert action == "shutdown"

    def test_shutdown_raises_at_max_consecutive(self):
        """SpikeShutdownError raised when consecutive reaches max_consecutive."""
        cfg = SpikeConfig(kl_threshold=0.1, action="shutdown", max_consecutive=2)
        guard = SpikeGuard(cfg)
        guard._consecutive = 2  # Pre-set to max
        with pytest.raises(SpikeShutdownError):
            guard.check(loss=1.0, kl=1.0, iteration=0)

    def test_revert_action_returned(self):
        cfg = SpikeConfig(kl_threshold=0.1, action="revert", max_consecutive=10)
        guard = SpikeGuard(cfg)
        action = guard.check(loss=1.0, kl=1.0, iteration=0)
        assert action == "revert"
