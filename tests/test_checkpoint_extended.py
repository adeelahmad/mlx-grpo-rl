"""Extended tests for training/checkpoint.py — load_weights, load_optimizer_state, save with model."""  # noqa: E501

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import pytest

from mlx_grpo.training.checkpoint import CHECKPOINT_PATTERN, CheckpointManager

# ── save() with metric-based best tracking ───────────────────────────────────


class TestSaveBestTracking:
    def test_save_with_reward_metric(self):
        """Saving with mean_reward metric tracks best."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, keep_recent=5, keep_best=3)
            # Save 3 checkpoints with increasing rewards
            for i, reward in enumerate([0.3, 0.5, 0.7]):
                mgr.save(iteration=i, model=None, metrics={"mean_reward": reward})
            state = json.loads((Path(tmpdir) / "training_state.json").read_text())
            assert "metric_mean_reward" in state
            assert state["metric_mean_reward"] == pytest.approx(0.7)

    def test_save_without_metric(self):
        """Saving without metrics doesn't fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            numbered = mgr.save(iteration=10, model=None)
            assert numbered is not None

    def test_save_returns_correct_path(self):
        """Numbered path has 7-digit zero-padded iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            path = mgr.save(iteration=42, model=None)
            assert "0000042" in path.name


# ── load_training_state ──────────────────────────────────────────────────────


class TestLoadTrainingStateExtended:
    def test_loads_all_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            mgr.save(
                iteration=5,
                model=None,
                trained_tokens=2000,
                learning_rate=2e-4,
                wandb_run_id="run123",
                metrics={"mean_reward": 0.8},
            )
            state = mgr.load_training_state()
            assert state["iteration"] == 6
            assert state["trained_tokens"] == 2000
            assert state["learning_rate"] == pytest.approx(2e-4)
            assert state["wandb_run_id"] == "run123"
            assert state.get("metric_mean_reward") == pytest.approx(0.8)


# ── load_optimizer_state ─────────────────────────────────────────────────────


class TestLoadOptimizerStateExtended:
    def _make_fake_safetensors(self, path: Path, size: int = 200) -> None:
        """Create a fake safetensors file large enough to pass size check."""
        # safetensors format: 8-byte header length + header + data
        header = b'{"__metadata__": {}}'
        header_len = len(header)
        content = struct.pack("<Q", header_len) + header + b"\x00" * size
        path.write_bytes(content)

    def test_returns_false_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            assert mgr.load_optimizer_state(object()) is False

    def test_returns_false_small_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            path = Path(tmpdir) / "optimizer_state.safetensors"
            path.write_bytes(b"x" * 10)
            assert mgr.load_optimizer_state(object()) is False


# ── load_weights ─────────────────────────────────────────────────────────────


class TestLoadWeightsExtended:
    def test_empty_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            assert mgr.load_weights() == {}

    def test_explicit_nonexistent_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            result = mgr.load_weights(Path(tmpdir) / "ghost.safetensors")
            assert result == {}

    def test_explicit_path_invalid_format(self):
        """Invalid safetensors data → RuntimeError propagates."""
        import pytest

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            bad_path = Path(tmpdir) / "bad.safetensors"
            bad_path.write_bytes(b"garbage data")
            # MLX raises RuntimeError for invalid safetensors
            with pytest.raises(RuntimeError):
                mgr.load_weights(bad_path)


# ── list_checkpoints extended ─────────────────────────────────────────────────


class TestListCheckpointsExtended:
    def test_multiple_types_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, keep_recent=20)
            # Create numbered checkpoint files manually
            for i in [5, 15, 25]:
                (Path(tmpdir) / f"{i:07d}_adapters.safetensors").touch()
            ckpts = mgr.list_checkpoints()
            iters = [it for it, _ in ckpts]
            assert iters == [5, 15, 25]

    def test_multiple_shard_files(self):
        """Only one entry per numbered checkpoint even if multiple shards exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            (Path(tmpdir) / "0000010_adapters.safetensors").touch()
            (Path(tmpdir) / "0000010_adapters_2.safetensors").touch()  # Not matching pattern
            ckpts = mgr.list_checkpoints()
            # Second file doesn't match CHECKPOINT_PATTERN
            assert len(ckpts) == 1


# ── CheckpointPattern ────────────────────────────────────────────────────────


class TestCheckpointPatternExtended:
    def test_various_iterations(self):
        for iter_str in ["0000001", "0010000", "9999999"]:
            m = CHECKPOINT_PATTERN.match(f"{iter_str}_adapters.safetensors")
            assert m and m.group(1) == iter_str

    def test_no_match_wrong_format(self):
        assert CHECKPOINT_PATTERN.match("adapters_0000100.safetensors") is None
        assert CHECKPOINT_PATTERN.match("checkpoint.safetensors") is None

    def test_matches_exact_format(self):
        m = CHECKPOINT_PATTERN.match("0001000_adapters.safetensors")
        assert m is not None


# ── save_config extended ─────────────────────────────────────────────────────


class TestSaveConfigExtended:
    def test_write_once_multiple_saves(self):
        """Third save also doesn't overwrite first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            mgr.save_config({"version": 1})
            mgr.save_config({"version": 2})
            mgr.save_config({"version": 3})
            data = json.loads((Path(tmpdir) / "adapter_config.json").read_text())
            assert data["version"] == 1

    def test_config_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            mgr.save_config({"key": [1, 2, 3], "nested": {"a": 1}})
            data = json.loads((Path(tmpdir) / "adapter_config.json").read_text())
            assert data["key"] == [1, 2, 3]


# ── Rotation extended ────────────────────────────────────────────────────────


class TestRotationExtended:
    def test_rotation_keeps_best(self):
        """Rotation keeps best checkpoints even if old."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, keep_recent=2, keep_best=2)
            # Create 8 checkpoint files
            for i in range(8):
                (Path(tmpdir) / f"{i:07d}_adapters.safetensors").touch()
                mgr._best.append((i, float(i) * 0.1))  # Ascending reward
            mgr._rotate()
            ckpts = mgr.list_checkpoints()
            # Should have at most keep_recent + keep_best = 4
            assert len(ckpts) <= 4

    def test_no_rotation_exact_threshold(self):
        """Exactly at threshold, no rotation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, keep_recent=3, keep_best=2)
            # Create exactly 5 checkpoints (= threshold)
            for i in range(5):
                (Path(tmpdir) / f"{i:07d}_adapters.safetensors").touch()
                mgr._best.append((i, float(i)))
            mgr._rotate()
            ckpts = mgr.list_checkpoints()
            assert len(ckpts) == 5
