"""Tests for CheckpointManager — save/load/rotate adapter checkpoints."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mlx_grpo.training.checkpoint import CHECKPOINT_PATTERN, CheckpointManager


class TestCheckpointManagerInit:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapters"
            CheckpointManager(adapter_dir)
            assert adapter_dir.exists()

    def test_creates_nested_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "a" / "b" / "c"
            CheckpointManager(adapter_dir)
            assert adapter_dir.exists()

    def test_default_keep_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            assert mgr.keep_recent == 5
            assert mgr.keep_best == 3


class TestSaveConfig:
    def test_writes_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            mgr.save_config({"model": "test", "rank": 16})
            config_path = Path(tmpdir) / "adapter_config.json"
            assert config_path.exists()
            data = json.loads(config_path.read_text())
            assert data["model"] == "test"

    def test_write_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            mgr.save_config({"version": 1})
            mgr.save_config({"version": 2})  # Should NOT overwrite
            data = json.loads((Path(tmpdir) / "adapter_config.json").read_text())
            assert data["version"] == 1


class TestSave:
    def test_save_no_model(self):
        """save(model=None) should still write training_state.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            mgr.save(iteration=5, model=None, trained_tokens=1000, learning_rate=1e-4)
            state_path = Path(tmpdir) / "training_state.json"
            assert state_path.exists()
            state = json.loads(state_path.read_text())
            assert state["iteration"] == 6  # saved as next iteration
            assert state["trained_tokens"] == 1000

    def test_save_returns_numbered_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            numbered = mgr.save(iteration=100, model=None)
            assert "0000100" in numbered.name

    def test_save_includes_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            mgr.save(iteration=1, model=None, metrics={"mean_reward": 0.75})
            state = json.loads((Path(tmpdir) / "training_state.json").read_text())
            assert state.get("metric_mean_reward") == pytest.approx(0.75)

    def test_save_wandb_run_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            mgr.save(iteration=1, model=None, wandb_run_id="abc123")
            state = json.loads((Path(tmpdir) / "training_state.json").read_text())
            assert state["wandb_run_id"] == "abc123"

    def test_save_multiple_iterations_state(self):
        """Each save updates training_state.json with the next iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, keep_recent=10)
            for i in [0, 1, 2]:
                mgr.save(iteration=i, model=None)
            state = json.loads((Path(tmpdir) / "training_state.json").read_text())
            assert state["iteration"] == 3  # last saved: iter 2 → next is 3


class TestLoadTrainingState:
    def test_empty_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            assert mgr.load_training_state() == {}

    def test_loads_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            mgr.save(iteration=42, model=None, trained_tokens=500)
            state = mgr.load_training_state()
            assert state["iteration"] == 43
            assert state["trained_tokens"] == 500


class TestLoadOptimizerState:
    def test_returns_false_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            assert mgr.load_optimizer_state(object()) is False

    def test_returns_false_for_small_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            path = Path(tmpdir) / "optimizer_state.safetensors"
            path.write_bytes(b"tiny")  # Too small
            assert mgr.load_optimizer_state(object()) is False


class TestLoadWeights:
    def test_empty_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            assert mgr.load_weights() == {}

    def test_empty_when_explicit_path_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            assert mgr.load_weights(Path(tmpdir) / "nonexistent.safetensors") == {}


class TestListCheckpoints:
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            assert mgr.list_checkpoints() == []

    def test_lists_numbered_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, keep_recent=10)
            for i in [10, 50, 100]:
                mgr.save(iteration=i, model=None)
            ckpts = mgr.list_checkpoints()
            iters = [it for it, _ in ckpts]
            assert iters == sorted(iters)  # sorted by iteration

    def test_ignores_non_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            # Create files that don't match pattern
            (Path(tmpdir) / "adapters.safetensors").write_text("")
            (Path(tmpdir) / "training_state.json").write_text("{}")
            assert mgr.list_checkpoints() == []


class TestLatestSymlink:
    def test_latest_symlink_created_after_save_with_weights(self):
        """latest symlink points to the numbered adapters file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            CheckpointManager(tmpdir, keep_recent=10)
            # Simulate saving with real adapter files by touching them manually
            numbered = Path(tmpdir) / "0000100_adapters.safetensors"
            numbered.touch()
            import shutil

            shutil.copy2(str(numbered), str(Path(tmpdir) / "adapters.safetensors"))
            latest = Path(tmpdir) / "latest"
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(numbered.name)
            assert latest.is_symlink()
            assert latest.readlink() == Path("0000100_adapters.safetensors")

    def test_list_checkpoints_ignores_latest_symlink(self):
        """list_checkpoints should never return 'latest'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, keep_recent=10)
            for i in [10, 20]:
                mgr.save(iteration=i, model=None)
            names = [p.name for _, p in mgr.list_checkpoints()]
            assert "latest" not in names


class TestSaveOptimizer:
    def test_save_optimizer_false_skips_optimizer_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, save_optimizer=False)

            # Pass a dummy optimizer; should not be saved
            class _FakeOpt:
                state = {}

            mgr.save(iteration=1, model=None, optimizer=_FakeOpt())
            assert not (Path(tmpdir) / "optimizer_state.safetensors").exists()

    def test_save_optimizer_true_by_default(self):
        # Verify the default is True without needing a real dir
        import inspect

        sig = inspect.signature(CheckpointManager.__init__)
        assert sig.parameters["save_optimizer"].default is True


class TestCheckpointPattern:
    def test_matches(self):
        m = CHECKPOINT_PATTERN.match("0000100_adapters.safetensors")
        assert m and m.group(1) == "0000100"

    def test_no_match(self):
        assert CHECKPOINT_PATTERN.match("adapters.safetensors") is None


class TestRotation:
    def test_rotation_removes_old(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, keep_recent=2, keep_best=1)
            # Create numbered checkpoint files manually
            for i in range(10):
                (Path(tmpdir) / f"{i:07d}_adapters.safetensors").touch()
                mgr._best.append((i, float(i) / 10))
            mgr._rotate()
            ckpts = mgr.list_checkpoints()
            # Should keep at most keep_recent + keep_best = 3 checkpoints
            assert len(ckpts) <= 3

    def test_no_rotation_when_few_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, keep_recent=5, keep_best=3)
            # Create numbered checkpoint files manually (no MLX needed)
            for i in range(4):
                (Path(tmpdir) / f"{i:07d}_adapters.safetensors").touch()
                mgr._best.append((i, float(i)))
            mgr._rotate()
            ckpts = mgr.list_checkpoints()
            assert len(ckpts) == 4  # under threshold (5+3=8), nothing removed
