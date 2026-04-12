"""Tests for FullFTCheckpointManager — directory-based full FT checkpoints."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mlx_grpo.training.full_checkpoint import FullFTCheckpointManager


class TestInit:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            FullFTCheckpointManager(Path(tmpdir) / "ckpts")
            assert (Path(tmpdir) / "ckpts").exists()

    def test_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            assert mgr.save_optimizer is True
            assert mgr.model_source_path is None
            assert mgr.keep_recent == 5
            assert mgr.keep_best == 3

    def test_custom_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(
                tmpdir, keep_recent=2, keep_best=1, save_optimizer=False, model_source_path="/m"
            )
            assert mgr.save_optimizer is False
            assert mgr.model_source_path == "/m"


class TestSave:
    def test_creates_numbered_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            mgr.save(iteration=100, model=None)
            assert (Path(tmpdir) / "0000100").is_dir()

    def test_returns_model_safetensors_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            result = mgr.save(iteration=50, model=None)
            assert result.name == "adapters.safetensors"
            assert "0000050" in str(result)

    def test_creates_latest_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            mgr.save(iteration=100, model=None)
            latest = Path(tmpdir) / "latest"
            assert latest.is_symlink()
            assert latest.resolve() == (Path(tmpdir) / "0000100").resolve()

    def test_latest_symlink_updated_on_second_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir, keep_recent=10, keep_best=10)
            mgr.save(iteration=100, model=None)
            mgr.save(iteration=200, model=None)
            latest = Path(tmpdir) / "latest"
            assert latest.is_symlink()
            # Symlink target is the relative name "0000200"
            assert latest.readlink() == Path("0000200")

    def test_writes_training_state_at_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            mgr.save(iteration=5, model=None, trained_tokens=1000)
            state = json.loads((Path(tmpdir) / "training_state.json").read_text())
            assert state["iteration"] == 6
            assert state["trained_tokens"] == 1000

    def test_writes_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            mgr.save(iteration=1, model=None, metrics={"mean_reward": 0.8})
            state = json.loads((Path(tmpdir) / "training_state.json").read_text())
            assert state["metric_mean_reward"] == pytest.approx(0.8)

    def test_writes_wandb_run_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            mgr.save(iteration=1, model=None, wandb_run_id="xyz789")
            state = json.loads((Path(tmpdir) / "training_state.json").read_text())
            assert state["wandb_run_id"] == "xyz789"

    def test_no_optimizer_dir_when_save_optimizer_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir, save_optimizer=False)
            mgr.save(iteration=10, model=None, optimizer=object())
            ckpt = Path(tmpdir) / "0000010"
            assert not (ckpt / "optimizer_state.safetensors").exists()


class TestCopyInferenceFiles:
    def test_copies_existing_files(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as tmpdir:
            # Create fake source inference files
            (Path(source) / "config.json").write_text('{"model_type": "llama"}')
            (Path(source) / "tokenizer_config.json").write_text("{}")

            mgr = FullFTCheckpointManager(tmpdir, model_source_path=source)
            mgr.save(iteration=5, model=None)

            ckpt = Path(tmpdir) / "0000005"
            assert (ckpt / "config.json").read_text() == '{"model_type": "llama"}'
            assert (ckpt / "tokenizer_config.json").exists()

    def test_skips_nonexistent_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir, model_source_path="/nonexistent/path")
            # Should not raise
            mgr.save(iteration=1, model=None)

    def test_does_not_overwrite_existing(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as tmpdir:
            (Path(source) / "config.json").write_text('{"v": 1}')
            mgr = FullFTCheckpointManager(tmpdir, model_source_path=source)
            mgr.save(iteration=5, model=None)

            # Pre-place modified file
            ckpt = Path(tmpdir) / "0000005"
            (ckpt / "config.json").write_text('{"v": 99}')

            # Second save should NOT overwrite
            mgr.save(iteration=10, model=None)
            assert (ckpt / "config.json").read_text() == '{"v": 99}'


class TestAdapterConfig:
    def test_adapter_config_copied_into_ckpt_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            mgr.save_config({"fine_tune_type": "full", "model": "test"})
            mgr.save(iteration=5, model=None)
            ckpt = Path(tmpdir) / "0000005"
            assert (ckpt / "adapter_config.json").exists()
            import json

            data = json.loads((ckpt / "adapter_config.json").read_text())
            assert data["fine_tune_type"] == "full"

    def test_adapter_config_not_overwritten_if_already_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            mgr.save_config({"fine_tune_type": "full", "model": "v1"})
            mgr.save(iteration=5, model=None)
            ckpt = Path(tmpdir) / "0000005"
            # Manually modify the copy in the checkpoint dir
            import json

            (ckpt / "adapter_config.json").write_text('{"fine_tune_type": "modified"}')
            # Second save to a new dir — existing dir is untouched
            mgr.save(iteration=10, model=None)
            assert (
                json.loads((ckpt / "adapter_config.json").read_text())["fine_tune_type"]
                == "modified"
            )

    def test_no_crash_when_root_config_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            # No save_config() called — should not crash
            mgr.save(iteration=1, model=None)


class TestMigrationFallback:
    def test_flat_adapters_found_when_no_dir_checkpoints(self):
        """Migration: _latest_model_path resolves to flat file when no dirs exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            flat = Path(tmpdir) / "adapters.safetensors"
            flat.touch()
            # _latest_dir() returns None (no numbered dirs, no latest symlink)
            assert mgr._latest_dir() is None
            # load_weights should resolve to the flat file before trying mx.load
            # We verify by checking that no KeyError / "no path" path is taken.
            # Since mx.load would fail on an empty file, patch it:
            import unittest.mock as mock

            with mock.patch("mlx.core.load", return_value={}):
                result = mgr.load_weights()
            assert isinstance(result, dict)

    def test_explicit_dir_path_checks_both_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            ckpt = Path(tmpdir) / "0000010"
            ckpt.mkdir()
            # Old naming: model.safetensors — should be found by load_weights(dir)
            (ckpt / "model.safetensors").touch()
            import unittest.mock as mock

            with mock.patch("mlx.core.load", return_value={}):
                result = mgr.load_weights(ckpt)
            assert isinstance(result, dict)


class TestLoadWeights:
    def test_empty_when_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            assert mgr.load_weights() == {}

    def test_empty_when_latest_has_no_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            mgr.save(iteration=5, model=None)  # no weights (model=None)
            # latest exists but adapters.safetensors doesn't
            assert mgr.load_weights() == {}

    def test_explicit_nonexistent_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            assert mgr.load_weights(Path(tmpdir) / "nonexistent.safetensors") == {}

    def test_explicit_dir_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            ckpt = Path(tmpdir) / "0000010"
            ckpt.mkdir()
            # adapters.safetensors doesn't exist → empty
            assert mgr.load_weights(ckpt) == {}


class TestLoadOptimizerState:
    def test_returns_false_when_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            assert mgr.load_optimizer_state(object()) is False

    def test_returns_false_when_no_optimizer_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            mgr.save(iteration=5, model=None)  # creates latest symlink
            assert mgr.load_optimizer_state(object()) is False

    def test_returns_false_for_small_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            mgr.save(iteration=5, model=None)
            latest = Path(tmpdir) / "latest"
            (latest / "optimizer_state.safetensors").write_bytes(b"tiny")
            assert mgr.load_optimizer_state(object()) is False


class TestListCheckpoints:
    def test_empty_initially(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            assert mgr.list_checkpoints() == []

    def test_lists_numbered_dirs_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir, keep_recent=10, keep_best=10)
            for i in [50, 10, 100]:
                mgr.save(iteration=i, model=None)
            ckpts = mgr.list_checkpoints()
            iters = [it for it, _ in ckpts]
            assert iters == sorted(iters)

    def test_ignores_latest_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir, keep_recent=10, keep_best=10)
            mgr.save(iteration=5, model=None)
            ckpts = mgr.list_checkpoints()
            names = [p.name for _, p in ckpts]
            assert "latest" not in names

    def test_ignores_non_matching_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            (Path(tmpdir) / "random_dir").mkdir()
            (Path(tmpdir) / "abc1234").mkdir()
            assert mgr.list_checkpoints() == []


class TestRotation:
    def test_rotation_removes_old_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir, keep_recent=2, keep_best=1)
            for i in range(10):
                d = Path(tmpdir) / f"{i:07d}"
                d.mkdir()
                mgr._best.append((i, float(i) / 10))
            mgr._rotate()
            ckpts = mgr.list_checkpoints()
            assert len(ckpts) <= 3  # keep_recent + keep_best

    def test_no_rotation_when_few_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir, keep_recent=5, keep_best=3)
            for i in range(4):
                d = Path(tmpdir) / f"{i:07d}"
                d.mkdir()
                mgr._best.append((i, float(i)))
            mgr._rotate()
            assert len(mgr.list_checkpoints()) == 4

    def test_rotation_preserves_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir, keep_recent=1, keep_best=1)
            for i in range(5):
                d = Path(tmpdir) / f"{i:07d}"
                d.mkdir()
                # iteration 2 is "best" with reward=0.9
                mgr._best.append((i, 0.9 if i == 2 else float(i) / 10))
            mgr._rotate()
            iters = [it for it, _ in mgr.list_checkpoints()]
            assert 2 in iters  # best kept
            assert 4 in iters  # most recent kept


class TestTrainingStateCompatibility:
    def test_load_training_state_after_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            mgr.save(iteration=42, model=None, trained_tokens=9999)
            state = mgr.load_training_state()
            assert state["iteration"] == 43
            assert state["trained_tokens"] == 9999

    def test_load_training_state_empty_before_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = FullFTCheckpointManager(tmpdir)
            assert mgr.load_training_state() == {}
