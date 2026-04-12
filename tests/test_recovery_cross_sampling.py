"""Tests for training/recovery.py and data/cross_sampling.py."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from mlx_grpo.data import Sample
from mlx_grpo.data.cross_sampling import CrossSampleConfig, CrossSampler
from mlx_grpo.training.recovery import EXIT_CODE_CORRUPTION, RecoveryManager, RecoveryState

# ── RecoveryManager ──────────────────────────────────────────────────────────


class TestRecoveryManagerLoadOrCreate:
    def test_creates_fresh_state_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            state = mgr.load_or_create()
            assert state.iteration == 0
            assert state.crash_count == 0

    def test_loads_existing_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState(iteration=42, crash_count=1)
            mgr.save_state(s)
            loaded = mgr.load_or_create()
            assert loaded.iteration == 42
            assert loaded.crash_count == 1

    def test_config_hash_mismatch_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState(iteration=42, config_hash="abc")
            mgr.save_state(s)
            fresh = mgr.load_or_create(config_hash="different")
            assert fresh.iteration == 0

    def test_config_hash_match_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState(iteration=42, config_hash="abc")
            mgr.save_state(s)
            loaded = mgr.load_or_create(config_hash="abc")
            assert loaded.iteration == 42

    def test_corrupted_json_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            mgr.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            mgr.state_path.write_text("not valid json{")
            state = mgr.load_or_create()
            assert state.iteration == 0

    def test_no_hash_in_file_loads_anyway(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState(iteration=10)
            mgr.save_state(s)
            # config_hash is empty in saved state, doesn't trigger mismatch
            loaded = mgr.load_or_create(config_hash="any")
            assert loaded.iteration == 10


class TestRecoveryManagerSaveState:
    def test_saves_all_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState(iteration=5, epoch=2, best_reward=0.9, total_tokens=1000)
            mgr.save_state(s)
            data = json.loads(mgr.state_path.read_text())
            assert data["iteration"] == 5
            assert data["epoch"] == 2
            assert data["best_reward"] == pytest.approx(0.9)

    def test_atomic_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            mgr.save_state(RecoveryState(iteration=1))
            assert mgr.state_path.exists()
            tmp_path = mgr.state_path.with_suffix(".tmp")
            assert not tmp_path.exists()  # Temp file cleaned up


class TestRecoveryManagerRecordCrash:
    def test_increments_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState()
            result = mgr.record_crash(s, "test error")
            assert s.crash_count == 1
            assert result is True  # Under max

    def test_returns_false_at_max(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState(crash_count=4)  # MAX_CRASHES=5, about to hit
            result = mgr.record_crash(s)
            assert result is False

    def test_stores_error_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState()
            mgr.record_crash(s, "GPU OOM error")
            assert "GPU OOM error" in s.extras["last_error"]

    def test_truncates_long_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState()
            mgr.record_crash(s, "x" * 1000)
            assert len(s.extras["last_error"]) <= 500


class TestRecoveryManagerShouldResume:
    def test_resumes_when_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState(iteration=5, last_checkpoint="/some/path")
            assert mgr.should_resume(s) is True

    def test_no_resume_when_iteration_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState(iteration=0, last_checkpoint="/some/path")
            assert mgr.should_resume(s) is False

    def test_no_resume_when_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState(iteration=5, last_checkpoint="")
            assert mgr.should_resume(s) is False

    def test_no_resume_when_max_crashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState(iteration=5, last_checkpoint="/path", crash_count=5)
            assert mgr.should_resume(s) is False


class TestRecoveryManagerFindLatest:
    def test_returns_none_when_no_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir + "/nonexistent")
            assert mgr.find_latest_checkpoint() is None

    def test_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            assert mgr.find_latest_checkpoint() is None

    def test_finds_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            ckpt = Path(tmpdir) / "checkpoint-100"
            ckpt.mkdir()
            result = mgr.find_latest_checkpoint()
            assert result is not None
            assert "checkpoint-100" in result


class TestRecoveryManagerCleanupCrashes:
    def test_resets_if_enough_time_passed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState(crash_count=3, last_crash_time=time.time() - 200)
            mgr.cleanup_old_crashes(s, cooldown=60)
            assert s.crash_count == 0

    def test_no_reset_if_too_recent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RecoveryManager(tmpdir)
            s = RecoveryState(crash_count=3, last_crash_time=time.time())
            mgr.cleanup_old_crashes(s, cooldown=600)
            assert s.crash_count == 3


# ── CrossSampler ─────────────────────────────────────────────────────────────


def _make_sample(prompt: str, answer: str, type_name: str = "math") -> Sample:
    return Sample(prompt=prompt, answer=answer, type_name=type_name)


class TestCrossSamplerInit:
    def test_indexes_by_type(self):
        samples = [
            _make_sample("q1", "a1", "math"),
            _make_sample("q2", "a2", "math"),
            _make_sample("q3", "a3", "mcq"),
        ]
        cs = CrossSampler(samples)
        assert "math" in cs._by_type
        assert len(cs._by_type["math"]) == 2


class TestCrossSamplerAugment:
    def test_augment_trigger(self):
        """With ratio=1.0, augmentation always fires."""
        samples = [
            _make_sample("q1", "answer one", "math"),
            _make_sample("q2", "answer two", "math"),
            _make_sample("q3", "answer three", "math"),
        ]
        cfg = CrossSampleConfig(ratio=1.0, seed=42)
        cs = CrossSampler(samples, cfg)
        augmented = cs.augment(samples[0])
        # With ratio=1.0, should augment (history injection)
        assert isinstance(augmented, Sample)

    def test_augment_no_trigger_zero_ratio(self):
        """With ratio=0.0, augmentation never fires."""
        samples = [
            _make_sample("q1", "a1"),
            _make_sample("q2", "a2"),
        ]
        cfg = CrossSampleConfig(ratio=0.0, seed=42)
        cs = CrossSampler(samples, cfg)
        result = cs.augment(samples[0])
        assert result is samples[0]

    def test_single_sample_pool_returns_original(self):
        samples = [_make_sample("q1", "a1")]
        cfg = CrossSampleConfig(ratio=1.0)
        cs = CrossSampler(samples, cfg)
        result = cs.augment(samples[0])
        assert result is samples[0]

    def test_same_type_only(self):
        """same_type_only=True only samples from same type."""
        samples = [
            _make_sample("q1", "a1", "math"),
            _make_sample("q2", "a2", "math"),
            _make_sample("q3", "a3", "mcq"),
        ]
        cfg = CrossSampleConfig(ratio=1.0, same_type_only=True, seed=42)
        cs = CrossSampler(samples, cfg)
        result = cs.augment(samples[2])  # mcq sample — only 1 mcq total
        # Only 1 sample in mcq type → not enough for augmentation
        assert result is samples[2]

    def test_cross_type_sampling(self):
        """same_type_only=False uses all samples."""
        samples = [
            _make_sample("q1", "a1", "math"),
            _make_sample("q2", "a2", "mcq"),
            _make_sample("q3", "a3", "general"),
        ]
        cfg = CrossSampleConfig(ratio=1.0, same_type_only=False, seed=0)
        cs = CrossSampler(samples, cfg)
        result = cs.augment(samples[0])
        assert isinstance(result, Sample)

    def test_augment_batch(self):
        samples = [_make_sample(f"q{i}", f"a{i}") for i in range(5)]
        cfg = CrossSampleConfig(ratio=0.5, seed=42)
        cs = CrossSampler(samples, cfg)
        results = cs.augment_batch(samples)
        assert len(results) == 5

    def test_reseed(self):
        samples = [_make_sample(f"q{i}", f"a{i}") for i in range(5)]
        cs = CrossSampler(samples, CrossSampleConfig(seed=42))
        cs.reseed(99)
        # After reseed, behavior should be reproducible
        r1 = cs.augment(samples[0])
        cs.reseed(99)
        r2 = cs.augment(samples[0])
        assert r1.prompt == r2.prompt

    def test_no_history_when_no_answers(self):
        """Samples without answers don't contribute history."""
        samples = [
            Sample(prompt="q1", answer=None, type_name="math"),
            Sample(prompt="q2", answer=None, type_name="math"),
        ]
        cfg = CrossSampleConfig(ratio=1.0, seed=42)
        cs = CrossSampler(samples, cfg)
        result = cs.augment(samples[0])
        # No answers → history_parts empty → return original
        assert result is samples[0]


# ── Constant ────────────────────────────────────────────────────────────────


def test_exit_code_corruption():
    assert EXIT_CODE_CORRUPTION == 42
