"""Extended tests for training/spike_guard.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mlx_grpo.training.spike_guard import (
    SkipRecord,
    SpikeConfig,
    SpikeGuard,
    SpikeShutdownError,
    _copy_tree,
    _flatten_values,
)

# ── SpikeConfig ──────────────────────────────────────────────────────────────


class TestSpikeConfig:
    def test_defaults(self):
        cfg = SpikeConfig()
        assert cfg.kl_threshold > 0
        assert cfg.action == "skip"
        assert cfg.max_consecutive == 5

    def test_custom(self):
        cfg = SpikeConfig(kl_threshold=2.0, loss_threshold=5.0, action="skip")
        assert cfg.kl_threshold == 2.0
        assert cfg.loss_threshold == 5.0

    def test_log_file(self):
        cfg = SpikeConfig(log_file="/tmp/test.jsonl")
        assert cfg.log_file == "/tmp/test.jsonl"


# ── SkipRecord ───────────────────────────────────────────────────────────────


class TestSkipRecord:
    def test_basic(self):
        record = SkipRecord(iteration=1, loss=2.5, kl=0.7, reason="spike", consecutive=1)
        assert record.iteration == 1
        assert record.loss == pytest.approx(2.5)
        assert record.kl == pytest.approx(0.7)
        assert record.reason == "spike"
        assert record.consecutive == 1


# ── SpikeGuard ───────────────────────────────────────────────────────────────


def _make_guard(kl_threshold=0.5, loss_threshold=None, max_consecutive=5, action="skip"):
    cfg = SpikeConfig(
        kl_threshold=kl_threshold,
        loss_threshold=loss_threshold,
        max_consecutive=max_consecutive,
        action=action,
    )
    return SpikeGuard(cfg)


class TestSpikeGuardNormalStep:
    def test_normal_kl_returns_none(self):
        guard = _make_guard()
        action = guard.check(loss=1.0, kl=0.1, iteration=0)
        assert action is None

    def test_kl_above_threshold_detected(self):
        guard = _make_guard(kl_threshold=0.3)
        action = guard.check(loss=1.0, kl=1.0, iteration=0)
        assert action == "skip"

    def test_loss_above_threshold_detected(self):
        guard = _make_guard(loss_threshold=2.0)
        action = guard.check(loss=5.0, kl=0.1, iteration=0)
        assert action == "skip"

    def test_kl_delta_spike(self):
        cfg = SpikeConfig(kl_threshold=100.0, kl_delta=0.5, min_window=3)
        guard = SpikeGuard(cfg)
        # Fill window with normal values
        for i in range(5):
            guard.check(loss=1.0, kl=0.1, iteration=i)
        # Now spike KL by 2.0 above mean
        action = guard.check(loss=1.0, kl=2.6, iteration=5)  # 0.1 + 0.5 = 0.6 threshold
        assert action == "skip"

    def test_no_spike_below_threshold(self):
        guard = _make_guard(kl_threshold=1.0)
        action = guard.check(loss=1.0, kl=0.5, iteration=0)
        assert action is None

    def test_consecutive_resets_on_clean(self):
        guard = _make_guard()
        guard._consecutive = 3
        guard.check(loss=1.0, kl=0.1, iteration=0)  # Normal step
        assert guard._consecutive == 0


class TestSpikeGuardShutdown:
    def test_max_consecutive_raises(self):
        """Exceeding max_consecutive raises SpikeShutdownError."""
        guard = _make_guard(kl_threshold=0.3, max_consecutive=2)
        guard._consecutive = 2  # Already at max
        with pytest.raises(SpikeShutdownError):
            guard.check(loss=1.0, kl=1.0, iteration=1)

    def test_action_returned_before_max(self):
        guard = _make_guard(kl_threshold=0.3, max_consecutive=5, action="skip")
        guard._consecutive = 0
        action = guard.check(loss=1.0, kl=1.0, iteration=0)
        assert action == "skip"

    def test_revert_action(self):
        guard = _make_guard(kl_threshold=0.3, action="revert")
        action = guard.check(loss=1.0, kl=1.0, iteration=0)
        assert action == "revert"


class TestSpikeGuardReset:
    def test_reset_accum(self):
        """reset_accum clears gradient accumulation state."""
        guard = _make_guard()

        class FakeTrainer:
            _accum_grads = {"w": [1, 2, 3]}
            _accum_count = 5

        trainer = FakeTrainer()
        guard.reset_accum(trainer)
        assert trainer._accum_count == 0
        assert trainer._accum_grads is None

    def test_reset_accum_no_attr(self):
        """reset_accum sets attrs even when not present."""
        guard = _make_guard()

        class MinimalTrainer:
            pass

        trainer = MinimalTrainer()
        guard.reset_accum(trainer)  # Should not raise
        assert trainer._accum_count == 0


class TestSpikeGuardLogging:
    def test_write_log_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "spike.jsonl")
            cfg = SpikeConfig(kl_threshold=0.1, log_file=log_path)
            guard = SpikeGuard(cfg)
            guard.check(loss=1.0, kl=1.0, iteration=1)  # Trigger spike
            assert Path(log_path).exists()
            line = json.loads(Path(log_path).read_text().strip())
            assert line["iteration"] == 1

    def test_write_log_appends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "spike.jsonl")
            cfg = SpikeConfig(kl_threshold=0.1, log_file=log_path, max_consecutive=100)
            guard = SpikeGuard(cfg)
            guard.check(loss=1.0, kl=1.0, iteration=1)
            guard.check(loss=1.0, kl=1.0, iteration=2)
            lines = Path(log_path).read_text().strip().split("\n")
            assert len(lines) == 2

    def test_no_log_file_no_crash(self):
        """Without log_file, _write_log does nothing."""
        guard = _make_guard()
        guard.check(loss=1.0, kl=1.0, iteration=0)  # Should not raise


class TestSpikeGuardProperties:
    def test_total_skipped(self):
        guard = _make_guard(kl_threshold=0.1)
        guard.check(loss=1.0, kl=1.0, iteration=0)
        assert guard.total_skipped == 1

    def test_skip_records(self):
        guard = _make_guard(kl_threshold=0.1)
        guard.check(loss=1.0, kl=1.0, iteration=5)
        records = guard.skip_records
        assert len(records) == 1
        assert records[0].iteration == 5

    def test_sample_info_stored(self):
        guard = _make_guard(kl_threshold=0.1)
        guard.check(loss=1.0, kl=1.0, iteration=0, sample_info={"prompt": "test"})
        assert guard.skip_records[0].sample == {"prompt": "test"}


# ── _flatten_values ──────────────────────────────────────────────────────────


class TestFlattenValues:
    def test_empty_dict(self):
        result = list(_flatten_values({}))
        assert result == []

    def test_flat_dict(self):
        result = list(_flatten_values({"a": 1, "b": 2}))
        assert sorted(result) == [1, 2]

    def test_nested_dict(self):
        result = list(_flatten_values({"outer": {"inner": 42}}))
        assert result == [42]

    def test_list(self):
        result = list(_flatten_values([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_nested_list(self):
        result = list(_flatten_values([[1, 2], [3, 4]]))
        assert result == [1, 2, 3, 4]

    def test_mixed(self):
        result = set(_flatten_values({"a": [1, 2], "b": {"c": 3}}))
        assert result == {1, 2, 3}

    def test_leaf(self):
        result = list(_flatten_values(99))
        assert result == [99]


# ── _copy_tree ────────────────────────────────────────────────────────────────


class TestCopyTree:
    def test_dict(self):
        tree = {"a": "val1", "b": "val2"}
        copy = _copy_tree(tree)
        assert copy["a"] == "val1"
        assert copy is not tree

    def test_list(self):
        tree = [1, 2, 3]
        copy = _copy_tree(tree)
        assert copy == [1, 2, 3]
        assert copy is not tree

    def test_nested(self):
        tree = {"layer": {"w": "weight_data"}}
        copy = _copy_tree(tree)
        assert copy["layer"]["w"] == "weight_data"
