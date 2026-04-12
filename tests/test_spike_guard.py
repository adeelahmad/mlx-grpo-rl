"""Tests for SpikeGuard — spike detection and mitigation."""

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

# ── SkipRecord ─────────────────────────────────────────────────────────────────


def test_skip_record_to_dict():
    rec = SkipRecord(iteration=5, loss=1.2, kl=0.6, reason="kl spike", consecutive=2)
    d = rec.to_dict()
    assert d["iteration"] == 5
    assert d["loss"] == 1.2
    assert d["kl"] == 0.6
    assert d["reason"] == "kl spike"
    assert d["consecutive"] == 2
    assert "timestamp" in d
    assert d["sample"] == {}


def test_skip_record_with_sample():
    rec = SkipRecord(
        iteration=1,
        loss=2.0,
        kl=0.5,
        reason="r",
        consecutive=1,
        sample={"prompt": "hello"},
    )
    d = rec.to_dict()
    assert d["sample"] == {"prompt": "hello"}


# ── _flatten_values ─────────────────────────────────────────────────────────────


def test_flatten_values_flat_dict():
    tree = {"a": 1, "b": 2}
    assert list(_flatten_values(tree)) == [1, 2]


def test_flatten_values_nested_dict():
    tree = {"outer": {"inner": 42}}
    assert list(_flatten_values(tree)) == [42]


def test_flatten_values_list():
    assert list(_flatten_values([1, 2, 3])) == [1, 2, 3]


def test_flatten_values_nested_list():
    assert list(_flatten_values([[1, 2], [3, 4]])) == [1, 2, 3, 4]


def test_flatten_values_mixed():
    tree = {"a": [1, 2], "b": {"c": 3}}
    result = list(_flatten_values(tree))
    assert set(result) == {1, 2, 3}


def test_flatten_values_leaf():
    assert list(_flatten_values(99)) == [99]


# ── _copy_tree ──────────────────────────────────────────────────────────────────


def test_copy_tree_dict():
    """Without MLX, leaf values are returned as-is."""
    tree = {"a": "val1", "b": "val2"}
    result = _copy_tree(tree)
    # Falls through to the except path: returns original value
    assert result == {"a": "val1", "b": "val2"}


def test_copy_tree_list():
    result = _copy_tree(["x", "y"])
    assert result == ["x", "y"]


def test_copy_tree_nested():
    result = _copy_tree({"x": {"y": "leaf"}})
    assert result == {"x": {"y": "leaf"}}


# ── SpikeConfig ─────────────────────────────────────────────────────────────────


def test_spike_config_defaults():
    cfg = SpikeConfig()
    assert cfg.kl_threshold == 0.5
    assert cfg.action == "skip"
    assert cfg.max_consecutive == 5


# ── SpikeGuard._spike_reason ────────────────────────────────────────────────────


class TestSpikeReason:
    def _guard(self, **kwargs) -> SpikeGuard:
        return SpikeGuard(SpikeConfig(**kwargs))

    def test_clean_below_threshold(self):
        g = self._guard(kl_threshold=0.5)
        assert g._spike_reason(loss=0.1, kl=0.1) is None

    def test_absolute_kl_spike(self):
        g = self._guard(kl_threshold=0.3)
        reason = g._spike_reason(loss=0.1, kl=0.5)
        assert reason is not None
        assert "kl" in reason

    def test_absolute_loss_spike(self):
        g = self._guard(loss_threshold=2.0)
        reason = g._spike_reason(loss=3.0, kl=0.01)
        assert reason is not None
        assert "loss" in reason

    def test_no_loss_threshold_by_default(self):
        g = self._guard()  # loss_threshold=None
        # Even very high loss should not trigger without threshold
        assert g._spike_reason(loss=999.0, kl=0.01) is None

    def test_delta_kl_below_min_window(self):
        """Delta checks require min_window steps first."""
        g = self._guard(kl_threshold=100.0, kl_delta=0.1, min_window=5)
        # Only 2 values in window — no delta spike
        g._kl_win.extend([0.1, 0.1])
        assert g._spike_reason(loss=0.1, kl=1.0) is None

    def test_delta_kl_fires_after_min_window(self):
        g = self._guard(kl_threshold=100.0, kl_delta=0.1, min_window=3, window=20)
        # Fill window with small values
        for _ in range(5):
            g._kl_win.append(0.1)
        # Now kl=0.5 >> mean(0.1) + delta(0.1)
        reason = g._spike_reason(loss=0.1, kl=0.5)
        assert reason is not None and "delta" in reason

    def test_delta_loss_fires(self):
        g = self._guard(kl_threshold=100.0, loss_delta=0.5, min_window=3, window=20)
        for _ in range(5):
            g._loss_win.append(0.2)
            g._kl_win.append(0.01)
        reason = g._spike_reason(loss=2.0, kl=0.01)
        assert reason is not None and "loss" in reason


# ── SpikeGuard.check ────────────────────────────────────────────────────────────


class TestSpikeGuardCheck:
    def _guard(self, **kwargs) -> SpikeGuard:
        return SpikeGuard(SpikeConfig(kl_threshold=0.5, **kwargs))

    def test_clean_step_returns_none(self):
        g = self._guard()
        assert g.check(loss=0.1, kl=0.1, iteration=0) is None

    def test_clean_step_resets_consecutive(self):
        g = self._guard()
        g._consecutive = 3
        g.check(loss=0.1, kl=0.1, iteration=0)
        assert g._consecutive == 0

    def test_spike_returns_action(self):
        g = self._guard(action="skip")
        action = g.check(loss=0.1, kl=0.9, iteration=1)
        assert action == "skip"

    def test_spike_increments_counter(self):
        g = self._guard(action="skip")
        g.check(loss=0.1, kl=0.9, iteration=1)
        assert g._consecutive == 1
        assert g._total_skipped == 1

    def test_multiple_spikes_tracked(self):
        g = self._guard(action="skip", max_consecutive=10)
        for i in range(3):
            g.check(loss=0.1, kl=0.9, iteration=i)
        assert g._total_skipped == 3
        assert g._consecutive == 3

    def test_consecutive_spike_shutdown(self):
        g = self._guard(action="skip", max_consecutive=3)
        with pytest.raises(SpikeShutdownError):
            for i in range(5):
                g.check(loss=0.1, kl=0.9, iteration=i)

    def test_shutdown_action_returned(self):
        """action='shutdown' is returned as the action string (caller raises on it)."""
        g = self._guard(action="shutdown", max_consecutive=100)
        action = g.check(loss=0.1, kl=0.9, iteration=0)
        assert action == "shutdown"

    def test_window_updated_on_clean(self):
        g = self._guard()
        g.check(loss=1.5, kl=0.1, iteration=0)
        assert g._loss_win[-1] == 1.5
        assert g._kl_win[-1] == 0.1

    def test_window_updated_on_spike(self):
        g = self._guard(action="skip")
        g.check(loss=0.5, kl=0.9, iteration=0)
        assert g._kl_win[-1] == 0.9

    def test_skip_record_appended(self):
        g = self._guard(action="skip")
        g.check(loss=0.2, kl=0.9, iteration=7, sample_info={"id": "test"})
        assert len(g.skip_records) == 1
        assert g.skip_records[0].iteration == 7
        assert g.skip_records[0].kl == 0.9

    def test_revert_action_returned(self):
        g = self._guard(action="revert")
        action = g.check(loss=0.1, kl=0.9, iteration=0)
        assert action == "revert"


# ── SpikeGuard.reset_accum ──────────────────────────────────────────────────────


def test_reset_accum():
    g = SpikeGuard(SpikeConfig())

    class FakeTrainer:
        _accum_grads = {"layer": [1, 2]}
        _accum_count = 5

    t = FakeTrainer()
    g.reset_accum(t)
    assert t._accum_grads is None
    assert t._accum_count == 0


# ── SpikeGuard._write_log ───────────────────────────────────────────────────────


def test_write_log_creates_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "spikes.jsonl"
        g = SpikeGuard(SpikeConfig(kl_threshold=0.3, log_file=str(log_path)))
        g.check(loss=0.5, kl=0.9, iteration=2)
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["iteration"] == 2


def test_write_log_appends():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "spikes.jsonl"
        g = SpikeGuard(
            SpikeConfig(
                kl_threshold=0.3, action="skip", max_consecutive=100, log_file=str(log_path)
            )
        )
        g.check(loss=0.1, kl=0.9, iteration=0)
        g.check(loss=0.1, kl=0.9, iteration=1)
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2


def test_write_log_no_file():
    """When log_file=None, no file is written."""
    g = SpikeGuard(SpikeConfig(kl_threshold=0.3))
    g.check(loss=0.1, kl=0.9, iteration=0)
    # No crash, no file written


# ── Properties ─────────────────────────────────────────────────────────────────


def test_total_skipped_property():
    g = SpikeGuard(SpikeConfig(kl_threshold=0.3, action="skip", max_consecutive=100))
    assert g.total_skipped == 0
    g.check(loss=0.1, kl=0.9, iteration=0)
    assert g.total_skipped == 1


def test_skip_records_property_returns_copy():
    g = SpikeGuard(SpikeConfig(kl_threshold=0.3, action="skip", max_consecutive=100))
    g.check(loss=0.1, kl=0.9, iteration=0)
    r = g.skip_records
    r.clear()  # Modifying return should not affect internal list
    assert g.total_skipped == 1
