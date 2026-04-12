"""Tests for StatsLogger — per-step CSV metrics logging."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from mlx_grpo.training import StepMetrics
from mlx_grpo.training.stats_logger import COLUMNS, StatsLogger


def _make_metrics(**kwargs) -> StepMetrics:
    defaults = dict(
        iteration=0,
        loss=1.5,
        mean_reward=0.6,
        reward_std=0.2,
        kl_divergence=0.03,
        grad_norm=0.8,
        tokens_per_sec=120.0,
        generated_tokens=256,
        elapsed_sec=2.5,
        peak_memory_mb=1024.0,
        rss_mb=512.0,
        learning_rate=1e-4,
    )
    defaults.update(kwargs)
    return StepMetrics(**defaults)


class _FakeTrainer:
    pass


class TestStatsLoggerInit:
    def test_creates_file_and_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            logger.close()
            assert path.exists()
            rows = list(csv.DictReader(path.open()))
            assert len(rows) == 0  # no data rows yet

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "stats.csv"
            logger = StatsLogger(path)
            logger.close()
            assert path.exists()

    def test_appends_to_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            # First logger writes header + 1 row
            l1 = StatsLogger(path)
            l1.step_hook(_FakeTrainer(), _make_metrics(iteration=0))
            l1.close()
            # Second logger appends without rewriting header
            l2 = StatsLogger(path)
            l2.step_hook(_FakeTrainer(), _make_metrics(iteration=1))
            l2.close()
            rows = list(csv.DictReader(path.open()))
            assert len(rows) == 2
            assert rows[0]["iteration"] == "0"
            assert rows[1]["iteration"] == "1"


class TestStatsLoggerStepHook:
    def test_writes_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            logger.step_hook(_FakeTrainer(), _make_metrics(iteration=5))
            logger.close()
            rows = list(csv.DictReader(path.open()))
            assert len(rows) == 1
            assert rows[0]["iteration"] == "5"

    def test_columns_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            logger.step_hook(_FakeTrainer(), _make_metrics())
            logger.close()
            with path.open() as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames
            assert set(COLUMNS).issubset(set(header or []))

    def test_loss_formatted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            logger.step_hook(_FakeTrainer(), _make_metrics(loss=1.234567))
            logger.close()
            rows = list(csv.DictReader(path.open()))
            assert rows[0]["loss"] == "1.234567"

    def test_type_from_extras(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            metrics = _make_metrics(
                extras={"type_names": ["math"], "type_stats": {}, "advantages": [0.5]}
            )
            logger.step_hook(_FakeTrainer(), metrics)
            logger.close()
            rows = list(csv.DictReader(path.open()))
            assert rows[0]["type"] == "math"

    def test_type_defaults_to_question_mark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            logger.step_hook(_FakeTrainer(), _make_metrics())
            logger.close()
            rows = list(csv.DictReader(path.open()))
            assert rows[0]["type"] == "?"

    def test_type_stats_aggregation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            type_stats = {
                "math": {"count": 2, "correct": 1, "incorrect": 1},
                "mcq": {"count": 2, "correct": 2, "incorrect": 0},
            }
            metrics = _make_metrics(extras={"type_stats": type_stats, "advantages": []})
            logger.step_hook(_FakeTrainer(), metrics)
            logger.close()
            rows = list(csv.DictReader(path.open()))
            assert rows[0]["num_correct"] == "3"
            assert rows[0]["num_incorrect"] == "1"

    def test_mean_advantage_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            logger.step_hook(_FakeTrainer(), _make_metrics(extras={"advantages": []}))
            logger.close()
            rows = list(csv.DictReader(path.open()))
            assert rows[0]["mean_advantage"] == "0.000000"

    def test_mean_advantage_computed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            logger.step_hook(_FakeTrainer(), _make_metrics(extras={"advantages": [0.4, 0.6]}))
            logger.close()
            rows = list(csv.DictReader(path.open()))
            assert float(rows[0]["mean_advantage"]) == pytest.approx(0.5)

    def test_epoch_computed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path, epochs=2, dataset_size=10, batch_size=2)
            logger.step_hook(_FakeTrainer(), _make_metrics(iteration=4))
            logger.close()
            rows = list(csv.DictReader(path.open()))
            # steps_per_epoch = 10/2=5; epoch = (4+1)/5 = 1.0
            assert float(rows[0]["epoch"]) == pytest.approx(1.0)

    def test_total_views_in_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            metrics = _make_metrics(extras={"total_views": 6})
            logger.step_hook(_FakeTrainer(), metrics)
            logger.close()
            rows = list(csv.DictReader(path.open()))
            assert rows[0]["total_views"] == "6"

    def test_no_writer_returns_early(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            logger._writer = None  # Simulate missing writer
            # Should not raise
            logger.step_hook(_FakeTrainer(), _make_metrics())
            logger.close()


class TestStatsLoggerScaffold:
    def test_log_scaffold_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)

            class FakeProbe:
                per_ratio_accuracy = {0.25: 0.6, 0.5: 0.8, 0.75: 0.9}
                per_ratio_reward_std = {0.25: 0.1, 0.5: 0.05, 0.75: 0.02}

            logger.log_scaffold(FakeProbe(), iteration=10)
            logger.close()

            scaffold_path = Path(tmpdir) / "scaffold_accuracy.csv"
            assert scaffold_path.exists()
            rows = list(csv.DictReader(scaffold_path.open()))
            assert len(rows) == 1
            assert rows[0]["iteration"] == "10"

    def test_log_scaffold_appends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)

            class FakeProbe:
                per_ratio_accuracy = {0.5: 0.7}
                per_ratio_reward_std = {0.5: 0.1}

            logger.log_scaffold(FakeProbe(), iteration=1)
            logger.log_scaffold(FakeProbe(), iteration=2)
            logger.close()

            scaffold_path = Path(tmpdir) / "scaffold_accuracy.csv"
            rows = list(csv.DictReader(scaffold_path.open()))
            assert len(rows) == 2


class TestStatsLoggerClose:
    def test_close_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            logger.close()
            logger.close()  # Should not raise


class TestStatsLoggerCkaDrift:
    def test_cka_drift_column_exists(self) -> None:
        assert "cka_drift" in COLUMNS

    def test_cka_drift_written_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            m = _make_metrics(extras={"cka_drift": 0.9123})
            logger.step_hook(None, m)  # type: ignore[arg-type]
            logger.close()
            rows = list(csv.DictReader(path.open()))
            assert rows[0]["cka_drift"] == "0.9123"

    def test_cka_drift_empty_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            m = _make_metrics()
            logger.step_hook(None, m)  # type: ignore[arg-type]
            logger.close()
            rows = list(csv.DictReader(path.open()))
            assert rows[0]["cka_drift"] == ""
