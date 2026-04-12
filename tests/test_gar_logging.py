"""Tests for GAR metrics logging across W&B, CSV, TUI, and JSONL backends."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlx_grpo.engine.result import RolloutResult, Segment, SegmentSource
from mlx_grpo.training import StepMetrics
from mlx_grpo.training.rollout_logger import RolloutLogger, StepContext
from mlx_grpo.training.stats_logger import COLUMNS, StatsLogger
from mlx_grpo.training.wandb_logger import WandbLogger

# ── Fixtures ──────────────────────────────────────────────────────────────────

_GAR_EXTRAS = {
    "gar_loss": 0.042,
    "gar_angles": {
        "20": {
            "angle_mean": 3.2,
            "angle_std": 1.5,
            "angle_max": 8.1,
            "loss": 0.012,
            "tokens_above": 12,
            "tokens_below": 28,
        },
        "21": {
            "angle_mean": 2.8,
            "angle_std": 1.1,
            "angle_max": 6.4,
            "loss": 0.010,
            "tokens_above": 8,
            "tokens_below": 32,
        },
    },
    "gar_scaffold_fraction": 0.65,
}


def _metrics(**extra_extras) -> StepMetrics:
    extras = dict(_GAR_EXTRAS)
    extras.update(extra_extras)
    return StepMetrics(
        iteration=1,
        loss=0.5,
        mean_reward=0.6,
        reward_std=0.1,
        kl_divergence=0.02,
        grad_norm=0.4,
        tokens_per_sec=100.0,
        generated_tokens=128,
        elapsed_sec=1.0,
        peak_memory_mb=512.0,
        rss_mb=256.0,
        learning_rate=1e-4,
        extras=extras,
    )


def _grpo_metrics() -> StepMetrics:
    """GRPO mode: no gar_loss key."""
    return StepMetrics(
        iteration=0,
        loss=0.3,
        mean_reward=0.7,
        reward_std=0.1,
        kl_divergence=0.01,
        grad_norm=0.3,
        tokens_per_sec=80.0,
        generated_tokens=64,
        elapsed_sec=0.5,
        peak_memory_mb=256.0,
        rss_mb=128.0,
        learning_rate=1e-4,
    )


def _make_rollout(with_gar: bool = False) -> RolloutResult:
    seg = Segment(
        tokens=[1, 2, 3],
        text="test output",
        source=SegmentSource.GENERATED,
        masked=False,
        tag="answer",
    )
    r = RolloutResult(segments=[seg], type_name="math")
    if with_gar:
        r.gar_activation_drift = {
            "20": {"angle_mean": 3.2, "angle_std": 1.5, "angle_max": 8.1},
        }
    return r


# ── W&B tests ────────────────────────────────────────────────────────────────


class TestGarWandbLogging:
    def _make_logger(self):
        with patch("wandb.init"):
            logger = WandbLogger.__new__(WandbLogger)
            logger._project = "test"
            logger._config = {}
            logger._run_id = None
            logger._tags = None
            logger._run = MagicMock()
            logger._available = True
        return logger

    def test_gar_wandb_logs_loss_metric(self):
        logger = self._make_logger()
        m = _metrics()
        with patch("wandb.log") as mock_log:
            logger.log_step(m, step=1)

        # Find the gar/loss log call
        def _has_gar_loss(c):  # noqa: SIM115
            return "gar/loss" in (c.args[0] if c.args else {})

        gar_calls = [c for c in mock_log.call_args_list if _has_gar_loss(c)]
        assert gar_calls, "Expected wandb.log call with gar/loss key"
        assert gar_calls[0].args[0]["gar/loss"] == pytest.approx(0.042)

    def test_gar_wandb_logs_scaffold_fraction(self):
        logger = self._make_logger()
        m = _metrics()
        with patch("wandb.log") as mock_log:
            logger.log_step(m, step=1)

        def _has_gar_loss(c):  # noqa: SIM115
            return "gar/loss" in (c.args[0] if c.args else {})

        gar_calls = [c for c in mock_log.call_args_list if _has_gar_loss(c)]
        assert gar_calls[0].args[0]["gar/scaffold_fraction"] == pytest.approx(0.65)

    def test_gar_wandb_logs_per_layer_angles(self):
        logger = self._make_logger()
        m = _metrics()
        with patch("wandb.log") as mock_log:
            logger.log_step(m, step=1)
        all_keys: set[str] = set()
        for c in mock_log.call_args_list:
            if c.args:
                all_keys.update(c.args[0].keys())
        assert "gar/angle_mean/20" in all_keys
        assert "gar/angle_std/20" in all_keys
        assert "gar/angle_max/20" in all_keys
        assert "gar/tokens_above_threshold/20" in all_keys
        assert "gar/tokens_below_threshold/20" in all_keys
        assert "gar/loss_per_layer/20" in all_keys

    def test_grpo_mode_no_gar_logging(self):
        logger = self._make_logger()
        m = _grpo_metrics()
        with patch("wandb.log") as mock_log:
            logger.log_step(m, step=0)
        all_keys: set[str] = set()
        for c in mock_log.call_args_list:
            if c.args:
                all_keys.update(c.args[0].keys())
        assert not any(k.startswith("gar/") for k in all_keys)

    def test_legacy_dict_path_unchanged(self):
        logger = self._make_logger()
        with patch("wandb.log") as mock_log:
            logger.log_step({"loss": 0.5, "reward": 0.8}, step=5)
        mock_log.assert_called_once_with({"loss": 0.5, "reward": 0.8}, step=5)


# ── CSV tests ─────────────────────────────────────────────────────────────────


class _FakeTrainer:
    pass


class TestGarCsvLogging:
    def test_gar_csv_columns_in_schema(self):
        assert "gar_loss" in COLUMNS
        assert "gar_angle_avg" in COLUMNS
        assert "gar_scaffold_fraction" in COLUMNS

    def test_gar_csv_has_gar_loss_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            logger.step_hook(_FakeTrainer(), _metrics())
            logger.close()
            rows = list(csv.DictReader(path.open()))
            assert float(rows[0]["gar_loss"]) == pytest.approx(0.042, rel=1e-4)

    def test_gar_csv_has_gar_angle_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            logger.step_hook(_FakeTrainer(), _metrics())
            logger.close()
            rows = list(csv.DictReader(path.open()))
            # Average of 3.2 and 2.8 = 3.0
            assert float(rows[0]["gar_angle_avg"]) == pytest.approx(3.0, rel=1e-3)

    def test_gar_csv_has_scaffold_fraction_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            logger.step_hook(_FakeTrainer(), _metrics())
            logger.close()
            rows = list(csv.DictReader(path.open()))
            assert float(rows[0]["gar_scaffold_fraction"]) == pytest.approx(0.65, rel=1e-3)

    def test_grpo_mode_gar_columns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            logger = StatsLogger(path)
            logger.step_hook(_FakeTrainer(), _grpo_metrics())
            logger.close()
            rows = list(csv.DictReader(path.open()))
            assert rows[0]["gar_loss"] == ""
            assert rows[0]["gar_angle_avg"] == ""
            assert rows[0]["gar_scaffold_fraction"] == ""


# ── Monitor / TUI tests ───────────────────────────────────────────────────────


class TestGarMonitorDisplay:
    def _make_monitor(self):
        from mlx_grpo.training.monitor import Monitor, MonitorConfig

        cfg = MonitorConfig(enabled=True)
        mon = Monitor(cfg)
        return mon

    def _make_trainer(self, iterations: int = 10):
        t = MagicMock()
        t.config.iterations = iterations
        return t

    def test_gar_monitor_displays_gar_section(self):
        from mlx_grpo.training.monitor import Monitor, MonitorConfig

        mon = Monitor(MonitorConfig(enabled=True))
        mon._log_fn = MagicMock()
        trainer = self._make_trainer()
        mon.step_hook(trainer, _metrics())
        logged = mon._log_fn.call_args[0][0]
        assert "gar=" in logged

    def test_gar_monitor_no_gar_in_grpo_mode(self):
        from mlx_grpo.training.monitor import Monitor, MonitorConfig

        mon = Monitor(MonitorConfig(enabled=True))
        mon._log_fn = MagicMock()
        trainer = self._make_trainer()
        mon.step_hook(trainer, _grpo_metrics())
        logged = mon._log_fn.call_args[0][0]
        assert "gar=" not in logged

    def test_gar_rich_section_added(self):
        """Rich render includes GAR rows when gar_loss present."""
        from mlx_grpo.training.monitor import _RICH, Monitor, MonitorConfig

        if not _RICH:
            pytest.skip("rich not installed")

        mon = Monitor(MonitorConfig(enabled=True))
        with patch("mlx_grpo.training.monitor.Live"):
            mon._console = MagicMock()
            prog = MagicMock()
            prog.add_task.return_value = 0
            mon._progress = prog
            mon._task_id = 0
            mon._repeat_task_id = 1
            mon._live = MagicMock()
            captured: list = []

            def fake_update(renderable):
                captured.append(renderable)

            mon._live.update = fake_update

            mon._render_rich(
                metrics=_metrics(),
                total=10,
                loss_avg=0.5,
                rew_avg=0.6,
                tps=100.0,
                mem=512.0,
                lr=1e-4,
                views=4,
                type_names="math",
                gar_angles=_GAR_EXTRAS["gar_angles"],
                gar_loss=0.042,
                gar_threshold=5.0,
            )
        assert captured  # _live.update was called

    def test_gar_monitor_marks_exceeded_threshold(self):
        """Layers with angle_mean > threshold get star marker in rich output."""
        from mlx_grpo.training.monitor import _RICH, Monitor, MonitorConfig

        if not _RICH:
            pytest.skip("rich not installed")

        mon = Monitor(MonitorConfig(enabled=True))
        mon._console = MagicMock()
        prog = MagicMock()
        prog.add_task.return_value = 0
        mon._progress = prog
        mon._task_id = 0
        mon._repeat_task_id = 1
        mon._live = MagicMock()

        captured_renderables: list = []
        mon._live.update = lambda r: captured_renderables.append(r)

        # Layer 20 angle_mean=8.1 > threshold=5.0, layer 21 angle_mean=2.8 < 5.0
        gar_angles = {
            "20": {"angle_mean": 8.1, "angle_std": 1.0, "angle_max": 9.0},
            "21": {"angle_mean": 2.8, "angle_std": 0.5, "angle_max": 4.0},
        }
        mon._render_rich(
            metrics=_grpo_metrics(),
            total=10,
            loss_avg=0.3,
            rew_avg=0.7,
            tps=80.0,
            mem=256.0,
            lr=1e-4,
            views=4,
            type_names="math",
            gar_angles=gar_angles,
            gar_loss=0.03,
            gar_threshold=5.0,
        )
        # Verify render was called (star logic runs — no exception thrown)
        assert captured_renderables

    def test_gar_monitor_shows_per_layer_drifts(self):
        """_render_rich handles up to 10 layers without error."""
        from mlx_grpo.training.monitor import _RICH, Monitor, MonitorConfig

        if not _RICH:
            pytest.skip("rich not installed")

        mon = Monitor(MonitorConfig(enabled=True))
        mon._console = MagicMock()
        prog = MagicMock()
        prog.add_task.return_value = 0
        mon._progress = prog
        mon._task_id = 0
        mon._repeat_task_id = 1
        mon._live = MagicMock()
        mon._live.update = lambda r: None

        gar_angles = {
            str(i): {"angle_mean": float(i), "angle_std": 0.5, "angle_max": float(i) + 1}
            for i in range(20, 32)
        }  # 12 layers — only top 10 shown
        mon._render_rich(
            metrics=_grpo_metrics(),
            total=10,
            loss_avg=0.3,
            rew_avg=0.7,
            tps=80.0,
            mem=256.0,
            lr=1e-4,
            views=4,
            type_names="math",
            gar_angles=gar_angles,
            gar_loss=0.05,
            gar_threshold=5.0,
        )


# ── JSONL tests ───────────────────────────────────────────────────────────────


class TestGarJsonlLogging:
    def test_gar_jsonl_sets_gar_mode_flag(self):
        r = _make_rollout(with_gar=True)
        entry = r.to_log_entry(include_tokens=False)
        assert entry["gar_mode"] is True

    def test_grpo_mode_no_gar_mode_flag(self):
        r = _make_rollout(with_gar=False)
        entry = r.to_log_entry(include_tokens=False)
        assert entry["gar_mode"] is False

    def test_gar_jsonl_has_activation_drift_dict(self):
        r = _make_rollout(with_gar=True)
        entry = r.to_log_entry(include_tokens=False)
        assert "gar_activation_drift" in entry
        assert "20" in entry["gar_activation_drift"]

    def test_gar_jsonl_no_activation_drift_in_grpo(self):
        r = _make_rollout(with_gar=False)
        entry = r.to_log_entry(include_tokens=False)
        assert "gar_activation_drift" not in entry

    def test_gar_jsonl_reward_advance_set_to_zero(self):
        """RolloutLogger sets reward=0.0 and advantage=0.0 for gar_mode entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path, log_tokens=False)
            r = _make_rollout(with_gar=True)
            logger.log(
                iteration=1,
                rollouts=[r],
                rewards=[0.9],  # Would normally be written
                advantages=[0.5],  # Would normally be written
            )
            logger.close()
            lines = path.read_text().strip().splitlines()
            entry = json.loads(lines[0])
            assert entry["reward"] == 0.0
            assert entry["advantage"] == 0.0

    def test_gar_jsonl_grpo_reward_unchanged(self):
        """In GRPO mode, reward and advantage pass through unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path, log_tokens=False)
            r = _make_rollout(with_gar=False)
            logger.log(
                iteration=0,
                rollouts=[r],
                rewards=[0.7],
                advantages=[0.3],
            )
            logger.close()
            lines = path.read_text().strip().splitlines()
            entry = json.loads(lines[0])
            assert entry["reward"] == pytest.approx(0.7)
            assert entry["advantage"] == pytest.approx(0.3)

    def test_gar_jsonl_includes_metadata(self):
        """StepContext with gar_threshold_deg and gar_normalize writes to step.metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            logger = RolloutLogger(path, log_tokens=False)
            ctx = StepContext(
                iteration=1,
                total_iterations=100,
                loss=0.5,
                gar_threshold_deg=5.0,
                gar_normalize=True,
            )
            r = _make_rollout(with_gar=True)
            logger.log(iteration=1, rollouts=[r], rewards=[0.0], advantages=[0.0], step_ctx=ctx)
            logger.close()
            lines = path.read_text().strip().splitlines()
            entry = json.loads(lines[0])
            assert entry["step"]["metadata"]["gar_threshold_deg"] == pytest.approx(5.0)
            assert entry["step"]["metadata"]["gar_normalize"] is True
