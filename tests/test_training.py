"""Tests for training loop and checkpoint manager."""

import json
import tempfile
from pathlib import Path

import pytest

from mlx_grpo.config.loader import TrainingConfig
from mlx_grpo.data import DatasetConfig, DatasetLoader
from mlx_grpo.training import Trainer
from mlx_grpo.training.checkpoint import CheckpointManager
from mlx_grpo.training.monitor import MetricsWindow, Monitor, MonitorConfig

# ── Trainer tests ──


class TestTrainer:
    def _make_dataset(self) -> DatasetLoader:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(10):
                f.write(json.dumps({"prompt": f"Q{i}?", "answer": "A"}) + "\n")
            f.flush()
            path = f.name
        return DatasetLoader(DatasetConfig(path=path, shuffle=False)).load()

    def test_runs_without_crash(self, mock_backend):
        config = TrainingConfig(iterations=3, group_size=2, save_every=0, eval_every=0)
        dataset = self._make_dataset()
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)
        trainer.train()
        assert trainer.iteration == 2  # 0-indexed last iteration

    def test_hooks_fire(self, mock_backend):
        config = TrainingConfig(iterations=2, group_size=2, save_every=0, eval_every=0)
        dataset = self._make_dataset()
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)

        fired = []
        trainer.on("after_step", lambda t, m: fired.append(m.iteration))
        trainer.train()
        assert fired == [0, 1]

    def test_metrics_populated(self, mock_backend):
        config = TrainingConfig(iterations=1, group_size=4, save_every=0, eval_every=0)
        dataset = self._make_dataset()
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)

        captured = []
        trainer.on("after_step", lambda t, m: captured.append(m))
        trainer.train()

        m = captured[0]
        assert m.iteration == 0
        assert len(m.rewards) == 4  # group_size
        assert m.generated_tokens > 0

    def test_progress(self, mock_backend):
        config = TrainingConfig(iterations=10, group_size=2, save_every=0, eval_every=0)
        dataset = self._make_dataset()
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)
        assert trainer.progress == 0.0


# ── Checkpoint tests ──


class TestCheckpointManager:
    def test_save_creates_training_state(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = CheckpointManager(d, keep_recent=5, keep_best=5)
            mgr.save(1, model=None, metrics={"mean_reward": 0.5})
            mgr.save(2, model=None, metrics={"mean_reward": 0.8})
            state = mgr.load_training_state()
            assert state["iteration"] == 3  # next iteration for resume
            assert state["metric_mean_reward"] == 0.8

    def test_training_state_fields(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = CheckpointManager(d)
            mgr.save(
                10,
                model=None,
                learning_rate=1e-5,
                wandb_run_id="abc123",
                metrics={"mean_reward": 0.9, "loss": 0.1},
            )
            state = mgr.load_training_state()
            assert state["iteration"] == 11  # next iteration for resume
            assert state["learning_rate"] == 1e-5
            assert state["wandb_run_id"] == "abc123"
            assert state["metric_mean_reward"] == 0.9
            assert state["metric_loss"] == 0.1

    def test_save_config(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = CheckpointManager(d)
            mgr.save_config({"model": "test", "lr": 1e-5})
            path = Path(d) / "adapter_config.json"
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["model"] == "test"

    def test_save_config_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = CheckpointManager(d)
            mgr.save_config({"version": 1})
            mgr.save_config({"version": 2})  # Should NOT overwrite
            data = json.loads((Path(d) / "adapter_config.json").read_text())
            assert data["version"] == 1

    def test_load_empty_training_state(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = CheckpointManager(d)
            state = mgr.load_training_state()
            assert state == {}


# ── Monitor tests ──


class TestMetricsWindow:
    def test_windowed_average(self):
        w = MetricsWindow(size=3)
        for v in [1.0, 2.0, 3.0, 4.0]:
            w.add(v)
        assert w.mean == pytest.approx(3.0)  # [2, 3, 4]
        assert w.latest == 4.0
        assert w.min == 2.0
        assert w.max == 4.0

    def test_empty(self):
        w = MetricsWindow()
        assert w.mean == 0.0
        assert len(w) == 0


class TestMonitor:
    def test_track_and_summary(self):
        mon = Monitor(MonitorConfig(enabled=False))
        mon.track("loss", 1.0)
        mon.track("loss", 0.5)
        s = mon.summary()
        assert s["loss"]["mean"] == 0.75
        assert s["loss"]["count"] == 2

    def test_kl_warning(self, capsys):
        mon = Monitor(MonitorConfig(kl_warning=0.01, kl_critical=0.1))

        # Simulate KL check via internal method
        class FakeTrainer:
            config = TrainingConfig()

        mon._check_kl(0.05, FakeTrainer())  # type: ignore
        captured = capsys.readouterr()
        assert "KL warning" in captured.err

    def test_kl_critical_stop(self):
        mon = Monitor(MonitorConfig(kl_critical=0.01, stop_on_critical=True))

        class FakeTrainer:
            config = TrainingConfig()

        with pytest.raises(RuntimeError, match="KL divergence"):
            mon._check_kl(0.5, FakeTrainer())  # type: ignore


# ── Snap-to-boundary token truncation ──


class _FakeTokenizer:
    """Char-level tokenizer: each token is one ASCII char."""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class TestSnapCut:
    """Token-level sentence boundary snapping via Trainer._snap_cut."""

    def _make_trainer(self):
        ds = DatasetLoader(
            DatasetConfig(
                path="/dev/null",
                shuffle=False,
            )
        )
        # Force-load with empty data to avoid file read
        ds._samples = []
        t = Trainer(
            config=TrainingConfig(iterations=1),
            dataset=ds,
            backend=None,
            tokenizer=_FakeTokenizer(),
            snap_to_boundary=True,
        )
        return t

    def test_prefix_snaps_to_period(self):
        t = self._make_trainer()
        # "Hello. World" → snap before token 10 should land at 7 ("Hello. ")
        comp = _FakeTokenizer().encode("Hello. World!!")
        snapped = t._snap_cut(comp, 10, direction="before")
        text = _FakeTokenizer().decode(comp[:snapped])
        assert text.rstrip()[-1] == "."

    def test_prefix_no_boundary_keeps_original(self):
        t = self._make_trainer()
        comp = _FakeTokenizer().encode("abcdefghij")
        assert t._snap_cut(comp, 5, direction="before") == 5

    def test_postfix_snaps_start_forward(self):
        t = self._make_trainer()
        # "First. Second. Third." → keep=10 from end, snap start to boundary
        comp = _FakeTokenizer().encode("First. Second. Third.")
        result = t._snap_cut(comp, 10, direction="after")
        assert result > 0  # adjusted

    def test_snap_respects_question_mark(self):
        t = self._make_trainer()
        comp = _FakeTokenizer().encode("Why? Because. Done.")
        snapped = t._snap_cut(comp, 6, direction="before")
        text = _FakeTokenizer().decode(comp[:snapped])
        assert text.rstrip()[-1] in ".?"

    def test_snap_respects_newline(self):
        t = self._make_trainer()
        comp = _FakeTokenizer().encode("Line one\nLine two")
        snapped = t._snap_cut(comp, 12, direction="before")
        text = _FakeTokenizer().decode(comp[:snapped])
        assert text.endswith("\n")
