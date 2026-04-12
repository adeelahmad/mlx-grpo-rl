"""Integration tests — end-to-end pipeline validation.

Tests the complete flow: config → data → trainer → loss → checkpoint → eval.
Uses MockBackend so no hardware required.
"""

import json
import tempfile
from pathlib import Path

from mlx_grpo.config.loader import TrainingConfig, load_config
from mlx_grpo.data import DatasetConfig, DatasetLoader, Sample
from mlx_grpo.engine.executor import RolloutExecutor
from mlx_grpo.loss import bnpo_loss, compute_advantages, dr_grpo_loss, grpo_loss, sft_loss
from mlx_grpo.training import Trainer
from mlx_grpo.training.checkpoint import CheckpointManager
from mlx_grpo.training.evaluation import EvalResult, Evaluator
from mlx_grpo.training.monitor import Monitor, MonitorConfig
from mlx_grpo.training.recovery import RecoveryManager

# ── Helpers ──


def _make_dataset(n: int = 20, type_name: str = "mcq") -> tuple[DatasetLoader, Path]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "prompt": f"What is {i} + {i}?",
                        "answer": str(i * 2),
                    }
                )
                + "\n"
            )
        f.flush()
        tmp_path = f.name
    loader = DatasetLoader(
        DatasetConfig(
            path=tmp_path,
            shuffle=False,
            type_name=type_name,
        )
    ).load()
    return loader, Path(tmp_path)


def _make_config_yaml(tmp_dir: Path, **overrides) -> Path:
    cfg = {
        "training": {
            "iterations": overrides.get("iterations", 5),
            "group_size": overrides.get("group_size", 2),
            "beta": 0.04,
            "save_every": overrides.get("save_every", 0),
            "eval_every": overrides.get("eval_every", 0),
            "loss_type": overrides.get("loss_type", "grpo"),
        },
        "generation": {
            "max_tokens": 64,
            "temperature": 0.7,
            "backend": "mock",
        },
    }
    path = tmp_dir / "test_config.yml"
    import yaml

    path.write_text(yaml.dump(cfg))
    return path


# ── End-to-end pipeline tests ──


class TestE2EPipeline:
    """Full pipeline: config → data → train → checkpoint."""

    def test_full_training_loop(self, mock_backend):
        """Train for 5 iterations and verify metrics flow."""
        dataset, _ = _make_dataset()
        config = TrainingConfig(
            iterations=5,
            group_size=3,
            save_every=0,
            eval_every=0,
        )
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)

        metrics_log = []
        trainer.on("after_step", lambda t, m: metrics_log.append(m))
        trainer.train()

        assert len(metrics_log) == 5
        for m in metrics_log:
            assert len(m.rewards) == 3  # group_size
            assert m.generated_tokens > 0
            assert m.elapsed_sec >= 0

    def test_training_with_checkpoints(self, mock_backend):
        """Train with checkpoint saving and verify training state created."""
        dataset, _ = _make_dataset()
        config = TrainingConfig(
            iterations=4,
            group_size=2,
            save_every=2,
            eval_every=0,
        )

        with tempfile.TemporaryDirectory() as ckpt_dir:
            ckpt_mgr = CheckpointManager(ckpt_dir)
            trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)
            trainer.on(
                "on_checkpoint",
                lambda t, m: ckpt_mgr.save(
                    m.iteration,
                    None,
                    metrics={"mean_reward": m.mean_reward, "loss": m.loss},
                ),
            )
            trainer.train()

            state = ckpt_mgr.load_training_state()
            assert state["iteration"] in (2, 4)  # stores next iter for resume

    def test_training_with_monitor(self, mock_backend):
        """Train with monitor and verify metrics tracked."""
        dataset, _ = _make_dataset()
        config = TrainingConfig(
            iterations=3,
            group_size=2,
            save_every=0,
            eval_every=0,
        )
        monitor = Monitor(MonitorConfig(enabled=True, log_every=100000))
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)
        trainer.on("after_step", monitor.step_hook)
        trainer.train()

        summary = monitor.summary()
        assert "reward" in summary
        assert summary["reward"]["count"] == 3

    def test_training_with_eval(self, mock_backend):
        """Train with periodic evaluation."""
        train_ds, _ = _make_dataset(20)
        eval_ds, _ = _make_dataset(10)
        config = TrainingConfig(
            iterations=4,
            group_size=2,
            save_every=0,
            eval_every=2,
        )

        eval_results = []
        trainer = Trainer(config=config, dataset=train_ds, backend=mock_backend)
        evaluator = Evaluator(trainer.executor, eval_dataset=eval_ds)
        trainer.on(
            "on_eval",
            lambda t, m: eval_results.append(
                evaluator.evaluate(iteration=m.iteration, max_samples=5)
            ),
        )
        trainer.train()

        assert len(eval_results) >= 1
        for er in eval_results:
            assert isinstance(er, EvalResult)

    def test_training_resume(self, mock_backend):
        """Train, checkpoint, and resume from checkpoint."""
        dataset, _ = _make_dataset()
        config = TrainingConfig(
            iterations=10,
            group_size=2,
            save_every=0,
            eval_every=0,
        )

        # First run: 5 iterations
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)
        iterations_run = []
        trainer.on("after_step", lambda t, m: iterations_run.append(m.iteration))
        trainer.train(resume_from=0)

        assert iterations_run == list(range(10))

        # Resume from iteration 5
        trainer2 = Trainer(config=config, dataset=dataset, backend=mock_backend)
        resumed = []
        trainer2.on("after_step", lambda t, m: resumed.append(m.iteration))
        trainer2.train(resume_from=5)

        assert resumed == [5, 6, 7, 8, 9]


# ── Loss pipeline tests ──


class TestLossPipeline:
    """Verify loss functions work with real-looking rollout data."""

    def _make_rollout_data(self, g: int = 4, t: int = 20):
        """Create synthetic log probs and masks."""
        import random

        rng = random.Random(42)
        log_probs = [[rng.gauss(-3.0, 0.5) for _ in range(t)] for _ in range(g)]
        ref_log_probs = [[rng.gauss(-3.0, 0.5) for _ in range(t)] for _ in range(g)]
        masks = [[rng.random() > 0.3 for _ in range(t)] for _ in range(g)]
        rewards = [rng.gauss(0.5, 0.3) for _ in range(g)]
        return log_probs, ref_log_probs, masks, rewards

    def test_grpo_loss_pipeline(self):
        lp, rlp, masks, rewards = self._make_rollout_data()
        advantages = compute_advantages(rewards)
        result = grpo_loss(lp, rlp, advantages, masks, beta=0.04)
        assert result.effective_tokens > 0
        assert result.loss != 0.0
        assert result.kl_divergence >= 0.0

    def test_bnpo_loss_pipeline(self):
        lp, rlp, masks, rewards = self._make_rollout_data()
        advantages = compute_advantages(rewards)
        result = bnpo_loss(lp, rlp, advantages, masks, beta=0.1, tau=0.1)
        assert result.effective_tokens > 0

    def test_dr_grpo_loss_pipeline(self):
        lp, rlp, masks, rewards = self._make_rollout_data()
        advantages = compute_advantages(rewards)
        result = dr_grpo_loss(lp, rlp, advantages, masks, beta=0.04)
        assert result.effective_tokens > 0

    def test_sft_loss(self):
        import random

        rng = random.Random(42)
        lp = [rng.gauss(-3.0, 0.5) for _ in range(20)]
        mask = [True] * 15 + [False] * 5
        loss = sft_loss(lp, mask)
        assert loss > 0.0  # Negative log probs → positive loss

    def test_advantage_clipping(self):
        rewards = [0.0, 0.5, 1.0, 2.0, 10.0]
        advs = compute_advantages(rewards, clip=2.0)
        assert all(-2.0 <= a <= 2.0 for a in advs)

    def test_empty_inputs(self):
        assert compute_advantages([]) == []
        result = grpo_loss([], [], [], [], beta=0.04)
        assert result.loss == 0.0


# ── Recovery pipeline tests ──


class TestRecoveryPipeline:
    def test_crash_recovery_flow(self):
        with tempfile.TemporaryDirectory() as d:
            rm = RecoveryManager(d)
            state = rm.load_or_create(config_hash="abc123")

            # Simulate training progress
            state.iteration = 50
            state.best_reward = 0.7
            rm.save_state(state)

            # Simulate crash
            can_continue = rm.record_crash(state, "OOM error")
            assert can_continue  # First crash

            # Reload
            loaded = rm.load_or_create(config_hash="abc123")
            assert loaded.iteration == 50
            assert loaded.crash_count == 1
            assert rm.should_resume(loaded) is False  # No checkpoint yet

            # With checkpoint
            loaded.last_checkpoint = "checkpoint-000050"
            rm.save_state(loaded)
            loaded2 = rm.load_or_create(config_hash="abc123")
            assert rm.should_resume(loaded2)

    def test_config_change_resets(self):
        with tempfile.TemporaryDirectory() as d:
            rm = RecoveryManager(d)
            state = rm.load_or_create(config_hash="v1")
            state.iteration = 100
            rm.save_state(state)

            # Different config → fresh start
            new_state = rm.load_or_create(config_hash="v2")
            assert new_state.iteration == 0


# ── Evaluator standalone tests ──


class TestEvaluator:
    def test_eval_with_mock_backend(self, mock_backend):
        dataset, _ = _make_dataset(10)
        executor = RolloutExecutor(mock_backend)
        evaluator = Evaluator(executor, eval_dataset=dataset)
        result = evaluator.evaluate(max_samples=5)
        assert isinstance(result, EvalResult)
        assert result.total_samples <= 5

    def test_eval_with_static_samples(self, mock_backend):
        samples = [
            Sample(prompt="What is 2+2?", answer="4", type_name="mcq"),
            Sample(prompt="What is 3+3?", answer="6", type_name="mcq"),
        ]
        executor = RolloutExecutor(mock_backend)
        evaluator = Evaluator(executor, eval_samples=samples)
        result = evaluator.evaluate(max_samples=10)
        assert result.total_samples <= 2

    def test_eval_per_type_tracking(self, mock_backend):
        samples = [
            Sample(prompt="Q1?", answer="A1", type_name="mcq"),
            Sample(prompt="Q2?", answer="A2", type_name="mcq"),
        ]
        executor = RolloutExecutor(mock_backend)
        evaluator = Evaluator(executor, eval_samples=samples)
        result = evaluator.evaluate()
        if result.total_samples > 0:
            assert "mcq" in result.per_type_reward


# ── Config → Training integration ──


class TestConfigIntegration:
    def test_config_loads_and_creates_trainer(self, mock_backend):
        with tempfile.TemporaryDirectory() as d:
            cfg_path = _make_config_yaml(Path(d))
            config = load_config(cfg_path)
            dataset, _ = _make_dataset(5)

            trainer = Trainer(
                config=config.training,
                dataset=dataset,
                backend=mock_backend,
            )
            assert trainer.config.iterations == 5
            assert trainer.config.group_size == 2

    def test_gradient_step_with_mock(self, mock_backend):
        """Verify _gradient_step returns valid metrics even with mock backend."""
        dataset, _ = _make_dataset(5)
        config = TrainingConfig(
            iterations=1,
            group_size=2,
            save_every=0,
            eval_every=0,
        )
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)

        captured = []
        trainer.on("after_step", lambda t, m: captured.append(m))
        trainer.train()

        m = captured[0]
        # With mock backend, we get reward-only step
        assert isinstance(m.loss, float)
        assert isinstance(m.kl_divergence, float)


# ── Cross-sampling integration ──


class TestCrossSamplingIntegration:
    def test_augmented_samples_work_with_trainer(self, mock_backend):
        from mlx_grpo.data.cross_sampling import CrossSampleConfig, CrossSampler

        dataset, _ = _make_dataset(20)
        sampler = CrossSampler(
            list(dataset),
            CrossSampleConfig(ratio=0.5, seed=42),
        )

        # Augment and verify samples still have required fields
        for s in list(dataset)[:5]:
            augmented = sampler.augment(s)
            assert augmented.prompt
            assert augmented.answer is not None
            assert augmented.type_name == "mcq"
