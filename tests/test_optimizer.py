"""Tests for configurable optimizer & LR schedule builder."""

import mlx.optimizers as optim
import pytest

from mlx_grpo.config.loader import TrainingConfig


def _cfg(**overrides) -> TrainingConfig:
    """Build a TrainingConfig with overrides."""
    from dataclasses import replace

    base = TrainingConfig(learning_rate=1e-4)
    return replace(base, **overrides)


def _has_schedule(opt: optim.Optimizer) -> bool:
    """Check if optimizer has an LR schedule attached."""
    return "learning_rate" in opt._schedulers


class TestBuildOptimizer:
    def test_default_adam(self):
        from mlx_grpo.training.optimizer import build_optimizer

        opt = build_optimizer(_cfg())
        assert isinstance(opt, optim.Adam)

    def test_adamw_with_kwargs(self):
        from mlx_grpo.training.optimizer import build_optimizer

        opt = build_optimizer(_cfg(optimizer={"name": "adamw", "weight_decay": 0.05}))
        assert isinstance(opt, optim.AdamW)

    def test_sgd(self):
        from mlx_grpo.training.optimizer import build_optimizer

        opt = build_optimizer(_cfg(optimizer={"name": "sgd", "momentum": 0.9}))
        assert isinstance(opt, optim.SGD)

    def test_lion(self):
        from mlx_grpo.training.optimizer import build_optimizer

        opt = build_optimizer(_cfg(optimizer={"name": "lion"}))
        assert isinstance(opt, optim.Lion)

    def test_adagrad(self):
        from mlx_grpo.training.optimizer import build_optimizer

        opt = build_optimizer(_cfg(optimizer={"name": "adagrad"}))
        assert isinstance(opt, optim.Adagrad)

    def test_adafactor(self):
        from mlx_grpo.training.optimizer import build_optimizer

        opt = build_optimizer(_cfg(optimizer={"name": "adafactor"}))
        assert isinstance(opt, optim.Adafactor)

    def test_adamax(self):
        from mlx_grpo.training.optimizer import build_optimizer

        opt = build_optimizer(_cfg(optimizer={"name": "adamax"}))
        assert isinstance(opt, optim.Adamax)

    def test_unknown_raises(self):
        from mlx_grpo.training.optimizer import build_optimizer

        with pytest.raises(ValueError, match="Unknown optimizer 'foo'"):
            build_optimizer(_cfg(optimizer={"name": "foo"}))

    def test_case_insensitive(self):
        from mlx_grpo.training.optimizer import build_optimizer

        opt = build_optimizer(_cfg(optimizer={"name": "AdamW"}))
        assert isinstance(opt, optim.AdamW)


class TestLRSchedule:
    def test_constant_lr(self):
        from mlx_grpo.training.optimizer import build_optimizer

        opt = build_optimizer(_cfg())
        assert not _has_schedule(opt)
        assert float(opt.learning_rate) == pytest.approx(1e-4, rel=1e-5)

    def test_cosine_schedule(self):
        from mlx_grpo.training.optimizer import build_optimizer

        opt = build_optimizer(
            _cfg(
                lr_schedule={
                    "name": "cosine_decay",
                    "decay_steps": 1000,
                }
            )
        )
        assert _has_schedule(opt)

    def test_warmup_plus_decay(self):
        from mlx_grpo.training.optimizer import build_optimizer

        opt = build_optimizer(
            _cfg(
                lr_schedule={
                    "name": "cosine_decay",
                    "decay_steps": 500,
                    "warmup_steps": 100,
                }
            )
        )
        assert _has_schedule(opt)
        # Initial LR should be near 0 (warmup starts from 0)
        assert float(opt.learning_rate) < 1e-5

    def test_linear_schedule(self):
        from mlx_grpo.training.optimizer import build_optimizer

        opt = build_optimizer(
            _cfg(
                lr_schedule={
                    "name": "linear",
                    "end_lr": 1e-7,
                    "steps": 500,
                }
            )
        )
        assert _has_schedule(opt)

    def test_step_decay(self):
        from mlx_grpo.training.optimizer import build_optimizer

        opt = build_optimizer(
            _cfg(
                lr_schedule={
                    "name": "step_decay",
                    "decay_rate": 0.5,
                    "step_size": 100,
                }
            )
        )
        assert _has_schedule(opt)

    def test_exponential_decay(self):
        from mlx_grpo.training.optimizer import build_optimizer

        opt = build_optimizer(
            _cfg(
                lr_schedule={
                    "name": "exponential_decay",
                    "decay_rate": 0.99,
                }
            )
        )
        assert _has_schedule(opt)

    def test_unknown_schedule_raises(self):
        from mlx_grpo.training.optimizer import build_optimizer

        with pytest.raises(ValueError, match="Unknown LR schedule"):
            build_optimizer(_cfg(lr_schedule={"name": "unknown"}))


class TestDescribeOptimizer:
    def test_describe_default(self):
        from mlx_grpo.training.optimizer import describe_optimizer

        result = describe_optimizer(optim.Adam(learning_rate=1e-4), _cfg())
        assert "adam" in result.lower()
        assert "1e-04" in result or "0.0001" in result

    def test_describe_with_schedule(self):
        from mlx_grpo.training.optimizer import describe_optimizer

        cfg = _cfg(
            optimizer={"name": "adamw", "weight_decay": 0.01},
            lr_schedule={"name": "cosine_decay", "decay_steps": 1000, "warmup_steps": 50},
        )
        result = describe_optimizer(optim.AdamW(learning_rate=1e-4), cfg)
        assert "adamw" in result.lower()
        assert "cosine_decay" in result
        assert "warmup=50" in result
