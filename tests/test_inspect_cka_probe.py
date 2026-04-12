"""Tests for mlx_grpo.inspect.cka_probe — uses mocks, no MLX hardware."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlx_grpo.inspect.cka_probe import CKAProbe, CKAProbeResult
from mlx_grpo.inspect.config import build_inspect_config


def _make_cfg(**kwargs: Any) -> Any:
    defaults = {"model_path": "/fake/model", "prompt": "test prompt"}
    defaults.update(kwargs)
    return build_inspect_config(defaults)


def _make_sketches(n_layers: int = 4, dim: int = 16, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {f"layer_{i}": rng.standard_normal(dim).astype(np.float32) for i in range(n_layers)}


class _MockCollectionResult:
    def __init__(self, sketches: dict[str, np.ndarray]) -> None:
        self.sketches = sketches


class TestCKAProbeResult:
    def test_frozen(self) -> None:
        result = CKAProbeResult(
            iteration=0,
            per_layer_cka={"layer_0": 0.9},
            mean_cka=0.9,
            min_cka_layer="layer_0",
            min_cka_value=0.9,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.iteration = 1  # type: ignore[misc]

    def test_slots(self) -> None:
        assert hasattr(CKAProbeResult, "__slots__")


class TestCKAProbeLoadBase:
    def test_load_base_from_npz(self) -> None:
        sketches = _make_sketches()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **sketches)
            path = f.name

        probe = CKAProbe(MagicMock(), MagicMock(), _make_cfg(), base_activations_path=path)
        loaded = probe._load_base(path)
        assert set(loaded.keys()) == set(sketches.keys())
        for key in sketches:
            assert np.allclose(loaded[key], sketches[key], atol=1e-5)

    def test_load_base_lazy(self) -> None:
        sketches = _make_sketches()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **sketches)
            path = f.name

        probe = CKAProbe(MagicMock(), MagicMock(), _make_cfg(), base_activations_path=path)
        assert probe._base_acts is None
        _ = probe._ensure_base()
        assert probe._base_acts is not None


class TestCKAProbeCompute:
    def _make_probe_with_base(self, base_sketches: dict[str, np.ndarray]) -> tuple[CKAProbe, str]:
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **base_sketches)
            path = f.name
        return CKAProbe(MagicMock(), MagicMock(), _make_cfg(), path), path

    def test_compute_identical_activations_high_cka(self) -> None:
        base = _make_sketches(seed=42)
        probe, path = self._make_probe_with_base(base)

        with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
            mock_instance = mock_coll.return_value
            mock_instance.collect.return_value = _MockCollectionResult(sketches=base)
            result = probe.compute(iteration=5)

        assert isinstance(result, CKAProbeResult)
        assert result.iteration == 5
        assert result.mean_cka > 0.9  # identical → near 1.0
        assert result.min_cka_layer in base

    def test_compute_different_activations_lower_cka(self) -> None:
        base = _make_sketches(seed=42)
        current = _make_sketches(seed=99)  # different random activations
        probe, path = self._make_probe_with_base(base)

        with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
            mock_instance = mock_coll.return_value
            mock_instance.collect.return_value = _MockCollectionResult(sketches=current)
            result = probe.compute(iteration=0)

        assert result.mean_cka >= 0.0

    def test_compute_empty_overlap(self) -> None:
        base = {"layer_0": np.ones(16, dtype=np.float32)}
        probe, path = self._make_probe_with_base(base)

        with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
            mock_instance = mock_coll.return_value
            # Return sketches with different keys → no overlap
            mock_instance.collect.return_value = _MockCollectionResult(
                sketches={"layer_99": np.ones(16, dtype=np.float32)}
            )
            result = probe.compute(iteration=0)

        assert result.mean_cka == 0.0
        assert result.min_cka_layer == ""

    def test_per_layer_cka_keys(self) -> None:
        base = _make_sketches(n_layers=3)
        probe, path = self._make_probe_with_base(base)

        with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
            mock_instance = mock_coll.return_value
            mock_instance.collect.return_value = _MockCollectionResult(sketches=base)
            result = probe.compute()

        assert set(result.per_layer_cka.keys()) == set(base.keys())


class TestCKAProbeOnEvalHook:
    def test_hook_called_with_metrics(self) -> None:
        base = _make_sketches()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **base)
            path = f.name

        probe = CKAProbe(MagicMock(), MagicMock(), _make_cfg(), path)

        with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
            mock_instance = mock_coll.return_value
            mock_instance.collect.return_value = _MockCollectionResult(sketches=base)
            trainer_mock = MagicMock()
            probe.on_eval_hook(trainer_mock, {"iteration": 10})
            # No exception = success; iteration counter incremented
            assert probe._iteration == 11

    def test_hook_logs_to_wandb(self) -> None:
        base = _make_sketches()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **base)
            path = f.name

        wandb_mock = MagicMock()
        probe = CKAProbe(MagicMock(), MagicMock(), _make_cfg(), path, wandb_logger=wandb_mock)

        with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
            mock_instance = mock_coll.return_value
            mock_instance.collect.return_value = _MockCollectionResult(sketches=base)
            probe.on_eval_hook(MagicMock(), {"iteration": 0})

        wandb_mock.log.assert_called_once()

    def test_hook_tolerates_compute_error(self) -> None:
        base = _make_sketches()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **base)
            path = f.name

        probe = CKAProbe(MagicMock(), MagicMock(), _make_cfg(), path)

        with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
            mock_instance = mock_coll.return_value
            mock_instance.collect.side_effect = RuntimeError("GPU exploded")
            # Should not raise
            probe.on_eval_hook(MagicMock(), {"iteration": 0})

    def test_hook_default_iteration(self) -> None:
        base = _make_sketches()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **base)
            path = f.name

        probe = CKAProbe(MagicMock(), MagicMock(), _make_cfg(), path)

        with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
            mock_instance = mock_coll.return_value
            mock_instance.collect.return_value = _MockCollectionResult(sketches=base)
            # No "iteration" key in metrics
            probe.on_eval_hook(MagicMock(), {})
            assert probe._iteration == 1


class TestCKAProbeSaveBaseActivations:
    def test_save_creates_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "base.npz")
            cfg = _make_cfg(probe_base=out_path)
            probe = CKAProbe(MagicMock(), MagicMock(), cfg, out_path)

            base = _make_sketches()
            with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
                mock_instance = mock_coll.return_value
                mock_instance.collect.return_value = _MockCollectionResult(sketches=base)
                probe.save_base_activations("test prompt", out_path)

            assert Path(out_path).exists()
            loaded = np.load(out_path)
            assert "layer_0" in loaded.files

    def test_save_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "subdir" / "nested" / "base.npz")
            cfg = _make_cfg(probe_base=out_path)
            probe = CKAProbe(MagicMock(), MagicMock(), cfg, out_path)

            base = _make_sketches()
            with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
                mock_instance = mock_coll.return_value
                mock_instance.collect.return_value = _MockCollectionResult(sketches=base)
                probe.save_base_activations("test", out_path)

            assert Path(out_path).exists()


class TestCKAProbeStepHook:
    def _make_probe_with_latest(self, sketches: dict) -> CKAProbe:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **sketches)
            path = f.name
        probe = CKAProbe(MagicMock(), MagicMock(), _make_cfg(), path)
        probe.latest_result = CKAProbeResult(
            iteration=5,
            per_layer_cka={"layer_0": 0.92, "layer_1": 0.88},
            mean_cka=0.90,
            min_cka_layer="layer_1",
            min_cka_value=0.88,
        )
        return probe

    def test_step_hook_injects_cka_drift(self) -> None:
        base = _make_sketches()
        probe = self._make_probe_with_latest(base)
        metrics = MagicMock()
        metrics.extras = {}
        probe.step_hook(MagicMock(), metrics)
        assert metrics.extras["cka_drift"] == pytest.approx(0.90)
        assert metrics.extras["cka_min_layer"] == "layer_1"

    def test_step_hook_noop_when_no_latest(self) -> None:
        base = _make_sketches()
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **base)
            path = f.name
        probe = CKAProbe(MagicMock(), MagicMock(), _make_cfg(), path)
        assert probe.latest_result is None
        metrics = MagicMock()
        metrics.extras = {}
        probe.step_hook(MagicMock(), metrics)
        assert "cka_drift" not in metrics.extras

    def test_step_hook_tolerates_no_extras(self) -> None:
        base = _make_sketches()
        probe = self._make_probe_with_latest(base)
        metrics = MagicMock(spec=[])  # no 'extras' attribute
        probe.step_hook(MagicMock(), metrics)  # should not raise


class TestCKAProbePerLayerWandB:
    def test_on_eval_hook_logs_per_layer(self) -> None:
        base = _make_sketches(n_layers=3)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **base)
            path = f.name

        wandb_mock = MagicMock()
        probe = CKAProbe(MagicMock(), MagicMock(), _make_cfg(), path, wandb_logger=wandb_mock)

        with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
            mock_instance = mock_coll.return_value
            mock_instance.collect.return_value = _MockCollectionResult(sketches=base)
            probe.on_eval_hook(MagicMock(), {"iteration": 0})

        log_dict = wandb_mock.log.call_args[0][0]
        assert "inspect/cka_drift" in log_dict
        # Should also have per-layer keys
        assert any("/layer_" in k for k in log_dict)

    def test_on_eval_hook_updates_latest_result(self) -> None:
        base = _make_sketches()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **base)
            path = f.name

        probe = CKAProbe(MagicMock(), MagicMock(), _make_cfg(), path)
        assert probe.latest_result is None

        with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
            mock_instance = mock_coll.return_value
            mock_instance.collect.return_value = _MockCollectionResult(sketches=base)
            probe.on_eval_hook(MagicMock(), {"iteration": 0})

        assert probe.latest_result is not None
        assert isinstance(probe.latest_result, CKAProbeResult)

    def test_on_eval_hook_accepts_stepmetrics_like(self) -> None:
        """on_eval_hook handles objects with .iteration attribute (not just dicts)."""
        base = _make_sketches()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **base)
            path = f.name

        probe = CKAProbe(MagicMock(), MagicMock(), _make_cfg(), path)
        metrics_obj = MagicMock()
        metrics_obj.iteration = 7

        with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
            mock_instance = mock_coll.return_value
            mock_instance.collect.return_value = _MockCollectionResult(sketches=base)
            probe.on_eval_hook(MagicMock(), metrics_obj)

        assert probe._iteration == 8
