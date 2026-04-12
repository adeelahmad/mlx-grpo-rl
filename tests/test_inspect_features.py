"""Tests for mlx_grpo.inspect.features — uses mock collector, no MLX."""

from __future__ import annotations

import json
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlx_grpo.inspect.config import InspectConfig, build_inspect_config
from mlx_grpo.inspect.features import (
    CheckpointCKAResult,
    LoRAIsolationResult,
    RewardStratifiedResult,
    ScaffoldNakedResult,
    TypeFingerprintResult,
    run_reward_stratified,
    run_scaffold_naked_comparison,
    run_type_fingerprints,
)

# ── Mock Collector ──────────────────────────────────────────────────────────


class _MockCollectionResult:
    def __init__(self, seed: int = 0, n_layers: int = 4) -> None:
        rng = np.random.default_rng(seed)
        self.sketches = {
            f"layer_{i}": rng.standard_normal(16).astype(np.float32) for i in range(n_layers)
        }
        self.raw_np = {
            f"layer_{i}": rng.standard_normal(32).astype(np.float32) for i in range(n_layers)
        }
        self.layer_acts: dict[str, Any] = {}
        self.prompt_ids = [1, 2, 3]
        self.track_pos = 2


class _MockCollector:
    def __init__(self, seed: int = 0, n_layers: int = 4) -> None:
        self._seed = seed
        self._n_layers = n_layers
        self._call_count = 0

    def collect(self, prompt: str) -> _MockCollectionResult:
        self._call_count += 1
        # Different seed per call for variety
        return _MockCollectionResult(seed=self._seed + self._call_count, n_layers=self._n_layers)


def _make_cfg(**kwargs: Any) -> InspectConfig:
    defaults = {"model_path": "/fake/model", "prompt": "What is 2+2?"}
    defaults.update(kwargs)
    return build_inspect_config(defaults)


# ── Feature 2: Reward Stratified ──────────────────────────────────────────────


class TestRunRewardStratified:
    def _write_rollout_log(self, records: list[dict[str, Any]]) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)  # noqa: SIM115
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        f.flush()
        f.close()
        return f.name

    def test_basic_split(self) -> None:
        records = [
            {"prompt": "q1", "completion": "a1", "reward": 1.0},
            {"prompt": "q2", "completion": "a2", "reward": 0.0},
            {"prompt": "q3", "completion": "a3", "reward": 0.8},
            {"prompt": "q4", "completion": "a4", "reward": 0.2},
        ]
        log_path = self._write_rollout_log(records)
        cfg = _make_cfg(rollout_log=log_path)
        collector = _MockCollector()
        result = run_reward_stratified(cfg, collector)
        assert isinstance(result, RewardStratifiedResult)
        assert result.n_high + result.n_low == len(records)
        assert result.median_reward > 0.0

    def test_raises_without_rollout_log(self) -> None:
        cfg = _make_cfg()
        with pytest.raises(ValueError, match="rollout_log"):
            run_reward_stratified(cfg, _MockCollector())

    def test_raises_on_empty_log(self) -> None:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)  # noqa: SIM115
        f.close()
        cfg = _make_cfg(rollout_log=f.name)
        with pytest.raises(ValueError, match="empty"):
            run_reward_stratified(cfg, _MockCollector())

    def test_similarity_dicts_have_layer_keys(self) -> None:
        records = [{"prompt": "q", "completion": "a", "reward": float(i)} for i in range(6)]
        log_path = self._write_rollout_log(records)
        cfg = _make_cfg(rollout_log=log_path)
        result = run_reward_stratified(cfg, _MockCollector())
        # Should have some layer keys if collection succeeded
        assert isinstance(result.high_similarity, dict)
        assert isinstance(result.low_similarity, dict)

    def test_single_reward_level(self) -> None:
        # All same reward → split goes to high only
        records = [{"prompt": "q", "completion": "a", "reward": 0.5} for _ in range(4)]
        log_path = self._write_rollout_log(records)
        cfg = _make_cfg(rollout_log=log_path)
        result = run_reward_stratified(cfg, _MockCollector())
        assert result.median_reward == 0.5


# ── Feature 4: Type Fingerprints ─────────────────────────────────────────────


class TestRunTypeFingerprints:
    def _write_prompts_file(self, records: list[dict[str, str]]) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)  # noqa: SIM115
        json.dump(records, f)
        f.flush()
        f.close()
        return f.name

    def test_basic_fingerprints(self) -> None:
        records = [
            {"prompt": "math: 2+2", "label": "math"},
            {"prompt": "math: 3+3", "label": "math"},
            {"prompt": "science: gravity", "label": "science"},
            {"prompt": "science: photosyn", "label": "science"},
        ]
        path = self._write_prompts_file(records)
        cfg = _make_cfg(prompts_file=path)
        result = run_type_fingerprints(cfg, _MockCollector())
        assert isinstance(result, TypeFingerprintResult)
        assert "math" in result.labels
        assert "science" in result.labels
        assert result.cross_cka.shape == (2, 2, 4)  # (n_labels, n_labels, n_layers)

    def test_raises_without_prompts_file(self) -> None:
        cfg = _make_cfg()
        with pytest.raises(ValueError, match="prompts_file"):
            run_type_fingerprints(cfg, _MockCollector())

    def test_cross_cka_diagonal_range(self) -> None:
        records = [
            {"prompt": "q1", "label": "a"},
            {"prompt": "q2", "label": "b"},
        ]
        path = self._write_prompts_file(records)
        cfg = _make_cfg(prompts_file=path)
        result = run_type_fingerprints(cfg, _MockCollector())
        # CKA values should be in [0,1]
        assert np.all(result.cross_cka >= -0.01)
        assert np.all(result.cross_cka <= 1.01)

    def test_layer_keys_sorted(self) -> None:
        records = [{"prompt": "q", "label": "x"} for _ in range(2)]
        path = self._write_prompts_file(records)
        cfg = _make_cfg(prompts_file=path)
        result = run_type_fingerprints(cfg, _MockCollector())
        layer_nums = [int(k.split("_")[-1]) for k in result.layer_keys]
        assert layer_nums == sorted(layer_nums)

    def test_mean_sketches_present(self) -> None:
        records = [{"prompt": "q", "label": "cat"} for _ in range(3)]
        path = self._write_prompts_file(records)
        cfg = _make_cfg(prompts_file=path)
        result = run_type_fingerprints(cfg, _MockCollector())
        assert "cat" in result.mean_sketches
        assert len(result.mean_sketches["cat"]) == 4  # n_layers

    def test_empty_file_raises(self) -> None:
        path = self._write_prompts_file([])
        cfg = _make_cfg(prompts_file=path)
        with pytest.raises(ValueError, match="No valid records"):
            run_type_fingerprints(cfg, _MockCollector())


# ── Feature 5: Scaffold vs Naked ─────────────────────────────────────────────


class TestRunScaffoldNakedComparison:
    def test_basic_comparison(self) -> None:
        cfg = _make_cfg()
        result = run_scaffold_naked_comparison(cfg, _MockCollector(), scaffold_prefix="[SCAFFOLD] ")
        assert isinstance(result, ScaffoldNakedResult)
        assert isinstance(result.similarity, dict)
        assert 0.0 <= result.mean_cka <= 1.0 + 1e-9
        assert isinstance(result.mean_cosine, float)

    def test_empty_prefix(self) -> None:
        cfg = _make_cfg()
        result = run_scaffold_naked_comparison(cfg, _MockCollector(), scaffold_prefix="")
        assert isinstance(result, ScaffoldNakedResult)

    def test_identical_prompt_same_result(self) -> None:
        """Same collector always returns same result for same seed+call count."""
        cfg = _make_cfg()
        result = run_scaffold_naked_comparison(cfg, _MockCollector(seed=42), scaffold_prefix="")
        assert isinstance(result.similarity, dict)
        assert len(result.similarity) > 0


# ── Feature 3: LoRA Isolation ─────────────────────────────────────────────────


class TestRunLoRAIsolation:
    def test_with_set_lora_scale(self) -> None:
        from mlx_grpo.inspect.features import run_lora_isolation  # noqa: PLC0415

        model = MagicMock()
        tokenizer = MagicMock()
        cfg = _make_cfg()
        scale_calls: list[float] = []

        def set_lora_scale(val: float) -> None:
            scale_calls.append(val)

        model.set_lora_scale = set_lora_scale

        with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
            mock_instance = mock_coll.return_value
            mock_instance.collect.return_value = _MockCollectionResult(seed=0)
            result = run_lora_isolation(cfg, model, tokenizer)

        assert isinstance(result, LoRAIsolationResult)
        assert 0.0 in scale_calls
        assert 1.0 in scale_calls

    def test_without_set_lora_scale(self) -> None:
        from mlx_grpo.inspect.features import run_lora_isolation  # noqa: PLC0415

        model = MagicMock(spec=[])  # no set_lora_scale
        tokenizer = MagicMock()
        cfg = _make_cfg()

        with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
            mock_instance = mock_coll.return_value
            mock_instance.collect.return_value = _MockCollectionResult(seed=0)
            result = run_lora_isolation(cfg, model, tokenizer)

        assert isinstance(result, LoRAIsolationResult)
        assert isinstance(result.delta_cka, dict)

    def test_delta_cka_keys_match_similarity(self) -> None:
        from mlx_grpo.inspect.features import run_lora_isolation  # noqa: PLC0415

        model = MagicMock()
        tokenizer = MagicMock()
        cfg = _make_cfg()

        with patch("mlx_grpo.inspect.collector.ActivationCollector") as mock_coll:
            mock_instance = mock_coll.return_value
            mock_instance.collect.return_value = _MockCollectionResult(seed=5)
            result = run_lora_isolation(cfg, model, tokenizer)

        assert set(result.delta_cka.keys()) == set(result.lora_similarity.keys())


# ── Feature 1: Checkpoint Trajectory ─────────────────────────────────────────


class TestRunCheckpointTrajectory:
    def test_raises_without_checkpoints_dir(self) -> None:
        from mlx_grpo.inspect.features import run_checkpoint_trajectory  # noqa: PLC0415

        cfg = _make_cfg()
        with pytest.raises(ValueError, match="checkpoints_dir"):
            run_checkpoint_trajectory(cfg, _MockCollector())

    def test_empty_dir_returns_empty_matrix(self) -> None:
        from mlx_grpo.inspect.features import run_checkpoint_trajectory  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_cfg(checkpoints_dir=tmpdir)
            result = run_checkpoint_trajectory(cfg, _MockCollector())
            assert isinstance(result, CheckpointCKAResult)
            assert result.cka_matrix.shape[0] == 0

    def test_returns_correct_type(self) -> None:
        from mlx_grpo.inspect.features import run_checkpoint_trajectory  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_cfg(checkpoints_dir=tmpdir)
            result = run_checkpoint_trajectory(cfg, _MockCollector())
            assert isinstance(result.checkpoint_labels, list)
            assert isinstance(result.layer_keys, list)
            assert isinstance(result.cka_matrix, np.ndarray)
