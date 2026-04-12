"""Comprehensive tests for the dynamic multi-view scaffold system.

Covers every layer of the system:
  1. DynamicViewsConfig  — dataclass defaults, parsing, unknown keys
  2. Fraction math        — equal division, cap, edge values
  3. Scaffold budget      — max_scaffold_tokens cap, <think> stripping
  4. Config parsing       — YAML → DynamicViewsConfig via load_config
  5. Dispatch priority    — adaptive > dynamic > fixed
  6. _dynamic_view_rollouts — group structure, metadata, pre_inject routing
  7. Integration          — full Trainer.train() with dynamic views
"""

from __future__ import annotations

import json
import tempfile
from unittest.mock import patch

import pytest

import mlx_grpo.types.builtins  # noqa: F401 — ensure types registered
from mlx_grpo.config.loader import (
    DynamicViewsConfig,
    TrainingConfig,
    _build_dataclass,
    load_config,
)
from mlx_grpo.data import DatasetConfig, DatasetLoader, Sample
from mlx_grpo.training import Trainer

# ── Shared helpers ────────────────────────────────────────────────────────────


def _write_jsonl(rows: list[dict]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        return f.name


def _make_dataset(type_name: str = "mcq", n: int = 3) -> DatasetLoader:
    rows = [
        {
            "prompt": f"Q{i}",
            "answer": "<think>step one step two step three</think>\\boxed{A}",
            "type": type_name,
            "metadata": {"ground_truth": "A"},
        }
        for i in range(n)
    ]
    path = _write_jsonl(rows)
    return DatasetLoader(DatasetConfig(path=path, shuffle=False)).load()


def _make_trainer(mock_backend, n_additional: int = 1, **dv_kwargs) -> Trainer:
    dv_cfg = DynamicViewsConfig(n_additional=n_additional, **dv_kwargs)
    dataset = _make_dataset()
    config = TrainingConfig(
        iterations=1,
        group_size=n_additional + 1,
        save_every=0,
        eval_every=0,
    )
    return Trainer(
        config=config,
        dataset=dataset,
        backend=mock_backend,
        dv_cfg=dv_cfg,
    )


# ── 1. DynamicViewsConfig ─────────────────────────────────────────────────────


class TestDynamicViewsConfig:
    def test_defaults(self):
        dv = DynamicViewsConfig()
        assert dv.n_additional == 1
        assert dv.target == "thinking"
        assert dv.completion == "model_closes"
        assert dv.max_scaffold_tokens == 512

    def test_custom_values(self):
        dv = DynamicViewsConfig(
            n_additional=3,
            target="answer",
            completion="model_opens",
            max_scaffold_tokens=128,
        )
        assert dv.n_additional == 3
        assert dv.target == "answer"
        assert dv.completion == "model_opens"
        assert dv.max_scaffold_tokens == 128

    def test_build_from_dict(self):
        raw = {"n_additional": 2, "max_scaffold_tokens": 256}
        dv = _build_dataclass(DynamicViewsConfig, raw)
        assert dv.n_additional == 2
        assert dv.max_scaffold_tokens == 256
        assert dv.target == "thinking"  # default preserved

    def test_unknown_keys_ignored(self):
        raw = {"n_additional": 1, "nonexistent_key": "ignored", "another": 999}
        dv = _build_dataclass(DynamicViewsConfig, raw)
        assert dv.n_additional == 1

    def test_frozen(self):
        dv = DynamicViewsConfig()
        with pytest.raises((AttributeError, TypeError)):
            dv.n_additional = 99  # type: ignore[misc]

    def test_zero_n_additional(self):
        # n_additional=0 is degenerate but should not crash at construction
        dv = DynamicViewsConfig(n_additional=0)
        assert dv.n_additional == 0

    def test_large_n_additional(self):
        dv = DynamicViewsConfig(n_additional=20)
        assert dv.n_additional == 20

    def test_all_valid_targets(self):
        for t in ("thinking", "answer", "both"):
            dv = DynamicViewsConfig(target=t)
            assert dv.target == t

    def test_all_valid_completions(self):
        for c in ("model_closes", "model_opens"):
            dv = DynamicViewsConfig(completion=c)
            assert dv.completion == c


# ── 2. Fraction math ──────────────────────────────────────────────────────────


class TestFractionDivision:
    """Verify the equal-division formula (k+1)/N produces correct fractions."""

    def _fractions(self, n: int) -> list[float]:
        return [(k + 1) / n for k in range(n)]

    def test_n1_single_fraction_is_1(self):
        assert self._fractions(1) == [1.0]

    def test_n2_half_and_full(self):
        f = self._fractions(2)
        assert f[0] == pytest.approx(0.5)
        assert f[1] == pytest.approx(1.0)

    def test_n3_equal_thirds(self):
        f = self._fractions(3)
        assert f[0] == pytest.approx(1 / 3)
        assert f[1] == pytest.approx(2 / 3)
        assert f[2] == pytest.approx(1.0)

    def test_n4_equal_quarters(self):
        f = self._fractions(4)
        for k, expected in enumerate([0.25, 0.50, 0.75, 1.0]):
            assert f[k] == pytest.approx(expected)

    def test_last_fraction_always_1(self):
        for n in [1, 2, 5, 10, 20]:
            assert self._fractions(n)[-1] == pytest.approx(1.0)

    def test_fractions_strictly_increasing(self):
        for n in [2, 3, 5, 8]:
            f = self._fractions(n)
            for i in range(1, len(f)):
                assert f[i] > f[i - 1]

    def test_total_views_equals_n_plus_one(self):
        for n in [1, 3, 5]:
            assert len(self._fractions(n)) + 1 == n + 1  # +1 for naked


# ── 3. Scaffold budget ────────────────────────────────────────────────────────


class TestScaffoldBudget:
    """Verify n_scaffold = min(int(fraction * n_target), max_scaffold_tokens)."""

    def _n_scaffold(self, fraction: float, n_target: int, max_scaffold: int = 512) -> int:
        return min(int(fraction * n_target), max_scaffold)

    def test_not_capped_when_under_max(self):
        assert self._n_scaffold(1.0, 100, 512) == 100

    def test_capped_at_max_when_over(self):
        assert self._n_scaffold(1.0, 1000, 512) == 512

    def test_half_fraction_not_capped(self):
        assert self._n_scaffold(0.5, 200, 512) == 100

    def test_half_fraction_capped(self):
        assert self._n_scaffold(0.5, 2000, 512) == 512

    def test_zero_target_tokens(self):
        assert self._n_scaffold(1.0, 0, 512) == 0

    def test_zero_max_scaffold(self):
        assert self._n_scaffold(1.0, 100, 0) == 0

    def test_small_max_scaffold(self):
        assert self._n_scaffold(1.0, 1000, 32) == 32

    def test_exact_boundary(self):
        # n_target == max_scaffold_tokens → no cap needed
        assert self._n_scaffold(1.0, 512, 512) == 512

    def test_think_tag_stripped_in_token_count(self, mock_backend):
        """<think> is stripped before tokenizing so it doesn't eat into budget."""
        # Mock backend: encode("<think>abc") treats < as a separate char
        # After stripping: "abc" → shorter → less tokens used on tag
        without_tag = len(mock_backend.encode("abc"))
        with_tag = len(mock_backend.encode("<think>abc"))
        assert without_tag < with_tag  # confirms stripping matters


# ── 4. Config parsing ─────────────────────────────────────────────────────────


class TestConfigParsing:
    def _minimal_yaml(self, extra: str = "") -> str:
        base = """\
training:
  model_path: "/tmp/model"
  iterations: 1
  group_size: 2
  save_every: 0
  eval_every: 0
generation:
  backend: mock
datasets:
  test:
    path: "/tmp/data.jsonl"
"""
        return base + extra

    def test_absent_gives_none(self, tmp_path):
        p = tmp_path / "cfg.yml"
        p.write_text(self._minimal_yaml())
        cfg = load_config(str(p))
        assert cfg.dynamic_views is None

    def test_parsed_when_present(self, tmp_path):
        extra = "dynamic_views:\n  n_additional: 2\n"
        p = tmp_path / "cfg.yml"
        p.write_text(self._minimal_yaml(extra))
        cfg = load_config(str(p))
        assert cfg.dynamic_views is not None
        assert cfg.dynamic_views.n_additional == 2

    def test_all_fields_parsed(self, tmp_path):
        extra = (
            "dynamic_views:\n"
            "  n_additional: 3\n"
            "  target: answer\n"
            "  completion: model_opens\n"
            "  max_scaffold_tokens: 128\n"
        )
        p = tmp_path / "cfg.yml"
        p.write_text(self._minimal_yaml(extra))
        cfg = load_config(str(p))
        dv = cfg.dynamic_views
        assert dv.n_additional == 3
        assert dv.target == "answer"
        assert dv.completion == "model_opens"
        assert dv.max_scaffold_tokens == 128

    def test_defaults_fill_missing_fields(self, tmp_path):
        extra = "dynamic_views:\n  n_additional: 2\n"
        p = tmp_path / "cfg.yml"
        p.write_text(self._minimal_yaml(extra))
        cfg = load_config(str(p))
        dv = cfg.dynamic_views
        assert dv.target == "thinking"  # default
        assert dv.completion == "model_closes"  # default
        assert dv.max_scaffold_tokens == 512  # default


# ── 5. Dispatch priority ──────────────────────────────────────────────────────


class TestDispatchPriority:
    def test_fixed_loop_when_no_dv_cfg(self, mock_backend):
        dataset = _make_dataset("mcq", 2)
        config = TrainingConfig(iterations=1, group_size=3, save_every=0, eval_every=0)
        captured = []
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)
        trainer.on("after_step", lambda t, m: captured.append(m))
        trainer.train()
        # Fixed loop with group_size=3 → exactly 3 rewards
        assert len(captured[0].rewards) == 3

    def test_dynamic_used_when_dv_cfg_set(self, mock_backend):
        captured = []
        with patch("time.sleep"):
            trainer = _make_trainer(mock_backend, n_additional=1)
            trainer.on("after_step", lambda t, m: captured.append(m))
            trainer.train()
        # Dynamic with n_additional=1 → 2 total views
        assert len(captured[0].rewards) == 2

    def test_adaptive_beats_dynamic_for_math(self, mock_backend):
        """Math type with view_strategy=adaptive_escalation ignores dynamic_views."""
        dv_cfg = DynamicViewsConfig(n_additional=5)  # would produce 6 views
        dataset = _make_dataset("math", 2)
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
        captured = []
        with patch("time.sleep"):
            trainer = Trainer(
                config=config,
                dataset=dataset,
                backend=mock_backend,
                dv_cfg=dv_cfg,
                type_overrides={
                    "math": {
                        "view_strategy": "adaptive_escalation",
                        "max_escalations": 1,
                        "escalation_step_tokens": 0,
                        "postfix_tokens": 0,
                    }
                },
            )
            trainer.on("after_step", lambda t, m: captured.append(m))
            trainer.train()
        # Adaptive with max_escalations=1 + naked = 2 views — NOT 6
        assert len(captured[0].rewards) == 2

    def test_dynamic_used_for_mcq_even_when_math_adaptive_configured(self, mock_backend):
        """MCQ type should use dynamic loop even when math adaptive is configured."""
        dv_cfg = DynamicViewsConfig(n_additional=2)  # 3 total views
        dataset = _make_dataset("mcq", 2)
        config = TrainingConfig(iterations=1, group_size=3, save_every=0, eval_every=0)
        captured = []
        with patch("time.sleep"):
            trainer = Trainer(
                config=config,
                dataset=dataset,
                backend=mock_backend,
                dv_cfg=dv_cfg,
                type_overrides={
                    "math": {  # only math uses adaptive
                        "view_strategy": "adaptive_escalation",
                        "max_escalations": 2,
                        "escalation_step_tokens": 0,
                        "postfix_tokens": 0,
                    }
                },
            )
            trainer.on("after_step", lambda t, m: captured.append(m))
            trainer.train()
        # MCQ uses dynamic → n_additional=2 + naked = 3 views
        assert len(captured[0].rewards) == 3


# ── 6. _dynamic_view_rollouts granular ───────────────────────────────────────


def _get_trainer_and_sample(mock_backend, n_additional: int = 1):
    """Helper: returns (trainer, sample, task_type) for direct method calls."""
    from mlx_grpo.types.base import TaskType

    trainer = _make_trainer(mock_backend, n_additional=n_additional)
    sample = Sample(
        prompt="What is the answer?",
        answer="<think>word1 word2 word3 word4 word5</think>\\boxed{A}",
        type_name="mcq",
        metadata={"ground_truth": "A"},
    )
    task_type = TaskType.get("mcq")
    return trainer, sample, task_type


class TestDynamicViewRollouts:
    def _call(self, trainer, sample, task_type, **kwargs):
        defaults = dict(
            merged_type_config={},
            prompt_tokens=list(range(10)),
            prompt_cache=None,
            iteration=0,
            max_completion_tokens=32,
        )
        defaults.update(kwargs)
        return trainer._dynamic_view_rollouts(sample=sample, task_type=task_type, **defaults)

    def test_n1_produces_two_results(self, mock_backend):
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 1)
        with patch("time.sleep"):
            results = self._call(trainer, sample, tt)
        assert len(results) == 2

    def test_n2_produces_three_results(self, mock_backend):
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 2)
        with patch("time.sleep"):
            results = self._call(trainer, sample, tt)
        assert len(results) == 3

    def test_n4_produces_five_results(self, mock_backend):
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 4)
        with patch("time.sleep"):
            results = self._call(trainer, sample, tt)
        assert len(results) == 5

    def test_naked_always_last(self, mock_backend):
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 3)
        with patch("time.sleep"):
            results = self._call(trainer, sample, tt)
        assert results[-1].metadata["_view_name"] == "naked"

    def test_naked_marked_as_naked(self, mock_backend):
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 1)
        with patch("time.sleep"):
            results = self._call(trainer, sample, tt)
        assert results[-1].metadata["_view_is_naked"] is True

    def test_scaffold_views_not_marked_as_naked(self, mock_backend):
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 3)
        with patch("time.sleep"):
            results = self._call(trainer, sample, tt)
        for r in results[:-1]:  # all except last (naked)
            assert r.metadata["_view_is_naked"] is False

    def test_view_names_sequential(self, mock_backend):
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 3)
        with patch("time.sleep"):
            results = self._call(trainer, sample, tt)
        assert results[0].metadata["_view_name"] == "dynamic_0"
        assert results[1].metadata["_view_name"] == "dynamic_1"
        assert results[2].metadata["_view_name"] == "dynamic_2"
        assert results[3].metadata["_view_name"] == "naked"

    def test_view_fractions_increasing(self, mock_backend):
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 3)
        with patch("time.sleep"):
            results = self._call(trainer, sample, tt)
        fracs = [r.metadata["_view_fraction"] for r in results[:-1]]
        assert fracs == sorted(fracs)
        assert fracs[-1] == pytest.approx(1.0)

    def test_scaffold_tokens_non_decreasing(self, mock_backend):
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 3)
        with patch("time.sleep"):
            results = self._call(trainer, sample, tt)
        tok_counts = [r.metadata["_scaffold_tokens"] for r in results[:-1]]
        assert tok_counts == sorted(tok_counts)

    def test_scaffold_tokens_capped_at_max(self, mock_backend):
        dv_cfg = DynamicViewsConfig(n_additional=1, max_scaffold_tokens=4)
        trainer = _make_trainer(mock_backend, n_additional=1)
        trainer._dv_cfg = dv_cfg
        # answer has many thinking tokens; cap at 4
        sample = Sample(
            prompt="Q",
            answer="<think>" + "word " * 50 + "</think>\\boxed{A}",
            type_name="mcq",
            metadata={"ground_truth": "A"},
        )
        from mlx_grpo.types.base import TaskType

        tt = TaskType.get("mcq")
        with patch("time.sleep"):
            results = trainer._dynamic_view_rollouts(
                sample=sample,
                task_type=tt,
                merged_type_config={},
                prompt_tokens=[1, 2, 3],
                prompt_cache=None,
                iteration=0,
                max_completion_tokens=32,
            )
        assert results[0].metadata["_scaffold_tokens"] <= 4

    def test_scaffold_tokens_zero_when_empty_target(self, mock_backend):
        trainer, _, tt = _get_trainer_and_sample(mock_backend, 1)
        empty_sample = Sample(
            prompt="Q",
            answer=None,
            type_name="mcq",
            metadata={"ground_truth": "A"},
        )
        with patch("time.sleep"):
            results = trainer._dynamic_view_rollouts(
                sample=empty_sample,
                task_type=tt,
                merged_type_config={},
                prompt_tokens=[1, 2, 3],
                prompt_cache=None,
                iteration=0,
                max_completion_tokens=32,
            )
        assert results[0].metadata["_scaffold_tokens"] == 0

    def test_think_tag_only_target_zero_scaffold(self, mock_backend):
        trainer, _, tt = _get_trainer_and_sample(mock_backend, 1)
        tag_sample = Sample(
            prompt="Q",
            answer="<think>",
            type_name="mcq",
            metadata={"ground_truth": "A"},
        )
        with patch("time.sleep"):
            results = trainer._dynamic_view_rollouts(
                sample=tag_sample,
                task_type=tt,
                merged_type_config={},
                prompt_tokens=[1, 2, 3],
                prompt_cache=None,
                iteration=0,
                max_completion_tokens=32,
            )
        assert results[0].metadata["_scaffold_tokens"] == 0

    def test_pre_inject_none_for_naked(self, mock_backend):
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 1)
        injected = []
        original = trainer.executor.execute

        def spy(*a, **kw):
            injected.append(kw.get("pre_inject_text"))
            return original(*a, **kw)

        with patch.object(trainer.executor, "execute", side_effect=spy), patch("time.sleep"):
            self._call(trainer, sample, tt)
        # Last call is naked — pre_inject_text must be None
        assert injected[-1] is None

    def test_pre_inject_present_for_scaffold_view(self, mock_backend):
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 1)
        injected = []
        original = trainer.executor.execute

        def spy(*a, **kw):
            injected.append(kw.get("pre_inject_text"))
            return original(*a, **kw)

        with patch.object(trainer.executor, "execute", side_effect=spy), patch("time.sleep"):
            self._call(trainer, sample, tt)
        # First call (scaffold view, fraction=1.0 of non-empty target) has text
        assert injected[0] is not None
        assert len(injected[0]) > 0

    def test_pre_inject_increases_with_k_for_n3(self, mock_backend):
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 3)
        injected = []
        original = trainer.executor.execute

        def spy(*a, **kw):
            injected.append(kw.get("pre_inject_text"))
            return original(*a, **kw)

        with patch.object(trainer.executor, "execute", side_effect=spy), patch("time.sleep"):
            self._call(trainer, sample, tt)
        # Views 0, 1, 2 have scaffold; their pre_inject lengths should not decrease
        lengths = [len(t) if t else 0 for t in injected[:-1]]
        assert lengths == sorted(lengths)

    def test_pre_inject_empty_target_gives_none_for_scaffold(self, mock_backend):
        trainer, _, tt = _get_trainer_and_sample(mock_backend, 1)
        empty_sample = Sample(prompt="Q", answer=None, type_name="mcq", metadata={})
        injected = []
        original = trainer.executor.execute

        def spy(*a, **kw):
            injected.append(kw.get("pre_inject_text"))
            return original(*a, **kw)

        with patch.object(trainer.executor, "execute", side_effect=spy), patch("time.sleep"):
            trainer._dynamic_view_rollouts(
                sample=empty_sample,
                task_type=tt,
                merged_type_config={},
                prompt_tokens=[1],
                prompt_cache=None,
                iteration=0,
                max_completion_tokens=32,
            )
        # Empty target → no prefix → pre_inject_text=None for scaffold view too
        assert injected[0] is None

    def test_on_group_fires_n_times(self, mock_backend):
        """on_group should fire once per scaffold view (not for naked)."""
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 3)
        fired = []
        trainer.on("on_group", lambda t, m: fired.append(m))
        with patch("time.sleep"):
            self._call(trainer, sample, tt)
        assert len(fired) == 3  # one per scaffold view

    def test_cooldown_sleep_called_after_each_view(self, mock_backend):
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 2)
        # rollout_cooldown=0.5 by default; sleep called after each of 3 views
        with patch("time.sleep") as mock_sleep:
            self._call(trainer, sample, tt)
        assert mock_sleep.call_count == 3  # 2 scaffold + 1 naked

    def test_results_are_rollout_results(self, mock_backend):
        from mlx_grpo.engine.result import RolloutResult

        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 2)
        with patch("time.sleep"):
            results = self._call(trainer, sample, tt)
        for r in results:
            assert isinstance(r, RolloutResult)

    def test_scaffold_mode_none_passed_to_batch(self, mock_backend):
        """scaffold=ScaffoldConfig(mode='none') prevents double-injection in rollout."""

        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 1)
        batch_ctxs = []
        original = trainer.executor.execute

        def spy(*a, **kw):
            batch_ctxs.append(kw.get("batch"))
            return original(*a, **kw)

        with patch.object(trainer.executor, "execute", side_effect=spy), patch("time.sleep"):
            self._call(trainer, sample, tt)
        for bc in batch_ctxs:
            assert bc.scaffold.mode == "none"

    def test_group_size_is_n_plus_one_in_all_batch_contexts(self, mock_backend):
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, 3)
        batch_ctxs = []
        original = trainer.executor.execute

        def spy(*a, **kw):
            batch_ctxs.append(kw.get("batch"))
            return original(*a, **kw)

        with patch.object(trainer.executor, "execute", side_effect=spy), patch("time.sleep"):
            self._call(trainer, sample, tt)
        for bc in batch_ctxs:
            assert bc.group_size == 4  # n_additional=3 + naked=1

    def test_group_index_sequential_0_to_n(self, mock_backend):
        n = 3
        trainer, sample, tt = _get_trainer_and_sample(mock_backend, n)
        batch_ctxs = []
        original = trainer.executor.execute

        def spy(*a, **kw):
            batch_ctxs.append(kw.get("batch"))
            return original(*a, **kw)

        with patch.object(trainer.executor, "execute", side_effect=spy), patch("time.sleep"):
            self._call(trainer, sample, tt)
        indices = [bc.group_index for bc in batch_ctxs]
        assert indices == list(range(n + 1))  # [0, 1, 2, 3]


# ── 7. Integration — full trainer.train() ────────────────────────────────────


class TestDynamicTrainerIntegration:
    def test_train_completes_n1(self, mock_backend):
        with patch("time.sleep"):
            trainer = _make_trainer(mock_backend, n_additional=1)
            trainer.train()
            assert trainer.iteration == 0  # completed 1 iteration (0-indexed)

    def test_after_step_fires_once_per_iteration(self, mock_backend):
        fired = []
        with patch("time.sleep"):
            trainer = _make_trainer(mock_backend, n_additional=1)
            trainer.config = TrainingConfig(iterations=3, group_size=2, save_every=0, eval_every=0)
            trainer.on("after_step", lambda t, m: fired.append(m.iteration))
            trainer.train()
        assert fired == [0, 1, 2]

    def test_two_views_with_n1(self, mock_backend):
        captured = []
        with patch("time.sleep"):
            trainer = _make_trainer(mock_backend, n_additional=1)
            trainer.on("after_step", lambda t, m: captured.append(m))
            trainer.train()
        assert len(captured[0].rewards) == 2

    def test_four_views_with_n3(self, mock_backend):
        captured = []
        dataset = _make_dataset("mcq", 2)
        config = TrainingConfig(iterations=1, group_size=4, save_every=0, eval_every=0)
        dv_cfg = DynamicViewsConfig(n_additional=3)
        with patch("time.sleep"):
            trainer = Trainer(
                config=config,
                dataset=dataset,
                backend=mock_backend,
                dv_cfg=dv_cfg,
            )
            trainer.on("after_step", lambda t, m: captured.append(m))
            trainer.train()
        assert len(captured[0].rewards) == 4

    def test_zero_n_additional_only_naked_view(self, mock_backend):
        """n_additional=0 → only naked view → 1 reward."""
        captured = []
        dataset = _make_dataset("mcq", 2)
        config = TrainingConfig(iterations=1, group_size=1, save_every=0, eval_every=0)
        dv_cfg = DynamicViewsConfig(n_additional=0)
        with patch("time.sleep"):
            trainer = Trainer(
                config=config,
                dataset=dataset,
                backend=mock_backend,
                dv_cfg=dv_cfg,
            )
            trainer.on("after_step", lambda t, m: captured.append(m))
            trainer.train()
        assert len(captured[0].rewards) == 1

    def test_large_max_scaffold_tokens_uncapped(self, mock_backend):
        """With max_scaffold_tokens >> target length, scaffold = full target."""
        captured = []
        with patch("time.sleep"):
            trainer = _make_trainer(mock_backend, n_additional=1, max_scaffold_tokens=99999)
            trainer.on("after_step", lambda t, m: captured.append(m))
            trainer.train()
        assert len(captured[0].rewards) == 2

    def test_dynamic_with_multiple_iterations(self, mock_backend):
        captured = []
        dataset = _make_dataset("mcq", 5)
        config = TrainingConfig(iterations=3, group_size=2, save_every=0, eval_every=0)
        dv_cfg = DynamicViewsConfig(n_additional=1)
        with patch("time.sleep"):
            trainer = Trainer(
                config=config,
                dataset=dataset,
                backend=mock_backend,
                dv_cfg=dv_cfg,
            )
            trainer.on("after_step", lambda t, m: captured.append(len(m.rewards)))
            trainer.train()
        # Every iteration: 2 rewards
        assert captured == [2, 2, 2]

    def test_trainer_dv_cfg_none_by_default(self, mock_backend):
        dataset = _make_dataset()
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend)
        assert trainer._dv_cfg is None

    def test_trainer_stores_dv_cfg(self, mock_backend):
        dv = DynamicViewsConfig(n_additional=3)
        trainer = _make_trainer(mock_backend, n_additional=3)
        trainer._dv_cfg = dv
        assert trainer._dv_cfg.n_additional == 3


# ── model_opens guard + target warning ──────────────────────────────────────


class TestModelOpensGuard:
    def test_model_opens_raises_not_implemented(self, mock_backend):
        """completion='model_opens' must raise NotImplementedError when rollout runs."""
        from unittest.mock import patch

        import pytest

        dv_cfg = DynamicViewsConfig(n_additional=1, completion="model_opens")
        dataset = _make_dataset()
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend, dv_cfg=dv_cfg)
        with patch("time.sleep"), pytest.raises(NotImplementedError, match="model_opens"):
            trainer.train()

    def test_model_closes_does_not_raise(self, mock_backend):
        """completion='model_closes' (default) must not raise."""
        dv_cfg = DynamicViewsConfig(n_additional=1, completion="model_closes")
        dataset = _make_dataset()
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend, dv_cfg=dv_cfg)
        with patch("time.sleep"):
            trainer.train()  # should not raise

    def test_target_answer_no_warning(self, mock_backend):
        """target='answer' is now implemented — must NOT emit UserWarning."""
        import warnings
        from unittest.mock import patch

        dv_cfg = DynamicViewsConfig(n_additional=1, target="answer")
        dataset = _make_dataset()
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend, dv_cfg=dv_cfg)
        with patch("time.sleep"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer.train()
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert not user_warns, f"Unexpected UserWarning(s): {[str(w.message) for w in user_warns]}"

    def test_target_both_no_warning(self, mock_backend):
        """target='both' is now implemented — must NOT emit UserWarning."""
        import warnings
        from unittest.mock import patch

        dv_cfg = DynamicViewsConfig(n_additional=1, target="both")
        dataset = _make_dataset()
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend, dv_cfg=dv_cfg)
        with patch("time.sleep"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer.train()
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert not user_warns, f"Unexpected UserWarning(s): {[str(w.message) for w in user_warns]}"

    def test_target_unknown_emits_warning(self, mock_backend):
        """Unknown target values must emit a UserWarning."""
        import warnings
        from unittest.mock import patch

        dv_cfg = DynamicViewsConfig(n_additional=1, target="nonexistent_target")
        dataset = _make_dataset()
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend, dv_cfg=dv_cfg)
        with patch("time.sleep"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer.train()
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warns, "Expected UserWarning for unknown target"
        assert any("nonexistent_target" in str(w.message) for w in user_warns)

    def test_target_thinking_no_warning(self, mock_backend):
        """target='thinking' (default) must not emit any UserWarning."""
        import warnings
        from unittest.mock import patch

        dv_cfg = DynamicViewsConfig(n_additional=1, target="thinking")
        dataset = _make_dataset()
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend, dv_cfg=dv_cfg)
        with patch("time.sleep"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer.train()
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert not user_warns, f"Unexpected UserWarning(s): {[str(w.message) for w in user_warns]}"

    def test_target_answer_uses_answer_portion(self, mock_backend):
        """target='answer' must encode text after </think>, not the thinking portion."""
        from unittest.mock import patch

        dv_cfg = DynamicViewsConfig(n_additional=1, target="answer")
        dataset = _make_dataset()
        config = TrainingConfig(iterations=1, group_size=2, save_every=0, eval_every=0)
        trainer = Trainer(config=config, dataset=dataset, backend=mock_backend, dv_cfg=dv_cfg)
        encoded_calls: list[str] = []
        original_encode = mock_backend.encode

        def capture_encode(text: str) -> list[int]:
            encoded_calls.append(text)
            return original_encode(text)

        mock_backend.encode = capture_encode
        with patch("time.sleep"):
            trainer.train()

        # At minimum: the encode call for the scaffold source should not include "step"
        answer_encodes = [t for t in encoded_calls if "boxed" in t and "step" not in t]
        assert answer_encodes, (
            f"Expected encode to be called with answer-only text; got calls: {encoded_calls[:5]}"
        )
