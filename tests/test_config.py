"""Tests for YAML config loading."""

import tempfile
import textwrap
from pathlib import Path

import pytest

from mlx_grpo.config.loader import _resolve_type_extends, load_config


def _write_yml(content: str) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(textwrap.dedent(content))
        f.flush()
        return Path(f.name)


class TestConfigLoader:
    def test_minimal(self):
        cfg = load_config(_write_yml("training:\n  iterations: 42"))
        assert cfg.training.iterations == 42

    def test_defaults(self):
        cfg = load_config(_write_yml(""))
        assert cfg.training.learning_rate == 1e-5
        assert cfg.generation.backend == "mlx"
        assert cfg.gradient.thinking_gradient_weight == 1.0

    def test_full_config(self):
        cfg = load_config(
            _write_yml("""
            training:
              model_path: "test/model"
              iterations: 100
              batch_size: 2
              group_size: 8
              beta: 0.02
              loss_type: bnpo
            generation:
              temperature: 0.5
              backend: mlx
            gradient:
              thinking_gradient_weight: 0.3
              sft_anchor_enabled: true
            monitor:
              kl_warning: 0.05
              stop_on_critical: true
        """)
        )
        assert cfg.training.model_path == "test/model"
        assert cfg.training.loss_type == "bnpo"
        assert cfg.generation.temperature == 0.5
        assert cfg.gradient.thinking_gradient_weight == 0.3
        assert cfg.gradient.sft_anchor_enabled is True
        assert cfg.monitor.stop_on_critical is True

    def test_generation_sampler_params(self):
        cfg = load_config(
            _write_yml("""
            generation:
              min_p: 0.05
              xtc_probability: 0.3
              xtc_threshold: 0.2
              min_tokens_to_keep: 2
              prefill_step_size: 256
              quantized_kv_start: 128
        """)
        )
        assert cfg.generation.min_p == 0.05
        assert cfg.generation.xtc_probability == 0.3
        assert cfg.generation.xtc_threshold == 0.2
        assert cfg.generation.min_tokens_to_keep == 2
        assert cfg.generation.prefill_step_size == 256
        assert cfg.generation.quantized_kv_start == 128

    def test_ignores_unknown_keys(self):
        cfg = load_config(
            _write_yml("""
            training:
              iterations: 10
              unknown_field: "ignored"
        """)
        )
        assert cfg.training.iterations == 10

    def test_yaml_types_registered(self):
        from mlx_grpo.types.base import TaskType

        load_config(
            _write_yml("""
            types:
              cfg_test_type:
                temperature: 0.9
                reward: accuracy
                rollout:
                  - generate: {max_tokens: 64}
        """)
        )
        assert "cfg_test_type" in TaskType.list_types()

    def test_yaml_rewards_registered(self):
        from mlx_grpo.rewards.base import BaseReward

        load_config(
            _write_yml("""
            rewards:
              cfg_test_reward: "lambda result, answer: 0.77"
        """)
        )
        r = BaseReward.get("cfg_test_reward")
        from mlx_grpo.engine.result import RolloutResult

        assert r.compute(RolloutResult(), None) == 0.77

    def test_yaml_callbacks_lambda(self):
        from mlx_grpo.types.base import TaskType

        load_config(
            _write_yml("""
            types:
              cfg_test_cb:
                reward: accuracy
                callbacks:
                  is_long: "lambda text: len(text) > 50"
                rollout:
                  - generate: {max_tokens: 32}
        """)
        )
        t = TaskType.get("cfg_test_cb")
        assert t.callbacks()["is_long"]("x" * 51) is True
        assert t.callbacks()["is_long"]("short") is False


class TestLambdaSafety:
    def test_rejects_import(self):
        from mlx_grpo.types.yaml_type import parse_lambda

        with pytest.raises(ValueError, match="Forbidden"):
            parse_lambda("lambda x: __import__('os').system('rm -rf /')")

    def test_rejects_dunder(self):
        from mlx_grpo.types.yaml_type import parse_lambda

        with pytest.raises(ValueError, match="Forbidden"):
            parse_lambda("lambda x: x.__class__.__bases__")

    def test_rejects_exec(self):
        from mlx_grpo.types.yaml_type import parse_lambda

        with pytest.raises(ValueError, match="Forbidden"):
            parse_lambda("lambda x: exec('bad')")

    def test_rejects_non_lambda(self):
        from mlx_grpo.types.yaml_type import parse_lambda

        with pytest.raises(ValueError, match="Expected 'lambda'"):
            parse_lambda("print('hello')")

    def test_allows_safe_lambda(self):
        from mlx_grpo.types.yaml_type import parse_lambda

        fn = parse_lambda("lambda x: len(x) > 5")
        assert fn("abcdef") is True
        assert fn("abc") is False


class TestProcessWorkerConfig:
    def test_rollout_worker_mode_default(self):
        cfg = load_config(_write_yml(""))
        assert cfg.training.rollout_worker_mode == "thread"

    def test_rollout_worker_mode_process(self):
        cfg = load_config(
            _write_yml("""
            training:
              rollout_workers: 2
              rollout_worker_mode: process
        """)
        )
        assert cfg.training.rollout_workers == 2
        assert cfg.training.rollout_worker_mode == "process"

    def test_rollout_worker_mode_thread(self):
        cfg = load_config(
            _write_yml("""
            training:
              rollout_worker_mode: thread
        """)
        )
        assert cfg.training.rollout_worker_mode == "thread"


class TestResolveTypeExtends:
    def test_no_extends_passthrough(self):
        raw = {"math": {"thinking_budget": 512}}
        assert _resolve_type_extends(raw) == {"math": {"thinking_budget": 512}}

    def test_single_extends(self):
        raw = {
            "math": {"thinking_budget": 512, "answer_max_tokens": 64},
            "math_hard": {"extends": "math", "thinking_budget": 1024},
        }
        result = _resolve_type_extends(raw)
        assert result["math_hard"]["thinking_budget"] == 1024
        assert result["math_hard"]["answer_max_tokens"] == 64  # inherited
        assert "extends" not in result["math_hard"]

    def test_extends_key_consumed(self):
        raw = {"a": {"x": 1}, "b": {"extends": "a", "y": 2}}
        result = _resolve_type_extends(raw)
        assert "extends" not in result["b"]

    def test_child_overrides_parent(self):
        raw = {"base": {"a": 1, "b": 2}, "child": {"extends": "base", "a": 99}}
        result = _resolve_type_extends(raw)
        assert result["child"]["a"] == 99
        assert result["child"]["b"] == 2

    def test_transitive_chain(self):
        raw = {
            "base": {"x": 1, "y": 2},
            "mid": {"extends": "base", "y": 20, "z": 3},
            "leaf": {"extends": "mid", "z": 30},
        }
        result = _resolve_type_extends(raw)
        assert result["leaf"]["x"] == 1  # from base
        assert result["leaf"]["y"] == 20  # from mid (overrides base)
        assert result["leaf"]["z"] == 30  # from leaf (overrides mid)

    def test_parent_not_in_overrides_returns_child_only(self):
        raw = {"child": {"extends": "unknown_parent", "a": 1}}
        result = _resolve_type_extends(raw)
        assert result["child"] == {"a": 1}

    def test_circular_raises(self):
        raw = {"a": {"extends": "b"}, "b": {"extends": "a"}}
        with pytest.raises(ValueError, match="Circular"):
            _resolve_type_extends(raw)

    def test_self_circular_raises(self):
        raw = {"a": {"extends": "a"}}
        with pytest.raises(ValueError, match="Circular"):
            _resolve_type_extends(raw)

    def test_via_yaml_load(self):
        cfg = load_config(
            _write_yml("""
            type_overrides:
              math:
                thinking_budget: 512
                answer_max_tokens: 64
              math_hard:
                extends: math
                thinking_budget: 1024
        """)
        )
        assert cfg.type_overrides["math_hard"]["thinking_budget"] == 1024
        assert cfg.type_overrides["math_hard"]["answer_max_tokens"] == 64
        assert "extends" not in cfg.type_overrides["math_hard"]

    def test_escalation_sub_dict_inherited(self):
        raw = {
            "base": {"escalation": {"horizontal": {"max_members": 4}}},
            "child": {"extends": "base", "thinking_budget": 256},
        }
        result = _resolve_type_extends(raw)
        assert result["child"]["escalation"]["horizontal"]["max_members"] == 4
        assert result["child"]["thinking_budget"] == 256


class TestMetalEstPrompt:
    def test_metal_est_prompt_default(self):
        """metal_est_prompt_tokens defaults to 900."""
        cfg = load_config(_write_yml(""))
        assert cfg.training.metal_est_prompt_tokens == 900

    def test_metal_est_prompt_configurable(self):
        """metal_est_prompt_tokens can be set in YAML."""
        cfg = load_config(_write_yml("training:\n  metal_est_prompt_tokens: 212"))
        assert cfg.training.metal_est_prompt_tokens == 212

    def test_rollout_worker_mode_default(self):
        """rollout_worker_mode defaults to 'thread'."""
        cfg = load_config(_write_yml(""))
        assert cfg.training.rollout_worker_mode == "thread"
