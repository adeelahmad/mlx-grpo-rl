"""Tests for type system metaprogramming and YAML loading."""

import tempfile

import pytest

from mlx_grpo.engine.result import RolloutResult
from mlx_grpo.types.base import TaskType


class TestAutoRegistration:
    def test_subclass_registers(self):
        class TestR1(TaskType, name="test_reg_1"):
            def rollout(self, ctx):
                pass

            def reward(self, result, answer=None):
                return 0.0

        assert "test_reg_1" in TaskType.list_types()

    def test_get_unknown(self):
        with pytest.raises(KeyError):
            TaskType.get("nonexistent_xyz")


class TestYAMLType:
    def _load(self, yml):
        from mlx_grpo.config.loader import load_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yml)
            f.flush()
            return load_config(f.name)

    def test_yaml_type_registered(self):
        yml = "types:\n  yt1:\n    reward: accuracy\n    rollout:\n"
        yml += "      - generate: {max_tokens: 64}\n"
        self._load(yml)
        assert "yt1" in TaskType.list_types()

    def test_yaml_callbacks(self):
        yml = "types:\n  yt2:\n    reward: accuracy\n    callbacks:\n"
        yml += '      short: "lambda t: len(t) < 100"\n'
        yml += "    rollout:\n      - generate: {max_tokens: 64}\n"
        self._load(yml)
        assert TaskType.get("yt2").callbacks()["short"]("hi")

    def test_yaml_reward(self):
        self._load('rewards:\n  one: "lambda r, a: 1.0"\n')
        from mlx_grpo.rewards.base import BaseReward

        assert BaseReward.get("one").compute(RolloutResult(), None) == 1.0
