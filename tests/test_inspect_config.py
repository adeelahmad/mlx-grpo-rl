"""Tests for mlx_grpo.inspect.config."""

from __future__ import annotations

import pytest

from mlx_grpo.inspect.config import InspectConfig, build_inspect_config


class TestInspectConfigDefaults:
    def test_default_construction(self) -> None:
        cfg = InspectConfig()
        assert cfg.model_path == ""
        assert cfg.model_b_path is None
        assert cfg.prompt == ""
        assert cfg.prompts_file is None
        assert cfg.control_prompt is None
        assert cfg.save_dir == "./inspect_out"
        assert cfg.capture_sublayers is False
        assert cfg.trace_generation is False
        assert cfg.max_new_tokens == 30
        assert cfg.top_k_logit_lens == 5
        assert cfg.layers_to_plot == ()
        assert cfg.circuit_layers == ()
        assert cfg.proj_dim == 128
        assert cfg.sketch_seed == 12345
        assert cfg.feature == "compare"
        assert cfg.checkpoints_dir is None
        assert cfg.rollout_log is None
        assert cfg.lora_path is None
        assert cfg.probe_base is None
        assert cfg.probe_wandb_key == "inspect/cka_drift"

    def test_frozen(self) -> None:
        cfg = InspectConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.model_path = "new"  # type: ignore[misc]

    def test_slots(self) -> None:
        assert hasattr(InspectConfig, "__slots__")


class TestBuildInspectConfig:
    def test_basic_build(self) -> None:
        cfg = build_inspect_config({"model_path": "/path/to/model"})
        assert cfg.model_path == "/path/to/model"

    def test_ignores_unknown_keys(self) -> None:
        cfg = build_inspect_config({"model_path": "x", "unknown_key": 42})
        assert cfg.model_path == "x"

    def test_coerces_int_string(self) -> None:
        cfg = build_inspect_config({"max_new_tokens": "50"})
        assert cfg.max_new_tokens == 50

    def test_coerces_bool(self) -> None:
        cfg = build_inspect_config({"capture_sublayers": True})
        assert cfg.capture_sublayers is True

    def test_list_to_tuple_layers(self) -> None:
        cfg = build_inspect_config({"layers_to_plot": [0, 1, 2, 5]})
        assert cfg.layers_to_plot == (0, 1, 2, 5)
        assert isinstance(cfg.layers_to_plot, tuple)

    def test_list_to_tuple_circuit_layers(self) -> None:
        cfg = build_inspect_config({"circuit_layers": [10, 20]})
        assert cfg.circuit_layers == (10, 20)

    def test_empty_list_to_empty_tuple(self) -> None:
        cfg = build_inspect_config({"layers_to_plot": []})
        assert cfg.layers_to_plot == ()

    def test_feature_values(self) -> None:
        for feat in [
            "compare",
            "checkpoint",
            "reward_strat",
            "lora",
            "fingerprint",
            "scaffold",
            "probe_save",
        ]:
            cfg = build_inspect_config({"feature": feat})
            assert cfg.feature == feat

    def test_invalid_feature_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid feature"):
            build_inspect_config({"feature": "bad_feature"})

    def test_model_b_path(self) -> None:
        cfg = build_inspect_config({"model_b_path": "/path/b"})
        assert cfg.model_b_path == "/path/b"

    def test_all_optional_features_fields(self) -> None:
        cfg = build_inspect_config(
            {
                "checkpoints_dir": "/ckpts",
                "rollout_log": "rollout.jsonl",
                "lora_path": "/lora",
                "probe_base": "base.npz",
                "probe_wandb_key": "custom/key",
            }
        )
        assert cfg.checkpoints_dir == "/ckpts"
        assert cfg.rollout_log == "rollout.jsonl"
        assert cfg.lora_path == "/lora"
        assert cfg.probe_base == "base.npz"
        assert cfg.probe_wandb_key == "custom/key"

    def test_proj_dim_and_seed(self) -> None:
        cfg = build_inspect_config({"proj_dim": 64, "sketch_seed": 99})
        assert cfg.proj_dim == 64
        assert cfg.sketch_seed == 99

    def test_layers_to_plot_from_tuple(self) -> None:
        cfg = build_inspect_config({"layers_to_plot": (0, 5, 10)})
        assert cfg.layers_to_plot == (0, 5, 10)

    def test_top_k_logit_lens(self) -> None:
        cfg = build_inspect_config({"top_k_logit_lens": 10})
        assert cfg.top_k_logit_lens == 10

    def test_save_dir(self) -> None:
        cfg = build_inspect_config({"save_dir": "/my/out"})
        assert cfg.save_dir == "/my/out"

    def test_non_list_layers_to_plot(self) -> None:
        # Non-list value → empty tuple
        cfg = build_inspect_config({"layers_to_plot": "not_a_list"})
        assert cfg.layers_to_plot == ()
