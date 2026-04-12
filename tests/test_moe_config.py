"""Tests for MoE configuration parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mlx_grpo.config.loader import (
    ExpertTypeMappingConfig,
    GateConfig,
    MoEConfig,
    RoutingConfig,
    load_config,
)


def _write_yaml(data: dict, tmp_path: Path) -> Path:
    p = tmp_path / "config.yml"
    p.write_text(yaml.dump(data))
    return p


@pytest.fixture
def minimal_yaml(tmp_path):
    """Minimal valid YAML that load_config can parse."""
    return {
        "training": {"model_path": "test-model", "iterations": 10},
        "generation": {"backend": "mlx"},
        "datasets": {},
    }


def test_config_without_moe(tmp_path, minimal_yaml):
    """Config without moe: block → moe field is None."""
    cfg = load_config(_write_yaml(minimal_yaml, tmp_path))
    assert cfg.moe is None


def test_config_with_minimal_moe(tmp_path, minimal_yaml):
    """Minimal moe: block uses all defaults."""
    minimal_yaml["moe"] = {"num_experts": 2}
    cfg = load_config(_write_yaml(minimal_yaml, tmp_path))
    assert cfg.moe is not None
    assert cfg.moe.num_experts == 2
    assert cfg.moe.lora_rank == 8  # default
    assert cfg.moe.lora_alpha == 16.0  # default
    assert isinstance(cfg.moe.gate, GateConfig)
    assert isinstance(cfg.moe.routing, RoutingConfig)
    assert cfg.moe.expert_types == ()


def test_config_gate_nested(tmp_path, minimal_yaml):
    """Nested gate config parses correctly."""
    minimal_yaml["moe"] = {
        "num_experts": 3,
        "gate": {
            "hidden_dim": 128,
            "pool_method": "first",
            "temperature": 2.0,
            "temperature_min": 0.2,
            "temperature_decay_steps": 300,
        },
    }
    cfg = load_config(_write_yaml(minimal_yaml, tmp_path))
    gate = cfg.moe.gate
    assert gate.hidden_dim == 128
    assert gate.pool_method == "first"
    assert gate.temperature == 2.0
    assert gate.temperature_min == 0.2
    assert gate.temperature_decay_steps == 300


def test_config_routing_nested(tmp_path, minimal_yaml):
    """Nested routing config parses correctly."""
    minimal_yaml["moe"] = {
        "num_experts": 3,
        "routing": {
            "supervised_steps": 200,
            "transition_steps": 50,
            "balance_loss_weight": 0.05,
        },
    }
    cfg = load_config(_write_yaml(minimal_yaml, tmp_path))
    r = cfg.moe.routing
    assert r.supervised_steps == 200
    assert r.transition_steps == 50
    assert r.balance_loss_weight == 0.05


def test_config_expert_types(tmp_path, minimal_yaml):
    """expert_types list parses into ExpertTypeMappingConfig tuples."""
    minimal_yaml["moe"] = {
        "num_experts": 3,
        "expert_types": [
            {"type_name": "mcq", "expert_index": 0, "lora_rank": 4},
            {"type_name": "math", "expert_index": 1, "lora_alpha": 32.0},
            {"type_name": "general_qna", "expert_index": 2},
        ],
    }
    cfg = load_config(_write_yaml(minimal_yaml, tmp_path))
    et = cfg.moe.expert_types
    assert len(et) == 3
    assert et[0].type_name == "mcq"
    assert et[0].expert_index == 0
    assert et[0].lora_rank == 4
    assert et[0].lora_alpha is None
    assert et[1].type_name == "math"
    assert et[1].lora_alpha == 32.0
    assert et[2].type_name == "general_qna"
    assert et[2].lora_rank is None


def test_config_lora_targets_as_list(tmp_path, minimal_yaml):
    """lora_targets list in YAML → tuple in dataclass."""
    minimal_yaml["moe"] = {
        "num_experts": 2,
        "lora_targets": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    }
    cfg = load_config(_write_yaml(minimal_yaml, tmp_path))
    assert isinstance(cfg.moe.lora_targets, tuple)
    assert len(cfg.moe.lora_targets) == 3


def test_moe_config_dataclass_defaults():
    """MoEConfig defaults are sane."""
    m = MoEConfig()
    assert m.num_experts == 3
    assert m.lora_rank == 8
    assert m.lora_alpha == 16.0
    assert m.lora_dropout == 0.0
    assert m.lora_targets == ("self_attn.q_proj", "self_attn.v_proj")
    assert m.lora_layers is None
    assert isinstance(m.gate, GateConfig)
    assert isinstance(m.routing, RoutingConfig)


def test_gate_config_defaults():
    g = GateConfig()
    assert g.hidden_dim == 64
    assert g.pool_method == "mean"
    assert g.temperature == 1.0
    assert g.temperature_min == 0.1
    assert g.noise_std == 0.0


def test_routing_config_defaults():
    r = RoutingConfig()
    assert r.supervised_steps == 100
    assert r.transition_steps == 100
    assert r.supervised_loss_weight == 1.0
    assert r.balance_loss_weight == 0.01


def test_expert_type_mapping_defaults():
    e = ExpertTypeMappingConfig()
    assert e.type_name == ""
    assert e.expert_index == 0
    assert e.lora_rank is None
    assert e.lora_alpha is None
