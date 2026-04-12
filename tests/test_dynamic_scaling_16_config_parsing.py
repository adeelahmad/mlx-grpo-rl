"""Tests for loading dynamic_gradient_scaling config from YAML."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mlx_grpo.config.dynamic_scaling import DynamicGradientScalingConfig
from mlx_grpo.config.loader import load_config


class TestConfigParsingDynamicScaling:
    """Test parsing YAML with dynamic_gradient_scaling section."""

    def test_load_with_enabled_true(self):
        """Test loading config with dynamic scaling enabled."""
        yaml_content = """
training:
  model: gpt2
  batch_size: 8
  dynamic_gradient_scaling:
    enabled: true
    angle_drift_threshold: 0.01
    max_multiplier: 5.0
    min_multiplier: 0.1
    ema_alpha: 0.25
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_config(Path(config_path)).training
            assert config.dynamic_gradient_scaling.enabled is True
            assert config.dynamic_gradient_scaling.angle_drift_threshold == 0.01
            assert config.dynamic_gradient_scaling.max_multiplier == 5.0
            assert config.dynamic_gradient_scaling.min_multiplier == 0.1
            assert config.dynamic_gradient_scaling.ema_alpha == 0.25
        finally:
            Path(config_path).unlink()

    def test_load_with_enabled_false(self):
        """Test loading config with dynamic scaling disabled."""
        yaml_content = """
training:
  model: gpt2
  batch_size: 8
  dynamic_gradient_scaling:
    enabled: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_config(Path(config_path)).training
            assert config.dynamic_gradient_scaling.enabled is False
        finally:
            Path(config_path).unlink()

    def test_load_defaults_when_missing(self):
        """Test that defaults are used when section is missing."""
        yaml_content = """
training:
  model: gpt2
  batch_size: 8
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_config(Path(config_path)).training
            assert config.dynamic_gradient_scaling.enabled is False
            assert config.dynamic_gradient_scaling.angle_drift_threshold == 0.005
            assert config.dynamic_gradient_scaling.max_multiplier == 5.0
            assert config.dynamic_gradient_scaling.min_multiplier == 0.1
            assert config.dynamic_gradient_scaling.ema_alpha == 0.25
        finally:
            Path(config_path).unlink()

    def test_load_partial_override(self):
        """Test that specified values override defaults."""
        yaml_content = """
training:
  model: gpt2
  batch_size: 8
  dynamic_gradient_scaling:
    enabled: true
    max_multiplier: 10.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_config(Path(config_path)).training
            assert config.dynamic_gradient_scaling.enabled is True
            assert config.dynamic_gradient_scaling.max_multiplier == 10.0
            assert config.dynamic_gradient_scaling.min_multiplier == 0.1  # Default
            assert config.dynamic_gradient_scaling.ema_alpha == 0.25  # Default
        finally:
            Path(config_path).unlink()

    def test_config_dataclass_frozen(self):
        """Test that DynamicGradientScalingConfig is frozen."""
        config = DynamicGradientScalingConfig(enabled=True)

        with pytest.raises(AttributeError):
            config.enabled = False

    def test_config_dataclass_has_all_fields(self):
        """Test that config dataclass has all required fields."""
        config = DynamicGradientScalingConfig()

        assert hasattr(config, "enabled")
        assert hasattr(config, "angle_drift_threshold")
        assert hasattr(config, "max_multiplier")
        assert hasattr(config, "min_multiplier")
        assert hasattr(config, "ema_alpha")
        assert hasattr(config, "update_interval")

    def test_config_in_training_config(self):
        """Test that DynamicGradientScalingConfig is part of TrainingConfig."""
        yaml_content = """
training:
  model: gpt2
  batch_size: 8
  dynamic_gradient_scaling:
    enabled: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_config(Path(config_path)).training
            assert isinstance(config.dynamic_gradient_scaling, DynamicGradientScalingConfig)
        finally:
            Path(config_path).unlink()

    def test_load_all_custom_values(self):
        """Test loading with all custom values."""
        yaml_content = """
training:
  model: gpt2
  batch_size: 8
  dynamic_gradient_scaling:
    enabled: true
    angle_drift_threshold: 0.001
    max_multiplier: 3.0
    min_multiplier: 0.2
    ema_alpha: 0.1
    update_interval: 1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_config(Path(config_path)).training
            assert config.dynamic_gradient_scaling.enabled is True
            assert config.dynamic_gradient_scaling.angle_drift_threshold == 0.001
            assert config.dynamic_gradient_scaling.max_multiplier == 3.0
            assert config.dynamic_gradient_scaling.min_multiplier == 0.2
            assert config.dynamic_gradient_scaling.ema_alpha == 0.1
            assert config.dynamic_gradient_scaling.update_interval == 1
        finally:
            Path(config_path).unlink()

    def test_top_level_dynamic_gradient_scaling(self):
        """Test loading when dynamic_gradient_scaling is at top level."""
        yaml_content = """
model: gpt2
batch_size: 8
dynamic_gradient_scaling:
  enabled: true
  angle_drift_threshold: 0.005
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_config(Path(config_path)).training
            assert config.dynamic_gradient_scaling.enabled is True
            assert config.dynamic_gradient_scaling.angle_drift_threshold == 0.005
        finally:
            Path(config_path).unlink()

    def test_nested_under_training(self):
        """Test that nested under training takes precedence."""
        yaml_content = """
dynamic_gradient_scaling:
  enabled: false
training:
  model: gpt2
  batch_size: 8
  dynamic_gradient_scaling:
    enabled: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_config(Path(config_path)).training
            # Should use the one from training section
            assert config.dynamic_gradient_scaling.enabled is True
        finally:
            Path(config_path).unlink()

    def test_update_interval_field_present(self):
        """Test that update_interval field is in config."""
        yaml_content = """
training:
  model: gpt2
  batch_size: 8
  dynamic_gradient_scaling:
    enabled: true
    update_interval: 2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_config(Path(config_path)).training
            assert config.dynamic_gradient_scaling.update_interval == 2
        finally:
            Path(config_path).unlink()

    def test_default_update_interval_is_one(self):
        """Test that default update_interval is 1."""
        config = DynamicGradientScalingConfig()
        assert config.update_interval == 1
