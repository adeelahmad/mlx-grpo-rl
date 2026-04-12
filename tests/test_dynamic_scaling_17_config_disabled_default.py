"""Tests that dynamic_gradient_scaling is disabled by default."""

from __future__ import annotations

import tempfile
from pathlib import Path

from mlx_grpo.config.dynamic_scaling import DynamicGradientScalingConfig
from mlx_grpo.config.loader import load_config


class TestConfigDisabledByDefault:
    """Test that scaling is disabled by default for backward compatibility."""

    def test_default_enabled_is_false(self):
        """Test that default enabled value is False."""
        config = DynamicGradientScalingConfig()
        assert config.enabled is False

    def test_minimal_config_disabled(self):
        """Test minimal config has scaling disabled."""
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
        finally:
            Path(config_path).unlink()

    def test_explicit_disable(self):
        """Test explicitly setting enabled to False."""
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

    def test_empty_section_defaults(self):
        """Test that empty section uses all defaults."""
        yaml_content = """
training:
  model: gpt2
  batch_size: 8
  dynamic_gradient_scaling: {}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_config(Path(config_path)).training
            # All defaults
            assert config.dynamic_gradient_scaling.enabled is False
            assert config.dynamic_gradient_scaling.angle_drift_threshold == 0.005
            assert config.dynamic_gradient_scaling.max_multiplier == 5.0
            assert config.dynamic_gradient_scaling.min_multiplier == 0.1
            assert config.dynamic_gradient_scaling.ema_alpha == 0.25
        finally:
            Path(config_path).unlink()

    def test_backward_compatibility_no_section(self):
        """Test that old configs without dynamic_gradient_scaling still work."""
        yaml_content = """
training:
  model: gpt2
  batch_size: 8
  learning_rate: 1e-4
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_config(Path(config_path)).training
            # Should have default config, disabled
            assert config.dynamic_gradient_scaling.enabled is False
            # Should load other fields
            assert config.learning_rate == 1e-4
        finally:
            Path(config_path).unlink()

    def test_no_training_data_loss(self):
        """Test that absence of scaling doesn't affect other training config."""
        yaml_content = """
training:
  model: gpt2
  batch_size: 16
  learning_rate: 5e-5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_config(Path(config_path)).training
            assert config.batch_size == 16
            assert config.learning_rate == 5e-5
            # Scaling disabled by default
            assert config.dynamic_gradient_scaling.enabled is False
        finally:
            Path(config_path).unlink()

    def test_all_defaults_independently(self):
        """Test each default value independently."""
        config = DynamicGradientScalingConfig()

        assert config.enabled is False
        assert config.angle_drift_threshold == 0.005
        assert config.max_multiplier == 5.0
        assert config.min_multiplier == 0.1
        assert config.ema_alpha == 0.25
        assert config.update_interval == 1

    def test_dataclass_field_defaults(self):
        """Test that dataclass field defaults match spec."""

        config = DynamicGradientScalingConfig()

        # Check critical defaults
        assert config.enabled is False  # Most important for backward compat
