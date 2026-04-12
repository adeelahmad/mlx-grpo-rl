"""Tests for PyTorch adapter fusion — mirrors MLX behavior."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from mlx_grpo.training.pytorch_adapter_migration import (
    _compute_delta,
    _get_float_weight,
    _get_submodule,
    _set_float_weight,
    fuse_old_adapter,
)


class TestGetSubmodule:
    """Test dotted path resolution."""

    def test_simple_path(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
        result = _get_submodule(model, "0")
        assert isinstance(result, nn.Linear)

    def test_nested_path(self):
        model = nn.Sequential(
            nn.Sequential(nn.Linear(10, 10)),
            nn.Linear(10, 5),
        )
        result = _get_submodule(model, "0.0")
        assert isinstance(result, nn.Linear)

    def test_invalid_path(self):
        model = nn.Linear(10, 10)
        result = _get_submodule(model, "nonexistent.path")
        assert result is None


class TestComputeDelta:
    """Test LoRA convention detection and delta computation."""

    def test_mlx_convention(self):
        """mlx_lm: A(in, rank), B(rank, out) -> delta = B.T @ A.T"""
        lora_a = torch.randn(10, 2)  # (in, rank)
        lora_b = torch.randn(2, 5)  # (rank, out)
        scale = 2.0

        delta = _compute_delta(lora_a, lora_b, scale, "test")
        assert delta is not None
        assert delta.shape == (5, 10)  # (out, in)

        # Verify computation: scale * (B.T @ A.T)
        expected = scale * (lora_b.T @ lora_a.T)
        assert torch.allclose(delta, expected)

    def test_custom_convention(self):
        """Custom DoRA: A(rank, in), B(out, rank) -> delta = B @ A"""
        lora_a = torch.randn(2, 10)  # (rank, in)
        lora_b = torch.randn(5, 2)  # (out, rank)
        scale = 2.0

        delta = _compute_delta(lora_a, lora_b, scale, "test")
        assert delta is not None
        assert delta.shape == (5, 10)  # (out, in)

        # Verify computation: scale * (B @ A)
        expected = scale * (lora_b @ lora_a)
        assert torch.allclose(delta, expected)

    def test_ambiguous_convention(self):
        """Ambiguous shapes should return None."""
        lora_a = torch.randn(10, 5)
        lora_b = torch.randn(3, 2)  # No matching rank dimension
        delta = _compute_delta(lora_a, lora_b, 1.0, "test")
        assert delta is None


class TestGetFloatWeight:
    """Test weight extraction from LoRA layers."""

    def test_simple_linear_weight(self):
        """Extract weight from plain linear layer."""
        layer = nn.Linear(10, 5)
        w_float, inner, is_quantized = _get_float_weight(layer)

        assert w_float is not None
        assert inner is layer
        assert is_quantized is False
        assert w_float.shape == (5, 10)

    def test_lora_wrapper(self):
        """Extract weight from LoRA wrapper (has base_layer attribute)."""
        inner_linear = nn.Linear(10, 5)
        wrapper = Mock()
        wrapper.base_layer = inner_linear
        wrapper.weight = inner_linear.weight

        w_float, inner, is_quantized = _get_float_weight(wrapper)
        assert w_float is not None
        assert inner is inner_linear
        assert is_quantized is False

    def test_missing_weight(self):
        """Return None if weight attribute missing."""
        layer = Mock(spec=[])
        w_float, inner, is_quantized = _get_float_weight(layer)
        assert w_float is None
        assert inner is None
        assert is_quantized is False


class TestSetFloatWeight:
    """Test weight writing."""

    def test_write_plain_float(self):
        """Write weight back to plain linear layer."""
        layer = nn.Linear(10, 5)
        original_device = layer.weight.device
        new_weight = torch.randn(5, 10)

        result = _set_float_weight(layer, new_weight, is_quantized=False)
        assert result is True
        assert layer.weight.device == original_device
        # Shape should match
        assert layer.weight.shape == new_weight.shape

    def test_write_to_missing_weight(self):
        """Return False if no weight attribute."""
        layer = Mock(spec=[])
        new_weight = torch.randn(5, 10)
        result = _set_float_weight(layer, new_weight, is_quantized=False)
        assert result is False


class TestFuseOldAdapter:
    """Test full adapter fusion pipeline."""

    def test_fuse_single_layer(self):
        """Fuse a single LoRA layer into base weight."""
        # Create a simple model with one LoRA-style linear layer
        model = nn.Linear(10, 5)

        # Create checkpoint weights
        checkpoint_weights = {
            "weight.lora_a": torch.randn(10, 2),  # (in, rank)
            "weight.lora_b": torch.randn(2, 5),  # (rank, out)
        }

        # Setup adapter config
        old_adapter_config = {
            "lora_parameters": {
                "scale": 0.5,
            }
        }

        # Manually create the structure expected by _get_submodule
        # We'll patch it to handle the flat "weight" case
        def mock_get_submodule(m, path):
            if path == "weight":
                return model
            return None

        with patch(
            "mlx_grpo.training.pytorch_adapter_migration._get_submodule", mock_get_submodule
        ):
            fused = fuse_old_adapter(
                model,
                checkpoint_weights,
                old_adapter_config,
                dropped_keys=set(),
            )

        assert fused > 0

    def test_fuse_with_dora_magnitude(self):
        """Fuse DoRA weights with magnitude."""
        model = nn.Linear(10, 5)

        # LoRA weights
        lora_a = torch.randn(10, 2)
        lora_b = torch.randn(2, 5)
        # DoRA magnitude per row
        magnitude = torch.ones(5)

        checkpoint_weights = {
            "weight.lora_a": lora_a,
            "weight.lora_b": lora_b,
            "weight.magnitude": magnitude,
        }

        old_adapter_config = {"lora_parameters": {"scale": 1.0}}

        def mock_get_submodule(m, path):
            return model if path == "weight" else None

        with patch(
            "mlx_grpo.training.pytorch_adapter_migration._get_submodule", mock_get_submodule
        ):
            fused = fuse_old_adapter(
                model,
                checkpoint_weights,
                old_adapter_config,
                dropped_keys={"weight.magnitude"},
            )

        # Should fuse because magnitude was dropped (DoRA -> LoRA scenario)
        assert fused > 0

    def test_fuse_skips_missing_layers(self):
        """Skip fusing when layer path not found."""
        model = nn.Linear(10, 5)

        checkpoint_weights = {
            "nonexistent_layer.lora_a": torch.randn(10, 2),
            "nonexistent_layer.lora_b": torch.randn(2, 5),
        }

        old_adapter_config = {"lora_parameters": {"scale": 1.0}}

        fused = fuse_old_adapter(
            model,
            checkpoint_weights,
            old_adapter_config,
            dropped_keys=set(),
        )

        assert fused == 0  # Nothing fused


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFuseGPU:
    """GPU-specific tests."""

    def test_device_handling(self):
        """Tensors moved to model device."""
        device = torch.device("cuda:0")
        model = nn.Linear(10, 5).to(device)

        # Checkpoint weights on CPU
        checkpoint_weights = {
            "weight.lora_a": torch.randn(10, 2),  # CPU
            "weight.lora_b": torch.randn(2, 5),  # CPU
        }

        old_adapter_config = {"lora_parameters": {"scale": 1.0}}

        def mock_get_submodule(m, path):
            return model if path == "weight" else None

        with patch(
            "mlx_grpo.training.pytorch_adapter_migration._get_submodule", mock_get_submodule
        ):
            fused = fuse_old_adapter(
                model,
                checkpoint_weights,
                old_adapter_config,
                dropped_keys=set(),
            )

        assert fused > 0
