"""Tests for dynamic_scaling_logger.log_multipliers_to_wandb."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from mlx_grpo.training.dynamic_scaling_logger import log_multipliers_to_wandb


class TestLogMultipliersToWandB:
    """Test W&B logging for per-layer multipliers."""

    def test_logs_per_layer_multipliers(self):
        """Test that per-layer multipliers are logged."""
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            multipliers = [
                2.1,
                0.5,
                3.5,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                1.2,
                1.8,
                2.8,
                3.2,
                0.8,
                1.6,
                2.4,
                3.6,
            ]
            layer_names = [f"L{20 + i}" for i in range(16)]

            log_multipliers_to_wandb(multipliers, layer_names, step=100)

            # Verify wandb.log was called
            mock_wandb.log.assert_called_once()

            # Get the logged dict
            logged_dict = mock_wandb.log.call_args[0][0]

            # Verify each layer is logged
            for name, mult in zip(layer_names, multipliers, strict=False):
                assert f"gradient_multipliers/{name}" in logged_dict
                assert logged_dict[f"gradient_multipliers/{name}"] == mult

    def test_logs_max_multiplier_used(self):
        """Test that max multiplier is logged."""
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            multipliers = [2.1, 0.5, 3.5, 1.0] * 4
            layer_names = [f"L{20 + i}" for i in range(16)]

            log_multipliers_to_wandb(multipliers, layer_names, step=50)

            logged_dict = mock_wandb.log.call_args[0][0]

            # Max should be 3.5
            assert logged_dict["gradient_scaling/max_multiplier_used"] == 3.5

    def test_logs_with_correct_step(self):
        """Test that step parameter is passed to wandb.log."""
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            multipliers = [1.0] * 16
            layer_names = [f"L{20 + i}" for i in range(16)]

            log_multipliers_to_wandb(multipliers, layer_names, step=999)

            # Verify step was passed
            call_kwargs = mock_wandb.log.call_args[1]
            assert call_kwargs["step"] == 999

    def test_empty_multipliers_no_op(self):
        """Test that empty input is no-op."""
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            log_multipliers_to_wandb([], [], step=100)

            # Should not call wandb.log
            mock_wandb.log.assert_not_called()

    def test_empty_names_no_op(self):
        """Test that empty names list is no-op."""
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            log_multipliers_to_wandb([1.0] * 16, [], step=100)

            # Should not call wandb.log
            mock_wandb.log.assert_not_called()

    def test_missing_wandb_no_error(self):
        """Test that missing wandb doesn't raise error."""
        with patch.dict("sys.modules", {"wandb": None}):
            # This should not raise an error
            multipliers = [1.0] * 16
            layer_names = [f"L{20 + i}" for i in range(16)]

            try:
                log_multipliers_to_wandb(multipliers, layer_names, step=100)
            except ImportError:
                pytest.fail("Should handle missing wandb gracefully")

    def test_all_ones_multipliers(self):
        """Test logging when all multipliers are 1.0."""
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            multipliers = [1.0] * 16
            layer_names = [f"L{20 + i}" for i in range(16)]

            log_multipliers_to_wandb(multipliers, layer_names, step=100)

            logged_dict = mock_wandb.log.call_args[0][0]

            # All should be 1.0
            for name, _mult in zip(layer_names, multipliers, strict=False):
                assert logged_dict[f"gradient_multipliers/{name}"] == 1.0

            # Max should be 1.0
            assert logged_dict["gradient_scaling/max_multiplier_used"] == 1.0

    def test_varied_multipliers(self):
        """Test logging with varied multiplier values."""
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            multipliers = [
                0.1,
                0.5,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                2.5,
                1.5,
                3.5,
                0.8,
                1.2,
                2.2,
                3.2,
                4.2,
                4.8,
            ]
            layer_names = [f"L{20 + i}" for i in range(16)]

            log_multipliers_to_wandb(multipliers, layer_names, step=200)

            logged_dict = mock_wandb.log.call_args[0][0]

            # Check max
            assert logged_dict["gradient_scaling/max_multiplier_used"] == 5.0

            # Check first and last
            assert logged_dict["gradient_multipliers/L20"] == 0.1
            assert logged_dict["gradient_multipliers/L35"] == 4.8

    def test_exact_16_layers(self):
        """Test with exactly 16 layers (L20-L35)."""
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            multipliers = [float(i) for i in range(1, 17)]
            layer_names = [f"L{20 + i}" for i in range(16)]

            log_multipliers_to_wandb(multipliers, layer_names, step=100)

            logged_dict = mock_wandb.log.call_args[0][0]

            # All 16 layers should be logged
            for i in range(16):
                name = f"L{20 + i}"
                assert f"gradient_multipliers/{name}" in logged_dict
                assert logged_dict[f"gradient_multipliers/{name}"] == float(i + 1)

    def test_min_max_multipliers(self):
        """Test logging with min and max multiplier values."""
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            multipliers = [0.1] * 8 + [5.0] * 8  # Min and max values
            layer_names = [f"L{20 + i}" for i in range(16)]

            log_multipliers_to_wandb(multipliers, layer_names, step=100)

            logged_dict = mock_wandb.log.call_args[0][0]

            assert logged_dict["gradient_scaling/max_multiplier_used"] == 5.0

    def test_step_zero(self):
        """Test logging at step 0."""
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            multipliers = [1.0] * 16
            layer_names = [f"L{20 + i}" for i in range(16)]

            log_multipliers_to_wandb(multipliers, layer_names, step=0)

            call_kwargs = mock_wandb.log.call_args[1]
            assert call_kwargs["step"] == 0

    def test_step_large_value(self):
        """Test logging with large step value."""
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            multipliers = [1.0] * 16
            layer_names = [f"L{20 + i}" for i in range(16)]

            log_multipliers_to_wandb(multipliers, layer_names, step=1000000)

            call_kwargs = mock_wandb.log.call_args[1]
            assert call_kwargs["step"] == 1000000

    def test_float_precision_preserved(self):
        """Test that float precision is preserved in logging."""
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            multipliers = [2.123456, 0.456789, 3.141592] + [1.0] * 13
            layer_names = [f"L{20 + i}" for i in range(16)]

            log_multipliers_to_wandb(multipliers, layer_names, step=100)

            logged_dict = mock_wandb.log.call_args[0][0]

            assert logged_dict["gradient_multipliers/L20"] == 2.123456
            assert logged_dict["gradient_multipliers/L21"] == 0.456789
            assert logged_dict["gradient_multipliers/L22"] == 3.141592

    def test_log_dict_structure(self):
        """Test the overall structure of logged dict."""
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            multipliers = [1.5] * 16
            layer_names = [f"L{20 + i}" for i in range(16)]

            log_multipliers_to_wandb(multipliers, layer_names, step=100)

            logged_dict = mock_wandb.log.call_args[0][0]

            # Should have 16 per-layer + 1 aggregate
            assert len(logged_dict) == 17

            # All keys should have correct format
            per_layer_keys = [k for k in logged_dict if k.startswith("gradient_multipliers/")]
            assert len(per_layer_keys) == 16

            # Should have one aggregate key
            assert "gradient_scaling/max_multiplier_used" in logged_dict
